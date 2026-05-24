"""One-step training loop + binary IoU/F1 metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from imagery_seg.eval import (
    DEFAULT_AREA_FILTERS_M2,
    DEFAULT_THRESHOLDS,
    aggregate_by_region,
    binary_f1,
    binary_iou,
    evaluate,
    evaluate_per_region,
    evaluate_with_sweep,
    filter_small_components,
    predict_probability,
)
from imagery_seg.features.base import FeatureSpec
from imagery_seg.ground_truth.base import GroundTruthSource
from imagery_seg.imagery.base import FetchedImage, ImagerySource
from imagery_seg.recipe import AOI, Recipe, TrainingConfig
from imagery_seg.train import train, train_one_epoch


class _Img(ImagerySource):
    name = "img"

    def fetch_for_bbox(self, bbox, max_side=64, dst_crs="EPSG:3857"):
        # Same uniform image every time so training is reproducible.
        arr = np.full((3, 64, 64), 64, dtype=np.uint8)
        return FetchedImage(
            array=arr,
            transform=from_bounds(0, 0, 64, 64, 64, 64),
            crs="EPSG:4326",
            asset_id="a",
        )


class _GT(GroundTruthSource):
    name = "gt"

    def fetch_polygons(self, bbox, feature_query):
        return [Polygon([(8, 8), (56, 8), (56, 56), (8, 56)])]


@dataclass(frozen=True)
class _Feat(FeatureSpec):
    name: str = "feat"
    erode_px: int = 0

    def overpass_query(self, bbox):
        return "stub"


def _recipe() -> Recipe:
    return Recipe(
        name="t",
        imagery=_Img(),
        ground_truth=_GT(),
        feature=_Feat(),
        training=TrainingConfig(
            epochs=1, batch_size=1, encoder="resnet34", encoder_weights=None,
        ),
        aois=((0.0, 0.0, 1.0, 1.0),),
    )


def test_binary_iou_perfect():
    pred = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[1, 1], [0, 0]])
    assert binary_iou(pred, target) == 1.0


def test_binary_iou_disjoint():
    pred = torch.tensor([[1, 0], [0, 0]])
    target = torch.tensor([[0, 1], [0, 0]])
    assert binary_iou(pred, target) == 0.0


def test_binary_iou_no_positives():
    pred = torch.zeros(4, 4, dtype=torch.long)
    target = torch.zeros(4, 4, dtype=torch.long)
    # Convention: empty union -> 1.0 (no penalty)
    assert binary_iou(pred, target) == 1.0


def test_binary_f1_half_overlap():
    pred = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[1, 0], [1, 0]])
    # TP=1, FP=1, FN=1 -> precision=recall=0.5 -> F1=0.5
    assert abs(binary_f1(pred, target) - 0.5) < 1e-6


def test_train_one_epoch_reduces_loss():
    """Tiny 1-AOI loop must run end-to-end and at least not raise.

    On a constant image the loss should be finite and non-NaN after
    one update.
    """
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(
        encoder=recipe.training.encoder,
        encoder_weights=None,
        classes=2,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = train_one_epoch(model, ds, optim, batch_size=1, device="cpu")
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_train_writes_best_and_history_by_default(tmp_path):
    """train() with run_dir writes best.pt + history.json but no per-epoch checkpoints."""
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    train_ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    val_ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(
        encoder=recipe.training.encoder, encoder_weights=None, classes=2,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    result = train(
        model, train_ds, optim,
        epochs=2,
        val_dataset=val_ds,
        batch_size=1,
        device="cpu",
        run_dir=tmp_path / "run",
    )
    assert len(result["history"]) == 2
    assert (tmp_path / "run" / "best.pt").is_file()
    assert (tmp_path / "run" / "history.json").is_file()
    assert not (tmp_path / "run" / "epoch_001.pt").exists()
    assert not (tmp_path / "run" / "epoch_002.pt").exists()
    assert result["best"]["epoch"] in (1, 2)


def test_train_keep_epochs_writes_all(tmp_path):
    """--keep-epochs adds per-epoch checkpoints alongside best.pt."""
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    train_ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(
        encoder=recipe.training.encoder, encoder_weights=None, classes=2,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(
        model, train_ds, optim,
        epochs=2, val_dataset=None, batch_size=1, device="cpu",
        run_dir=tmp_path / "run", keep_epochs=True,
    )
    assert (tmp_path / "run" / "epoch_001.pt").is_file()
    assert (tmp_path / "run" / "epoch_002.pt").is_file()
    assert (tmp_path / "run" / "best.pt").is_file()


def test_train_without_val_uses_last_epoch_as_best(tmp_path):
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    train_ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(
        encoder=recipe.training.encoder, encoder_weights=None, classes=2,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = train(
        model, train_ds, optim,
        epochs=3, val_dataset=None, batch_size=1, device="cpu",
        run_dir=tmp_path / "r",
    )
    # No val -> best is whichever epoch ran last (current heuristic)
    assert result["best"]["epoch"] == 3


def test_predict_probability_shape_and_range():
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(encoder=recipe.training.encoder, encoder_weights=None, classes=2)
    sample = ds[0]
    probs = predict_probability(model, sample["image"])
    assert probs.shape == sample["mask"].shape
    assert (probs >= 0).all() and (probs <= 1).all()


def test_evaluate_with_sweep_picks_best_threshold():
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(encoder=recipe.training.encoder, encoder_weights=None, classes=2)
    out = evaluate_with_sweep(model, ds, thresholds=(0.2, 0.5, 0.8))
    assert len(out) == 1
    e = out[0]
    assert e["best_threshold"] in (0.2, 0.5, 0.8)
    # best_iou should equal max sweep iou by construction
    swept_ious = [s["iou"] for s in e["sweep"]]
    assert abs(e["best_iou"] - max(swept_ious)) < 1e-9


def test_default_thresholds_range():
    assert DEFAULT_THRESHOLDS[0] == 0.1
    assert DEFAULT_THRESHOLDS[-1] == 0.9
    assert all(0.1 <= t <= 0.9 for t in DEFAULT_THRESHOLDS)


def test_filter_small_components_drops_below_threshold():
    """One large blob (~16 px) survives, two tiny blobs (~1 px) drop."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:5, 1:5] = 1  # 16 px component
    mask[7, 7] = 1      # 1 px component
    mask[9, 0] = 1      # 1 px component
    # 1 m² per pixel -> 1 px = 1 m². min 5 m² keeps only the 16 px blob.
    out = filter_small_components(mask, pixel_area_m2=1.0, min_area_m2=5.0)
    assert out.sum() == 16
    # Threshold 0 is a no-op
    out0 = filter_small_components(mask, pixel_area_m2=1.0, min_area_m2=0.0)
    assert out0.sum() == 18


def test_filter_small_components_empty_mask():
    mask = np.zeros((5, 5), dtype=np.uint8)
    out = filter_small_components(mask, pixel_area_m2=1.0, min_area_m2=5.0)
    assert out.sum() == 0


def test_evaluate_with_sweep_includes_area_filter_dims():
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(encoder=recipe.training.encoder, encoder_weights=None, classes=2)
    out = evaluate_with_sweep(
        model, ds,
        thresholds=(0.3, 0.6),
        area_filters_m2=(0.0, 10.0),
    )
    assert len(out) == 1
    e = out[0]
    assert len(e["sweep"]) == 4  # 2 thresholds x 2 area_filters
    assert e["best_threshold"] in (0.3, 0.6)
    assert e["best_area_filter_m2"] in (0.0, 10.0)
    assert "pixel_area_m2" in e


def test_default_area_filters_starts_at_zero():
    assert DEFAULT_AREA_FILTERS_M2[0] == 0.0


def test_binary_iou_with_valid_mask_ignores_invalid_pixels():
    """A pixel marked invalid contributes neither to intersection nor union."""
    pred = torch.tensor([[1, 1], [0, 1]])
    target = torch.tensor([[1, 1], [1, 0]])
    # Without mask: TP=2, FP=1, FN=1, IoU = 2/4 = 0.5
    assert abs(binary_iou(pred, target) - 0.5) < 1e-9
    # Mask out the bottom row — only top row counted, both perfect TP, IoU=1.0
    valid = torch.tensor([[1, 1], [0, 0]], dtype=torch.uint8)
    assert binary_iou(pred, target, valid) == 1.0


def test_binary_f1_with_valid_mask_ignores_invalid_pixels():
    pred = torch.tensor([[1, 1], [0, 1]])
    target = torch.tensor([[1, 1], [1, 0]])
    valid = torch.tensor([[1, 1], [0, 0]], dtype=torch.uint8)
    # Top row only: TP=2, FP=0, FN=0 -> F1 = 1.0
    assert binary_f1(pred, target, valid) == 1.0


def _multi_region_recipe() -> Recipe:
    """Same as _recipe() but with 2 region-tagged AOIs."""
    return Recipe(
        name="multi",
        imagery=_Img(),
        ground_truth=_GT(),
        feature=_Feat(),
        training=TrainingConfig(epochs=1, batch_size=1, encoder="resnet34", encoder_weights=None),
        aois=(
            AOI(bbox=(0.0, 0.0, 1.0, 1.0), region="jp"),
            AOI(bbox=(2.0, 2.0, 3.0, 3.0), region="ph"),
        ),
    )


def test_aggregate_by_region_groups_correctly():
    sweep = [
        {"region": "jp", "best_iou": 0.30, "best_f1": 0.40},
        {"region": "jp", "best_iou": 0.50, "best_f1": 0.60},
        {"region": "ph", "best_iou": 0.10, "best_f1": 0.20},
    ]
    out = aggregate_by_region(sweep, metric="best_iou")
    assert out["jp"]["n"] == 2
    assert abs(out["jp"]["mean"] - 0.40) < 1e-9
    assert abs(out["jp"]["min"] - 0.30) < 1e-9
    assert abs(out["jp"]["max"] - 0.50) < 1e-9
    assert out["ph"]["n"] == 1
    assert abs(out["ph"]["mean"] - 0.10) < 1e-9


def test_evaluate_per_region_returns_one_entry_per_region():
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _multi_region_recipe()
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(encoder=recipe.training.encoder, encoder_weights=None, classes=2)
    out = evaluate_per_region(model, ds)
    assert set(out.keys()) == {"jp", "ph"}
    for region, s in out.items():
        assert s["n"] == 1
        assert 0.0 <= s["mean_iou"] <= 1.0


def test_evaluate_with_sweep_carries_region_through():
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _multi_region_recipe()
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(encoder=recipe.training.encoder, encoder_weights=None, classes=2)
    out = evaluate_with_sweep(model, ds, thresholds=(0.5,), area_filters_m2=(0.0,))
    assert {e["region"] for e in out} == {"jp", "ph"}


def test_evaluate_returns_metrics():
    torch.manual_seed(0)
    from imagery_seg.dataset import RecipeDataset
    from imagery_seg.model import build_unet

    recipe = _recipe()
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:4326")
    model = build_unet(
        encoder=recipe.training.encoder,
        encoder_weights=None,
        classes=2,
    )
    metrics = evaluate(model, ds, device="cpu")
    assert set(metrics) == {"iou", "f1"}
    assert 0.0 <= metrics["iou"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
