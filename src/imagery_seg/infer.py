"""Single-AOI inference: vectorise prediction + render visual diagnostics.

`imagery-seg infer` wraps this. The output of one inference is a directory
containing:

  imagery.png         the raw (post-reproject) RGB chip
  comparison.png      2x2 panel: imagery / GT overlay / pred overlay /
                      confusion (TP green, FP blue, FN red)
  prediction.geojson  vectorised prediction polygons in WGS84
  metadata.json       AOI bbox, region, notes + threshold/area_filter
                      + IoU/F1 + polygon counts

The confusion panel is the one to read when diagnosing a poor IoU:
  - mostly red  → model misses real buildings (under-detection)
  - mostly blue → model hallucinates buildings (over-detection)
  - mixed       → boundary misalignment or class confusion
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
import rasterio.features
import torch
from PIL import Image
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from .aoi_cache import AOICache
from .dataset import RecipeDataset
from .eval import binary_f1, binary_iou, filter_small_components, predict_probability
from .recipe import Recipe

logger = logging.getLogger("imagery_seg.infer")


def run_inference(
    model: torch.nn.Module,
    recipe: Recipe,
    aoi_index: int,
    *,
    threshold: float = 0.5,
    area_filter_m2: float = 0.0,
    out_dir: Path,
    cache: AOICache | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run model on one AOI and emit visualisation + GeoJSON. Returns metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    single = dc_replace(recipe, aois=(recipe.aois[aoi_index],), val_regions=())
    ds = RecipeDataset(
        single, max_side=recipe.training.max_side,
        dst_crs="EPSG:3857", cache=cache, augment=False,
    )
    sample = ds[0]
    image_t = sample["image"]
    mask_t = sample["mask"]
    valid_t = sample.get("valid_mask")
    transform = sample["transform"]
    crs = sample["crs"]
    pixel_area = float(sample["pixel_area_m2"])

    probs = predict_probability(model, image_t, device=device)
    pred_bin = (probs >= threshold).long().numpy().astype(np.uint8)
    if area_filter_m2 > 0:
        pred_bin = filter_small_components(
            pred_bin, pixel_area_m2=pixel_area, min_area_m2=area_filter_m2,
        )
    # Zero out predictions in no-data regions — predicting "building" where
    # there is no imagery is meaningless and adds noise to the GeoJSON.
    if valid_t is not None:
        pred_bin = pred_bin * valid_t.numpy().astype(np.uint8)

    # Vectorise prediction -> WGS84 GeoJSON
    pred_polys = _vectorise_mask(pred_bin, transform)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": transform_geom(crs, "EPSG:4326", mapping(p)),
            }
            for p in pred_polys
        ],
    }
    (out_dir / "prediction.geojson").write_text(json.dumps(fc), encoding="utf-8")

    # Visuals
    arr_uint8 = (image_t.numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(arr_uint8).save(out_dir / "imagery.png")
    valid_np = valid_t.numpy().astype(np.uint8) if valid_t is not None else None
    _render_comparison(
        arr_uint8,
        mask_gt=mask_t.numpy().astype(np.uint8),
        mask_pred=pred_bin,
        valid_mask=valid_np,
        out_path=out_dir / "comparison.png",
        threshold=threshold,
        area_filter_m2=area_filter_m2,
    )

    aoi = single.aois[0]
    iou = float(binary_iou(torch.from_numpy(pred_bin), mask_t, valid_t))
    f1 = float(binary_f1(torch.from_numpy(pred_bin), mask_t, valid_t))
    valid_pixels = int(valid_t.sum().item()) if valid_t is not None else int(mask_t.numel())
    total_pixels = int(mask_t.numel())
    meta = {
        "aoi_index": aoi_index,
        "bbox": list(aoi.bbox),
        "region": aoi.region,
        "notes": aoi.notes,
        "threshold": threshold,
        "area_filter_m2": area_filter_m2,
        "iou": iou,
        "f1": f1,
        "n_pred_polys": len(pred_polys),
        "n_gt_pixels": int(mask_t.sum().item()),
        "n_pred_pixels": int(pred_bin.sum()),
        "n_valid_pixels": valid_pixels,
        "n_total_pixels": total_pixels,
        "valid_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
        "pixel_area_m2": pixel_area,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _vectorise_mask(mask: np.ndarray, transform) -> list[BaseGeometry]:
    """Binary mask + raster transform -> shapely polygons in the raster's CRS."""
    if mask.sum() == 0:
        return []
    feats = rasterio.features.shapes(
        mask, mask=mask.astype(bool), transform=transform,
    )
    return [shape(geom) for geom, val in feats if val == 1]


def _render_comparison(
    arr_uint8: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    valid_mask: np.ndarray | None,
    out_path: Path,
    threshold: float,
    area_filter_m2: float,
) -> None:
    """2x3 grid: imagery / GT / pred / confusion / valid_mask / metrics.

    Confusion panel restricts TP/FP/FN to valid pixels — invalid pixels
    are rendered as semi-transparent grey so the no-data layout is
    visually obvious.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w, _ = arr_uint8.shape
    valid_b = (valid_mask.astype(bool) if valid_mask is not None
               else np.ones((h, w), dtype=bool))

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    axes[0, 0].imshow(arr_uint8)
    axes[0, 0].set_title("imagery")
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(arr_uint8)
    axes[0, 1].imshow(
        np.ma.masked_where(mask_gt == 0, mask_gt),
        cmap="autumn", alpha=0.5, vmin=0, vmax=1,
    )
    axes[0, 1].set_title(f"GT (OSM)  n_pos_px = {int(mask_gt.sum())}")
    axes[0, 1].set_axis_off()

    axes[0, 2].imshow(arr_uint8)
    axes[0, 2].imshow(
        np.ma.masked_where(mask_pred == 0, mask_pred),
        cmap="winter", alpha=0.5, vmin=0, vmax=1,
    )
    axes[0, 2].set_title(
        f"pred (thr={threshold}, area>={area_filter_m2:.0f}m²)  "
        f"n_pos_px = {int(mask_pred.sum())}"
    )
    axes[0, 2].set_axis_off()

    # Confusion: only count valid pixels
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    gt_b = mask_gt.astype(bool) & valid_b
    pr_b = mask_pred.astype(bool) & valid_b
    tp = gt_b & pr_b
    fp = pr_b & ~gt_b
    fn = gt_b & ~pr_b
    overlay[tp] = [0.0, 1.0, 0.0, 0.55]   # green = TP
    overlay[fp] = [0.0, 0.4, 1.0, 0.55]   # blue  = FP
    overlay[fn] = [1.0, 0.0, 0.0, 0.55]   # red   = FN
    # Show invalid pixels as semi-transparent grey
    overlay[~valid_b] = [0.5, 0.5, 0.5, 0.7]
    axes[1, 0].imshow(arr_uint8)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("confusion  TP=green  FP=blue  FN=red  (grey=no-data)")
    axes[1, 0].set_axis_off()

    # Valid mask panel
    axes[1, 1].imshow(arr_uint8)
    invalid_overlay = np.zeros((h, w, 4), dtype=np.float32)
    invalid_overlay[~valid_b] = [1.0, 0.0, 1.0, 0.5]  # magenta = no-data
    axes[1, 1].imshow(invalid_overlay)
    valid_pct = 100.0 * valid_b.sum() / valid_b.size
    axes[1, 1].set_title(f"valid_mask  valid={valid_pct:.1f}%  no-data=magenta")
    axes[1, 1].set_axis_off()

    # Metrics text panel
    axes[1, 2].set_axis_off()
    tp_n = int(tp.sum()); fp_n = int(fp.sum()); fn_n = int(fn.sum())
    valid_n = int(valid_b.sum())
    inv_n = int((~valid_b).sum())
    inter = tp_n
    union = tp_n + fp_n + fn_n
    iou_v = inter / union if union else 1.0
    prec = tp_n / (tp_n + fp_n) if (tp_n + fp_n) else 0.0
    rec = tp_n / (tp_n + fn_n) if (tp_n + fn_n) else 0.0
    f1_v = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    text = (
        f"valid pixels   = {valid_n:>8}\n"
        f"no-data pixels = {inv_n:>8}\n"
        f"GT pos pixels  = {int(gt_b.sum()):>8}\n"
        f"pred pos px    = {int(pr_b.sum()):>8}\n"
        f"\n"
        f"TP = {tp_n:>8}\n"
        f"FP = {fp_n:>8}\n"
        f"FN = {fn_n:>8}\n"
        f"\n"
        f"IoU       = {iou_v:.4f}\n"
        f"precision = {prec:.4f}\n"
        f"recall    = {rec:.4f}\n"
        f"F1        = {f1_v:.4f}"
    )
    axes[1, 2].text(0.02, 0.98, text, family="monospace", fontsize=11,
                    verticalalignment="top", transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


__all__ = ["run_inference"]
