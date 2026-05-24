"""RecipeDataset: ties imagery + GT + feature into (image, mask) pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from imagery_seg.dataset import RecipeDataset
from imagery_seg.features.base import FeatureSpec
from imagery_seg.ground_truth.base import GroundTruthSource
from imagery_seg.imagery.base import FetchedImage, ImagerySource
from imagery_seg.recipe import AOI, Recipe


class _Imagery(ImagerySource):
    name = "test_img"

    def fetch_for_bbox(self, bbox, max_side=64, dst_crs="EPSG:3857"):
        # 3 x 32 x 32 deterministic image. We pin the synthetic CRS to
        # WGS84 so the polygon-coordinate space and the raster coordinate
        # space line up without any reprojection in the test.
        arr = np.full((3, 32, 32), 128, dtype=np.uint8)
        return FetchedImage(
            array=arr,
            transform=from_bounds(0, 0, 32, 32, 32, 32),
            crs="EPSG:4326",
            asset_id="testasset",
        )


class _GT(GroundTruthSource):
    name = "test_gt"

    def fetch_polygons(self, bbox, feature_query):
        # A single polygon covering pixel-space rows/cols 4..28
        return [Polygon([(4, 4), (28, 4), (28, 28), (4, 28)])]


@dataclass(frozen=True)
class _Feat(FeatureSpec):
    name: str = "test_feat"
    erode_px: int = 0

    def overpass_query(self, bbox):
        return "stub"


def _recipe():
    return Recipe(
        name="ds-unit",
        imagery=_Imagery(),
        ground_truth=_GT(),
        feature=_Feat(),
        aois=(
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 2.0, 2.0),
        ),
    )


def test_dataset_len_matches_aoi_count():
    ds = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857")
    assert len(ds) == 2


def test_dataset_returns_image_and_mask_tensors():
    ds = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857")
    sample = ds[0]
    image, mask = sample["image"], sample["mask"]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.long
    # Float pixels normalised to [0, 1]
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    # Mask has both classes (the centred polygon -> 1s, edges -> 0s)
    assert int(mask.min()) == 0
    assert int(mask.max()) == 1


def test_dataset_aoi_id_in_sample():
    ds = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857")
    sample = ds[1]
    assert "aoi_index" in sample and sample["aoi_index"] == 1


def test_dataset_index_out_of_range():
    ds = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857")
    with pytest.raises(IndexError):
        _ = ds[42]


def test_augmentation_off_is_deterministic():
    ds = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=False)
    a = ds[0]
    b = ds[0]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["mask"], b["mask"])


def test_augmentation_on_produces_variation():
    torch.manual_seed(123)
    ds = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=True)
    samples = [ds[0] for _ in range(10)]
    images = [s["image"] for s in samples]
    masks = [s["mask"] for s in samples]
    # At least one pair of samples must differ on EITHER axis (geometric or
    # photometric) — 10 draws on D4 × brightness make collision astronomically rare.
    assert any(not torch.equal(images[0], img) for img in images[1:])
    # Geometric transform applies to image AND mask together: if one flips/rotates,
    # so does the other.
    for s in samples:
        # When the geometric op is identity, image[mask==1] are positive class pixels
        # — same positions across image and mask. With ANY transform, this still holds
        # because they're transformed together. Easiest invariant: mask shape unchanged.
        assert s["mask"].shape == masks[0].shape
        assert s["image"].shape == images[0].shape


def test_augmentation_keeps_mask_class_count():
    """Geometric augmentation should not lose foreground pixels."""
    torch.manual_seed(0)
    ds_off = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=False)
    ds_on = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=True)
    base_positive = int(ds_off[0]["mask"].sum())
    for _ in range(10):
        # Flips and rot90 preserve pixel counts exactly.
        assert int(ds_on[0]["mask"].sum()) == base_positive


def test_augmentation_preserves_shape_on_non_square():
    """rot90 by odd k swaps H/W and breaks batching; on non-square chips,
    augmentation must restrict to shape-preserving rotations."""
    class _RectImg(_Imagery):
        name = "rect"
        def fetch_for_bbox(self, bbox, max_side=64, dst_crs="EPSG:3857"):
            # 3 x 16 x 32 (non-square)
            arr = np.full((3, 16, 32), 64, dtype=np.uint8)
            return FetchedImage(
                array=arr,
                transform=from_bounds(0, 0, 32, 16, 32, 16),
                crs="EPSG:4326",
                asset_id="rect",
            )

    recipe = Recipe(
        name="rect",
        imagery=_RectImg(),
        ground_truth=_GT(),
        feature=_Feat(),
        aois=(AOI(bbox=(0.0, 0.0, 1.0, 1.0), region="t"),),
    )
    torch.manual_seed(0)
    ds_on = RecipeDataset(recipe, max_side=32, dst_crs="EPSG:4326", augment=True)
    base_shape = ds_on[0]["image"].shape
    for _ in range(20):
        sample = ds_on[0]
        assert sample["image"].shape == base_shape
        assert sample["mask"].shape == base_shape[1:]


def test_augmentation_geometric_only_keeps_pixel_values_exact():
    """color_jitter=False -> augmented image is one of 8 D4-symmetric copies
    of the base image (pixel values unchanged, only spatially rearranged)."""
    torch.manual_seed(11)
    ds_off = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=False)
    ds_geom = RecipeDataset(
        _recipe(), max_side=32, dst_crs="EPSG:3857",
        augment=True, color_jitter=False,
    )
    base = ds_off[0]["image"]

    # Enumerate the 8 D4-symmetric transforms of base.
    candidates = set()
    for k in range(4):
        rot = torch.rot90(base, k, dims=(-2, -1))
        candidates.add(tuple(rot.flatten().tolist()))
        candidates.add(tuple(rot.flip(-1).flatten().tolist()))
        candidates.add(tuple(rot.flip(-2).flatten().tolist()))
        candidates.add(tuple(rot.flip(-1).flip(-2).flatten().tolist()))

    for _ in range(8):
        got = tuple(ds_geom[0]["image"].flatten().tolist())
        assert got in candidates


def test_augmentation_with_color_jitter_changes_values():
    """color_jitter=True can produce image values that aren't in the
    original D4 candidate set (because of brightness scaling)."""
    torch.manual_seed(11)
    ds_off = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=False)
    ds_full = RecipeDataset(
        _recipe(), max_side=32, dst_crs="EPSG:3857",
        augment=True, color_jitter=True,
    )
    base = ds_off[0]["image"]
    candidates = set()
    for k in range(4):
        rot = torch.rot90(base, k, dims=(-2, -1))
        candidates.add(tuple(rot.flatten().tolist()))
        candidates.add(tuple(rot.flip(-1).flatten().tolist()))
        candidates.add(tuple(rot.flip(-2).flatten().tolist()))
        candidates.add(tuple(rot.flip(-1).flip(-2).flatten().tolist()))

    # At least one of N draws must land outside the D4 set (brightness != 1.0).
    seen_outside = False
    for _ in range(20):
        got = tuple(ds_full[0]["image"].flatten().tolist())
        if got not in candidates:
            seen_outside = True
            break
    assert seen_outside, "color_jitter=True should sometimes produce non-D4-equivalent images"


def test_augmentation_image_and_mask_transform_in_sync():
    """If H-flip happens, mask must flip horizontally too — same for V/rot90."""
    torch.manual_seed(7)
    ds_on = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=True)
    ds_off = RecipeDataset(_recipe(), max_side=32, dst_crs="EPSG:3857", augment=False)
    base_mask = ds_off[0]["mask"]

    # Each augmented mask should be one of the 8 D4-symmetry transforms of base_mask.
    candidates = set()
    for k in range(4):
        rot = torch.rot90(base_mask, k, dims=(-2, -1))
        candidates.add(tuple(rot.flatten().tolist()))
        candidates.add(tuple(rot.flip(-1).flatten().tolist()))
        candidates.add(tuple(rot.flip(-2).flatten().tolist()))
        candidates.add(tuple(rot.flip(-1).flip(-2).flatten().tolist()))

    for _ in range(8):
        got = tuple(ds_on[0]["mask"].flatten().tolist())
        assert got in candidates
