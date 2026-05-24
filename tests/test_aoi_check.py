"""Phase B v2: check_aoi (feature-aware) + validate_recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from imagery_seg.aoi_check import (
    AOICheck,
    DEFAULT_MIN_VALID_FRACTION,
    MIN_TRAIN_VAL_DISTANCE_M,
    bbox_area_km2,
    check_aoi,
    validate_recipe,
)
from imagery_seg.features.base import FeatureSpec
from imagery_seg.ground_truth.base import GroundTruthSource
from imagery_seg.imagery.base import FetchedImage, ImagerySource
from imagery_seg.recipe import AOI, Recipe


# ---------------------- fakes ------------------------------------------------


class _FakeImagery(ImagerySource):
    name = "fake_img"

    def __init__(self, valid_fraction: float = 1.0, raise_on_fetch: bool = False):
        self.valid_fraction = valid_fraction
        self.raise_on_fetch = raise_on_fetch

    def fetch_for_bbox(self, bbox, max_side=64, dst_crs="EPSG:3857"):
        if self.raise_on_fetch:
            raise RuntimeError("imagery unavailable")
        h, w = 32, 32
        arr = np.full((3, h, w), 100, dtype=np.uint8)
        valid = np.ones((h, w), dtype=np.uint8)
        invalid_rows = int(h * (1 - self.valid_fraction))
        if invalid_rows > 0:
            valid[-invalid_rows:, :] = 0
        # Stay in WGS84 so the transform matches the polygon CRS used
        # by _FakeGT, avoiding a reprojection mismatch in tests.
        return FetchedImage(
            array=arr,
            transform=from_bounds(*bbox, width=w, height=h),
            crs="EPSG:4326",
            asset_id="fake-asset",
            valid_mask=valid,
        )


class _FakeGT(GroundTruthSource):
    """GT that returns N square polygons of given side fraction (relative to bbox)."""
    name = "fake_gt"

    def __init__(self, n_polygons: int = 0, side_fraction: float = 0.1):
        self.n_polygons = n_polygons
        self.side_fraction = side_fraction

    def fetch_polygons(self, bbox, feature_query):
        return [g for g, _ in self.fetch_with_tags(bbox, feature_query)]

    def fetch_with_tags(self, bbox, feature_query):
        w, s, e, n = bbox
        sx = (e - w) * self.side_fraction
        sy = (n - s) * self.side_fraction
        out = []
        for i in range(self.n_polygons):
            # arrange polygons in a row inside the bbox
            x0 = w + i * sx * 1.1
            y0 = s + 0.1 * (n - s)
            x1 = min(x0 + sx, e)
            y1 = min(y0 + sy, n)
            out.append((Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), {}))
        return out


@dataclass(frozen=True)
class _CountFeature(FeatureSpec):
    """Feature with count-based threshold only (building-like)."""
    name: str = "count_only"
    erode_px: int = 0
    min_positive_polygon_count: int = 5
    min_positive_pixel_fraction: float = 0.0

    def overpass_query(self, bbox):
        return "stub"


@dataclass(frozen=True)
class _FractionFeature(FeatureSpec):
    """Feature with pixel-fraction threshold (parking-like)."""
    name: str = "fraction_only"
    erode_px: int = 0
    min_positive_polygon_count: int = 1
    min_positive_pixel_fraction: float = 0.10

    def overpass_query(self, bbox):
        return "stub"


def _recipe(aois: tuple[AOI, ...]) -> Recipe:
    return Recipe(
        name="t",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_CountFeature(),
        aois=aois,
    )


# ---------------------- bbox_area_km2 -----------------------------------------


def test_bbox_area_km2_known_value():
    area = bbox_area_km2((139.0, 35.0, 139.01, 35.01))
    assert 0.8 < area < 1.2


# ---------------------- check_aoi (feature-aware) -----------------------------


def test_check_aoi_imagery_fetch_failure():
    img = _FakeImagery(raise_on_fetch=True)
    out = check_aoi((0, 0, 1, 1), img, _FakeGT(), _CountFeature(), max_side=32)
    assert out.hard_pass is False
    assert any("imagery fetch failed" in r for r in out.reasons)


def test_check_aoi_pass_count_only_feature():
    img = _FakeImagery(valid_fraction=1.0)
    gt = _FakeGT(n_polygons=10, side_fraction=0.05)
    out = check_aoi((139.0, 35.0, 139.01, 35.01), img, gt, _CountFeature(), max_side=32)
    assert out.hard_pass is True
    assert out.positive_polygon_count == 10
    assert out.feature_name == "count_only"


def test_check_aoi_fail_low_polygon_count():
    img = _FakeImagery(valid_fraction=1.0)
    gt = _FakeGT(n_polygons=2)  # below feature threshold of 5
    out = check_aoi((139.0, 35.0, 139.01, 35.01), img, gt, _CountFeature(), max_side=32)
    assert out.hard_pass is False
    assert any("polygon_count" in r for r in out.reasons)


def test_check_aoi_fraction_feature_pass_with_low_count_high_coverage():
    """Parking-like feature: 1 polygon covering 20% of bbox passes
    even though the count threshold is only 1."""
    img = _FakeImagery(valid_fraction=1.0)
    gt = _FakeGT(n_polygons=1, side_fraction=0.5)  # 50% side -> 25% pixels
    out = check_aoi((139.0, 35.0, 139.01, 35.01), img, gt, _FractionFeature(), max_side=32)
    assert out.hard_pass is True
    assert out.positive_polygon_count == 1
    assert out.positive_pixel_fraction >= 0.10


def test_check_aoi_fraction_feature_fail_when_coverage_too_low():
    """Same fraction-feature, but polygon is small (1% pixels) — fails."""
    img = _FakeImagery(valid_fraction=1.0)
    gt = _FakeGT(n_polygons=1, side_fraction=0.05)  # 0.25% pixels
    out = check_aoi((139.0, 35.0, 139.01, 35.01), img, gt, _FractionFeature(), max_side=32)
    assert out.hard_pass is False
    assert any("positive_pixel_fraction" in r for r in out.reasons)


def test_check_aoi_and_semantic_both_thresholds_must_pass():
    """Combo feature with min_count=5 AND min_fraction=0.20.
    Count alone passes but fraction fails -> hard fail."""

    @dataclass(frozen=True)
    class _Both(FeatureSpec):
        name: str = "both"
        erode_px: int = 0
        min_positive_polygon_count: int = 5
        min_positive_pixel_fraction: float = 0.20

        def overpass_query(self, bbox):
            return "stub"

    img = _FakeImagery(valid_fraction=1.0)
    # 8 polygons each covering ~0.5% -> total ~4% pixels
    gt = _FakeGT(n_polygons=8, side_fraction=0.05)
    out = check_aoi((139.0, 35.0, 139.01, 35.01), img, gt, _Both(), max_side=32)
    assert out.hard_pass is False  # count passes but fraction fails
    assert any("pixel_fraction" in r for r in out.reasons)
    assert not any("polygon_count" in r for r in out.reasons)


def test_check_aoi_overrides_take_priority_over_feature_defaults():
    img = _FakeImagery(valid_fraction=1.0)
    gt = _FakeGT(n_polygons=2)
    # Feature wants 5 polygons, override allows 1 -> should pass
    out = check_aoi(
        (139.0, 35.0, 139.01, 35.01), img, gt, _CountFeature(),
        max_side=32, min_polygon_count=1,
    )
    assert out.hard_pass is True


def test_check_aoi_zero_thresholds_skip_check():
    @dataclass(frozen=True)
    class _NoFilter(FeatureSpec):
        name: str = "nofilter"
        erode_px: int = 0
        min_positive_polygon_count: int = 0
        min_positive_pixel_fraction: float = 0.0

        def overpass_query(self, bbox):
            return "stub"

    img = _FakeImagery(valid_fraction=1.0)
    gt = _FakeGT(n_polygons=0)  # zero polygons
    out = check_aoi((139.0, 35.0, 139.01, 35.01), img, gt, _NoFilter(), max_side=32)
    # Both checks skipped -> hard_pass True (valid_frac still must pass)
    assert out.hard_pass is True


def test_check_aoi_low_valid_fraction_fails():
    img = _FakeImagery(valid_fraction=0.30)  # below DEFAULT 0.50
    gt = _FakeGT(n_polygons=20)
    out = check_aoi((139.0, 35.0, 139.01, 35.01), img, gt, _CountFeature(), max_side=32)
    assert out.hard_pass is False
    assert any("valid_pixel_fraction" in r for r in out.reasons)


# ---------------------- validate_recipe ---------------------------------------


def test_validate_recipe_passes_clean():
    a1 = AOI(bbox=(139.000, 35.000, 139.005, 35.005), region="r",
             phenotype="japan_suburban")
    a2 = AOI(bbox=(139.020, 35.020, 139.025, 35.025), region="r",
             phenotype="japan_suburban", holdout=True)
    r = _recipe((a1, a2))
    rep = validate_recipe(r)
    assert rep["passes"] is True
    assert rep["errors"] == []


def test_validate_recipe_fails_on_phenotype_mismatch():
    a1 = AOI(bbox=(139.0, 35.0, 139.005, 35.005), region="r",
             phenotype="japan_suburban")
    a2 = AOI(bbox=(120.0, 14.0, 120.005, 14.005), region="r2",
             phenotype="philippines_colonial", holdout=True)
    r = _recipe((a1, a2))
    rep = validate_recipe(r)
    assert rep["passes"] is False
    assert any("multiple phenotypes" in e for e in rep["errors"])


def test_validate_recipe_fails_on_short_train_val_distance():
    a1 = AOI(bbox=(139.0, 35.0, 139.001, 35.001), region="r",
             phenotype="x")
    a2 = AOI(bbox=(139.001, 35.0, 139.002, 35.001), region="r",
             phenotype="x", holdout=True)
    r = _recipe((a1, a2))
    rep = validate_recipe(r)
    assert rep["passes"] is False
    assert any("apart" in e for e in rep["errors"])


def test_validate_recipe_warning_for_mixed_tagged_untagged():
    a1 = AOI(bbox=(139.0, 35.0, 139.005, 35.005), region="r",
             phenotype="japan_suburban")
    a2 = AOI(bbox=(139.020, 35.020, 139.025, 35.025), region="r",
             phenotype="", holdout=True)
    r = _recipe((a1, a2))
    rep = validate_recipe(r)
    assert rep["passes"] is True
    assert any("phenotype tag" in w for w in rep["warnings"])
