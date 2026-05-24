"""AOI selection criteria — see Phase B design notes.

Catches the failure modes we've observed in this PoC:

  - non-RGB COGs (1-band float DSM masquerading as imagery)
  - no-data regions inside the COG bbox (DRONEBIRD flight-path edges)
  - sparse-building AOIs where IoU signal is noise
  - cross-phenotype AOIs paired with the wrong model

`check_aoi(bbox, imagery, gt, feature)` runs hard filters + soft scores
and returns a structured `AOICheck` report. The CLI wraps it as
`imagery-seg aoi-check` (single bbox) and `imagery-seg aoi-grid`
(tile a bbox into NxM and score each cell).

Thresholds are draft (chosen from a small set of observed AOIs). Expect
to tune after running against more real data — that's the whole point
of running Phase B before Phase 2 (AOI catalog).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .features.base import FeatureSpec
from .ground_truth.base import GroundTruthSource
from .imagery.base import ImagerySource
from .rasterize import polygons_to_mask
from .recipe import Bbox

logger = logging.getLogger("imagery_seg.aoi_check")

# Default thresholds (feature-agnostic). Per-feature signal thresholds
# (polygon count + pixel fraction) come from FeatureSpec — see Phase B
# v2 design notes.
DEFAULT_MIN_VALID_FRACTION = 0.50
SOFT_VALID_FRACTION_FULL = 1.00
SOFT_DENSITY_GOOD_LO = 100.0
SOFT_DENSITY_GOOD_HI = 2000.0


@dataclass(frozen=True)
class AOICheck:
    """One AOI's evaluation against the Phase B selection criteria.

    `hard_pass` is True iff every hard filter passes. `reasons` lists
    the specific failures (empty when `hard_pass` is True). `warnings`
    are concerns that don't block usability.

    `soft_scores` is a dict of per-criterion scores in [0, 1];
    `quality_score` is their unweighted mean. Use it to rank AOIs.

    Fields now reflect the post-Phase-B-v2 design where positive-class
    signal is feature-aware (polygon count + pixel fraction both
    considered, with thresholds inherited from the FeatureSpec).
    """

    bbox: Bbox
    hard_pass: bool
    feature_name: str = ""
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Imagery facts
    asset_id: str = ""
    imagery_bands: int = 0
    imagery_dtype: str = ""
    # Coverage
    valid_pixel_fraction: float = 0.0
    # GT facts (renamed from building_* to be feature-agnostic; the
    # legacy building_count alias is dropped — Phase B is the only
    # caller and the rename is non-breaking for downstream JSON because
    # we expose both names in to_dict.)
    positive_polygon_count: int = 0
    positive_pixel_fraction: float = 0.0
    polygon_density_per_km2: float = 0.0
    bbox_area_km2: float = 0.0
    # Scoring
    soft_scores: dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "hard_pass": self.hard_pass,
            "feature_name": self.feature_name,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "asset_id": self.asset_id,
            "imagery_bands": self.imagery_bands,
            "imagery_dtype": self.imagery_dtype,
            "valid_pixel_fraction": self.valid_pixel_fraction,
            "positive_polygon_count": self.positive_polygon_count,
            "positive_pixel_fraction": self.positive_pixel_fraction,
            "polygon_density_per_km2": self.polygon_density_per_km2,
            "bbox_area_km2": self.bbox_area_km2,
            "soft_scores": dict(self.soft_scores),
            "quality_score": self.quality_score,
        }


def bbox_area_km2(bbox: Bbox) -> float:
    """Approximate WGS84 bbox area in km², using local-flat approximation."""
    w, s, e, n = bbox
    lat_center = (s + n) / 2.0
    width_km = (e - w) * 111.32 * math.cos(math.radians(lat_center))
    height_km = (n - s) * 110.57
    return max(0.0, width_km * height_km)


def _soft_valid_fraction(v: float) -> float:
    """0 at DEFAULT_MIN_VALID_FRACTION, 1 at SOFT_VALID_FRACTION_FULL, linear."""
    if v <= DEFAULT_MIN_VALID_FRACTION:
        return 0.0
    if v >= SOFT_VALID_FRACTION_FULL:
        return 1.0
    return (v - DEFAULT_MIN_VALID_FRACTION) / (
        SOFT_VALID_FRACTION_FULL - DEFAULT_MIN_VALID_FRACTION
    )


def check_aoi(
    bbox: Bbox,
    imagery: ImagerySource,
    ground_truth: GroundTruthSource,
    feature: FeatureSpec,
    *,
    max_side: int = 1024,
    dst_crs: str = "EPSG:3857",
    min_polygon_count: int | None = None,
    min_pixel_fraction: float | None = None,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> AOICheck:
    """Evaluate one bbox against the Phase B selection criteria.

    Per-feature thresholds: when `min_polygon_count` / `min_pixel_fraction`
    are None, the corresponding `feature.min_positive_*` is used. A
    threshold of 0 means "skip this check" (e.g. BuildingFeature uses
    count-only with fraction=0). Both checks are combined with AND
    semantics — for an AOI to pass it must clear every non-zero
    threshold.

    Network: STAC + COG for imagery, Overpass for the actual feature
    polygons (we used to do a fast count query, but pixel-fraction
    computation forces fetching the geometries anyway, so just go
    straight there). ~15-45 seconds per AOI uncached.
    """
    min_count = (
        feature.min_positive_polygon_count
        if min_polygon_count is None else min_polygon_count
    )
    min_fraction = (
        feature.min_positive_pixel_fraction
        if min_pixel_fraction is None else min_pixel_fraction
    )

    reasons: list[str] = []
    warnings: list[str] = []
    area_km2 = bbox_area_km2(bbox)

    # 1. Imagery fetch
    try:
        img = imagery.fetch_for_bbox(bbox, max_side=max_side, dst_crs=dst_crs)
    except Exception as e:
        return AOICheck(
            bbox=bbox,
            hard_pass=False,
            feature_name=feature.name,
            reasons=[f"imagery fetch failed: {type(e).__name__}: {e}"],
            bbox_area_km2=area_km2,
        )

    bands = int(img.array.shape[0])
    dtype = str(img.array.dtype)

    # 2. Valid pixel fraction
    if img.valid_mask is not None:
        total = int(img.valid_mask.size)
        valid_count = int(img.valid_mask.sum())
        valid_frac = valid_count / total if total else 0.0
    else:
        valid_frac = 1.0  # no validity info -> assume all valid

    if valid_frac < min_valid_fraction:
        reasons.append(
            f"valid_pixel_fraction {valid_frac:.3f} < threshold {min_valid_fraction}"
        )

    # 3. Fetch + post-process feature polygons (this is what training
    # actually uses, so the count/fraction we compute matches what the
    # dataset would feed into the model).
    polygon_count = 0
    pixel_fraction = 0.0
    try:
        geoms_with_tags = ground_truth.fetch_with_tags(
            bbox, feature.overpass_query(bbox),
        )
        polys = feature.to_polygons(geoms_with_tags)
        polygon_count = len(polys)

        if polys:
            mask = polygons_to_mask(
                polys,
                src_crs="EPSG:4326",
                dst_crs=img.crs,
                transform=img.transform,
                height=img.height,
                width=img.width,
                erode_px=feature.erode_px,
            )
            if img.valid_mask is not None:
                valid_b = img.valid_mask.astype(bool)
                positive_in_valid = int((mask.astype(bool) & valid_b).sum())
                valid_pixels = int(valid_b.sum())
            else:
                positive_in_valid = int(mask.astype(bool).sum())
                valid_pixels = int(mask.size)
            pixel_fraction = (
                positive_in_valid / valid_pixels if valid_pixels else 0.0
            )
    except Exception as e:
        warnings.append(
            f"GT fetch/rasterise failed: {type(e).__name__}: {e}"
        )

    # 4. Feature-aware hard filter (AND semantic, 0 thresholds skip)
    if min_count > 0 and polygon_count < min_count:
        reasons.append(
            f"polygon_count {polygon_count} < feature.min_positive_polygon_count {min_count}"
        )
    if min_fraction > 0 and pixel_fraction < min_fraction:
        reasons.append(
            f"positive_pixel_fraction {pixel_fraction:.4f} < "
            f"feature.min_positive_pixel_fraction {min_fraction}"
        )

    density = (polygon_count / area_km2) if area_km2 > 0 else 0.0
    if density > 0 and density < SOFT_DENSITY_GOOD_LO:
        warnings.append(
            f"polygon_density_per_km2 {density:.0f} is low (rural? sparse OSM?)"
        )
    if density > SOFT_DENSITY_GOOD_HI:
        warnings.append(
            f"polygon_density_per_km2 {density:.0f} is ultra-dense; "
            f"IoU will be bound by polygon boundary noise"
        )

    # 5. Soft scores — scale relative to the feature's own thresholds so
    # they're comparable across features. min_count*5 saturates the
    # count score; min_fraction*3 saturates the fraction score.
    count_sat = max(min_count * 5, 1)
    fraction_sat = max(min_fraction * 3, 0.01)
    soft = {
        "valid_fraction": _soft_valid_fraction(valid_frac),
        "polygon_count": min(1.0, polygon_count / count_sat) if min_count > 0 else 1.0,
        "pixel_fraction": (
            min(1.0, pixel_fraction / fraction_sat) if min_fraction > 0 else 1.0
        ),
    }
    quality = sum(soft.values()) / len(soft) if soft else 0.0

    return AOICheck(
        bbox=bbox,
        hard_pass=not reasons,
        feature_name=feature.name,
        reasons=reasons,
        warnings=warnings,
        asset_id=img.asset_id,
        imagery_bands=bands,
        imagery_dtype=dtype,
        valid_pixel_fraction=valid_frac,
        positive_polygon_count=polygon_count,
        positive_pixel_fraction=pixel_fraction,
        polygon_density_per_km2=density,
        bbox_area_km2=area_km2,
        soft_scores=soft,
        quality_score=quality,
    )


# -- Recipe-level validation ---------------------------------------------------


def _bbox_center(bbox: Bbox) -> tuple[float, float]:
    w, s, e, n = bbox
    return ((w + e) / 2.0, (s + n) / 2.0)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


MIN_TRAIN_VAL_DISTANCE_M = 200.0


def validate_recipe(recipe) -> dict[str, Any]:  # noqa: ANN001 (avoids Recipe circular)
    """Recipe-level checks. Returns `{passes, errors, warnings}`.

      errors:    blocking issues — recipe shouldn't be trained as-is
      warnings:  concerns — train at your own risk

    Currently checks:
      - all train+val AOIs share the same `phenotype` tag (or all empty)
      - every train AOI is >= MIN_TRAIN_VAL_DISTANCE_M from every val AOI
        (great-circle distance between bbox centres)
    """
    errors: list[str] = []
    warnings: list[str] = []

    train_aois = recipe.effective_train_aois
    val_aois = recipe.effective_val_aois
    all_aois = tuple(train_aois) + tuple(val_aois)

    if not all_aois:
        errors.append("recipe has no AOIs")
        return {"passes": False, "errors": errors, "warnings": warnings}

    # Phenotype consistency
    phenotypes = {a.phenotype for a in all_aois}
    if len(phenotypes) > 1:
        # If empty + nonempty mixed, that's mostly a "not yet tagged" state
        nonempty = phenotypes - {""}
        if len(nonempty) > 1:
            errors.append(
                f"AOIs span multiple phenotypes {sorted(nonempty)}; "
                f"phenotype-specific models should train on one phenotype"
            )
        elif "" in phenotypes:
            warnings.append(
                f"some AOIs lack phenotype tag (others = {sorted(nonempty)}); "
                f"recommend tagging all AOIs"
            )

    # Train-val spatial separation
    for ti, t in enumerate(train_aois):
        tlon, tlat = _bbox_center(t.bbox)
        for vi, v in enumerate(val_aois):
            vlon, vlat = _bbox_center(v.bbox)
            d = _haversine_m(tlon, tlat, vlon, vlat)
            if d < MIN_TRAIN_VAL_DISTANCE_M:
                errors.append(
                    f"train[{ti}] and val[{vi}] are only {d:.0f}m apart "
                    f"(< {MIN_TRAIN_VAL_DISTANCE_M:.0f}m); spatial leakage risk"
                )

    return {"passes": not errors, "errors": errors, "warnings": warnings}


__all__ = [
    "AOICheck",
    "check_aoi",
    "bbox_area_km2",
    "validate_recipe",
    "MIN_TRAIN_VAL_DISTANCE_M",
    "DEFAULT_MIN_VALID_FRACTION",
]
