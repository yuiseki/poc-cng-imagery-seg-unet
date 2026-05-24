"""AOICache: round-trip + namespacing + RecipeDataset integration."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from imagery_seg.aoi_cache import AOICache
from imagery_seg.dataset import RecipeDataset
from imagery_seg.features.base import FeatureSpec
from imagery_seg.ground_truth.base import GroundTruthSource
from imagery_seg.imagery.base import FetchedImage, ImagerySource
from imagery_seg.recipe import Recipe, TrainingConfig


class _CountingImagery(ImagerySource):
    name = "count_img"

    def __init__(self) -> None:
        self.calls = 0

    def fetch_for_bbox(self, bbox, max_side=64, dst_crs="EPSG:3857"):
        self.calls += 1
        arr = np.full((3, 8, 8), 7, dtype=np.uint8)
        return FetchedImage(
            array=arr,
            transform=from_bounds(*bbox, width=8, height=8),
            crs=dst_crs,
            asset_id=f"asset-{self.calls}",
        )


class _CountingGT(GroundTruthSource):
    name = "count_gt"

    def __init__(self) -> None:
        self.calls = 0

    def fetch_polygons(self, bbox, feature_query):
        self.calls += 1
        west, south, east, north = bbox
        return [Polygon([(west, south), (east, south), (east, north), (west, north)])]


@dataclass(frozen=True)
class _Feat(FeatureSpec):
    name: str = "f"

    def overpass_query(self, bbox):
        return "stub"


def _recipe(imagery: ImagerySource, gt: GroundTruthSource) -> Recipe:
    return Recipe(
        name="cache-test",
        imagery=imagery,
        ground_truth=gt,
        feature=_Feat(),
        training=TrainingConfig(),
        aois=((0.0, 0.0, 1.0, 1.0),),
    )


def test_image_roundtrip(tmp_path: Path):
    cache = AOICache(root=tmp_path)
    img = FetchedImage(
        array=np.full((3, 8, 8), 11, dtype=np.uint8),
        transform=from_bounds(0, 0, 1, 1, 8, 8),
        crs="EPSG:3857",
        asset_id="asset-xyz",
    )
    cache.put_image("count_img", (0.0, 0.0, 1.0, 1.0), 64, "EPSG:3857", img)
    hit = cache.get_image("count_img", (0.0, 0.0, 1.0, 1.0), 64, "EPSG:3857")
    assert hit is not None
    assert hit.array.shape == img.array.shape
    assert (hit.array == img.array).all()
    assert hit.asset_id == "asset-xyz"
    # meta sidecar captures asset_id
    meta_files = list((tmp_path / "images" / "count_img").glob("*.meta.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text())
    assert meta["asset_id"] == "asset-xyz"


def test_polygon_roundtrip(tmp_path: Path):
    cache = AOICache(root=tmp_path)
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    cache.put_polygons("count_gt__f", (0.0, 0.0, 1.0, 1.0), polys)
    hit = cache.get_polygons("count_gt__f", (0.0, 0.0, 1.0, 1.0))
    assert hit is not None
    assert len(hit) == 1
    assert abs(hit[0].area - 1.0) < 1e-9


def test_image_miss_returns_none(tmp_path: Path):
    cache = AOICache(root=tmp_path)
    assert cache.get_image("nope", (0.0, 0.0, 1.0, 1.0), 64, "EPSG:3857") is None
    assert cache.get_polygons("nope", (0.0, 0.0, 1.0, 1.0)) is None


def test_dataset_with_cache_avoids_redundant_fetches(tmp_path: Path):
    """Re-reading the same AOI hits cache; source.fetch_* is not called again."""
    imagery = _CountingImagery()
    gt = _CountingGT()
    cache = AOICache(root=tmp_path)
    ds = RecipeDataset(_recipe(imagery, gt), max_side=64, dst_crs="EPSG:3857", cache=cache)

    _ = ds[0]
    _ = ds[0]
    _ = ds[0]
    assert imagery.calls == 1
    assert gt.calls == 1


def test_dataset_without_cache_fetches_every_time(tmp_path: Path):
    """Sanity: default (cache=None) keeps the pre-cache behavior."""
    imagery = _CountingImagery()
    gt = _CountingGT()
    ds = RecipeDataset(_recipe(imagery, gt), max_side=64, dst_crs="EPSG:3857")

    _ = ds[0]
    _ = ds[0]
    assert imagery.calls == 2
    assert gt.calls == 2


def test_polygon_cache_shared_across_recipes_with_same_gt(tmp_path: Path):
    """Two recipes differing only in imagery share the polygon cache."""
    cache = AOICache(root=tmp_path)
    gt = _CountingGT()

    class _OtherImagery(_CountingImagery):
        name = "other_img"

    r1 = _recipe(_CountingImagery(), gt)
    r2 = _recipe(_OtherImagery(), gt)
    ds1 = RecipeDataset(r1, max_side=64, dst_crs="EPSG:3857", cache=cache)
    ds2 = RecipeDataset(r2, max_side=64, dst_crs="EPSG:3857", cache=cache)

    _ = ds1[0]
    _ = ds2[0]
    # Imagery cache is namespaced separately -> each imagery fetched once
    assert r1.imagery.calls == 1
    assert r2.imagery.calls == 1
    # Polygon cache is keyed by (gt, feature) only -> gt fetched once total
    assert gt.calls == 1


def test_concurrent_dataset_reads_do_not_double_fetch(tmp_path: Path):
    """Two threads racing on the same AOI: source.fetch_* called once."""
    imagery = _CountingImagery()
    gt = _CountingGT()
    cache = AOICache(root=tmp_path)
    recipe = _recipe(imagery, gt)
    ds = RecipeDataset(recipe, max_side=64, dst_crs="EPSG:3857", cache=cache)

    results: list = []

    def worker():
        results.append(ds[0])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    # With file_lock + double-checked get, exactly one fill per axis.
    assert imagery.calls == 1
    assert gt.calls == 1
