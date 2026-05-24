"""On-disk AOI cache for Recipe-driven datasets.

Two-tier cache, split by what each piece depends on:

  imagery  cache (root/images/{imagery_ns}/{key}.tif + .meta.json)
      keyed by (imagery_ns, bbox, max_side, dst_crs)
      stored as a GeoTIFF preserving transform+crs+asset_id

  polygon  cache (root/polygons/{gt_feature_ns}/{key}.geojson)
      keyed by (gt_feature_ns, bbox)
      stored as a WGS84 GeoJSON FeatureCollection

The split lets two recipes that share imagery (or share GT+feature) hit
the same cached file. This is the missing piece from sibling stage 6's
single-axis cache: in poc-cng-imagery-seg-unet, imagery and GT come from
independent axes, so the cache namespacing is too.

Concurrency: each (key) gets a sidecar .lock; `file_lock` blocks
concurrent fills so two epochs sharing an AOI don't double-fetch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from .cache import cache_key, file_lock, write_atomic
from .imagery.base import FetchedImage

logger = logging.getLogger("imagery_seg.aoi_cache")

DEFAULT_CACHE_ROOT = Path("tmp/cache")


@dataclass(frozen=True)
class AOICache:
    """File-backed cache for `(image, polygons)` pairs.

    Construct one per training/eval run and pass to RecipeDataset.
    """

    root: Path = DEFAULT_CACHE_ROOT

    # -- imagery -----------------------------------------------------------

    def _image_paths(
        self,
        imagery_ns: str,
        bbox: tuple[float, float, float, float],
        max_side: int,
        dst_crs: str,
    ) -> dict[str, Path]:
        h = cache_key(imagery_ns, bbox, max_side=max_side, dst_crs=dst_crs)
        d = self.root / "images" / imagery_ns
        return {
            "tif": d / f"{h}.tif",
            "meta": d / f"{h}.meta.json",
            "lock": d / f"{h}.lock",
        }

    def get_image(
        self,
        imagery_ns: str,
        bbox: tuple[float, float, float, float],
        max_side: int,
        dst_crs: str,
    ) -> FetchedImage | None:
        paths = self._image_paths(imagery_ns, bbox, max_side, dst_crs)
        if not (paths["tif"].is_file() and paths["meta"].is_file()):
            return None
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        with rasterio.open(paths["tif"]) as ds:
            arr = ds.read()
            transform = ds.transform
            crs = str(ds.crs) if ds.crs else dst_crs
        # Validity layer stored as the last band when count == bands + 1
        # (e.g. 4 bands = RGB + validity). Legacy 3-band entries fall back
        # to None (caller treats as all-valid).
        valid_mask = None
        if arr.shape[0] >= 4:
            valid_mask = arr[-1].astype(np.uint8)
            arr = arr[:-1]
        return FetchedImage(
            array=arr,
            transform=transform,
            crs=crs,
            asset_id=meta.get("asset_id", ""),
            valid_mask=valid_mask,
        )

    def put_image(
        self,
        imagery_ns: str,
        bbox: tuple[float, float, float, float],
        max_side: int,
        dst_crs: str,
        image: FetchedImage,
    ) -> None:
        paths = self._image_paths(imagery_ns, bbox, max_side, dst_crs)
        paths["tif"].parent.mkdir(parents=True, exist_ok=True)
        _write_geotiff_atomic(paths["tif"], image)
        meta = {
            "asset_id": image.asset_id,
            "bbox": list(bbox),
            "max_side": int(max_side),
            "dst_crs": dst_crs,
            "imagery_ns": imagery_ns,
        }
        write_atomic(paths["meta"], json.dumps(meta, indent=2).encode("utf-8"))

    # -- polygons ----------------------------------------------------------

    def _polygon_paths(
        self,
        polygon_ns: str,
        bbox: tuple[float, float, float, float],
    ) -> dict[str, Path]:
        h = cache_key(polygon_ns, bbox)
        d = self.root / "polygons" / polygon_ns
        return {
            "geojson": d / f"{h}.geojson",
            "lock": d / f"{h}.lock",
        }

    def get_polygons(
        self,
        polygon_ns: str,
        bbox: tuple[float, float, float, float],
    ) -> list[BaseGeometry] | None:
        paths = self._polygon_paths(polygon_ns, bbox)
        if not paths["geojson"].is_file():
            return None
        fc = json.loads(paths["geojson"].read_text(encoding="utf-8"))
        feats = fc.get("features", []) or []
        return [shape(f["geometry"]) for f in feats if f.get("geometry")]

    def put_polygons(
        self,
        polygon_ns: str,
        bbox: tuple[float, float, float, float],
        polygons: list[BaseGeometry],
    ) -> None:
        paths = self._polygon_paths(polygon_ns, bbox)
        paths["geojson"].parent.mkdir(parents=True, exist_ok=True)
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": mapping(p)}
                for p in polygons
            ],
        }
        write_atomic(
            paths["geojson"],
            json.dumps(fc).encode("utf-8"),
        )

    # -- locks -------------------------------------------------------------

    def image_lock(
        self,
        imagery_ns: str,
        bbox: tuple[float, float, float, float],
        max_side: int,
        dst_crs: str,
    ):
        """Context manager: held while filling the image cache for this key."""
        paths = self._image_paths(imagery_ns, bbox, max_side, dst_crs)
        return file_lock(paths["lock"])

    def polygon_lock(
        self,
        polygon_ns: str,
        bbox: tuple[float, float, float, float],
    ):
        """Context manager: held while filling the polygon cache for this key."""
        paths = self._polygon_paths(polygon_ns, bbox)
        return file_lock(paths["lock"])


def _write_geotiff_atomic(path: Path, image: FetchedImage) -> None:
    """Write `image.array` to `path` as a tiled deflate GeoTIFF, then
    rename into place. rasterio doesn't expose an atomic write itself, so
    we go via a tmp file and `Path.replace`.
    """
    import os

    array = np.asarray(image.array)
    if array.ndim == 2:
        array = array[None, :, :]
    # Append validity as the last band so cache files round-trip the
    # no-data layout. Stored as uint8 {0, 1}.
    if image.valid_mask is not None:
        vm = np.asarray(image.valid_mask).astype(array.dtype)
        if vm.shape != array.shape[1:]:
            raise ValueError(
                f"valid_mask shape {vm.shape} doesn't match image {array.shape[1:]}"
            )
        array = np.concatenate([array, vm[None, ...]], axis=0)
    bands, h, w = array.shape
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "count": bands,
        "height": h,
        "width": w,
        "dtype": str(array.dtype),
        "transform": image.transform,
        "crs": image.crs,
        "compress": "deflate",
        "predictor": 2,
    }
    # GTiff requires block sizes that are multiples of 16 (and bigger than
    # the image itself triggers RasterBlockError too). Skip tiling for
    # tiny images — they don't benefit from it anyway.
    if h >= 16 and w >= 16:
        profile.update(
            tiled=True,
            blockxsize=min(256, (w // 16) * 16),
            blockysize=min(256, (h // 16) * 16),
        )
    try:
        with rasterio.open(tmp, "w", **profile) as ds:
            ds.write(array)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


__all__ = ["AOICache", "DEFAULT_CACHE_ROOT"]
