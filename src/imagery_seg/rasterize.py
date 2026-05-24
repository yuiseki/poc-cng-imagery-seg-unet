"""Polygons (WGS84) + raster transform/CRS -> binary uint8 mask."""

from __future__ import annotations

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from scipy.ndimage import binary_erosion
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry


def polygons_to_mask(
    polygons: list[BaseGeometry],
    *,
    src_crs: str,
    dst_crs: str,
    transform: Affine,
    height: int,
    width: int,
    erode_px: int = 0,
) -> np.ndarray:
    """Rasterise polygons into a (height, width) uint8 {0,1} mask.

    polygons are in `src_crs`; the raster is in `dst_crs`. If they
    differ, polygons are reprojected with rasterio.warp.transform_geom.

    erode_px>0 shrinks the positive region by that many pixels via
    a square structuring element; useful for keeping ambiguous label
    boundaries from dominating the loss.
    """
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)

    if src_crs != dst_crs:
        shapes = [
            (transform_geom(src_crs, dst_crs, mapping(p)), 1) for p in polygons
        ]
    else:
        shapes = [(mapping(p), 1) for p in polygons]

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    if erode_px > 0:
        struct = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), dtype=bool)
        mask = binary_erosion(mask.astype(bool), structure=struct).astype(np.uint8)

    return mask


__all__ = ["polygons_to_mask"]
