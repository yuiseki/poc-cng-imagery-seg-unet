"""Polygons + transform -> binary mask."""

from __future__ import annotations

import numpy as np
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping, shape

from imagery_seg.rasterize import polygons_to_mask


def _reproject(polys, src_crs, dst_crs):
    return [shape(transform_geom(src_crs, dst_crs, mapping(p))) for p in polys]


def test_polygons_to_mask_square_in_3857():
    """A WGS84 polygon covering the whole 4x4 raster -> all-ones mask."""
    bbox = (139.0, 35.0, 140.0, 36.0)
    # Output transform lives in EPSG:3857; reproject the polygon to match.
    west, south, east, north = (
        15473131.0, 4163881.0, 15584728.0, 4302966.0,  # approx atami area in 3857
    )
    transform = from_bounds(west, south, east, north, 4, 4)
    poly = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    mask = polygons_to_mask(
        [poly], src_crs="EPSG:4326", dst_crs="EPSG:3857",
        transform=transform, height=4, width=4,
    )
    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8
    # The square fully covers the raster -> all pixels positive.
    assert int(mask.sum()) == 16


def test_polygons_to_mask_empty_when_no_polygons():
    transform = from_bounds(0, 0, 1, 1, 8, 8)
    mask = polygons_to_mask(
        [], src_crs="EPSG:4326", dst_crs="EPSG:4326",
        transform=transform, height=8, width=8,
    )
    assert mask.shape == (8, 8)
    assert int(mask.sum()) == 0


def test_polygons_to_mask_erode():
    """1px erode peels the boundary off a small polygon."""
    transform = from_bounds(0, 0, 10, 10, 10, 10)  # 1 unit per pixel
    poly = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])  # 6x6 box
    base = polygons_to_mask(
        [poly], src_crs="EPSG:4326", dst_crs="EPSG:4326",
        transform=transform, height=10, width=10,
    )
    eroded = polygons_to_mask(
        [poly], src_crs="EPSG:4326", dst_crs="EPSG:4326",
        transform=transform, height=10, width=10, erode_px=1,
    )
    assert int(eroded.sum()) < int(base.sum())
