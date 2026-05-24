"""Vector-tile ground-truth source for `tile.yuiseki.net` (stub).

Resolves the bbox to the set of covering MVT tiles, fetches them,
decodes the requested layer with mapbox_vector_tile, reprojects to
WGS84, and returns polygons.

Currently a stub: tests use it to assert the abstraction shape is
right but reading + decoding MVT pulls in mapbox_vector_tile, which
we'd rather defer until the recipe actually needs it.
"""

from __future__ import annotations

import logging

from shapely.geometry.base import BaseGeometry

from .base import GroundTruthSource

logger = logging.getLogger("imagery_seg.ground_truth.vector_tile")


class VectorTileGT(GroundTruthSource):
    name = "vector_tile"

    def __init__(
        self,
        url_template: str = "https://tile.yuiseki.net/{z}/{x}/{y}.pbf",
        zoom: int = 14,
        timeout: float = 30.0,
    ) -> None:
        self.url_template = url_template
        self.zoom = zoom
        self.timeout = timeout

    def fetch_polygons(
        self,
        bbox: tuple[float, float, float, float],
        feature_query: object,
    ) -> list[BaseGeometry]:
        # When implemented:
        #   1. mercantile.tiles(west, south, east, north, [self.zoom]) -> tile list
        #   2. For each tile: GET url_template.format(z, x, y)
        #   3. mapbox_vector_tile.decode(payload)[feature_query] -> features
        #   4. Reproject from per-tile MVT extent to WGS84 lon/lat
        #   5. Clip to bbox, return as shapely polygons
        raise NotImplementedError(
            "VectorTileGT is a scaffold stub. Wire mapbox_vector_tile + "
            "mercantile when the recipe actually needs tile.yuiseki.net "
            "(see comments in this file for the integration sketch)."
        )
