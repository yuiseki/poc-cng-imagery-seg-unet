"""FeatureSpec abstract base.

A FeatureSpec is the "what to look for" knob: it produces a query
that a GroundTruthSource can resolve to polygons, and carries class
metadata (name, expected positive ratio, optional erode/dilate for
mask hygiene).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class FeatureSpec(ABC):
    """Identifies the geometric feature to be segmented.

    name: short identifier used in cache keys, recipe files, and the
        output mask filename.
    erode_px: pixels to erode the rasterised mask by, to avoid
        ambiguous label boundaries dominating the loss. 0 = no erosion.

    min_positive_polygon_count / min_positive_pixel_fraction:
        Phase B hard-filter thresholds, used by `check_aoi` to decide
        whether an AOI has enough positive-class signal for training.
        AND semantic — both must be satisfied, with a 0 value meaning
        "skip this check". Defaults match BuildingFeature so existing
        recipes keep their behavior; features whose positive-class
        density looks different (parks, roads, parking lots) override.
    """
    name: str
    erode_px: int = 0
    min_positive_polygon_count: int = 30
    min_positive_pixel_fraction: float = 0.0

    @abstractmethod
    def overpass_query(self, bbox: tuple[float, float, float, float]) -> str:
        """Return Overpass-QL fetching this feature inside bbox.

        Implementations that don't support Overpass should raise
        NotImplementedError.
        """
        raise NotImplementedError

    def to_polygons(
        self,
        geoms_with_tags: list[tuple[BaseGeometry, dict[str, str]]],
    ) -> list[BaseGeometry]:
        """Post-process raw OSM geometries into the final polygon mask
        input. Default: passes through any geometry that is already a
        Polygon and drops everything else (LineString, Point, ...).

        Features that work on LineStrings (RoadFeature) override this
        to do tag-aware buffering. Features that work on polygons
        (BuildingFeature, ParkFeature) inherit the default.
        """
        return [g for g, _ in geoms_with_tags if isinstance(g, Polygon)]

    def vector_tile_layer(self) -> str:
        """Return MVT layer name when fetched via vector tiles, or
        raise NotImplementedError if this feature isn't expressible as
        a single layer.
        """
        raise NotImplementedError(
            f"{self.name} does not declare a vector-tile layer"
        )
