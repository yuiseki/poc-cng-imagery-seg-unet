"""GroundTruthSource abstract base.

A GroundTruthSource produces geometries (in WGS84 lon/lat) for a bbox
+ a feature query. The feature query is opaque to the source
(FeatureSpec controls what gets asked for), so the same source can
be reused for buildings, parks, roads, etc.

Two methods:
  fetch_polygons   : the polygon-only convenience, kept for backward compat
  fetch_with_tags  : geometry + OSM tags, needed by features that have to
                     post-process (e.g. roads = LineStrings buffered by
                     a width derived from the highway tag)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shapely.geometry.base import BaseGeometry


class GroundTruthSource(ABC):
    """Pluggable ground-truth backend (Overpass / vector tile / ...)."""

    name: str = "abstract"

    @abstractmethod
    def fetch_polygons(
        self,
        bbox: tuple[float, float, float, float],
        feature_query: object,
    ) -> list[BaseGeometry]:
        """Return polygons in WGS84 lon/lat matching feature_query
        inside bbox.

        The feature_query type is up to the FeatureSpec implementation;
        each GroundTruthSource is paired with FeatureSpec variants it
        understands (Overpass-QL string, MVT layer name, etc.).
        """
        raise NotImplementedError

    def fetch_with_tags(
        self,
        bbox: tuple[float, float, float, float],
        feature_query: object,
    ) -> list[tuple[BaseGeometry, dict[str, str]]]:
        """Return (geometry, OSM-tags) pairs in WGS84.

        Default implementation wraps `fetch_polygons` with empty tags
        — sufficient for features whose post-processing is identity
        (BuildingFeature, ParkFeature). Sources that have richer
        metadata available (OverpassGT) should override.
        """
        return [(g, {}) for g in self.fetch_polygons(bbox, feature_query)]
