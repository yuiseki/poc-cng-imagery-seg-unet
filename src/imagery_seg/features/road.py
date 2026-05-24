"""Road feature: OSM `highway=*` ways buffered into polygons.

Roads are tagged as LineStrings in OSM (not Polygons like buildings),
so this Feature has to buffer each line by a width derived from the
`highway` tag before the polygon→mask rasterise step can use it.

Width source priority:
  1. explicit `width` tag (rounded numeric metres),
  2. `lanes` × 3.5 m as a fallback,
  3. table lookup keyed on `highway` tag value.

Pedestrian / cycleway / footway / path are intentionally excluded —
their imagery signal differs enough (narrow, often shaded under tree
canopy) that mixing them into the binary "road" class hurts the
segmentation signal for the dominant drive-able-road class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from .base import FeatureSpec

# Approximate metric road widths, in metres, for the common drive-able
# OSM highway tag values. Conservative — gets the buffer in the right
# ballpark on Japan-style urban grids; real roads vary by jurisdiction.
DEFAULT_ROAD_WIDTHS_M: dict[str, float] = {
    "motorway": 12.0,       "motorway_link": 8.0,
    "trunk": 10.0,          "trunk_link": 7.0,
    "primary": 8.0,         "primary_link": 6.0,
    "secondary": 7.0,       "secondary_link": 5.0,
    "tertiary": 6.0,        "tertiary_link": 4.0,
    "unclassified": 5.0,
    "residential": 5.0,
    "living_street": 4.0,
    "service": 3.5,
}

# Tag values we route to OSM via overpass — same list as the width table,
# kept separate so users can subset or extend.
DEFAULT_HIGHWAY_VALUES: tuple[str, ...] = tuple(DEFAULT_ROAD_WIDTHS_M.keys())


def _parse_width_metres(width_tag: str | None) -> float | None:
    """Extract a metric width from the OSM `width` tag if present.
    The OSM convention is metres by default; we accept a leading
    numeric prefix and ignore unit suffixes.
    """
    if not width_tag:
        return None
    try:
        # tolerate "4.5", "4.5 m", "4.5m"
        head = width_tag.strip().split()[0].rstrip("m")
        v = float(head)
        return v if v > 0 else None
    except (ValueError, IndexError):
        return None


def _lanes_to_width(lanes_tag: str | None) -> float | None:
    """OSM `lanes` tag → assumed 3.5 m per lane (urban norm)."""
    if not lanes_tag:
        return None
    try:
        n = int(lanes_tag.split(";")[0])  # tolerate "2;3" multi-values
        return n * 3.5 if n > 0 else None
    except ValueError:
        return None


def _width_for_tags(tags: dict[str, str]) -> float:
    """Best-guess road width in metres for an OSM way's tags."""
    w = _parse_width_metres(tags.get("width"))
    if w is not None:
        return w
    w = _lanes_to_width(tags.get("lanes"))
    if w is not None:
        return w
    return DEFAULT_ROAD_WIDTHS_M.get(tags.get("highway", ""), 4.0)


@dataclass(frozen=True)
class RoadFeature(FeatureSpec):
    """Buffer OSM highway LineStrings into a binary 'road' polygon mask."""

    name: str = "road"
    erode_px: int = 0
    # Roads are LineStrings buffered into thin polygons; their per-polygon
    # area is small but they cover meaningful ground area. Hard-filter on
    # both count (at least 10 way segments) and pixel fraction (>= 3%).
    min_positive_polygon_count: int = 10
    min_positive_pixel_fraction: float = 0.03
    highway_values: tuple[str, ...] = field(default_factory=lambda: DEFAULT_HIGHWAY_VALUES)

    def overpass_query(self, bbox: tuple[float, float, float, float]) -> str:
        west, south, east, north = bbox
        b = f"{south},{west},{north},{east}"
        regex = "|".join(self.highway_values)
        # `out geom tags;` so the parser can read both the LineString and
        # the highway/width/lanes tags for width inference.
        return (
            "[out:json][timeout:60];"
            f'way["highway"~"^({regex})$"]({b});'
            "out geom tags;"
        )

    def vector_tile_layer(self) -> str:
        return "road"

    def to_polygons(
        self,
        geoms_with_tags: list[tuple[BaseGeometry, dict[str, str]]],
    ) -> list[BaseGeometry]:
        """Buffer each LineString by half its tag-derived width.

        The buffer is in WGS84 degrees, scaled by the mean of latitude
        and longitude metric scales at the bbox centre. This is good to
        ~±10% at Japan / temperate latitudes; for tropical or polar AOIs
        revisit with a local UTM projection.
        """
        if not geoms_with_tags:
            return []

        # Pick a single reference latitude from the centroid of the first
        # geometry — for AOI-sized bboxes (a few hundred m) the lat is
        # effectively constant within the AOI.
        first_geom = next((g for g, _ in geoms_with_tags), None)
        if first_geom is None:
            return []
        lat_ref = float(first_geom.centroid.y)
        # 1° lat ≈ 110540 m; 1° lon ≈ 111320 * cos(lat) m
        m_per_deg_lat = 110540.0
        m_per_deg_lon = 111320.0 * max(0.001, math.cos(math.radians(lat_ref)))
        avg_m_per_deg = 0.5 * (m_per_deg_lat + m_per_deg_lon)

        out: list[BaseGeometry] = []
        for geom, tags in geoms_with_tags:
            if not isinstance(geom, LineString):
                # Polygons (rare for highways but tolerated) pass through.
                if not geom.is_empty:
                    out.append(geom)
                continue
            half_m = _width_for_tags(tags) / 2.0
            if half_m <= 0:
                continue
            buffer_deg = half_m / avg_m_per_deg
            buffered = geom.buffer(buffer_deg, cap_style=2)  # flat caps
            if not buffered.is_empty:
                out.append(buffered)
        return out


__all__ = [
    "RoadFeature",
    "DEFAULT_ROAD_WIDTHS_M",
    "DEFAULT_HIGHWAY_VALUES",
]
