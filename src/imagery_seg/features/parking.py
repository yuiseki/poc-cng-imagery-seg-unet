"""Parking feature: OSM `amenity=parking` (+ `parking=surface`) ways.

Parking lots in OSM are stored as closed polygons (like buildings),
so this Feature uses the default identity `to_polygons` — no buffering
required. The catch is that parking polygons are relatively rare per
AOI (a typical residential Japanese cell has 1-5, vs. hundreds of
buildings), and individual polygons are large. The Phase B hard filter
gates this case via `min_positive_polygon_count=2` plus
`min_positive_pixel_fraction=0.02`: don't require many polygons, but
do require at least 2% of valid pixels to be tagged as parking.

We intentionally pull both `amenity=parking` (the canonical tag) and
`parking=surface` (an extra qualifier sometimes used without the
amenity tag). Multi-storey and underground parking are excluded —
they aren't a single ground-level polygon you can pick out of aerial
imagery in a meaningful way for binary segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import FeatureSpec


@dataclass(frozen=True)
class ParkingFeature(FeatureSpec):
    """OSM amenity=parking ways → binary parking mask."""

    name: str = "parking"
    erode_px: int = 0  # parking edges are usually crisp in OSM mapping
    min_positive_polygon_count: int = 2
    min_positive_pixel_fraction: float = 0.02

    def overpass_query(self, bbox: tuple[float, float, float, float]) -> str:
        west, south, east, north = bbox
        b = f"{south},{west},{north},{east}"
        # Match amenity=parking OR parking=surface, but exclude
        # multi-storey / underground variants (not aerial-detectable).
        return (
            "[out:json][timeout:60];"
            "("
            f'way["amenity"="parking"]({b});'
            f'way["parking"="surface"]({b});'
            ");"
            "out geom tags;"
        )

    def vector_tile_layer(self) -> str:
        return "parking"


__all__ = ["ParkingFeature"]
