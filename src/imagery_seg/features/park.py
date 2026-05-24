"""Park feature: OSM `leisure=park` and friends."""

from __future__ import annotations

from dataclasses import dataclass

from .base import FeatureSpec


@dataclass(frozen=True)
class ParkFeature(FeatureSpec):
    name: str = "park"
    erode_px: int = 0  # parks tend to have soft edges already
    # A single large park polygon can dominate the AOI; gate on coverage
    # more than count.
    min_positive_polygon_count: int = 3
    min_positive_pixel_fraction: float = 0.05

    def overpass_query(self, bbox: tuple[float, float, float, float]) -> str:
        west, south, east, north = bbox
        b = f"{south},{west},{north},{east}"
        # Union the most common park-ish tags. recreation_ground catches
        # the Japanese "公園" tagged as landuse=recreation_ground in
        # places where leisure=park hasn't been applied.
        return (
            "[out:json][timeout:60];"
            "("
            f'way["leisure"="park"]({b});'
            f'way["landuse"="recreation_ground"]({b});'
            f'relation["leisure"="park"]({b});'
            ");"
            "out geom;"
        )

    def vector_tile_layer(self) -> str:
        return "park"
