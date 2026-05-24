"""Building feature: OSM `way["building"]`."""

from __future__ import annotations

from dataclasses import dataclass

from .base import FeatureSpec


@dataclass(frozen=True)
class BuildingFeature(FeatureSpec):
    name: str = "building"
    erode_px: int = 1  # 1px erode is what the sister repo settled on

    def overpass_query(self, bbox: tuple[float, float, float, float]) -> str:
        west, south, east, north = bbox
        # Overpass bbox order: (south, west, north, east)
        return (
            "[out:json][timeout:60];"
            f'way["building"]({south},{west},{north},{east});'
            "out geom;"
        )

    def vector_tile_layer(self) -> str:
        # Aligns with tile.yuiseki.net (planetiler-style) layer naming.
        return "building"
