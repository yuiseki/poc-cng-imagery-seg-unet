"""Overpass + BuildingFeature tests.

Unit tests parse a committed Overpass JSON. Integration tests hit
overpass.yuiseki.net (self-hosted, no rate limit worth worrying about).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from imagery_seg.features.building import BuildingFeature
from imagery_seg.ground_truth.overpass import OverpassGT, _polygons_from_payload

FIXTURES = Path(__file__).parent / "fixtures"


def test_building_overpass_query_contains_building_tag():
    bbox = (139.075, 35.115, 139.080, 35.120)
    q = BuildingFeature().overpass_query(bbox)
    assert 'way["building"]' in q
    # Overpass bbox order is (south, west, north, east)
    assert "35.115" in q and "139.075" in q
    assert "35.12" in q and "139.08" in q


def test_polygons_from_payload_parses_ways():
    payload = json.loads((FIXTURES / "overpass_buildings_atami.json").read_text())
    polys = _polygons_from_payload(payload)
    assert len(polys) >= 1
    for p in polys:
        assert isinstance(p, BaseGeometry)
        assert p.geom_type == "Polygon"
        west, south, east, north = p.bounds
        # Within Atami AOI ish
        assert 139.07 <= west < east <= 139.09
        assert 35.11 <= south < north <= 35.13


def test_polygons_from_payload_skips_short_ways():
    payload = {
        "elements": [
            {"type": "way", "geometry": [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}]},
            {"type": "way", "geometry": [
                {"lat": 0, "lon": 0},
                {"lat": 0, "lon": 1},
                {"lat": 1, "lon": 1},
                {"lat": 0, "lon": 0},
            ]},
            {"type": "node", "lat": 0, "lon": 0},
        ]
    }
    polys = _polygons_from_payload(payload)
    assert len(polys) == 1


def test_overpass_source_name():
    assert OverpassGT().name == "overpass"


# ---- E2E ----------------------------------------------------------------


@pytest.mark.integration
def test_e2e_overpass_buildings_atami():
    """Hit overpass.yuiseki.net for real.

    Same bbox as the fixture; the live count should be in the same
    order of magnitude.
    """
    src = OverpassGT()
    bbox = (139.075, 35.115, 139.080, 35.120)
    polys = src.fetch_polygons(bbox, BuildingFeature().overpass_query(bbox))
    assert len(polys) >= 1
    assert all(isinstance(p, Polygon) for p in polys)
