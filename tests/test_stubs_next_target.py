"""Tests for the "next natural evolution" backends:
Sentinel-2 imagery via Microsoft Planetary Computer + Park feature.

These are scaffolds — Sentinel-2 STAC search is implemented and tested
(unit + integration), Sentinel-2 COG read is left as NotImplementedError
because MPC requires SAS-token signing that we haven't wired in yet.
Park feature is implemented because the Overpass query for parks is
trivial and useful for the smoke pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagery_seg.features.park import ParkFeature
from imagery_seg.ground_truth.vector_tile import VectorTileGT
from imagery_seg.imagery.sentinel2 import Sentinel2Imagery, _items_from_mpc_payload

FIXTURES = Path(__file__).parent / "fixtures"


def test_sentinel2_source_name():
    assert Sentinel2Imagery().name == "sentinel2"


def test_sentinel2_parse_mpc_search():
    payload = json.loads((FIXTURES / "mpc_sentinel2_atami.json").read_text())
    items = _items_from_mpc_payload(payload)
    assert len(items) >= 1
    assert items[0].id.startswith("S2")
    # MPC visual asset href contains blob.core.windows.net
    assert "blob.core.windows.net" in items[0].visual_href


def test_sentinel2_fetch_for_bbox_not_implemented():
    """Read path is stubbed; we want the failure to be loud + explicit."""
    src = Sentinel2Imagery()
    with pytest.raises(NotImplementedError, match="SAS"):
        src.fetch_for_bbox((139.07, 35.10, 139.10, 35.13), max_side=128)


def test_park_feature_query_contains_park_tags():
    bbox = (139.075, 35.115, 139.080, 35.120)
    q = ParkFeature().overpass_query(bbox)
    assert 'leisure"="park' in q
    # ParkFeature also picks up landuse=recreation_ground per OSM convention
    assert "recreation_ground" in q or "landuse" in q


def test_park_feature_vector_tile_layer():
    # tile.yuiseki.net (planetiler-style) uses 'park' as a layer name
    assert ParkFeature().vector_tile_layer() == "park"


def test_vector_tile_gt_is_stub():
    """tile.yuiseki.net read is not implemented yet; should fail loudly."""
    src = VectorTileGT()
    with pytest.raises(NotImplementedError):
        src.fetch_polygons((0, 0, 1, 1), feature_query="building")


# ---- E2E ----------------------------------------------------------------


@pytest.mark.integration
def test_e2e_mpc_sentinel2_search():
    """MPC STAC search is rate-limited but unauthenticated for /search."""
    src = Sentinel2Imagery()
    items = src.search(
        (139.07, 35.10, 139.10, 35.13),
        datetime="2024-09-01/2024-09-30",
        limit=2,
    )
    assert len(items) >= 1
    assert items[0].id.startswith("S2")
