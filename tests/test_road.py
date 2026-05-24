"""RoadFeature + OverpassGT.fetch_with_tags plumbing tests."""

from __future__ import annotations

from shapely.geometry import LineString, Polygon

from imagery_seg.features.road import (
    DEFAULT_ROAD_WIDTHS_M,
    RoadFeature,
    _lanes_to_width,
    _parse_width_metres,
    _width_for_tags,
)
from imagery_seg.ground_truth.overpass import _geoms_with_tags_from_payload


# ---- width parsing ----------------------------------------------------------


def test_parse_width_simple_number():
    assert _parse_width_metres("4.5") == 4.5
    assert _parse_width_metres("12") == 12.0


def test_parse_width_with_metres_suffix():
    assert _parse_width_metres("4.5 m") == 4.5
    assert _parse_width_metres("8m") == 8.0


def test_parse_width_rejects_garbage():
    assert _parse_width_metres(None) is None
    assert _parse_width_metres("") is None
    assert _parse_width_metres("wide") is None
    assert _parse_width_metres("-3") is None


def test_lanes_to_width():
    assert _lanes_to_width("2") == 7.0
    assert _lanes_to_width("4") == 14.0
    assert _lanes_to_width("2;3") == 7.0  # multi-value tolerated
    assert _lanes_to_width(None) is None
    assert _lanes_to_width("zero") is None


def test_width_for_tags_priority():
    # width tag wins over lanes
    assert _width_for_tags({"width": "9.5", "lanes": "2", "highway": "primary"}) == 9.5
    # lanes wins over highway default
    assert _width_for_tags({"lanes": "4", "highway": "primary"}) == 14.0
    # highway default when no width / lanes
    assert _width_for_tags({"highway": "primary"}) == DEFAULT_ROAD_WIDTHS_M["primary"]
    # unknown highway -> 4 m fallback
    assert _width_for_tags({"highway": "unknown_xyz"}) == 4.0


# ---- Overpass parser ---------------------------------------------------------


def test_overpass_parser_extracts_linestring_and_tags():
    payload = {
        "elements": [
            {
                "type": "way",
                "geometry": [
                    {"lon": 139.020, "lat": 35.184},
                    {"lon": 139.021, "lat": 35.185},
                    {"lon": 139.022, "lat": 35.186},
                ],
                "tags": {"highway": "residential", "name": "Foo street"},
            },
        ]
    }
    out = _geoms_with_tags_from_payload(payload)
    assert len(out) == 1
    geom, tags = out[0]
    assert isinstance(geom, LineString)
    assert tags == {"highway": "residential", "name": "Foo street"}


def test_overpass_parser_returns_polygon_for_closed_way():
    payload = {
        "elements": [
            {
                "type": "way",
                "geometry": [
                    {"lon": 139.0, "lat": 35.0},
                    {"lon": 139.0, "lat": 35.001},
                    {"lon": 139.001, "lat": 35.001},
                    {"lon": 139.0, "lat": 35.0},  # closed
                ],
                "tags": {"building": "yes"},
            },
        ]
    }
    out = _geoms_with_tags_from_payload(payload)
    assert len(out) == 1
    geom, tags = out[0]
    assert isinstance(geom, Polygon)
    assert tags == {"building": "yes"}


# ---- RoadFeature.to_polygons -------------------------------------------------


def test_road_feature_buffers_linestring_into_polygon():
    feat = RoadFeature()
    line = LineString([(139.020, 35.184), (139.025, 35.184)])
    out = feat.to_polygons([(line, {"highway": "primary"})])
    assert len(out) == 1
    assert isinstance(out[0], Polygon)
    assert out[0].area > 0


def test_road_feature_width_via_tag():
    feat = RoadFeature()
    line = LineString([(139.020, 35.184), (139.025, 35.184)])
    narrow = feat.to_polygons([(line, {"highway": "service"})])[0]
    wide = feat.to_polygons([(line, {"highway": "motorway"})])[0]
    assert wide.area > narrow.area  # 12m wider than 3.5m


def test_road_feature_explicit_width_overrides_highway():
    feat = RoadFeature()
    line = LineString([(139.020, 35.184), (139.025, 35.184)])
    a = feat.to_polygons([(line, {"highway": "service", "width": "12"})])[0]
    b = feat.to_polygons([(line, {"highway": "motorway"})])[0]
    # Same width tag -> similar areas
    assert abs(a.area - b.area) / b.area < 0.05


def test_road_feature_overpass_query_lists_drivable_only():
    feat = RoadFeature()
    q = feat.overpass_query((139.0, 35.0, 139.01, 35.01))
    # The drive-able set is in the regex
    assert '"highway"' in q
    assert "residential" in q
    assert "motorway" in q
    # footway is not (intentionally excluded)
    assert "footway" not in q
    assert "out geom tags" in q


def test_road_feature_drops_unbufferable_input():
    feat = RoadFeature()
    out = feat.to_polygons([])
    assert out == []


# ---- FeatureSpec default to_polygons (BuildingFeature path) ------------------


def test_default_to_polygons_keeps_polygons_drops_lines():
    """Building-like features rely on the default to_polygons which passes
    Polygons through and discards LineStrings."""
    from imagery_seg.features.building import BuildingFeature
    feat = BuildingFeature()
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    line = LineString([(0, 0), (1, 1)])
    out = feat.to_polygons([(poly, {"building": "yes"}), (line, {"highway": "x"})])
    assert len(out) == 1
    assert isinstance(out[0], Polygon)
