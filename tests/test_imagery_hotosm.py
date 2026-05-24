"""HOTOSM imagery source tests.

`test_stac_parse_*` use a committed STAC fixture so unit-test runs
don't touch the network. `test_e2e_*` are marked `integration` and
fire real HTTP against api.imagery.hotosm.org + S3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imagery_seg.imagery.hotosm import (
    HotosmImagery,
    HotosmSTAC,
    _pick_cog_href,
    _read_cog_window,
    _UnusableCogError,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_stac_parse_search_response():
    """Fixture-based: parse a bbox-search response into HotosmItems."""
    payload = _load("hotosm_stac_atami.json")
    items = HotosmSTAC._items_from_payload(payload)
    assert len(items) >= 1
    first = items[0]
    assert first.id
    assert first.cog_url.startswith("https://")
    assert len(first.bbox) == 4
    # Datetime-sorted desc; newest-first means each item's datetime is
    # >= the next one's (treating None as "").
    dts = [(it.datetime or "") for it in items]
    assert dts == sorted(dts, reverse=True)


def test_stac_parse_item_response():
    payload = _load("hotosm_stac_item_atami.json")
    items = HotosmSTAC._items_from_payload(payload)
    assert len(items) == 1
    item = items[0]
    assert item.id == "60e5afbe5bc2dc00058bbe06"
    assert item.cog_url.endswith(".tif")


def test_pick_cog_href_prefers_visual():
    item = {
        "assets": {
            "visual": {"href": "https://example.org/v.tif", "type": "image/tiff"},
            "thumbnail": {"href": "https://example.org/t.png"},
        }
    }
    assert _pick_cog_href(item) == "https://example.org/v.tif"


def test_pick_cog_href_falls_back_to_geotiff_type():
    item = {
        "assets": {
            "other": {"href": "https://example.org/o.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized"},
        }
    }
    assert _pick_cog_href(item).endswith("/o.tif")


def test_pick_cog_href_none_when_no_asset():
    assert _pick_cog_href({"assets": {}}) is None


def test_hotosm_imagery_name():
    assert HotosmImagery().name == "hotosm"


def test_unusable_cog_raised_on_single_band_float(tmp_path: Path):
    """A 1-band float32 COG (e.g. DSM) should be rejected so the
    next STAC candidate gets tried — see the Manila incident where
    two items shared a title/instruments/bbox but only one was RGB."""
    import rasterio
    from rasterio.transform import from_bounds
    bad = tmp_path / "bad.tif"
    arr = np.full((1, 32, 32), 8.5, dtype=np.float32)
    with rasterio.open(
        bad, "w", driver="GTiff", count=1, height=32, width=32,
        dtype="float32", crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 32, 32),
    ) as ds:
        ds.write(arr)
    from imagery_seg.imagery.hotosm import HotosmItem
    item = HotosmItem(id="bad", collection=None, bbox=(0, 0, 1, 1),
                     datetime=None, cog_url=str(bad))
    with pytest.raises(_UnusableCogError):
        _read_cog_window(item, (0.0, 0.0, 1.0, 1.0), 64, "EPSG:4326")


def test_unusable_cog_raised_on_two_band_uint8(tmp_path: Path):
    """Anything < 3 bands is rejected even if dtype is uint8."""
    import rasterio
    from rasterio.transform import from_bounds
    bad = tmp_path / "two_band.tif"
    arr = np.full((2, 32, 32), 100, dtype=np.uint8)
    with rasterio.open(
        bad, "w", driver="GTiff", count=2, height=32, width=32,
        dtype="uint8", crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 32, 32),
    ) as ds:
        ds.write(arr)
    from imagery_seg.imagery.hotosm import HotosmItem
    item = HotosmItem(id="bad2", collection=None, bbox=(0, 0, 1, 1),
                     datetime=None, cog_url=str(bad))
    with pytest.raises(_UnusableCogError):
        _read_cog_window(item, (0.0, 0.0, 1.0, 1.0), 64, "EPSG:4326")


def test_dsm_items_are_skipped():
    """Items with 'DSM' in instruments OR title are skipped — they
    share a bbox with the paired RGB item and would otherwise be
    randomly picked when both items match a search."""
    payload = {
        "features": [
            {
                "id": "rgb-item",
                "bbox": [139.0, 35.0, 139.1, 35.1],
                "properties": {
                    "datetime": None,
                    "instruments": ["Optical"],
                    "title": "Aihara01, Sagamihara",
                },
                "assets": {"visual": {"href": "https://example.org/rgb.tif"}},
            },
            {
                "id": "dsm-by-instruments",
                "bbox": [139.0, 35.0, 139.1, 35.1],
                "properties": {
                    "datetime": None,
                    "instruments": ["Optical/DSM"],
                    "title": "Aihara01DSM, Sagamihara",
                },
                "assets": {"visual": {"href": "https://example.org/dsm1.tif"}},
            },
            {
                "id": "dsm-by-title-only",
                "bbox": [139.0, 35.0, 139.1, 35.1],
                "properties": {
                    "datetime": None,
                    "title": "Some DSM survey",  # title-only signal
                },
                "assets": {"visual": {"href": "https://example.org/dsm2.tif"}},
            },
        ]
    }
    items = HotosmSTAC._items_from_payload(payload)
    ids = {it.id for it in items}
    assert ids == {"rgb-item"}


# ---- E2E ----------------------------------------------------------------


@pytest.mark.integration
def test_e2e_stac_search_atami():
    stac = HotosmSTAC()
    try:
        items = stac.search((139.07, 35.10, 139.10, 35.13), limit=3)
    finally:
        stac.close()
    assert len(items) >= 1
    assert items[0].cog_url.startswith("https://")


@pytest.mark.integration
def test_e2e_fetch_for_bbox_atami(tmp_path: Path):
    """Real /vsicurl/ read against an HOTOSM COG.

    Uses a small max_side so the byte budget stays modest even when
    the upstream COG is hundreds of MB.
    """
    src = HotosmImagery()
    img = src.fetch_for_bbox(
        (139.075, 35.115, 139.080, 35.120),
        max_side=256,
        dst_crs="EPSG:3857",
    )
    assert isinstance(img.array, np.ndarray)
    assert img.array.shape[0] in (3, 4)
    assert img.array.shape[-1] <= 256 and img.array.shape[-2] <= 256
    assert img.crs == "EPSG:3857"
    assert img.asset_id  # stable id used in cache keys
    # At least some non-zero pixels — bbox is inside the AOI.
    assert int(img.array.sum()) > 0
