"""HOTOSM OpenAerialMap imagery source.

Thin wrapper around the HOTOSM STAC API (`api.imagery.hotosm.org/stac`)
that exposes a single `fetch_for_bbox()` call returning a FetchedImage.

Reads COGs via `/vsicurl/` so only the requested window's bytes are
pulled. This is a direct port of the production-ready logic in the
sister poc-cng-hotosm-vector-seg repo, refactored to fit the
ImagerySource ABC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.warp import transform_bounds as warp_transform_bounds

from .base import FetchedImage, ImagerySource

logger = logging.getLogger("imagery_seg.imagery.hotosm")

STAC_API_URL = "https://api.imagery.hotosm.org/stac"


@dataclass(frozen=True)
class HotosmItem:
    id: str
    collection: str | None
    bbox: tuple[float, float, float, float]
    datetime: str | None
    cog_url: str


def _pick_cog_href(item: dict[str, Any]) -> str | None:
    """Return the HTTPS URL of the COG asset, preferring 'visual'."""
    assets = item.get("assets") or {}
    asset = assets.get("visual")
    if not asset:
        for cand in assets.values():
            media = (cand.get("type") or "").lower()
            if "cloud-optimized" in media or "geotiff" in media:
                asset = cand
                break
    if not asset:
        return None
    href = asset.get("href")
    if href and href.startswith(("http://", "https://")):
        return href
    alternates = asset.get("alternate") or {}
    for alt in alternates.values():
        alt_href = alt.get("href") if isinstance(alt, dict) else None
        if alt_href and alt_href.startswith(("http://", "https://")):
            return alt_href
    return href


class HotosmSTAC:
    """Minimal HOTOSM STAC client (POST /search only)."""

    def __init__(self, api_url: str = STAC_API_URL, timeout: float = 30.0) -> None:
        self.api_url = api_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def search(
        self,
        bbox: tuple[float, float, float, float],
        limit: int = 20,
        datetime: str | None = None,
    ) -> list[HotosmItem]:
        body: dict[str, Any] = {"bbox": list(bbox), "limit": limit}
        if datetime is not None:
            body["datetime"] = datetime
        resp = self._client.post(f"{self.api_url}/search", json=body)
        resp.raise_for_status()
        return self._items_from_payload(resp.json())

    @staticmethod
    def _items_from_payload(payload: dict[str, Any]) -> list[HotosmItem]:
        """Pure parser, broken out so tests can feed it fixtures.

        Skips items whose `properties.instruments` contains "DSM": those
        are digital surface models (single-band elevation rasters) that
        share the same bbox as their paired RGB item but would silently
        train a model on elevation values instead of pixel colours.
        """
        items: list[HotosmItem] = []
        for f in payload.get("features", []) or []:
            if _is_dsm_item(f):
                continue
            cog = _pick_cog_href(f)
            if not cog:
                continue
            bbox_field = f.get("bbox")
            if not bbox_field or len(bbox_field) != 4:
                continue
            items.append(
                HotosmItem(
                    id=f.get("id", ""),
                    collection=f.get("collection"),
                    bbox=tuple(bbox_field),  # type: ignore[arg-type]
                    datetime=(f.get("properties") or {}).get("datetime"),
                    cog_url=cog,
                )
            )
        items.sort(key=lambda i: i.datetime or "", reverse=True)
        return items


class HotosmImagery(ImagerySource):
    """ImagerySource backed by HOTOSM OpenAerialMap COGs.

    Each fetch_for_bbox call picks the newest item that covers the
    bbox and reads a window from its visual COG via /vsicurl/, then
    reprojects to dst_crs at most max_side px on the long edge.
    """

    name = "hotosm"

    def __init__(self, stac_api_url: str = STAC_API_URL, timeout: float = 30.0) -> None:
        self.stac_api_url = stac_api_url
        self.timeout = timeout
        # STAC client + cache live lazily so tests can construct the
        # source without going to the network.
        self._stac: HotosmSTAC | None = None

    def _client(self) -> HotosmSTAC:
        if self._stac is None:
            self._stac = HotosmSTAC(self.stac_api_url, timeout=self.timeout)
        return self._stac

    def close(self) -> None:
        if self._stac is not None:
            self._stac.close()
            self._stac = None

    def fetch_for_bbox(
        self,
        bbox: tuple[float, float, float, float],
        max_side: int = 1024,
        dst_crs: str = "EPSG:3857",
    ) -> FetchedImage:
        """Iterate STAC candidates, return the first that delivers usable
        RGB imagery. STAC frequently lists multiple items for the same
        bbox (e.g. an RGB ortho + a separate single-band DSM with
        identical title and instruments), so blindly taking items[0] can
        silently train on elevation data. See _read_cog_window for the
        validation rules.
        """
        items = self._client().search(bbox, limit=20)
        if not items:
            raise RuntimeError(f"no HOTOSM items cover bbox {bbox}")
        skipped: list[tuple[str, str]] = []
        for item in items:
            try:
                return _read_cog_window(item, bbox, max_side, dst_crs)
            except _UnusableCogError as e:
                logger.info("HOTOSM item %s rejected: %s", item.id, e)
                skipped.append((item.id, str(e)))
        raise RuntimeError(
            f"no usable HOTOSM RGB item for bbox {bbox}; rejected: {skipped}"
        )


class _UnusableCogError(Exception):
    """Raised when a STAC item's COG isn't usable RGB imagery (wrong band
    count, wrong dtype, etc.). Caught by `HotosmImagery.fetch_for_bbox`
    so the next candidate item gets tried."""


_OK_DTYPES = {"uint8"}  # downstream normalises by /255; non-uint8 would distort


def _read_cog_window(
    item: HotosmItem,
    bbox_wgs84: tuple[float, float, float, float],
    max_side: int,
    dst_crs: str,
) -> FetchedImage:
    """Single-stage reproject from the source COG to dst_crs.

    Avoids the double-bilinear bug in earlier code (read-then-reproject)
    by feeding rasterio.band(src, i+1) directly into reproject(). Raises
    `_UnusableCogError` if the source isn't 3+ band uint8/uint16 — caught
    upstream so the next STAC candidate gets tried.
    """
    vsi = _vsicurl(item.cog_url)
    logger.info("HOTOSM /vsicurl read: %s", item.cog_url)
    with rasterio.open(vsi) as src:
        if src.count < 3:
            raise _UnusableCogError(
                f"COG has {src.count} band(s); RGB needs >= 3"
            )
        if any(d not in _OK_DTYPES for d in src.dtypes[:3]):
            raise _UnusableCogError(
                f"COG dtypes {src.dtypes[:3]} not in {_OK_DTYPES} "
                f"(likely a DSM / elevation product masquerading as imagery)"
            )
        src_crs = src.crs
        west_s, south_s, east_s, north_s = warp_transform_bounds(
            "EPSG:4326", dst_crs, *bbox_wgs84
        )
        # Compute output size such that max(out_w, out_h) == max_side.
        w = east_s - west_s
        h = north_s - south_s
        if w <= 0 or h <= 0:
            raise RuntimeError(f"degenerate bbox in {dst_crs}: {bbox_wgs84}")
        ratio = w / h
        if ratio >= 1.0:
            out_w = max_side
            out_h = max(1, int(round(max_side / ratio)))
        else:
            out_h = max_side
            out_w = max(1, int(round(max_side * ratio)))
        dst_transform = from_bounds(west_s, south_s, east_s, north_s, out_w, out_h)

        n_bands = min(src.count, 3)
        dst = np.zeros((n_bands, out_h, out_w), dtype="uint8")
        for i in range(n_bands):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=dst[i],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    # Pixels where every band is exactly 0 fell outside the COG's actual
    # flight path. Bilinear interpolation makes boundary pixels (1, 0, 0)
    # or similar — they get counted as valid, which is fine; the false-
    # positive boundary is 1-2 px and negligible compared to interior
    # solid-black regions in DRONEBIRD COGs.
    valid_mask = (dst.sum(axis=0) > 0).astype("uint8")
    return FetchedImage(
        array=dst,
        transform=dst_transform,
        crs=dst_crs,
        asset_id=item.id,
        valid_mask=valid_mask,
    )


def _vsicurl(url: str) -> str:
    if url.startswith(("http://", "https://")):
        return f"/vsicurl/{url}"
    return url


def _is_dsm_item(feature: dict[str, Any]) -> bool:
    """Detect HOTOSM STAC items that ship a DSM raster rather than RGB.

    Looks at properties.instruments (e.g. "Optical/DSM" vs "Optical")
    and the title (DRONEBIRD items typically append "DSM"). Either
    signal alone is enough — we'd rather skip an item than train on
    elevation data by mistake.
    """
    props = feature.get("properties") or {}
    instruments = props.get("instruments") or []
    if any("DSM" in str(inst) for inst in instruments):
        return True
    title = str(props.get("title") or "")
    if "DSM" in title:
        return True
    return False


__all__ = ["HotosmImagery", "HotosmItem", "HotosmSTAC", "_pick_cog_href"]
