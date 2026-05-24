"""Microsoft Planetary Computer Sentinel-2 L2A imagery (stub).

STAC search is fully implemented. COG read is left as a stub because
MPC requires SAS-token signing (via `planetary_computer.sign`) to
fetch the underlying blobs, and we don't yet need it for scaffold
verification. Wire signing in when the recipe actually trains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .base import FetchedImage, ImagerySource

logger = logging.getLogger("imagery_seg.imagery.sentinel2")

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class Sentinel2Item:
    id: str
    bbox: tuple[float, float, float, float]
    datetime: str | None
    visual_href: str
    cloud_cover: float | None


def _items_from_mpc_payload(payload: dict[str, Any]) -> list[Sentinel2Item]:
    items: list[Sentinel2Item] = []
    for f in payload.get("features", []) or []:
        assets = f.get("assets") or {}
        visual = assets.get("visual") or {}
        href = visual.get("href")
        bbox = f.get("bbox")
        if not href or not bbox or len(bbox) != 4:
            continue
        props = f.get("properties") or {}
        items.append(
            Sentinel2Item(
                id=f.get("id", ""),
                bbox=tuple(bbox),  # type: ignore[arg-type]
                datetime=props.get("datetime"),
                visual_href=href,
                cloud_cover=props.get("eo:cloud_cover"),
            )
        )
    # Prefer least-cloudy, then newest.
    items.sort(
        key=lambda it: (it.cloud_cover if it.cloud_cover is not None else 100,
                        -(int((it.datetime or "0").replace("-", "").replace(":", "").replace("T", "").replace("Z", "")[:14] or 0))),
    )
    return items


class Sentinel2Imagery(ImagerySource):
    name = "sentinel2"

    def __init__(self, stac_url: str = MPC_STAC, timeout: float = 30.0) -> None:
        self.stac_url = stac_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def search(
        self,
        bbox: tuple[float, float, float, float],
        datetime: str | None = None,
        limit: int = 10,
    ) -> list[Sentinel2Item]:
        params: dict[str, Any] = {
            "collections": "sentinel-2-l2a",
            "bbox": ",".join(str(v) for v in bbox),
            "limit": limit,
        }
        if datetime:
            params["datetime"] = datetime
        resp = self._http().get(f"{self.stac_url}/search", params=params)
        resp.raise_for_status()
        return _items_from_mpc_payload(resp.json())

    def fetch_for_bbox(
        self,
        bbox: tuple[float, float, float, float],
        max_side: int = 1024,
        dst_crs: str = "EPSG:3857",
    ) -> FetchedImage:
        # MPC blob URLs need a SAS token from
        # planetarycomputer.microsoft.com/api/sas/v1/token. Wire that
        # in (via planetary_computer.sign or a manual GET) when this
        # source moves past scaffold and into actual training.
        raise NotImplementedError(
            "Sentinel2Imagery.fetch_for_bbox: needs SAS-token signing "
            "for MPC blob URLs. STAC search works; signing TBD."
        )


__all__ = ["Sentinel2Imagery", "Sentinel2Item", "_items_from_mpc_payload"]
