"""Overpass-based ground-truth source.

Defaults to `overpass.yuiseki.net` (yuiseki's self-hosted mirror,
no rate-limit concerns) so training jobs don't get 504'd by the
shared public endpoint. Can be pointed elsewhere by passing
`endpoint=...`.

POSTs to /api/interpreter — the self-hosted mirror accepts GET too
but the public overpass-api.de mirror 406s GET from non-browser
User-Agent strings; POST works on every endpoint we care about.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from .base import GroundTruthSource

logger = logging.getLogger("imagery_seg.ground_truth.overpass")

DEFAULT_ENDPOINT = "https://overpass.yuiseki.net/api/interpreter"


class OverpassGT(GroundTruthSource):
    name = "overpass"

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "imagery-seg-unet/0.1"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def fetch_polygons(
        self,
        bbox: tuple[float, float, float, float],
        feature_query: object,
    ) -> list[BaseGeometry]:
        return [g for g, _ in self.fetch_with_tags(bbox, feature_query)
                if isinstance(g, Polygon)]

    def fetch_with_tags(
        self,
        bbox: tuple[float, float, float, float],
        feature_query: object,
    ) -> list[tuple[BaseGeometry, dict[str, str]]]:
        if not isinstance(feature_query, str):
            raise TypeError(
                "OverpassGT expects feature_query as an Overpass-QL string "
                "(see FeatureSpec.overpass_query)"
            )
        logger.info("Overpass POST %s", self.endpoint)
        resp = self._http().post(
            self.endpoint,
            data={"data": feature_query},
        )
        resp.raise_for_status()
        return _geoms_with_tags_from_payload(resp.json())


def _polygons_from_payload(payload: dict[str, Any]) -> list[Polygon]:
    """Polygon-only convenience built on top of _geoms_with_tags_from_payload."""
    return [g for g, _ in _geoms_with_tags_from_payload(payload)
            if isinstance(g, Polygon)]


def _geoms_with_tags_from_payload(
    payload: dict[str, Any],
) -> list[tuple[BaseGeometry, dict[str, str]]]:
    """Pure parser: Overpass JSON -> list of (geometry, tags) pairs in WGS84.

    Auto-detects open vs closed ways:
      - closed (first == last point and >= 4 vertices) → Polygon
      - open (LineString candidate) → LineString if >= 2 vertices
    Relations / multipolygons aren't handled (buildings are overwhelmingly
    ways in practice, roads always are).
    """
    out: list[tuple[BaseGeometry, dict[str, str]]] = []
    for el in payload.get("elements", []) or []:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in geom]
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        is_closed = len(coords) >= 4 and coords[0] == coords[-1]
        try:
            if is_closed:
                shp: BaseGeometry = Polygon(coords)
            else:
                shp = LineString(coords)
        except Exception:  # pragma: no cover -- defensive
            continue
        if not shp.is_valid or shp.is_empty:
            continue
        out.append((shp, tags))
    return out
