"""ImagerySource abstract base.

An ImagerySource is responsible for producing a numpy array of pixel
values + a rasterio-style transform + CRS for an arbitrary bbox.
Implementations decide how they search for source assets (STAC, fixed
URL template, local files), how they read them (COG /vsicurl/, tile
mosaic), and what bands/channels they expose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine


@dataclass(frozen=True)
class FetchedImage:
    """A fetched window of imagery.

    array: (C, H, W) uint8 or uint16. The exact channel layout is
        defined by the source (e.g. HOTOSM = RGB, Sentinel-2 = B04/B03/B02
        true colour for now).
    transform: rasterio Affine in `crs`'s units (i.e. for EPSG:3857
        the unit is metre).
    crs: EPSG string, e.g. "EPSG:3857".
    asset_id: stable identifier of the underlying asset(s), used in
        cache keys.
    valid_mask: (H, W) uint8 {0,1} marking which output pixels were
        sourced from real data vs. fell outside the COG's flight path
        (typical for DRONEBIRD COGs that have a rectangular bbox but
        an irregular interior coverage). When None, callers should
        treat all pixels as valid.
    """
    array: np.ndarray
    transform: Affine
    crs: str
    asset_id: str
    valid_mask: np.ndarray | None = None

    @property
    def height(self) -> int:
        return int(self.array.shape[-2])

    @property
    def width(self) -> int:
        return int(self.array.shape[-1])


class ImagerySource(ABC):
    """Pluggable imagery backend.

    Sources should be cheap to construct (no network on __init__);
    network/IO happens inside fetch_for_bbox so callers can wrap that
    in caches and retries.
    """

    #: short identifier used in cache keys + recipe files
    name: str = "abstract"

    @abstractmethod
    def fetch_for_bbox(
        self,
        bbox: tuple[float, float, float, float],
        max_side: int = 1024,
        dst_crs: str = "EPSG:3857",
    ) -> FetchedImage:
        """Return imagery covering bbox (lon/lat WGS84), reprojected
        into dst_crs, at most max_side pixels on the longest side.

        Must be deterministic for the same (bbox, max_side, dst_crs)
        so the cache layer can key on these arguments.
        """
        raise NotImplementedError
