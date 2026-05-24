"""PyTorch Dataset wrapping (image, mask) pairs produced by a Recipe.

Pulls imagery + GT polygons for each AOI in the recipe and rasterises
the polygons into the imagery's grid. Per-sample dict layout matches
what segmentation_models_pytorch examples expect.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .aoi_cache import AOICache
from .imagery.base import FetchedImage
from .rasterize import polygons_to_mask
from .recipe import Recipe


def _pixel_area_m2(transform, crs: str, bbox: tuple[float, float, float, float]) -> float:
    """Approximate real-world pixel area in m² for a fetched window.

    For EPSG:3857 the transform's a/e are in pseudo-metres that are stretched
    by 1/cos(lat); we undo that with cos²(lat) at the bbox centre. For other
    projected CRSes we assume the units are already metres.
    """
    pixel_w = abs(transform.a)
    pixel_h = abs(transform.e)
    if crs.upper().replace(" ", "") == "EPSG:3857":
        lat_center = (bbox[1] + bbox[3]) / 2.0
        cos_lat = math.cos(math.radians(lat_center))
        return float(pixel_w * pixel_h * cos_lat * cos_lat)
    return float(pixel_w * pixel_h)


class RecipeDataset(Dataset):
    """One sample per AOI declared on the recipe.

    With `cache=None` every __getitem__ call hits imagery + GT sources.
    Pass an AOICache to reuse window rasters across epochs / runs.

    `augment=True` applies geometric (H/V flip + rot90 ∈ {0..3}) and
    photometric (brightness ±20%) transforms on the fly. The cache is
    populated with the un-augmented imagery — augmentation runs after
    the cache read so the on-disk cache is recipe-agnostic. Val datasets
    should always use `augment=False` so val metrics stay deterministic.
    """

    def __init__(
        self,
        recipe: Recipe,
        max_side: int = 1024,
        dst_crs: str = "EPSG:3857",
        cache: AOICache | None = None,
        augment: bool = False,
        color_jitter: bool = True,
    ) -> None:
        self.recipe = recipe
        self.max_side = max_side
        self.dst_crs = dst_crs
        self.cache = cache
        self.augment = augment
        self.color_jitter = color_jitter

    def __len__(self) -> int:
        return len(self.recipe.aois)

    def _load_image(self, bbox: tuple[float, float, float, float]) -> FetchedImage:
        if self.cache is None:
            return self.recipe.imagery.fetch_for_bbox(
                bbox, max_side=self.max_side, dst_crs=self.dst_crs,
            )
        ns = self.recipe.imagery_cache_namespace
        hit = self.cache.get_image(ns, bbox, self.max_side, self.dst_crs)
        if hit is not None:
            return hit
        # Single-process: serialise concurrent fills against this key.
        with self.cache.image_lock(ns, bbox, self.max_side, self.dst_crs):
            hit = self.cache.get_image(ns, bbox, self.max_side, self.dst_crs)
            if hit is not None:
                return hit
            img = self.recipe.imagery.fetch_for_bbox(
                bbox, max_side=self.max_side, dst_crs=self.dst_crs,
            )
            self.cache.put_image(ns, bbox, self.max_side, self.dst_crs, img)
            return img

    def _load_polygons(self, bbox: tuple[float, float, float, float]):
        """Fetch raw OSM geoms + tags, apply the feature's to_polygons.

        The on-disk cache stores the *post-processed* polygons (e.g.
        roads buffered to widths) so future hits skip both the network
        and the buffering pass. Cache invalidation lives at the
        polygon_cache_namespace level (= ground_truth + feature) so a
        feature spec change (e.g. wider road widths) wants the cache
        cleared.
        """
        if self.cache is None:
            geoms_with_tags = self.recipe.ground_truth.fetch_with_tags(
                bbox, self.recipe.feature.overpass_query(bbox),
            )
            return self.recipe.feature.to_polygons(geoms_with_tags)
        ns = self.recipe.polygon_cache_namespace
        hit = self.cache.get_polygons(ns, bbox)
        if hit is not None:
            return hit
        with self.cache.polygon_lock(ns, bbox):
            hit = self.cache.get_polygons(ns, bbox)
            if hit is not None:
                return hit
            geoms_with_tags = self.recipe.ground_truth.fetch_with_tags(
                bbox, self.recipe.feature.overpass_query(bbox),
            )
            polys = self.recipe.feature.to_polygons(geoms_with_tags)
            self.cache.put_polygons(ns, bbox, polys)
            return polys

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.recipe.aois):
            raise IndexError(index)
        aoi = self.recipe.aois[index]
        bbox = aoi.bbox

        img = self._load_image(bbox)
        polys = self._load_polygons(bbox)
        mask = polygons_to_mask(
            polys,
            src_crs="EPSG:4326",
            dst_crs=img.crs,
            transform=img.transform,
            height=img.height,
            width=img.width,
            erode_px=self.recipe.feature.erode_px,
        )

        # uint8 [0, 255] -> float32 [0, 1]
        arr = img.array.astype(np.float32) / 255.0
        # Ensure 3 channels (smp.Unet default expects in_channels=3)
        if arr.shape[0] > 3:
            arr = arr[:3]
        elif arr.shape[0] < 3:
            pad = np.zeros((3 - arr.shape[0], arr.shape[1], arr.shape[2]),
                           dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        image_t = torch.from_numpy(arr)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        # valid_mask defaults to all-ones for backward compatibility with
        # legacy cache entries (3-band, no validity stored).
        if img.valid_mask is not None:
            valid_t = torch.from_numpy(img.valid_mask.astype(np.uint8))
        else:
            valid_t = torch.ones_like(mask_t, dtype=torch.uint8)
        if self.augment:
            image_t, mask_t, valid_t = _apply_augmentation(
                image_t, mask_t, valid_t, color_jitter=self.color_jitter,
            )

        return {
            "image": image_t,
            "mask": mask_t,
            "valid_mask": valid_t,
            "aoi_index": index,
            "asset_id": img.asset_id,
            "bbox": bbox,
            "region": aoi.region,
            "notes": aoi.notes,
            "pixel_area_m2": _pixel_area_m2(img.transform, img.crs, bbox),
            # Transform + crs are needed by `imagery-seg infer` to vectorise
            # the prediction mask back to WGS84 GeoJSON. DataLoader's custom
            # collate only stacks image+mask so non-tensor fields stay safe.
            "transform": img.transform,
            "crs": img.crs,
        }


def _apply_augmentation(
    image: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor,
    *,
    color_jitter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """D4 symmetry (H/V flip + rot90) + optional brightness jitter ±20%.

    image: (C, H, W) float in [0, 1]
    mask:  (H, W) int64 {0, 1}
    valid: (H, W) uint8 {0, 1}
    Mask + valid get the same geometric ops as image — color jitter is
    image-only (would corrupt class labels / validity).
    """
    # Horizontal flip (last axis)
    if torch.rand(()).item() < 0.5:
        image = image.flip(-1)
        mask = mask.flip(-1)
        valid = valid.flip(-1)
    # Vertical flip (second-to-last axis)
    if torch.rand(()).item() < 0.5:
        image = image.flip(-2)
        mask = mask.flip(-2)
        valid = valid.flip(-2)
    # rot90: 90/270° swap H and W, which breaks batching for non-square chips.
    # Allow {0,90,180,270} for square images; only {0, 180} otherwise.
    h, w = int(image.shape[-2]), int(image.shape[-1])
    if h == w:
        k = int(torch.randint(4, ()).item())
    else:
        k = int(torch.randint(2, ()).item()) * 2  # {0, 2} = identity or 180°
    if k:
        image = torch.rot90(image, k, dims=(-2, -1))
        mask = torch.rot90(mask, k, dims=(-2, -1))
        valid = torch.rot90(valid, k, dims=(-2, -1))
    # Brightness jitter: factor in [0.8, 1.2]. Optional — skip for HOTOSM
    # same-COG training where there's no inherent illumination variation.
    if color_jitter:
        brightness = 1.0 + (torch.rand(()).item() - 0.5) * 0.4
        image = (image * brightness).clamp(0.0, 1.0)
    return image, mask, valid


__all__ = ["RecipeDataset"]
