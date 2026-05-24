"Segmentation inference API — GET /seg/{feature_type}/{z}/{x}/{y}.mvt"
from __future__ import annotations

import asyncio
import io
import logging
import os
from pathlib import Path

import httpx
import mapbox_vector_tile
import mercantile
import numpy as np
import rasterio.features
import rasterio.transform
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from imagery_seg.model import build_unet, pad_to_multiple

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
IMAGERY_TILE_BASE = os.getenv(
    "IMAGERY_TILE_BASE", "https://hotosm-imagery-tile.yuiseki.com"
)
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
EMPTY_MVT = b""

# Map URL segment → (model filename, MVT layer name)
FEATURE_CONFIG: dict[str, tuple[str, str]] = {
    "buildings": ("buildings.pt", "building"),
    "roads":     ("roads.pt",     "road"),
    "parkings":  ("parkings.pt",  "parking"),
    "parks":     ("parks.pt",     "park"),
}

# Loaded models: feature_type → (torch.nn.Module, threshold)
_models: dict[str, tuple[torch.nn.Module, float]] = {}

_wm_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

app = FastAPI(title="poc-cng-imagery-seg-unet")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def _load_model(pt_path: Path) -> tuple[torch.nn.Module, float]:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    encoder = meta.get("encoder", "resnet34")
    encoder_weights = meta.get("encoder_weights", "imagenet")
    classes = meta.get("classes", 2)
    threshold = float(ckpt.get("best_threshold", DEFAULT_THRESHOLD))
    model = build_unet(encoder, encoder_weights, classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, threshold


@app.on_event("startup")
def _startup() -> None:
    for feature_type, (filename, _layer) in FEATURE_CONFIG.items():
        pt_path = MODELS_DIR / filename
        if pt_path.exists():
            logger.info("Loading model %s from %s", feature_type, pt_path)
            try:
                _models[feature_type] = _load_model(pt_path)
                logger.info("Loaded %s (threshold=%.3f)", feature_type, _models[feature_type][1])
            except Exception as exc:
                logger.error("Failed to load %s: %s", feature_type, exc)
        else:
            logger.info("No model for %s (expected %s)", feature_type, pt_path)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "loaded": list(_models.keys())}


def _tile_transform(x: int, y: int, z: int, size: int) -> rasterio.transform.Affine:
    b = mercantile.xy_bounds(mercantile.Tile(x, y, z))
    return rasterio.transform.from_bounds(b.left, b.bottom, b.right, b.top, size, size)


def _wm_geom_to_wgs84(geom_dict: dict) -> dict:
    s = shape(geom_dict)
    s4326 = shapely_transform(_wm_to_wgs84.transform, s)
    return mapping(s4326)


def _infer(model: torch.nn.Module, img_t: torch.Tensor, threshold: float) -> np.ndarray:
    padded, undo = pad_to_multiple(img_t, 32)
    with torch.no_grad():
        logits = model(padded)          # (1, 2, H', W')
        probs = torch.softmax(logits, dim=1)[0, 1]
        probs = undo(probs)             # (H, W)
    return (probs.numpy() > threshold).astype(np.uint8)


MAX_GEOM_AREA_M2 = float(os.getenv("MAX_GEOM_AREA_M2", "10000"))  # ~1ha; larger → false positive


@app.get("/seg/{feature_type}/{z}/{x}/{y}.mvt")
async def seg_tile(feature_type: str, z: int, x: int, y: int) -> Response:
    if z <= 14:
        return Response(content=EMPTY_MVT, media_type="application/vnd.mapbox-vector-tile")
    if feature_type not in FEATURE_CONFIG:
        return Response(status_code=404, content=f"Unknown feature type: {feature_type}")
    if feature_type not in _models:
        return Response(status_code=503, content=f"Model not loaded: {feature_type}")

    model, threshold = _models[feature_type]
    _layer_name = FEATURE_CONFIG[feature_type][1]

    # 1. Fetch imagery tile
    url = f"{IMAGERY_TILE_BASE}/tiles/{z}/{x}/{y}.png"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        return Response(content=EMPTY_MVT, media_type="application/vnd.mapbox-vector-tile")

    # 2. Decode + normalize to float32 [0, 1]
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    size = img.width
    img_arr = np.array(img, dtype=np.float32) / 255.0       # (H, W, 3)
    img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # 3. Inference in thread pool (avoid blocking async event loop)
    loop = asyncio.get_event_loop()
    mask = await loop.run_in_executor(None, _infer, model, img_t, threshold)

    if mask.sum() == 0:
        return Response(content=EMPTY_MVT, media_type="application/vnd.mapbox-vector-tile")

    # 4. Vectorize: pixel → Web Mercator polygons; drop large shapes (false positives)
    transform = _tile_transform(x, y, z, size)
    shapes = [
        (geom, val)
        for geom, val in rasterio.features.shapes(mask, mask=mask, transform=transform)
        if val == 1 and shape(geom).area <= MAX_GEOM_AREA_M2
    ]
    if not shapes:
        return Response(content=EMPTY_MVT, media_type="application/vnd.mapbox-vector-tile")

    # 5. Reproject Web Mercator → WGS84
    features = [
        {"geometry": _wm_geom_to_wgs84(geom), "properties": {}}
        for geom, _ in shapes
    ]

    # 6. Encode MVT
    bounds = mercantile.bounds(mercantile.Tile(x, y, z))
    mvt_bytes: bytes = mapbox_vector_tile.encode(
        {"name": _layer_name, "features": features},
        default_options={
            "quantize_bounds": (bounds.west, bounds.south, bounds.east, bounds.north),
            "extents": 4096,
        },
    )

    return Response(
        content=mvt_bytes,
        media_type="application/vnd.mapbox-vector-tile",
        headers={"Cache-Control": "public, max-age=3600"},
    )
