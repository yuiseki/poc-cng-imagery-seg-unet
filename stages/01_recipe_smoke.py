"""End-to-end scaffold smoke test.

Loads a recipe, pulls the first AOI for real (HOTOSM COG via
/vsicurl/ + Overpass via overpass.yuiseki.net), rasterises the GT,
runs one training step, evaluates the same AOI, and writes
image.png + mask.png + metrics.json under tmp/<recipe-name>/.

Usage:
    uv run python stages/01_recipe_smoke.py recipes/hotosm_buildings.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from imagery_seg.dataset import RecipeDataset  # noqa: E402
from imagery_seg.eval import evaluate  # noqa: E402
from imagery_seg.model import build_unet  # noqa: E402
from imagery_seg.recipe import load_recipe  # noqa: E402
from imagery_seg.train import train_one_epoch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe_path", type=Path)
    parser.add_argument("--max-side", type=int, default=512,
                        help="cap on the imagery long edge (default 512)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("smoke")

    r = load_recipe(args.recipe_path)
    log.info("loaded recipe: %s", r.name)
    log.info("  %s + %s + %s -> %d AOIs",
             r.imagery.name, r.ground_truth.name, r.feature.name, len(r.aois))

    out_dir = REPO_ROOT / "tmp" / r.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Truncate AOIs to just the first one so the smoke stays cheap.
    smoke_recipe = r.__class__(
        name=r.name,
        imagery=r.imagery,
        ground_truth=r.ground_truth,
        feature=r.feature,
        training=r.training,
        aois=r.aois[:1],
    )
    ds = RecipeDataset(smoke_recipe, max_side=args.max_side, dst_crs="EPSG:3857")

    t0 = time.perf_counter()
    sample = ds[0]
    log.info("sample fetched in %.2fs", time.perf_counter() - t0)

    image_u8 = (sample["image"].numpy() * 255).clip(0, 255).astype(np.uint8)
    mask_u8 = (sample["mask"].numpy().astype(np.uint8) * 255)

    Image.fromarray(image_u8.transpose(1, 2, 0)).save(out_dir / "image.png")
    Image.fromarray(mask_u8).save(out_dir / "mask.png")
    log.info("wrote %s/image.png + mask.png (%dx%d)",
             out_dir, image_u8.shape[-1], image_u8.shape[-2])

    model = build_unet(
        encoder=r.training.encoder,
        encoder_weights=None,  # offline: skip imagenet download for smoke
        classes=2,
    )
    optim = torch.optim.Adam(model.parameters(), lr=r.training.lr)
    t0 = time.perf_counter()
    losses = train_one_epoch(model, ds, optim, batch_size=1, device=args.device)
    log.info("one-step train: loss=%.4f in %.2fs", losses[0], time.perf_counter() - t0)

    metrics = evaluate(model, ds, device=args.device)
    log.info("post-step metrics: IoU=%.4f F1=%.4f", metrics["iou"], metrics["f1"])

    summary = {
        "recipe": r.name,
        "aoi": list(smoke_recipe.aois[0]),
        "imagery": r.imagery.name,
        "ground_truth": r.ground_truth.name,
        "feature": r.feature.name,
        "shape": list(image_u8.shape),
        "loss": losses[0],
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s/metrics.json", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
