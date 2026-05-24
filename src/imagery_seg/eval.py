"""Binary IoU / F1 + a top-level evaluate(model, dataset).

The dataset is expected to yield class-index masks (0 = background,
1 = positive); models output 2-channel logits.

Two paths:
  evaluate(...)            : argmax-based, scaffold-grade (1 call, 1 number)
  evaluate_with_sweep(...) : returns per-AOI best threshold + full sweep,
                             matching the sister repo's stage 16 pattern
                             (per-AOI threshold calibration).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label
from torch.utils.data import Dataset

from .model import pad_to_multiple

# Sibling stages/16 sweep range. Coarse enough to be fast on CPU,
# dense enough to land on a good operating point.
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.1 + 0.05 * i, 2) for i in range(17))

# Sibling stages/14 sweep. 0.0 is the no-filter baseline.
DEFAULT_AREA_FILTERS_M2: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0)


def filter_small_components(
    mask: np.ndarray | torch.Tensor,
    *,
    pixel_area_m2: float,
    min_area_m2: float,
) -> np.ndarray:
    """Drop connected components below `min_area_m2`. No-op when
    min_area_m2 <= 0 (lets callers sweep including the baseline).
    Returns a uint8 {0,1} ndarray of the same shape.
    """
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = mask
    arr = arr.astype(np.uint8, copy=False)
    if min_area_m2 <= 0 or arr.sum() == 0:
        return arr
    labeled, n = cc_label(arr)
    if n == 0:
        return arr
    counts = np.bincount(labeled.ravel())
    keep_label = (counts * pixel_area_m2) >= min_area_m2
    keep_label[0] = False  # background never reported as foreground
    return keep_label[labeled].astype(np.uint8)


def binary_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Class-1 IoU restricted to valid pixels. Empty union returns 1.0
    (no penalty for unanimously correct negatives — this matters when
    an AOI has zero ground-truth positives).

    If `valid_mask` is provided, invalid pixels are excluded from both
    intersection and union — useful for HOTOSM COGs whose bbox includes
    no-data regions outside the actual flight path.
    """
    pred_b = pred.bool()
    target_b = target.bool()
    if valid_mask is not None:
        v = valid_mask.bool()
        pred_b = pred_b & v
        target_b = target_b & v
    inter = (pred_b & target_b).sum().item()
    union = (pred_b | target_b).sum().item()
    if union == 0:
        return 1.0
    return inter / union


def binary_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    pred_b = pred.bool()
    target_b = target.bool()
    if valid_mask is not None:
        v = valid_mask.bool()
        pred_b = pred_b & v
        target_b = target_b & v
    tp = (pred_b & target_b).sum().item()
    fp = (pred_b & ~target_b).sum().item()
    fn = (~pred_b & target_b).sum().item()
    if tp == 0 and (fp == 0 or fn == 0):
        return 1.0 if (tp == fp == fn == 0) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Mean IoU + mean F1 across the dataset (one sample at a time)."""
    model.eval()
    model.to(device)
    ious: list[float] = []
    f1s: list[float] = []
    for i in range(len(dataset)):  # type: ignore[arg-type]
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"]
        valid = sample.get("valid_mask")
        padded, undo = pad_to_multiple(image, multiple=32)
        logits = undo(model(padded))
        pred = logits.argmax(dim=1).squeeze(0).cpu()
        ious.append(binary_iou(pred, target, valid))
        f1s.append(binary_f1(pred, target, valid))
    return {
        "iou": sum(ious) / len(ious) if ious else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Return the class-1 probability map for a single image.

    Input: image of shape (C, H, W) on CPU.
    Output: (H, W) float tensor on CPU with values in [0, 1].
    """
    model.eval()
    model.to(device)
    batched = image.unsqueeze(0).to(device)
    padded, undo = pad_to_multiple(batched, multiple=32)
    logits = undo(model(padded))  # (1, 2, H, W)
    probs = F.softmax(logits, dim=1)[:, 1]  # (1, H, W)
    return probs.squeeze(0).cpu()


@torch.no_grad()
def evaluate_with_sweep(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    area_filters_m2: Iterable[float] = (0.0,),
    device: str = "cpu",
) -> list[dict]:
    """Run a per-AOI (threshold x area_filter) sweep.

    Returns a list (one entry per sample) of:
      {
        "aoi_index": int,
        "bbox": tuple,
        "asset_id": str | None,
        "pixel_area_m2": float,
        "best_threshold": float,
        "best_area_filter_m2": float,
        "best_iou": float,
        "best_f1": float,
        "sweep": [{"threshold": t, "area_filter_m2": a, "iou": ..., "f1": ...}, ...],
      }

    The caller is expected to persist this (see CLI `sweep`) so the
    serving side can apply the per-AOI (threshold, area_filter) pair
    without re-running.
    """
    thresholds = list(thresholds)
    area_filters_m2 = list(area_filters_m2)
    results: list[dict] = []
    for i in range(len(dataset)):  # type: ignore[arg-type]
        sample = dataset[i]
        probs = predict_probability(model, sample["image"], device=device)
        target = sample["mask"]
        valid = sample.get("valid_mask")
        pixel_area_m2 = float(sample.get("pixel_area_m2", 0.0))
        per_combo: list[dict] = []
        best_t, best_a = thresholds[0], area_filters_m2[0]
        best_iou, best_f1 = -1.0, 0.0
        for t in thresholds:
            base_pred = (probs >= t).long()
            for a in area_filters_m2:
                if a > 0 and pixel_area_m2 > 0:
                    pred_np = filter_small_components(
                        base_pred, pixel_area_m2=pixel_area_m2, min_area_m2=float(a),
                    )
                    pred = torch.from_numpy(pred_np)
                else:
                    pred = base_pred
                iou = binary_iou(pred, target, valid)
                f1 = binary_f1(pred, target, valid)
                per_combo.append({
                    "threshold": float(t),
                    "area_filter_m2": float(a),
                    "iou": float(iou),
                    "f1": float(f1),
                })
                if iou > best_iou:
                    best_iou, best_f1 = iou, f1
                    best_t, best_a = float(t), float(a)
        results.append({
            "aoi_index": int(sample.get("aoi_index", i)),
            "bbox": tuple(sample.get("bbox", ())),
            "region": sample.get("region", ""),
            "asset_id": sample.get("asset_id"),
            "pixel_area_m2": pixel_area_m2,
            "best_threshold": best_t,
            "best_area_filter_m2": best_a,
            "best_iou": best_iou,
            "best_f1": best_f1,
            "sweep": per_combo,
        })
    return results


def aggregate_by_region(
    sweep_results: list[dict],
    *,
    metric: str = "best_iou",
) -> dict[str, dict[str, float | int]]:
    """Per-region summary of sweep_results.

    Returns `{region: {n, mean, median, min, max, p25, p75}}` over the
    chosen metric (default "best_iou"; also useful: "best_f1"). The
    "" region (untagged AOIs) is reported separately so untagged data
    doesn't quietly poison region-stratified comparisons.
    """
    by_region: dict[str, list[float]] = {}
    for entry in sweep_results:
        region = entry.get("region", "")
        value = float(entry.get(metric, 0.0))
        by_region.setdefault(region, []).append(value)

    summary: dict[str, dict[str, float | int]] = {}
    for region, values in by_region.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[region] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p25": float(np.quantile(arr, 0.25)),
            "p75": float(np.quantile(arr, 0.75)),
        }
    return summary


@torch.no_grad()
def evaluate_per_region(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: str = "cpu",
) -> dict[str, dict[str, float | int]]:
    """Compute per-region IoU/F1 using the simple argmax path
    (no threshold sweep). Returns `{region: {n, mean_iou, median_iou,
    min_iou, mean_f1, ...}}`. For a sweeping variant, use
    `aggregate_by_region(evaluate_with_sweep(...))`.
    """
    model.eval()
    model.to(device)
    per_region_iou: dict[str, list[float]] = {}
    per_region_f1: dict[str, list[float]] = {}
    for i in range(len(dataset)):  # type: ignore[arg-type]
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"]
        valid = sample.get("valid_mask")
        padded, undo = pad_to_multiple(image, multiple=32)
        logits = undo(model(padded))
        pred = logits.argmax(dim=1).squeeze(0).cpu()
        region = sample.get("region", "")
        per_region_iou.setdefault(region, []).append(binary_iou(pred, target, valid))
        per_region_f1.setdefault(region, []).append(binary_f1(pred, target, valid))

    summary: dict[str, dict[str, float | int]] = {}
    for region in per_region_iou:
        ious = np.asarray(per_region_iou[region], dtype=np.float64)
        f1s = np.asarray(per_region_f1[region], dtype=np.float64)
        summary[region] = {
            "n": int(ious.size),
            "mean_iou": float(ious.mean()),
            "median_iou": float(np.median(ious)),
            "min_iou": float(ious.min()),
            "mean_f1": float(f1s.mean()),
            "median_f1": float(np.median(f1s)),
            "min_f1": float(f1s.min()),
        }
    return summary


__all__ = [
    "binary_iou", "binary_f1", "evaluate",
    "predict_probability", "evaluate_with_sweep",
    "evaluate_per_region", "aggregate_by_region",
    "filter_small_components",
    "DEFAULT_THRESHOLDS", "DEFAULT_AREA_FILTERS_M2",
]
