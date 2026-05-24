"""Training loop. Scaffold-quality: 1 epoch, 1 batch at a time.

Wrap with proper schedulers / AMP / multi-AOI batching once a real
recipe needs it. The point of this version is to give CLI users
something to run end-to-end without faking it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .eval import evaluate
from .model import pad_to_multiple

logger = logging.getLogger("imagery_seg.train")


def train_one_epoch(
    model: torch.nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 1,
    device: str = "cpu",
) -> list[float]:
    """Run one epoch and return the per-batch losses.

    Uses cross-entropy against the class index mask (classes=2 -> 0/1).
    """
    model.train()
    model.to(device)
    losses: list[float] = []
    loader: Iterable = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        valid = batch.get("valid_mask")
        padded, undo = pad_to_multiple(image, multiple=32)
        logits = model(padded)
        logits = undo(logits)
        if valid is not None:
            # Mask out invalid (no-data) pixels so they don't contribute
            # to gradient. Mean over valid pixels only.
            valid = valid.to(device).bool()
            per_pixel = F.cross_entropy(logits, mask, reduction="none")
            denom = valid.sum().clamp(min=1).float()
            loss = (per_pixel * valid.float()).sum() / denom
        else:
            loss = F.cross_entropy(logits, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        logger.info("train loss=%.4f", loss.item())
    return losses


def _pad_to(t: torch.Tensor, h: int, w: int, value: float = 0.0) -> torch.Tensor:
    """Right/bottom-pad tensor (..., H, W) to (h, w)."""
    ph = h - t.shape[-2]
    pw = w - t.shape[-1]
    if ph == 0 and pw == 0:
        return t
    return torch.nn.functional.pad(t, (0, pw, 0, ph), value=value)


def _collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Pad-then-stack collate that handles mixed H×W within a batch.

    AOIs with different aspect ratios produce images whose long edge is
    max_side but whose short edge varies. Right/bottom padding to the
    batch-local maximum aligns them without resizing.
    """
    max_h = max(s["image"].shape[-2] for s in samples)
    max_w = max(s["image"].shape[-1] for s in samples)
    batched = {
        "image": torch.stack([_pad_to(s["image"], max_h, max_w, 0.0) for s in samples]),
        "mask":  torch.stack([_pad_to(s["mask"],  max_h, max_w, 0)   for s in samples]),
    }
    if all("valid_mask" in s for s in samples):
        # Pad region is outside the captured tile — mark as invalid.
        batched["valid_mask"] = torch.stack(
            [_pad_to(s["valid_mask"], max_h, max_w, 0) for s in samples]
        )
    return batched


def train(
    model: torch.nn.Module,
    train_dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    val_dataset: Dataset | None = None,
    batch_size: int = 1,
    device: str = "cpu",
    run_dir: Path | None = None,
    model_meta: dict | None = None,
    keep_epochs: bool = False,
) -> dict:
    """Run `epochs` of train_one_epoch with optional val + checkpointing.

    Returns a history dict with per-epoch mean loss + val IoU/F1 and the
    best epoch index. If `run_dir` is given, writes:

      {run_dir}/best.pt            the epoch with the highest val IoU
                                   (or last epoch if no val_dataset)
      {run_dir}/history.json       the returned history dict

    When `keep_epochs=True`, additionally writes `{run_dir}/epoch_{n:03d}.pt`
    every epoch. Default off — a single checkpoint is ~100MB for ResNet34/UNet
    so 10 epochs eat ~1GB of disk that the history.json already summarises.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best = {"epoch": -1, "val_iou": -1.0}

    for epoch in range(1, epochs + 1):
        losses = train_one_epoch(
            model, train_dataset, optimizer,
            batch_size=batch_size, device=device,
        )
        mean_loss = sum(losses) / len(losses) if losses else float("nan")

        val_metrics: dict[str, float] = {}
        if val_dataset is not None and len(val_dataset) > 0:  # type: ignore[arg-type]
            val_metrics = evaluate(model, val_dataset, device=device)
            logger.info(
                "epoch %d val IoU=%.4f F1=%.4f",
                epoch, val_metrics["iou"], val_metrics["f1"],
            )

        epoch_entry = {
            "epoch": epoch,
            "train_loss_mean": mean_loss,
            "val_iou": val_metrics.get("iou"),
            "val_f1": val_metrics.get("f1"),
        }
        history.append(epoch_entry)

        score = val_metrics.get("iou", -1.0) if val_metrics else float(epoch)
        is_best = score > best["val_iou"] if val_metrics else True
        if is_best:
            best = {"epoch": epoch, "val_iou": float(score)}

        if run_dir is not None:
            payload = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss_mean": mean_loss,
                "val_iou": val_metrics.get("iou"),
                "val_f1": val_metrics.get("f1"),
                "meta": model_meta or {},
            }
            if keep_epochs:
                torch.save(payload, run_dir / f"epoch_{epoch:03d}.pt")
            if is_best:
                torch.save(payload, run_dir / "best.pt")

    result = {"history": history, "best": best}
    if run_dir is not None:
        (run_dir / "history.json").write_text(
            json.dumps(result, indent=2, default=float),
            encoding="utf-8",
        )
    return result


__all__ = ["train_one_epoch", "train"]
