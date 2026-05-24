"""smp.Unet builder + a pad-to-multiple helper.

ResNet34 has 5 downsample stages so input H/W must be divisible by 32
or the skip connections mismatch. pad_to_multiple returns a callable
that crops back to the original size after inference.
"""

from __future__ import annotations

from typing import Callable

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def build_unet(
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    classes: int = 2,
    in_channels: int = 3,
) -> torch.nn.Module:
    """Return a smp.Unet ready for training or eval.

    encoder_weights=None gives a fast offline-friendly path for tests.
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Right/bottom-pad a (..., H, W) tensor to a multiple of `multiple`.

    Returns (padded, undo) where undo(y) crops y back to the original
    (..., H, W) so callers can run inference on the padded tensor and
    map predictions back to the input grid losslessly.
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, lambda y: y

    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    def undo(y: torch.Tensor) -> torch.Tensor:
        return y[..., :h, :w]

    return padded, undo


__all__ = ["build_unet", "pad_to_multiple"]
