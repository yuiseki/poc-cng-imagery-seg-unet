"""Model build + one-pass forward."""

from __future__ import annotations

import torch

from imagery_seg.model import build_unet, pad_to_multiple


def test_build_unet_shape():
    """smp.Unet(resnet34/imagenet) accepts a 3x256x256 input and outputs
    2 channels at the same spatial resolution."""
    model = build_unet(encoder="resnet34", encoder_weights=None, classes=2)
    model.eval()
    x = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 256, 256)


def test_pad_to_multiple_pads_even_dims():
    x = torch.zeros(1, 3, 100, 130)
    padded, undo = pad_to_multiple(x, 32)
    assert padded.shape[-1] % 32 == 0
    assert padded.shape[-2] % 32 == 0
    assert padded.shape[-2] >= 100 and padded.shape[-1] >= 130
    # The undo crop returns to the original spatial size
    restored = undo(padded)
    assert restored.shape == x.shape


def test_pad_to_multiple_noop_when_aligned():
    x = torch.zeros(1, 3, 64, 64)
    padded, undo = pad_to_multiple(x, 32)
    assert padded.shape == x.shape
    assert torch.equal(undo(padded), x)
