"""imagery_seg: pluggable U-Net training pipeline.

Compose an ImagerySource, a GroundTruthSource, and a FeatureSpec into
a `Recipe`. The pipeline takes the recipe + a list of bboxes and
produces (image, mask) training pairs, then trains a smp.Unet on top.

See README.md for the world view.
"""

__version__ = "0.1.0"

from .recipe import Recipe

__all__ = ["Recipe"]
