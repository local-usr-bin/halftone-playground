"""halftone-playground: variable-width line halftoning experiments.

Stripe mode provides the variable-width stripe core: grayscale-driven
geometry produces a boolean mask, and a separate renderer turns that mask into
an image.  An optional preprocessing step rescales the source image
(``image_scale``) before the geometry runs; the scale and the stripe period
are independent parameters.  Spiral mode is intentionally not implemented yet.
"""

from __future__ import annotations

from .preprocess import resize_grayscale, scaled_size
from .render import render_black_on_white
from .smoke import GRADIENT_SIZE, make_gradient, save_gray_png
from .stripe import stripe_mask

__version__ = "0.0.1"

__all__ = [
    "GRADIENT_SIZE",
    "__version__",
    "make_gradient",
    "render_black_on_white",
    "resize_grayscale",
    "save_gray_png",
    "scaled_size",
    "stripe_mask",
]
