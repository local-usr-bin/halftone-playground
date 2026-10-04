"""halftone-playground: variable-width line halftoning experiments.

Both modes share one core: grayscale-driven geometry produces a boolean mask,
and a separate renderer turns that mask into an image.  Stripe mode draws
variable-width stripes directly; Spiral mode unwraps the Cartesian image into
linear polar coordinates, runs the very same Stripe core there and maps the
result back, which turns the stripes into a multi-arm spiral.  An optional
preprocessing step rescales the source image (``image_scale``) before the
geometry runs; the scale and the line period are independent parameters.
"""

from __future__ import annotations

from .preprocess import resize_grayscale, scaled_size
from .render import render_black_on_white
from .smoke import GRADIENT_SIZE, make_gradient, save_gray_png
from .spiral import spiral_mask
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
    "spiral_mask",
    "stripe_mask",
]
