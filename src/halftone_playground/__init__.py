"""halftone-playground: variable-width line halftoning experiments.

Both modes share one core: grayscale-driven geometry produces a boolean mask,
and a separate renderer turns that mask into an image.  Stripe mode draws
variable-width stripes directly; Spiral mode unwraps the Cartesian image into
linear polar coordinates, runs the very same Stripe core there and maps the
result back, which turns the stripes into a multi-arm spiral.  An optional
preprocessing step rescales the source image (``image_scale``) before the
geometry runs; the scale and the line period are independent parameters.

Geometry and color are fully separated.  The cores only ever emit a ``bool``
mask (``True`` = line, ``False`` = background); the renderers in ``render.py``
decide what a line pixel looks like -- black on white, white on black, or the
source image's own colour on white -- without touching the mask.
"""

from __future__ import annotations

from .preprocess import (
    circular_support,
    invert_luminance,
    resize_grayscale,
    resize_rgb,
    scaled_size,
)
from .render import (
    render_black_on_white,
    render_source_color_on_white,
    render_white_on_black,
)
from .smoke import GRADIENT_SIZE, make_gradient, save_gray_png
from .spiral import spiral_fixed_mask, spiral_mask
from .stripe import stripe_fixed_mask, stripe_mask

__version__ = "0.0.1"

__all__ = [
    "GRADIENT_SIZE",
    "__version__",
    "circular_support",
    "invert_luminance",
    "make_gradient",
    "render_black_on_white",
    "render_source_color_on_white",
    "render_white_on_black",
    "resize_grayscale",
    "resize_rgb",
    "save_gray_png",
    "scaled_size",
    "spiral_fixed_mask",
    "spiral_mask",
    "stripe_fixed_mask",
    "stripe_mask",
]
