"""halftone-playground: variable-width line halftoning experiments.

Stripe mode Round 1 provides the variable-width stripe core: grayscale-driven
geometry produces a boolean mask, and a separate renderer turns that mask into
an image.  Spiral mode is intentionally not implemented yet.
"""

from __future__ import annotations

from .render import render_black_on_white
from .smoke import GRADIENT_SIZE, make_gradient, save_gray_png
from .stripe import stripe_mask

__version__ = "0.0.1"

__all__ = [
    "GRADIENT_SIZE",
    "__version__",
    "make_gradient",
    "render_black_on_white",
    "save_gray_png",
    "stripe_mask",
]
