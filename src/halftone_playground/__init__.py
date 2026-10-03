"""halftone-playground: variable-width line halftoning experiments.

Phase 0 baseline: environment and image I/O smoke utilities only.
Halftoning algorithms (stripe / spiral) are intentionally not implemented yet.
"""

from __future__ import annotations

from .smoke import GRADIENT_SIZE, make_gradient, save_gray_png

__version__ = "0.0.1"

__all__ = ["GRADIENT_SIZE", "__version__", "make_gradient", "save_gray_png"]
