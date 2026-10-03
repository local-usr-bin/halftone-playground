"""Phase 0 smoke utilities.

Only environment / I/O checks live here: a NumPy gradient generator and a
Pillow PNG writer. No halftoning (stripe, spiral, threshold) logic exists in
this project yet, by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

GRADIENT_SIZE = 256

PathLike = Union[str, Path]


def make_gradient(size: int = GRADIENT_SIZE) -> np.ndarray:
    """Return a simple left-to-right grayscale gradient as ``(size, size)`` uint8.

    The left edge is black (0) and the right edge is white (255). This is a
    plain synthetic image used to verify NumPy/Pillow I/O only; it involves no
    halftone, stripe, or threshold processing.
    """
    ramp = np.linspace(0.0, 255.0, num=size, endpoint=True)
    return np.tile(ramp, (size, 1)).astype(np.uint8)


def save_gray_png(array: np.ndarray, path: PathLike) -> Image.Image:
    """Save a 2-D uint8 array as a grayscale PNG (mode ``L``) and return it."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {array.shape}")
    img = Image.fromarray(array, mode="L")
    img.save(path, format="PNG")
    return img
