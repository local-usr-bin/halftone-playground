"""Rendering layer: turn a geometric mask into an actual image.

The renderer is deliberately kept separate from the halftone geometry.  A mask
only says *where* the lines are; a renderer decides what a line *looks like*
(black on white today, inverted / colored / transparent later).

Round 1 ships exactly one renderer, and it is strictly bi-level: no
antialiasing, no scaling, no post-processing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["render_black_on_white"]

_LINE_VALUE = 0
_BACKGROUND_VALUE = 255


def render_black_on_white(mask: np.ndarray) -> np.ndarray:
    """Render a boolean mask as a strict black-on-white ``uint8`` image.

    Parameters
    ----------
    mask:
        A non-empty 2-D boolean array.  ``True`` marks a line pixel.

    Returns
    -------
    numpy.ndarray
        A ``uint8`` array with the same shape as ``mask`` in which line pixels
        are ``0`` and background pixels are ``255``.  Only those two values can
        ever appear; the image is strictly bi-level by construction.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy.ndarray, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (height, width), got shape {mask.shape}")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    if mask.dtype != np.bool_:
        raise ValueError(f"mask must have dtype bool, got {mask.dtype}")

    out = np.full(mask.shape, _BACKGROUND_VALUE, dtype=np.uint8)
    out[mask] = _LINE_VALUE
    return out
