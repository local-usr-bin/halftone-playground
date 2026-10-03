"""Preprocessing layer: optional spatial rescaling of the grayscale source.

The Stripe core itself only ever sees a plain 2-D ``uint8`` grayscale array;
this module is the small, optional step *before* that core.  It maps a user
``image_scale`` factor onto a target canvas size and resizes the grayscale
image accordingly.

Product rules frozen for this round:

* ``image_scale`` accepts any positive finite float (non-integer values
  included); there is **no arbitrary upper limit** -- ``100.0`` is as legal
  as ``1.3``;
* the resample filter is fixed to Pillow BICUBIC and is *not* a parameter;
* ``image_scale`` and the stripe ``period`` are fully independent: the period
  is always measured in pixels of the *scaled* image and is never multiplied,
  divided or otherwise adjusted by the scale factor;
* ``image_scale == 1.0`` returns the input unchanged (a copy), so a no-op
  scale can never alter pixel values;
* no resource management happens here: computing a target size never
  allocates an image, and if a requested output is too large for the
  machine, that is left to the ordinary failure of the allocation itself.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

__all__ = ["scaled_size", "resize_grayscale"]

try:  # Pillow >= 9.1
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - older Pillow fallback
    _BICUBIC = Image.BICUBIC  # type: ignore[attr-defined]


def scaled_size(width: int, height: int, image_scale: float) -> tuple[int, int]:
    """Return the target ``(width, height)`` after applying ``image_scale``.

    Parameters
    ----------
    width, height:
        Positive integers, the source image dimensions.
    image_scale:
        Any positive finite real number.  Non-integer values are fine.

    Returns
    -------
    tuple[int, int]
        The output dimensions, computed per axis with the half-up rule

            target = floor(original * image_scale + 0.5)

        (deliberately not Python's banker's ``round``).

    Raises
    ------
    TypeError
        If the scale is not a real number (strings, ``None``, booleans) or a
        dimension is not an integer.
    ValueError
        If the scale is zero, negative or non-finite; if the product
        overflows to a non-finite value; or if a rounded dimension would be
        smaller than 1 ("scaled dimensions must be at least 1x1").  No silent
        clamping to 1 ever happens.

    Notes
    -----
    This function is pure arithmetic: it never allocates an image.  A future
    GUI can call it to preview the output resolution (e.g. source 512x512
    with ``image_scale=1.3`` -> 666x666) before any memory is committed.
    There is deliberately no upper bound on the result: 4000x3000 at
    ``image_scale=100.0`` happily returns 400000x300000.
    """
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    scale = _validate_image_scale(image_scale)

    target_width = _scaled_axis(width, scale)
    target_height = _scaled_axis(height, scale)

    if target_width < 1 or target_height < 1:
        raise ValueError(
            "scaled dimensions must be at least 1x1, got "
            f"{target_width}x{target_height} "
            f"(width={width}, height={height}, image_scale={image_scale!r})"
        )
    return target_width, target_height


def resize_grayscale(gray: np.ndarray, image_scale: float) -> np.ndarray:
    """Resize a 2-D ``uint8`` grayscale array by ``image_scale``.

    Parameters
    ----------
    gray:
        A non-empty 2-D ``uint8`` array (height, width).
    image_scale:
        Any positive finite real number.

    Returns
    -------
    numpy.ndarray
        A 2-D ``uint8`` array of exactly ``scaled_size(*gray.shape[::-1],
        image_scale)``.  With ``image_scale == 1.0`` the result is a copy with
        pixel values identical to the input (no resampling is performed, so a
        no-op scale cannot introduce even one changed pixel).  For any other
        scale the image is resampled with the fixed internal Pillow BICUBIC
        filter.  The output is never RGB, never float, never alpha.
    """
    _validate_gray(gray)
    scale = _validate_image_scale(image_scale)

    height, width = gray.shape
    target_width, target_height = scaled_size(width, height, scale)

    if scale == 1.0:
        return gray.copy()

    image = Image.fromarray(gray, mode="L")
    resized = image.resize((target_width, target_height), resample=_BICUBIC)
    return np.array(resized, dtype=np.uint8)


def _scaled_axis(original: int, scale: float) -> int:
    """One axis of the half-up rounding rule, with overflow detection."""
    product = original * scale
    if not math.isfinite(product):
        raise ValueError(
            "scaled dimension overflows to a non-finite value "
            f"(original={original}, image_scale={scale!r})"
        )
    return int(math.floor(product + 0.5))


def _validate_dimension(value: int, name: str) -> int:
    """Reject anything that is not a positive plain integer dimension."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {int(value)}")
    return int(value)


def _validate_image_scale(image_scale: float) -> float:
    """Reject anything that is not a positive finite real number."""
    if isinstance(image_scale, bool) or not isinstance(
        image_scale, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"image_scale must be a real number, got {type(image_scale).__name__}"
        )
    scale = float(image_scale)
    if not math.isfinite(scale):
        raise ValueError(f"image_scale must be finite, got {scale!r}")
    if scale <= 0.0:
        raise ValueError(f"image_scale must be positive, got {scale!r}")
    return scale


def _validate_gray(gray: np.ndarray) -> None:
    """Enforce the same strict input contract as the Stripe core."""
    if not isinstance(gray, np.ndarray):
        raise TypeError(f"gray must be a numpy.ndarray, got {type(gray).__name__}")
    if gray.ndim != 2:
        raise ValueError(f"gray must be 2-D (height, width), got shape {gray.shape}")
    if gray.size == 0:
        raise ValueError("gray must not be empty")
    if gray.dtype != np.uint8:
        raise ValueError(f"gray must have dtype uint8, got {gray.dtype}")
