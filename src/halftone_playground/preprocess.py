"""Preprocessing layer: optional spatial rescaling of the source image.

The geometry cores only ever see a plain 2-D ``uint8`` grayscale array; this
module is the small, optional step *before* those cores.  It maps a user
``image_scale`` factor onto a target canvas size and resizes the source image
accordingly -- the grayscale image that feeds geometry, and (Renderer Round 1)
an optional Cartesian RGB source that feeds the source-colour compositor.

Both resizers share the *same* target-size arithmetic and the *same* fixed
filter, so a grayscale image and its RGB companion scaled by the same
``image_scale`` stay pixel-aligned and a mask computed from one lines up with
the other.

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
* ``resize_grayscale`` and ``resize_rgb`` share one target size and one fixed
  filter, so a grayscale image and its RGB companion never drift apart;
* no resource management happens here: computing a target size never
  allocates an image, and if a requested output is too large for the
  machine, that is left to the ordinary failure of the allocation itself.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

__all__ = ["scaled_size", "resize_grayscale", "resize_rgb"]

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


def resize_rgb(rgb: np.ndarray, image_scale: float) -> np.ndarray:
    """Resize a 3-D ``uint8`` RGB array by ``image_scale``.

    This is the RGB companion of :func:`resize_grayscale`.  When
    ``image_scale != 1`` and the user picks the source-colour output variant,
    the RGB source has to land on exactly the same canvas the geometry ran on
    -- otherwise the mask and the RGB pixels cannot be aligned.  Both resizers
    therefore share the very same :func:`scaled_size` target (half-up per
    axis) and the same fixed BICUBIC filter, so they are guaranteed to
    produce matching dimensions.

    Parameters
    ----------
    rgb:
        A non-empty array of shape ``(height, width, 3)`` with dtype
        ``uint8``.
    image_scale:
        Any positive finite real number.

    Returns
    -------
    numpy.ndarray
        A ``(scaled_H, scaled_W, 3)`` ``uint8`` RGB array whose spatial size
        is exactly ``scaled_size(*rgb.shape[:2][::-1], image_scale)``.  With
        ``image_scale == 1.0`` the result is a copy whose pixel values are
        identical to the input (``np.array_equal`` holds; no resampling is
        performed at all).  For any other scale the image is resampled with
        the fixed internal Pillow BICUBIC filter.  The output is never
        grayscale, never float, never alpha.

    Raises
    ------
    TypeError
        If ``rgb`` is not an ``ndarray``, or the scale is not a real number.
    ValueError
        If ``rgb`` is not 3-D, empty, has a channel count other than 3, or is
        not ``uint8``; if the scale is zero / negative / non-finite; or if the
        target size collapses below 1x1.

    Notes
    -----
    The scale is *not* tied to the Stripe / Spiral ``period``: passing
    ``image_scale=2.0`` doubles the canvas but leaves ``period`` alone, exactly
    as with the grayscale path.
    """
    _validate_rgb(rgb)
    scale = _validate_image_scale(image_scale)

    height, width = rgb.shape[:2]
    target_width, target_height = scaled_size(width, height, scale)

    if scale == 1.0:
        return rgb.copy()

    image = Image.fromarray(rgb, mode="RGB")
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


def _validate_rgb(rgb: np.ndarray) -> None:
    """Enforce the strict input contract for :func:`resize_rgb`."""
    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"rgb must be a numpy.ndarray, got {type(rgb).__name__}")
    if rgb.ndim != 3:
        raise ValueError(
            f"rgb must be 3-D (height, width, 3), got shape {rgb.shape}"
        )
    if rgb.size == 0:
        raise ValueError("rgb must not be empty")
    if rgb.shape[2] != 3:
        raise ValueError(
            f"rgb must have exactly 3 channels (RGB), got shape {rgb.shape}"
        )
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {rgb.dtype}")
