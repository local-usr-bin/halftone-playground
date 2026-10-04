"""Rendering layer: turn a geometric mask into an actual image.

The renderer is deliberately kept separate from the halftone geometry.  A mask
only says *where* the lines are; a renderer decides what a line *looks like*.

Layering contract
-----------------

Geometry cores (Stripe, Spiral) produce one thing and one thing only: a
``bool`` mask where ``True`` means "line" and ``False`` means "background".
Every renderer in this module consumes *exactly that mask* and nothing else:

* it never re-derives gray levels or line widths,
* it never smooths, resizes, blurs or otherwise rewrites the mask,
* it never lets color influence geometry.

Consequently the *same* mask can be handed to any variant below and the
geometry is bit-for-bit identical across the outputs -- only the pixel values
change.  This is the whole point of the round: geometry and color are fully
separated.

Available variants (Renderer / Compositor Round 1)
--------------------------------------------------

======================  ====================  ==================================
function                ``mask=True``         ``mask=False``
======================  ====================  ==================================
``render_black_on_white``   ``0``               ``255``
``render_white_on_black``   ``255``             ``0``
``render_source_color_on_white``  source RGB    ``(255, 255, 255)``
======================  ====================  ==================================

``render_black_on_white`` and ``render_white_on_black`` are strict bi-level
(only ``0`` / ``255`` can appear).  The two are exact per-pixel complements:
``black_on_white + white_on_black == 255`` everywhere, because inversion is a
*renderer* property and never a change of the mask.

Still bi-level, still no anti-aliasing, no scaling and no post-processing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "render_black_on_white",
    "render_source_color_on_white",
    "render_white_on_black",
]

_LINE_VALUE = 0
_BACKGROUND_VALUE = 255

_RGB_CHANNELS = 3
_WHITE = np.array([255, 255, 255], dtype=np.uint8)


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
    _validate_mask(mask)

    out = np.full(mask.shape, _BACKGROUND_VALUE, dtype=np.uint8)
    out[mask] = _LINE_VALUE
    return out


def render_white_on_black(mask: np.ndarray) -> np.ndarray:
    """Render a boolean mask as a strict white-on-black ``uint8`` image.

    This is the *binary inversion* variant.  Inversion is a property of the
    renderer, not of the geometry: the mask is consumed unchanged and no
    ``invert`` parameter exists anywhere in the Stripe or Spiral cores.

    Parameters
    ----------
    mask:
        A non-empty 2-D boolean array.  ``True`` marks a line pixel.

    Returns
    -------
    numpy.ndarray
        A ``uint8`` array with the same shape as ``mask`` in which line pixels
        are ``255`` and background pixels are ``0``.  Only those two values can
        ever appear.  For the very same ``mask`` it holds exactly that

            render_black_on_white(mask) + render_white_on_black(mask) == 255

        for every pixel (the addition being done in a wider integer type).

    Notes
    -----
    The output is the exact inverse of :func:`render_black_on_white`, so it is
    implemented as that complement.  The whole canvas background becomes
    black, including any region outside a shape's support (e.g. the corners
    outside the Spiral disc, where the mask is ``False``).
    """
    return np.uint8(_BACKGROUND_VALUE) - render_black_on_white(mask)


def render_source_color_on_white(
    mask: np.ndarray, source_rgb: np.ndarray
) -> np.ndarray:
    """Paint line pixels with the source image's own colour on a white field.

    This is the *variable-width source-colour* variant: the existing boolean
    mask still decides the geometry (line width still follows the local
    grayscale), only the line pixel value changes -- from black to the
    corresponding source pixel.

    Parameters
    ----------
    mask:
        A non-empty 2-D boolean array.  ``True`` marks a line pixel.
    source_rgb:
        A non-empty ``uint8`` array of shape ``(H, W, 3)`` -- the *Cartesian*
        RGB source.  Its first two dimensions must match ``mask.shape``
        exactly; the compositor never resizes, so a size mismatch is rejected
        (spatial alignment is a preprocessing concern, see
        :func:`halftone_playground.preprocess.resize_rgb`).

    Returns
    -------
    numpy.ndarray
        An ``H x W x 3`` ``uint8`` RGB array where ``mask=True`` pixels are the
        **exact, byte-for-byte** ``source_rgb`` pixel at the same ``(y, x)``
        and ``mask=False`` pixels are pure white ``(255, 255, 255)``.

    Notes
    -----
    The colour is copied straight from the Cartesian source at the *final*
    Cartesian coordinates.  For Spiral this means the RGB data never enters
    the polar pipeline at all: there is no forward polar unwrap, no inverse
    polar unwrap and therefore no extra colour resampling.  No grayscale
    re-tinting, palette quantization, colour correction, gamma, interpolation,
    alpha blending or premultiplication is applied -- a line pixel is a direct
    copy of the Cartesian source pixel.

    Neither input array is modified.
    """
    _validate_mask(mask)
    _validate_source_rgb(source_rgb, mask.shape)

    out = np.empty((*mask.shape, _RGB_CHANNELS), dtype=np.uint8)
    out[...] = _WHITE
    out[mask] = source_rgb[mask]
    return out


# --------------------------------------------------------------------------
# validation helpers (private)
# --------------------------------------------------------------------------


def _validate_mask(mask: np.ndarray) -> None:
    """Enforce the strict boolean-mask contract shared by every renderer."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy.ndarray, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (height, width), got shape {mask.shape}")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    if mask.dtype != np.bool_:
        raise ValueError(f"mask must have dtype bool, got {mask.dtype}")


def _validate_source_rgb(source_rgb: np.ndarray, mask_shape: tuple[int, int]) -> None:
    """Enforce the strict Cartesian RGB source contract of the compositor."""
    if not isinstance(source_rgb, np.ndarray):
        raise TypeError(
            f"source_rgb must be a numpy.ndarray, got {type(source_rgb).__name__}"
        )
    if source_rgb.ndim != 3:
        raise ValueError(
            "source_rgb must be 3-D (height, width, 3), got shape "
            f"{source_rgb.shape}"
        )
    if source_rgb.size == 0:
        raise ValueError("source_rgb must not be empty")
    if source_rgb.shape[2] != _RGB_CHANNELS:
        raise ValueError(
            "source_rgb must have exactly 3 channels (RGB), got shape "
            f"{source_rgb.shape}"
        )
    if source_rgb.dtype != np.uint8:
        raise ValueError(
            f"source_rgb must have dtype uint8, got {source_rgb.dtype}"
        )
    if source_rgb.shape[:2] != mask_shape:
        raise ValueError(
            "source_rgb spatial shape must match mask exactly, got "
            f"{source_rgb.shape[:2]} vs mask {mask_shape}; the compositor "
            "never resizes -- align the RGB source in preprocessing instead"
        )
