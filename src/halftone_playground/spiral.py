"""Spiral mode (Round 1): linear-polar variable-width spiral core.

Construction
------------

Spiral mode deliberately does **not** rasterize an Archimedean spiral through
its parametric equation.  It recycles the exact same idea as Stripe mode by
changing the coordinate system:

    Cartesian grayscale
      -> linear polar unwrap            (``cv2.warpPolar``)
      -> shared variable-width Stripe core on the polar image
      -> inverse linear polar           (``cv2.warpPolar``, ``WARP_INVERSE_MAP``)
      -> circular support mask
      -> ``bool`` mask

On the polar canvas the *radius* becomes the horizontal axis and the *angle*
becomes the vertical axis.  A straight, tilted stripe drawn there is a
constant-pitch family of turns: unwrapping it back to Cartesian produces a
multi-arm spiral whose line width is driven by the local grayscale, exactly
like the stripes.  The spiral pitch, the number of arms and the chirality all
fall out of the *angle* of that tilt -- there is no spiral equation anywhere
in this module.

Stripe reuse
------------

The variable-width core is the existing :func:`halftone_playground.stripe.
stripe_mask`.  Spiral does not reimplement width computation, gray mapping,
integer-width quantization or rasterization; it only chooses the effective
polar angle.  The frozen Stripe round-2 width semantics therefore apply
unchanged to the polar image::

    W = P * (1 - G)          N = floor(W + 0.5)          (tie-left)

Chirality
---------

The default (and only) chirality is **clockwise when read from the centre
outwards**, i.e. looking at the picture the arms sweep towards positive
screen angle (right -> down -> left -> up).  This follows from using a
*positive* polar stripe slope; no ``chirality`` parameter is exposed.

Coordinate conventions
----------------------

* Polar rows (``y``) are **angle**, polar columns (``x``) are **radius**.
* Angle ``0`` points along ``+x`` (screen right); the angle grows towards
  ``+y`` (screen down), which is visually clockwise.
* ``cv2.warpPolar`` takes ``dsize = (radius_samples, angle_samples)`` because
  the destination has ``rows == angle`` and ``cols == radius``.

Geometry (frozen by probes 01 / 01b / 01c; not re-derived here)
--------------------------------------------------------------

==================  ==========================================
``center``          ``((N - 1) / 2, (N - 1) / 2)``
``warp_radius``     ``min(h, w) / 2``      (forward/inverse mapping only)
``support_radius``  ``(min(h, w) - 1) / 2`` (visible disc only)
``radius_samples``  ``ceil(warp_radius)``
``angle_samples``   ``floor(2 * pi * warp_radius + 0.5)``
``theta_eff``       ``degrees(atan2(angle_samples, arms * period))``
==================  ==========================================

``warp_radius`` and ``support_radius`` are two *distinct* concepts and are
intentionally never merged.  ``angle_samples`` uses an explicit half-up rule
(``floor(v + 0.5)``) rather than Python's banker's rounding so the sampling
strategy is deterministic for every caller.

Round 1 scope
-------------

* input must be a **square** 2-D ``uint8`` grayscale array;
* a ``bool`` mask is the only output -- rendering, inversion and color all
  belong to later layers;
* OpenCV must be able to allocate the polar image (see
  :data:`OPENCV_MAX_DIMENSION`).

The square restriction is a *Round 1 product boundary*, not a claim that a
polar unwrap is inherently square-only: the fit / crop / letterbox strategy
for non-square sources is simply not frozen yet.  ``image_scale`` is likewise
not a Spiral parameter; call :func:`halftone_playground.preprocess.
resize_grayscale` before :func:`spiral_mask` if a scale is wanted.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .stripe import stripe_mask

__all__ = ["spiral_mask"]

#: Hard limit of the OpenCV ``warpPolar`` / ``remap`` implementation: both the
#: source and the destination image dimensions must stay strictly below this
#: value.  It is a *library* limit, not an arbitrary product cap.
OPENCV_MAX_DIMENSION = 32767

#: Smallest square side that still yields meaningful polar geometry.  A 1x1
#: image has a zero-radius disc and is rejected explicitly rather than given a
#: special-case algorithm.
MIN_SIDE = 2

_FORWARD_FLAGS = (
    cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR
)
_INVERSE_FLAGS = (
    cv2.WARP_POLAR_LINEAR
    | cv2.WARP_INVERSE_MAP
    | cv2.WARP_FILL_OUTLIERS
    | cv2.INTER_NEAREST
)


def spiral_mask(gray: np.ndarray, period: int, arms: int) -> np.ndarray:
    """Return a boolean mask marking a variable-width spiral line area.

    Parameters
    ----------
    gray:
        A non-empty **square** 2-D ``uint8`` grayscale array of shape
        ``(n, n)``.  Black is 0 and white is 255.
    period:
        Positive integer, the distance between adjacent stripe lines in the
        *polar* image; measured in pixels of the Cartesian output image.  As
        with Stripe mode it is a nominal geometric spacing: it is never
        rescaled to fill anything and it is not tied to ``image_scale``.
    arms:
        Positive integer, the number of spiral arms.  The turns are generated
        purely from the polar stripe slope, so any ``arms >= 1`` works; a
        complex rosette at the centre for large ``arms`` is an accepted
        feature of the construction, not a defect.

    Returns
    -------
    numpy.ndarray
        A ``bool`` array with the same shape as ``gray``.  ``True`` marks a
        spiral line pixel; everything outside the circular support is always
        ``False`` (corners included).  The mask carries geometry only and
        never color.

    Raises
    ------
    TypeError
        If ``gray`` is not an ``ndarray``, or ``period`` / ``arms`` is not an
        integer (booleans included).
    ValueError
        If ``gray`` is not 2-D, empty, not ``uint8``, or not square (Round 1
        boundary); if a dimension is smaller than :data:`MIN_SIDE`; if
        ``period`` or ``arms`` is not positive; or if a source or polar
        dimension reaches :data:`OPENCV_MAX_DIMENSION`.
    """
    _validate_inputs(gray, period, arms)

    side = gray.shape[0]
    period = int(period)
    arms = int(arms)

    center = _polar_center(side)
    warp_radius = _warp_radius(side)
    support_radius = _support_radius(side)
    radius_samples = _radius_samples(warp_radius)
    angle_samples = _angle_samples(warp_radius)
    theta_effective = _effective_stripe_angle(period, arms, angle_samples)

    _validate_opencv_dimensions(
        side=side,
        radius_samples=radius_samples,
        angle_samples=angle_samples,
    )

    # Cartesian grayscale -> linear polar.  A slightly blurred source keeps
    # the grey sampling a point sample of the underlying signal: this is the
    # gray *unwrapping* filter, never anti-aliasing of the binary output.
    polar_gray = cv2.warpPolar(
        gray,
        (radius_samples, angle_samples),
        center,
        warp_radius,
        _FORWARD_FLAGS,
    )

    # Shared variable-width Stripe core on the polar image.  The requested
    # ``arms`` is realised entirely through this angle.
    polar_mask = stripe_mask(polar_gray, period, theta_effective)

    # Binarize before the inverse warp so the whole round trip stays strictly
    # bi-level.
    polar_lines = np.where(polar_mask, np.uint8(255), np.uint8(0))

    # Inverse linear polar.  ``WARP_FILL_OUTLIERS`` is mandatory: without it
    # destination pixels whose inverse radius exceeds ``warp_radius`` keep
    # uninitialised data and the corners become non-deterministic garbage.
    inverse = cv2.warpPolar(
        polar_lines,
        (side, side),
        center,
        warp_radius,
        _INVERSE_FLAGS,
    )

    return (inverse > 0) & _circular_support(side, center, support_radius)


# --------------------------------------------------------------------------
# geometry helpers (private)
# --------------------------------------------------------------------------


def _polar_center(side: int) -> tuple[float, float]:
    """Return the polar transform centre ``((N - 1) / 2, (N - 1) / 2)``.

    The half-pixel offset matters: an integer centre makes the mapping
    asymmetric and leaves edge artifacts.  512x512 therefore uses
    ``(255.5, 255.5)``.
    """
    c = (side - 1) / 2.0
    return (c, c)


def _warp_radius(side: int) -> float:
    """Radius used by the forward / inverse polar mapping (``N / 2``)."""
    return side / 2.0


def _support_radius(side: int) -> float:
    """Radius of the visible circular support (``(N - 1) / 2``).

    Smaller than :func:`_warp_radius` on purpose.  The polar image only has
    ``ceil(warp_radius)`` radial columns, so the outermost mapped radius is
    ``warp_radius`` while the last *sampled* radius is
    ``(radius_samples - 1) * warp_radius / radius_samples``.  Pixels between
    those two radii inside a ``warp_radius`` disc would have no valid radial
    sample; using ``(N - 1) / 2`` keeps the disc free of such pixels (probe
    01c: 880 bad pixels at ``support_radius = N / 2``, 0 at ``(N - 1) / 2``)
    while still passing the pure white / pure black strict checks.
    """
    return (side - 1) / 2.0


def _radius_samples(warp_radius: float) -> int:
    """Number of radial columns: ``ceil(warp_radius)``."""
    return int(math.ceil(warp_radius))


def _angle_samples(warp_radius: float) -> int:
    """Number of angular rows: ``floor(2 * pi * warp_radius + 0.5)``.

    The outer circle has roughly ``2 * pi * R`` pixels, so this gives about
    one angular sample per pixel of arc at the rim -- no uncovered gaps, no
    angular stretching.  The half-up rounding is explicit (``floor(v + 0.5)``
    with ``v > 0``) so the value never depends on banker's rounding.
    """
    return int(math.floor(2.0 * math.pi * warp_radius + 0.5))


def _effective_stripe_angle(period: int, arms: int, angle_samples: int) -> float:
    """Return the polar stripe angle that yields ``arms`` spiral arms.

    The Stripe core offsets row ``y`` by ``shift(y) = y * cot(theta)``.  One
    full trip around the polar image is ``angle_samples`` rows, so the total
    offset is ``angle_samples * cot(theta)`` periods.  The number of lines
    crossed on that trip *is* the number of arms, hence

        slope = cot(theta_effective) = arms * period / angle_samples

    and, with a strictly positive slope (frozen clockwise chirality),

        theta_effective = degrees(atan2(angle_samples, arms * period))

    A ``period`` that divides ``arms * period`` makes the trip close on an
    integer number of periods, so the ``0`` / ``+x`` seam lines up exactly.
    """
    return math.degrees(math.atan2(angle_samples, arms * period))


def _circular_support(
    side: int, center: tuple[float, float], support_radius: float
) -> np.ndarray:
    """Boolean disc of radius ``support_radius`` around ``center``.

    Pixel ``(y, x)`` is inside when its *centre* distance to ``center`` is
    ``<= support_radius``.  Integer pixel coordinates keep the disc exactly
    mirror-symmetric; ``warpPolar`` is never trusted to decide the product's
    disc boundary.
    """
    coords = np.arange(side, dtype=np.float64)
    dx = coords - center[0]
    dy = coords - center[1]
    distance_squared = dy[:, None] ** 2 + dx[None, :] ** 2
    return distance_squared <= support_radius * support_radius


# --------------------------------------------------------------------------
# validation helpers (private)
# --------------------------------------------------------------------------


def _validate_inputs(gray: np.ndarray, period: int, arms: int) -> None:
    """Enforce the strict Round 1 contract for :func:`spiral_mask`."""
    if not isinstance(gray, np.ndarray):
        raise TypeError(f"gray must be a numpy.ndarray, got {type(gray).__name__}")
    if gray.ndim != 2:
        raise ValueError(f"gray must be 2-D (height, width), got shape {gray.shape}")
    if gray.size == 0:
        raise ValueError("gray must not be empty")
    if gray.dtype != np.uint8:
        raise ValueError(f"gray must have dtype uint8, got {gray.dtype}")

    height, width = gray.shape
    if height != width:
        raise ValueError(
            "spiral_mask requires a square grayscale image in Round 1, got "
            f"shape {gray.shape}; the fit/crop/letterbox policy for non-square "
            "sources is not frozen yet"
        )
    if height < MIN_SIDE:
        raise ValueError(
            f"spiral_mask requires side >= {MIN_SIDE}, got {height}"
        )

    _validate_positive_int(period, "period")
    _validate_positive_int(arms, "arms")


def _validate_positive_int(value: int, name: str) -> None:
    """Reject anything that is not a positive plain integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {int(value)}")


def _validate_opencv_dimensions(
    *, side: int, radius_samples: int, angle_samples: int
) -> None:
    """Reject layouts that OpenCV's ``warpPolar`` / ``remap`` cannot allocate.

    Current OpenCV requires every image dimension to stay strictly below
    :data:`OPENCV_MAX_DIMENSION`.  This is a hard implementation limit of the
    library, so the Spiral entry point reports it clearly instead of tiling,
    chunking or silently shrinking the request.
    """
    for name, value in (
        ("height", side),
        ("width", side),
        ("radius_samples", radius_samples),
        ("angle_samples", angle_samples),
    ):
        if value >= OPENCV_MAX_DIMENSION:
            raise ValueError(
                f"{name}={value} reaches the OpenCV warpPolar/remap dimension "
                f"limit (dimensions must be < {OPENCV_MAX_DIMENSION}); this is "
                "an OpenCV implementation limit, not a halftone-playground cap"
            )
