"""Stripe mode (Round 2): variable-width stripe halftone core.

The core idea, shared with the future Spiral mode, is that the *width* of a
line is driven by the local grayscale of the source image, so that the
black/white **area ratio** reproduces the perceived tone.

Scope:

* 2-D ``uint8`` grayscale input only;
* a fixed stripe-center spacing ``period`` measured in output pixels;
* one stripe orientation angle;
* a strict boolean mask as the single output;
* a plain linear gray -> width mapping.

Everything that turns that mask into a picture (black on white, inverted,
colored, transparent, ...) belongs to a separate renderer layer and is *not*
part of this module.  The mask never encodes color.

Round 2 rasterization semantics
--------------------------------

Each cell first maps its mean gray to a continuous width

    W_continuous = P * (1 - G)

and then quantizes it to an explicit **integer line width**

    N = floor(W_continuous + 0.5), clamped to [0, P]

(half-up rounding, deliberately *not* Python's banker's ``round``).  A full
cell must draw exactly ``N`` consecutive black pixels -- not "approximately
N".  The integer block is positioned so that its geometric center
``s + N/2`` (pixel ``x`` covering ``[x, x+1)`` with center ``x + 0.5``) is as
close as possible to the continuous stripe center; an exact tie chooses the
smaller start index (tie-left).  Odd ``N`` can therefore carry a fixed,
deterministic bias of at most half a pixel -- no alternating, dithering or
error diffusion is used.  Edge cells place the block on the nominal canvas
first and clip afterwards, so a clipped edge cell may show *fewer* than ``N``
black pixels and is never shifted inward.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["stripe_fixed_mask", "stripe_mask"]


def stripe_mask(gray: np.ndarray, period: int, angle_deg: float = 90.0) -> np.ndarray:
    """Return a boolean mask marking the stripe line area of ``gray``.

    Parameters
    ----------
    gray:
        A non-empty 2-D ``uint8`` grayscale array of shape ``(height, width)``.
        Black is 0 and white is 255.
    period:
        Distance between adjacent stripe **centers**, in output pixels.  Must
        be a positive integer.  It is a nominal, geometric spacing: it is never
        rescaled to "fill" the canvas, and the final cell of a row is allowed to
        be cut off by the canvas edge.
    angle_deg:
        Stripe orientation in image coordinates, in degrees.

        * ``90`` -> vertical stripes
        * ``0`` / ``180`` -> horizontal stripes

        The orientation repeats every 180 degrees; any finite value is accepted
        and normalized into ``[0, 180)``.

    Returns
    -------
    numpy.ndarray
        A ``bool`` array with the same shape as ``gray``.  ``True`` means "this
        output pixel belongs to a stripe line".  The mask carries geometry
        only, never color.

    Notes
    -----
    Cell ownership is decided on *pixel centers*: output pixel ``x`` covers
    the continuous interval ``[x, x + 1)`` and has center ``c = x + 0.5``.  A
    pixel belongs to the cell whose continuous interval contains its center.
    The line itself is an explicit integer-width block (see the module
    docstring): ``N = floor(W + 0.5)`` consecutive pixels whose geometric
    center is nearest to the continuous stripe center, ties to the smaller
    start index.
    """
    _validate_inputs(gray, period, angle_deg)

    height, width = gray.shape
    theta = _normalize_angle(angle_deg)

    if _is_horizontal(theta):
        # Horizontal stripes are vertical stripes on the transposed image.
        # Reuse the very same routine instead of maintaining a second,
        # independent horizontal algorithm.
        return stripe_mask(gray.T, period, 90.0).T

    period_f = float(period)
    cot = _stripe_cot(theta)
    mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for left, x_start, x_stop in _row_cells(y, cot, period_f, width):
            if x_start >= x_stop:
                continue

            # Step 1: mean gray of the source pixels inside this cell.
            mean_gray = float(gray[y, x_start:x_stop].mean())
            # Step 2: normalize to G with black = 0, white = 1.
            tone = mean_gray / 255.0
            # Step 3: naive linear mapping.  W uses the nominal period P even
            # for partial edge cells.
            w_continuous = period_f * (1.0 - tone)

            # Step 4 (Round 2): quantize to an explicit integer line width.
            # Half-up rounding of the continuous width:
            #     4.49 -> 4    4.50 -> 5    4.51 -> 5
            # deliberately not Python's banker's round().  With G in [0, 1]
            # the value is already inside [0, P]; the clamp keeps that
            # guarantee explicit against floating-point residue.
            line_width = _quantize_line_width(w_continuous, period)

            if line_width > 0:
                _place_block(mask, y, left, period_f, line_width, width)

    return mask


def stripe_fixed_mask(
    shape: tuple[int, int],
    period: int,
    line_width: int,
    angle_deg: float = 90.0,
) -> np.ndarray:
    """Return a boolean mask of **fixed-width** stripes.

    The geometry (cell ownership, phase, angle semantics, raster placement,
    tie-left, edge clipping) is exactly the frozen variable-width Stripe
    geometry; the only difference is where the per-cell width comes from:

    =========================  ============================================
    :func:`stripe_mask`        cell ``N = floor(P * (1 - G) + 0.5)``
    :func:`stripe_fixed_mask`  cell ``N = line_width`` (constant)
    =========================  ============================================

    Parameters
    ----------
    shape:
        The output shape as ``(height, width)``, two positive integers.  A
        *shape* is accepted rather than a grayscale array because fixed-width
        geometry never looks at the source grayscale.
    period:
        Distance between adjacent stripe **centers**, in output pixels.
        Positive integer; the same nominal spacing as :func:`stripe_mask`.
    line_width:
        Constant line width in output pixels.  Positive integer with
        ``1 <= line_width <= period``.  There is deliberately **no automatic
        clamping**: a value outside the range raises instead of being silently
        adjusted.
    angle_deg:
        Stripe orientation in image coordinates, in degrees.  Same semantics as
        :func:`stripe_mask`: ``90`` -> vertical, ``0`` / ``180`` -> horizontal,
        orientation repeating every 180 degrees.

    Returns
    -------
    numpy.ndarray
        A 2-D ``bool`` array of ``shape``.  ``True`` marks a line pixel.  Every
        *full* cell draws exactly ``line_width`` consecutive pixels, placed so
        the block center is nearest to the continuous stripe center (exact ties
        to the smaller start index); edge cells are clipped on the nominal
        canvas and may therefore show fewer visible pixels.  The mask carries
        geometry only, never color.

    Notes
    -----
    The constant width enters the *integer* rasterizer directly as
    ``line_width``.  It is never routed through a synthetic constant grayscale
    image: ``uint8`` gray has only 256 levels, so for ``period > 255`` such a
    route could not represent every integer width and would silently lose
    width levels.
    """
    height, width = _validate_fixed_shape(shape)
    period = _validate_period(period)
    line_width = _validate_line_width(line_width, period)
    _validate_angle(angle_deg)

    theta = _normalize_angle(angle_deg)

    if _is_horizontal(theta):
        # Same transpose reuse as the variable-width core: horizontal stripes
        # are vertical stripes on the transposed canvas.
        return stripe_fixed_mask((width, height), period, line_width, 90.0).T

    period_f = float(period)
    cot = _stripe_cot(theta)
    mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for left, x_start, x_stop in _row_cells(y, cot, period_f, width):
            if x_start >= x_stop:
                continue
            _place_block(mask, y, left, period_f, line_width, width)

    return mask


# --------------------------------------------------------------------------
# shared frozen raster geometry (private)
# --------------------------------------------------------------------------


def _stripe_cot(theta: float) -> float:
    """Return ``cot(theta)`` for a normalized, non-horizontal orientation.

    Exactly vertical stripes must give ``phase(y) == 0`` on every row, so the
    ``90`` degree case returns a hard ``0.0``: that keeps the tiny
    floating-point residue of ``cot(90 deg)`` from accumulating into a per-row
    drift.
    """
    if theta == 90.0:
        return 0.0
    return math.cos(math.radians(theta)) / math.sin(math.radians(theta))


def _row_cells(y: int, cot: float, period_f: float, width: int):
    """Enumerate the cells of output row ``y`` as ``(left, x_start, x_stop)``.

    This is the frozen raster placement, shared by the variable-width and
    fixed-width Stripe cores so the two can never drift apart:

    * Row phase, directly from the frozen definition::

          shift(y) = y * cot(theta)        phase(y) = shift(y) mod P

      Because ``period_f > 0`` the modulo always lands the phase inside a single
      period.  The shift is deliberately *not* reduced first: keeping the
      expression identical to the specification makes it trivial to audit.

    * Cell ``k`` spans ``[phase + k*P, phase + (k+1)*P)`` in continuous x; its
      center is at ``phase + (k + 0.5) * P``.  Enumeration starts one cell to
      the left of the origin (``k = -1``), whose right edge is at most ``P > 0``,
      so the leftmost partial cell is always visited; when ``phase == 0`` that
      cell is exactly ``[-P, 0)`` and simply contains no canvas pixel.  This is
      what lets every angle share one loop: the phase is already normalized into
      ``[0, P)``, so the enumeration no longer depends on the sign or magnitude
      of ``cot(theta)`` -- there is no special case for angles beyond 90
      degrees.

    * Continuous cell interval -> integer column indices.  Pixel ``x`` covers
      ``[x, x + 1)`` and has center ``x + 0.5``, so the pixel belongs to this
      cell exactly when::

          left <= x + 0.5 < right   <=>   left - 0.5 <= x < right - 0.5

      giving ``x_start = ceil(left - 0.5)`` and
      ``x_stop = ceil(right - 0.5)`` (exclusive).  This is *not*
      ``ceil(left) / ceil(right)``: those would drop whole pixels whenever a
      cell boundary sits just past an integer.  Only the canvas is clipped
      afterwards, so clipped edge cells are still measured on the pixels they
      actually own and nothing outside the canvas is treated as white, black or
      zero.
    """
    phase = (y * cot) % period_f
    left = phase - period_f
    while True:
        if left >= width:
            # This cell starts at or past the right edge: nothing of it can
            # intersect the canvas, and later cells are even further right.
            break
        right = left + period_f
        x_start = _snap_to_int(math.ceil(_snap(left - 0.5)))
        x_stop = _snap_to_int(math.ceil(_snap(right - 0.5)))
        yield left, max(0, x_start), min(width, x_stop)
        left = right


def _quantize_line_width(w_continuous: float, period: int) -> int:
    """Round a continuous width to an integer in ``[0, period]`` (half-up).

    ``N = floor(W + 0.5)`` -- deliberately not Python's banker's ``round``.
    With ``G`` in ``[0, 1]`` the value is already inside ``[0, P]``; the clamp
    keeps that guarantee explicit against floating-point residue.
    """
    line_width = int(math.floor(w_continuous + 0.5))
    if line_width < 0:
        return 0
    if line_width > period:
        return period
    return line_width


def _place_block(
    mask: np.ndarray,
    y: int,
    left: float,
    period_f: float,
    line_width: int,
    width: int,
) -> None:
    """Write one ``line_width``-pixel block into ``mask`` row ``y``.

    The block is placed on the *nominal* canvas and only then clipped to the
    real one, the shared frozen rule for both cores:

    * its geometric center ``s + N/2`` is nearest to the continuous stripe
      center ``left + P/2``; the unconstrained optimum start is ``center - N/2``
      and the nearest integer ``s`` with ties towards the *smaller* index is
      ``ceil(center - N/2 - 0.5)`` (``ceil(d - 0.5)`` rounds half-down, which is
      exactly tie-left).  ``_snap`` absorbs the representation residue around
      integer values so the tie decision stays deterministic.
    * an edge cell may show fewer than ``N`` visible black pixels and is never
      shifted inward, because shifting would move the stripe center.
    """
    center = left + period_f * 0.5
    start = _snap_to_int(math.ceil(_snap(center - line_width * 0.5 - 0.5)))
    stop = start + line_width

    lo = max(start, 0)
    hi = min(stop, width)
    if lo < hi:
        mask[y, lo:hi] = True


def _snap(value: float) -> float:
    """Snap a value that is within round-off of an integer onto that integer.

    This is the *only* tolerance in the geometric pipeline, kept from Round 1.
    It exists purely to absorb the representation residue of floating-point
    arithmetic (values like ``7.999999999999999`` that stand for the integer
    8), so that the exact ``ceil`` formulae land on the intended sheet of the
    integer lattice.  Round 2 uses it in exactly two places: turning cell
    boundaries into integer column indices, and turning the continuous block
    start ``center - N/2 - 0.5`` into the integer block start -- the latter is
    what keeps the tie-left rule deterministic, because without the snap a
    mathematically exact tie could round ``up`` on one row and stay ``down``
    on the next purely through representation noise.  The bound is 1e-9, nine
    orders of magnitude below the 1 pixel quantization step, and the distance
    to the nearest integer is kept so that a genuine fractional boundary can
    never be dragged across an integer.
    """
    return round(value) if abs(value - round(value)) < 1e-9 else value


def _snap_to_int(value: int) -> int:
    """Return ``value`` as a plain Python int (``math.ceil`` already gives one)."""
    return int(value)


def _normalize_angle(angle_deg: float) -> float:
    """Normalize an orientation into ``[0, 180)`` degrees."""
    return float(angle_deg) % 180.0


def _is_horizontal(theta: float) -> bool:
    """Return True when the orientation is horizontal (0 degrees)."""
    return theta == 0.0


def _validate_inputs(gray: np.ndarray, period: int, angle_deg: float) -> None:
    """Reject inputs that are outside the frozen Round 1 contract.

    The contract is intentionally strict: no silent casting of other dtypes,
    no coercion of non-integer periods, no "best effort" handling of
    non-finite angles.
    """
    if not isinstance(gray, np.ndarray):
        raise TypeError(f"gray must be a numpy.ndarray, got {type(gray).__name__}")
    if gray.ndim != 2:
        raise ValueError(f"gray must be 2-D (height, width), got shape {gray.shape}")
    if gray.size == 0:
        raise ValueError("gray must not be empty")
    if gray.dtype != np.uint8:
        raise ValueError(f"gray must have dtype uint8, got {gray.dtype}")

    _validate_period(period)
    _validate_angle(angle_deg)


def _validate_period(period: int) -> int:
    """Reject anything that is not a positive plain integer period."""
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)):
        raise TypeError(f"period must be an integer, got {type(period).__name__}")
    if int(period) <= 0:
        raise ValueError(f"period must be a positive integer, got {int(period)}")
    return int(period)


def _validate_angle(angle_deg: float) -> None:
    """Reject anything that is not a finite real-number angle."""
    if isinstance(angle_deg, bool) or not isinstance(angle_deg, (int, float, np.integer, np.floating)):
        raise TypeError(f"angle_deg must be a real number, got {type(angle_deg).__name__}")
    if not math.isfinite(float(angle_deg)):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")


def _validate_fixed_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate a ``(height, width)`` output shape for :func:`stripe_fixed_mask`.

    Uses the same strict style as the rest of the module: a length-2 tuple of
    positive plain integers (booleans rejected).  No grayscale array is
    required, because fixed-width geometry never reads the source image.
    """
    if isinstance(shape, np.ndarray) or not isinstance(shape, tuple):
        raise TypeError(f"shape must be a tuple, got {type(shape).__name__}")
    if len(shape) != 2:
        raise ValueError(f"shape must have length 2 (height, width), got {shape!r}")
    height, width = shape
    for name, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"shape {name} must be an integer, got {type(value).__name__}"
            )
        if int(value) <= 0:
            raise ValueError(f"shape {name} must be positive, got {int(value)}")
    return int(height), int(width)


def _validate_line_width(line_width: int, period: int) -> int:
    """Reject anything but an integer ``1 <= line_width <= period``.

    There is deliberately no clamping: an out-of-range width is an error, not a
    value to be quietly adjusted.
    """
    if isinstance(line_width, bool) or not isinstance(line_width, (int, np.integer)):
        raise TypeError(
            f"line_width must be an integer, got {type(line_width).__name__}"
        )
    width = int(line_width)
    if width <= 0:
        raise ValueError(f"line_width must be a positive integer, got {width}")
    if width > period:
        raise ValueError(
            f"line_width must not exceed period, got line_width={width} > "
            f"period={period}"
        )
    return width
