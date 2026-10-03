"""Stripe mode (Round 1): variable-width stripe halftone core.

The core idea, shared with the future Spiral mode, is that the *width* of a
line is driven by the local grayscale of the source image, so that the
black/white **area ratio** reproduces the perceived tone.

Round 1 scope (deliberately small):

* 2-D ``uint8`` grayscale input only;
* a fixed stripe-center spacing ``period`` measured in output pixels;
* one stripe orientation angle;
* a strict boolean mask as the single output;
* a plain linear gray -> width mapping.

Everything that turns that mask into a picture (black on white, inverted,
colored, transparent, ...) belongs to a separate renderer layer and is *not*
part of this module.  The mask never encodes color.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["stripe_mask"]


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
    The algorithm works on *pixel centers*.  Output pixel ``x`` covers the
    continuous interval ``[x, x + 1)`` and has center ``c = x + 0.5``.  A pixel
    is part of a line only when its center ``c`` falls inside the continuous
    line interval ``[center - W/2, center + W/2)``.  Deciding membership by
    "does the pixel rectangle merely touch the line interval" would
    systematically thicken thin lines, so it is deliberately avoided here.
    """
    _validate_inputs(gray, period, angle_deg)

    height, width = gray.shape
    theta = _normalize_angle(angle_deg)

    if _is_horizontal(theta):
        # Horizontal stripes are vertical stripes on the transposed image.
        # Reuse the very same routine instead of maintaining a second,
        # independent horizontal algorithm.
        return stripe_mask(gray.T, period, 90.0).T

    # General, non-horizontal case.  The image is never rotated: instead each
    # row y gets a horizontal phase offset, which is what the old construction
    # called shift(y) = y * cot(theta).
    if theta == 90.0:
        # Exactly vertical stripes must give phase(y) == 0 for every row.  Take
        # this branch explicitly so that the tiny floating-point residue of
        # cot(90 deg) cannot accumulate into a per-row drift.
        cot = 0.0
    else:
        cot = math.cos(math.radians(theta)) / math.sin(math.radians(theta))

    period_f = float(period)
    mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        # Row phase, directly from the frozen definition:
        #
        #     shift(y) = y * cot(theta)        phase(y) = shift(y) mod P
        #
        # Because period_f > 0 the modulo always lands the phase inside a single
        # period.  The shift is deliberately *not* reduced first: keeping the
        # expression identical to the specification makes it trivial to audit
        # against the work order.
        phase = (y * cot) % period_f

        # Cell k spans [phase + k*P, phase + (k+1)*P) in continuous x; its
        # center is at phase + (k + 0.5) * P.
        #
        # Start one cell to the left of the origin.  Its right edge is at most
        # P > 0, so with k = -1 the leftmost partial cell is always visited, and
        # when phase == 0 that cell is exactly [-P, 0) which simply contains no
        # canvas pixel.  This is also what makes every angle use one and the
        # same loop: because the phase is already normalized into [0, P), the
        # enumeration no longer depends on the sign or the magnitude of
        # cot(theta) at all, so there is no special case for angles beyond 90
        # degrees.
        k = -1
        left = phase - period_f
        while True:
            right = left + period_f
            if left >= width:
                # This cell starts at or past the right edge: nothing of it can
                # intersect the canvas.  Later cells are even further right.
                break

            # Continuous cell interval -> integer column indices.
            #
            # Pixel x covers [x, x + 1) and has center x + 0.5, so the pixel
            # belongs to this cell exactly when
            #
            #     left <= x + 0.5 < right      <=>      left - 0.5 <= x < right - 0.5
            #
            # The integral columns satisfying that are, for half-open ranges,
            #
            #     x_start = ceil(left  - 0.5)
            #     x_stop  = ceil(right - 0.5)
            #
            # with x_stop used as an exclusive slice bound.  Note that this
            # derivation is *not* ceil(left) / ceil(right): those would drop
            # whole pixels whenever a cell boundary happens to sit just past an
            # integer.  Only the canvas is clipped afterwards, so the pixels of
            # clipped edge cells are still averaged, and nothing outside the
            # canvas is ever treated as white, black or zero.
            x_start = _snap_to_int(math.ceil(_snap(left - 0.5)))
            x_stop = _snap_to_int(math.ceil(_snap(right - 0.5)))
            x_start = max(0, x_start)
            x_stop = min(width, x_stop)

            if x_start < x_stop:
                # Step 1: mean gray of the source pixels inside this cell.
                mean_gray = float(gray[y, x_start:x_stop].mean())
                # Step 2: normalize to G with black = 0, white = 1.
                tone = mean_gray / 255.0
                # Step 3: naive linear mapping.  W uses the nominal period P
                # even for partial edge cells.
                line_width = period_f * (1.0 - tone)

                if line_width > 0.0:
                    center = left + period_f * 0.5
                    half_width = line_width * 0.5
                    span_left = center - half_width
                    span_right = center + half_width

                    # The same pixel-center rule, now for the line interval:
                    # pixel x is a line pixel iff span_left <= x + 0.5 < span_right.
                    first_px = _snap_to_int(math.ceil(_snap(span_left - 0.5)))
                    last_px = _snap_to_int(math.ceil(_snap(span_right - 0.5))) - 1
                    first_px = max(0, first_px)
                    last_px = min(width - 1, last_px)
                    if first_px <= last_px:
                        mask[y, first_px : last_px + 1] = True

            left = right
            k += 1

    return mask


def _snap(value: float) -> float:
    """Snap a value that is within round-off of an integer onto that integer.

    This is the *only* tolerance in the geometric pipeline.  It exists purely
    to absorb the representation residue of ``x +/- 0.5`` (values like
    ``7.999999999999999`` that stand for the integer 8), so that the exact
    ``ceil`` formula above lands on the intended sheet of the integer lattice.
    The bound is 1e-9, nine orders of magnitude below the 1 pixel quantization
    step, and the distance to the nearest integer is kept so that a genuine
    fractional boundary can never be dragged across an integer.
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

    if isinstance(period, bool) or not isinstance(period, (int, np.integer)):
        raise TypeError(f"period must be an integer, got {type(period).__name__}")
    if int(period) <= 0:
        raise ValueError(f"period must be a positive integer, got {int(period)}")

    if isinstance(angle_deg, bool) or not isinstance(angle_deg, (int, float, np.integer, np.floating)):
        raise TypeError(f"angle_deg must be a real number, got {type(angle_deg).__name__}")
    if not math.isfinite(float(angle_deg)):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")
