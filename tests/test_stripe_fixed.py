"""Fixed-width Stripe tests.

These pin the fixed-width geometry contract on top of the *frozen* variable-
width raster placement:

* every full cell draws exactly ``line_width`` consecutive pixels;
* the block center is nearest to the continuous stripe center, exact ties to
  the smaller start index (tie-left) -- inherited from the shared rasterizer;
* edge cells clip on the nominal canvas and are never shifted inward;
* the constant width is applied directly as an integer and is never routed
  through a synthetic grayscale image;
* ``line_width == period`` fills every covered cell, so at any angle a canvas
  tiled from full cells becomes entirely ``True``;
* all standard angles produce the right shape / dtype / phase behaviour.

No grayscale array is required anywhere: ``stripe_fixed_mask`` takes a shape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from halftone_playground import stripe_fixed_mask

P = 16
ALL_ANGLES = [0.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(row_mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(row_mask)))
    return out


def _snap(value: float) -> float:
    return round(value) if abs(value - round(value)) < 1e-9 else value


def _cot(theta: float) -> float:
    if theta == 90.0:
        return 0.0
    return math.cos(math.radians(theta)) / math.sin(math.radians(theta))


def _block_start(center: float, line_width: int) -> int:
    """Spec transcription: nearest block start, ties to the smaller index."""
    return math.ceil(_snap(center - line_width * 0.5 - 0.5))


def _full_cells(width: int, height: int, period: int, angle: float):
    """Yield ``(y, left, x_start, x_stop)`` for every full cell of the canvas.

    A "full cell" is one whose integer column set has exactly ``period`` pixels
    and lies entirely inside ``[0, width)``.  These are the cells on which the
    exact-width contract is asserted; the two edge cells are allowed to clip.
    """
    cot = _cot(angle)
    for y in range(height):
        phase = (y * cot) % period
        left = phase - period
        while True:
            if left >= width:
                break
            right = left + period
            xs = math.ceil(_snap(left - 0.5))
            xe = math.ceil(_snap(right - 0.5))
            if xs >= 0 and xe <= width and xe - xs == period:
                yield y, left, xs, xe
            left = right


# --------------------------------------------------------------------------
# 1. integer widths 1..period, odd widths included
# --------------------------------------------------------------------------


@pytest.mark.parametrize("line_width", [1, 2, 3, 5, 7, 8, 15, 16])
def test_full_cell_draws_exactly_line_width(line_width: int) -> None:
    """Every full cell of P=16 at 90 degrees shows exactly ``line_width`` px."""
    height, width = 8, 4 * P
    mask = stripe_fixed_mask((height, width), P, line_width, 90.0)

    for y in range(height):
        for _, _, xs, xe in _full_cells(width, height, P, 90.0):
            cell = mask[y, xs:xe]
            assert int(cell.sum()) == line_width, (line_width, y, xs, xe)
            runs = _runs(cell)
            assert len(runs) == 1
            assert runs[0][1] - runs[0][0] == line_width


def test_all_integer_widths_reachable_including_odd() -> None:
    """The full 1..P ladder is reachable directly -- no 256-level gray limit."""
    for line_width in range(1, P + 1):
        mask = stripe_fixed_mask((4, 4 * P), P, line_width, 90.0)
        counts = {int(mask[y, xs:xe].sum())
                  for y, _, xs, xe in _full_cells(4 * P, 4, P, 90.0)}
        assert counts == {line_width}, line_width


def test_direct_integer_beats_period_over_255() -> None:
    """A period > 255 still exposes every integer width.

    A synthetic uint8 grayscale route would collapse width levels here; the
    direct integer path must not.
    """
    period = 300
    for line_width in (1, 150, 151, 299, 300):
        mask = stripe_fixed_mask((2, 2 * period), period, line_width, 90.0)
        counts = {int(mask[0, xs:xe].sum())
                  for _, _, xs, xe in _full_cells(2 * period, 2, period, 90.0)}
        assert counts == {line_width}, (period, line_width)


# --------------------------------------------------------------------------
# 2. tie-left and nearest block center (shared raster placement)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("line_width,expected_start", [(5, 5), (7, 4)])
def test_odd_width_full_cell_position_p16(line_width: int, expected_start: int) -> None:
    """N=5 ties to the smaller index ([5,10)); N=7 is uniquely [4,11)."""
    mask = stripe_fixed_mask((1, P), P, line_width, 90.0)
    cell = mask[0, :P]
    assert int(cell.sum()) == line_width
    assert _runs(cell) == [(expected_start, expected_start + line_width)]


def test_block_center_nearest_and_tie_left_for_every_width() -> None:
    """For P=16 the chosen start is distance-minimal, ties to the smaller index."""
    for line_width in range(1, P + 1):
        mask = stripe_fixed_mask((1, P), P, line_width, 90.0)
        start = _runs(mask[0, :P])[0][0]
        center = 8.0
        d_impl = abs(start + line_width / 2 - center)
        for s in range(start - 2, start + 3):
            d = abs(s + line_width / 2 - center)
            assert d >= d_impl - 1e-9, (line_width, s)
            if abs(d - d_impl) <= 1e-9:
                assert s >= start, (line_width, s)


# --------------------------------------------------------------------------
# 3. angles: shape, dtype, contiguous blocks, fractional phase, edge safety
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", ALL_ANGLES)
def test_shape_dtype_and_contiguous_blocks(angle: float) -> None:
    height, width = 37, 53
    line_width = 5
    mask = stripe_fixed_mask((height, width), 8, line_width, angle)
    assert mask.shape == (height, width)
    assert mask.dtype == np.bool_
    assert set(np.unique(mask).tolist()) <= {False, True}

    # Each full cell draws exactly line_width pixels, and they form one block.
    if angle in (0.0, 180.0):
        # Horizontal stripes are the 90-degree pattern on the transposed
        # canvas, so the "width" axis is the row (height) axis here.
        for x in range(width):
            for _, _, xs, xe in _full_cells(height, width, 8, 90.0):
                cell = mask[xs:xe, x]
                assert int(cell.sum()) == line_width, (angle, x, xs, xe)
    else:
        for y, _, xs, xe in _full_cells(width, height, 8, angle):
            assert int(mask[y, xs:xe].sum()) == line_width, (angle, y, xs, xe)


@pytest.mark.parametrize("angle", ALL_ANGLES)
def test_edge_cells_never_exceed_line_width(angle: float) -> None:
    height, width = 24, 40
    mask = stripe_fixed_mask((height, width), P, 7, angle)

    if angle in (0.0, 180.0):
        # Horizontal stripes: a run along a row is the full stripe, so the
        # thickness lives along a column and must never exceed line_width.
        for y in range(height):
            for start, stop in _runs(mask[:, 0]):
                assert stop - start <= 7
    else:
        for y in range(height):
            for start, stop in _runs(mask[y]):
                assert stop - start <= 7


def test_edge_cell_is_clipped_not_shifted() -> None:
    """A 10px canvas, P=16, width=8: block [4,12) clips to [4,10), not shifted."""
    mask = stripe_fixed_mask((4, 10), P, 8, 90.0)
    assert mask.shape == (4, 10)
    for y in range(4):
        row = mask[y]
        assert int(row.sum()) == 6
        assert row[:4].tolist() == [False] * 4
        assert row[4:].tolist() == [True] * 6


def test_left_edge_cell_is_clipped_not_shifted() -> None:
    """45 degrees, row 8: the k=-1 cell centres at x=0 and clips at the left."""
    mask = stripe_fixed_mask((16, 16), P, 8, 45.0)
    row = mask[8]
    assert row[:4].tolist() == [True] * 4
    assert row[4:12].tolist() == [False] * 8
    assert row[12:].tolist() == [True] * 4


# --------------------------------------------------------------------------
# 4. periodic pattern: no grayscale needed, row-constant at 90 degrees
# --------------------------------------------------------------------------


def test_vertical_pattern_is_row_identical() -> None:
    mask = stripe_fixed_mask((40, 64), P, 3, 90.0)
    first = mask[0]
    for y in range(1, mask.shape[0]):
        assert np.array_equal(mask[y], first)


@pytest.mark.parametrize("angle", [30.0, 45.0, 60.0])
def test_oblique_phase_moves_smoothly(angle: float) -> None:
    """Non-90-degree phases follow shift(y) = y * cot(theta) mod P."""
    height, width = 32, 64
    line_width = 5
    mask = stripe_fixed_mask((height, width), P, line_width, angle)
    cot = _cot(angle)
    for y in range(height):
        phase = (y * cot) % P
        left = phase - P
        while left < 0:
            left += P
        if left + P > width:
            continue
        start = _block_start(left + P * 0.5, line_width)
        assert mask[y, start : start + line_width].all(), (angle, y, left)


# --------------------------------------------------------------------------
# 5. hard property: line_width == period fills everything
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [30.0, 45.0, 90.0, 135.0])
def test_line_width_equals_period_is_all_true(angle: float) -> None:
    """With line_width == P every cell fills, so the canvas is all True.

    The 37x53 canvas is deliberately not a whole number of periods wide at any
    tested angle, so both edge columns exercise clipped cells too.
    """
    mask = stripe_fixed_mask((37, 53), 8, 8, angle)
    assert mask.all(), f"uncovered pixels at {angle} degrees: " + str(
        [tuple(idx) for idx in np.argwhere(~mask)][:10]
    )


@pytest.mark.parametrize("period", [1, 4, 8, 16, 17, 64])
def test_line_width_equals_period_all_true_many_periods(period: int) -> None:
    mask = stripe_fixed_mask((48, 80), period, period, 90.0)
    assert mask.all()


# --------------------------------------------------------------------------
# 6. validation
# --------------------------------------------------------------------------


class TestShapeValidation:
    @pytest.mark.parametrize("shape", [(0, 4), (4, 0), (-1, 4), (4, -1)])
    def test_rejects_non_positive_dimensions(self, shape):
        with pytest.raises(ValueError):
            stripe_fixed_mask(shape, P, 5)

    @pytest.mark.parametrize("shape", [(4,), (4, 4, 4), ()])
    def test_rejects_wrong_length(self, shape):
        with pytest.raises(ValueError):
            stripe_fixed_mask(shape, P, 5)

    @pytest.mark.parametrize("shape", [[4, 4], "44", 4, np.array([4, 4])])
    def test_rejects_non_tuple(self, shape):
        with pytest.raises(TypeError):
            stripe_fixed_mask(shape, P, 5)

    @pytest.mark.parametrize("shape", [(True, 4), (4, False), (4.0, 4)])
    def test_rejects_bool_or_float_dimensions(self, shape):
        with pytest.raises(TypeError):
            stripe_fixed_mask(shape, P, 5)


class TestParameterValidation:
    @pytest.mark.parametrize("period", [0, -1, -16])
    def test_rejects_non_positive_period(self, period):
        with pytest.raises(ValueError, match="period must be a positive integer"):
            stripe_fixed_mask((8, 8), period, 1)

    @pytest.mark.parametrize("period", [16.0, True, "16"])
    def test_rejects_non_integer_period(self, period):
        with pytest.raises((TypeError, ValueError)):
            stripe_fixed_mask((8, 8), period, 1)

    @pytest.mark.parametrize("line_width", [0, -1, -5])
    def test_rejects_non_positive_line_width(self, line_width):
        with pytest.raises(ValueError, match="line_width must be a positive integer"):
            stripe_fixed_mask((8, 8), P, line_width)

    @pytest.mark.parametrize("line_width", [17, 32, 100])
    def test_rejects_line_width_above_period(self, line_width):
        with pytest.raises(ValueError, match="must not exceed period"):
            stripe_fixed_mask((8, 8), P, line_width)

    @pytest.mark.parametrize("line_width", [5.0, True, "5"])
    def test_rejects_non_integer_line_width(self, line_width):
        with pytest.raises((TypeError, ValueError)):
            stripe_fixed_mask((8, 8), P, line_width)

    def test_does_not_clamp_line_width(self):
        # 17 > P=16 must raise, never silently become 16.
        with pytest.raises(ValueError):
            stripe_fixed_mask((8, 8), P, 17)
        # P == line_width is the inclusive upper bound.
        assert stripe_fixed_mask((8, 8), P, P, 90.0).all()

    @pytest.mark.parametrize("angle", [float("inf"), float("-inf"), float("nan")])
    def test_rejects_non_finite_angle(self, angle):
        with pytest.raises(ValueError, match="finite"):
            stripe_fixed_mask((8, 8), P, 5, angle)

    @pytest.mark.parametrize("angle", ["90", None, True])
    def test_rejects_non_numeric_angle(self, angle):
        with pytest.raises(TypeError):
            stripe_fixed_mask((8, 8), P, 5, angle)

    def test_accepts_numpy_integers(self):
        mask = stripe_fixed_mask((np.int64(8), np.int64(8)), np.int64(16), np.int64(5))
        assert mask.dtype == np.bool_


# --------------------------------------------------------------------------
# 7. determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", ALL_ANGLES)
def test_deterministic(angle: float) -> None:
    first = stripe_fixed_mask((48, 64), P, 5, angle)
    second = stripe_fixed_mask((48, 64), P, 5, angle)
    assert np.array_equal(first, second)
