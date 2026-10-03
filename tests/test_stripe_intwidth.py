"""Stripe Round 2 tests: integer line-width rasterization semantics.

These tests pin the *frozen* Round 2 contract:

* ``N = floor(W_continuous + 0.5)`` clamped to ``[0, P]`` (half-up, not
  banker's rounding);
* a full cell draws exactly ``N`` consecutive black pixels;
* the block start ``s`` minimizes ``|s + N/2 - center|`` with exact ties
  resolved towards the smaller index (tie-left), deterministically, with no
  alternating bias, dithering or error diffusion;
* edge cells place the block on the nominal canvas first and clip afterwards,
  so they may show fewer than ``N`` black pixels and are never shifted inward;
* pure white stays empty and pure black covers the whole canvas at every
  standard angle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from halftone_playground import stripe_mask

P = 16


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _constant(value: int, height: int, width: int) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _cell_source(n_black: int, height: int = 1, width: int = P,
                 period: int = P) -> np.ndarray:
    """A canvas whose only full cell holds ``n_black`` black pixels.

    With ``width == period`` the first cell ``[0, P)`` is the one and only
    full cell, so the rest of the canvas cannot interfere with the
    measurement.
    """
    gray = np.full((height, width), 255, dtype=np.uint8)
    gray[:, :n_black] = 0
    return gray


def _block_start(center: float, line_width: int) -> int:
    """Spec transcription: nearest block start, ties to the smaller index."""
    return math.ceil(_snap(center - line_width * 0.5 - 0.5))


def _snap(value: float) -> float:
    return round(value) if abs(value - round(value)) < 1e-9 else value


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


def _cot(theta: float) -> float:
    if theta == 90.0:
        return 0.0
    return math.cos(math.radians(theta)) / math.sin(math.radians(theta))


# --------------------------------------------------------------------------
# 1. the 17 integer levels 0..P (the key regression test)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_black", range(P + 1))
def test_full_cell_produces_every_integer_width(n_black: int) -> None:
    """N black + (P-N) white pixels in a cell must draw exactly N black pixels.

    The cell mean maps to ``W_continuous = N`` exactly, so the mask must show
    precisely ``N`` consecutive black pixels -- the full 0..P ladder of
    integer widths, odd values included, has to be reachable.
    """
    gray = _cell_source(n_black)
    mask = stripe_mask(gray, P, 90.0)

    cell = mask[0, :P]
    assert int(cell.sum()) == n_black
    runs = _runs(cell)
    if n_black == 0:
        assert runs == []
    else:
        assert len(runs) == 1
        assert runs[0][1] - runs[0][0] == n_black


# --------------------------------------------------------------------------
# 2. half-up rounding of non-integer continuous widths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_black",
    # Sums chosen so that W = 16 - S/255 sweeps across the 4.5 boundary and
    # several other fractional levels: S=2945 -> W=4.451, S=2933 -> W=4.498,
    # S=2932 -> W=4.502, S=2757 -> W=5.2, S=2601 -> W=5.8, S=2867 -> W=4.757.
    [2945, 2933, 2932, 2757, 2601, 2867],
)
def test_rounding_is_floor_half_up(n_black: int) -> None:
    """actual width must equal floor(W_continuous + 0.5) for fractional W."""
    # A cell of 16 pixels summing to n_black: mix 0s and 255s so the sum is
    # exact regardless of position.
    gray = _cell_from_sum(n_black)
    mask = stripe_mask(gray, P, 90.0)

    mean_gray = float(gray[0, :P].mean())
    w_continuous = P * (1.0 - mean_gray / 255.0)
    expected = math.floor(w_continuous + 0.5)

    actual = int(mask[0, :P].sum())
    assert actual == expected
    assert expected in (4, 5, 6)  # sanity: the case really is fractional


def _cell_from_sum(total: int) -> np.ndarray:
    """A 1x16 cell whose pixel sum is exactly ``total`` (mixed 0/255 values)."""
    assert 0 <= total <= 16 * 255
    n_white, remainder = divmod(total, 255)
    values = [0] * (16 - n_white) + [255] * n_white
    if remainder:
        values[0] = remainder  # one intermediate pixel absorbs the rest
    return np.array([values], dtype=np.uint8)


def test_half_up_boundary_examples() -> None:
    """4.49 -> 4 and 4.51 -> 5, checked through exact pixel sums.

    With P = 16 a cell sum S gives W = 16 - S/255, so S = 2933 -> W = 4.498
    (rounds down) and S = 2932 -> W = 4.502 (rounds up).  The two cases sit
    on opposite sides of the 4.5 boundary and must differ by one pixel.
    """
    low = int(stripe_mask(_cell_from_sum(2933), P, 90.0)[0, :P].sum())
    high = int(stripe_mask(_cell_from_sum(2932), P, 90.0)[0, :P].sum())
    assert low == 4
    assert high == 5


# --------------------------------------------------------------------------
# 3. odd widths: nearest block center, tie-left
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_black,expected_start",
    [(5, 5), (7, 4)],
)
def test_odd_width_full_cell_position_p16(n_black: int, expected_start: int) -> None:
    """P=16, N=5 and N=7 in a 90-degree full cell.

    The cell spans [0, 16) with center 8.  For N = 5 both s = 5 (block center
    7.5) and s = 6 (block center 8.5) are exactly 0.5 px away: the tie must
    go to the smaller index, so the block is [5, 10).  For N = 7 the nearest
    block is uniquely [4, 11) (block center 7.5, distance 0.5).
    """
    gray = _cell_source(n_black)
    mask = stripe_mask(gray, P, 90.0)

    cell = mask[0, :P]
    assert int(cell.sum()) == n_black
    assert len(runs := _runs(cell)) == 1
    assert runs[0] == (expected_start, expected_start + n_black)

    # The chosen start really is the distance-minimal one, and among equally
    # good starts it is the smallest index.
    center = 8.0
    distances = {
        s: abs(s + n_black / 2 - center) for s in (expected_start - 1, expected_start, expected_start + 1)
    }
    best = min(distances.values())
    assert abs(distances[expected_start] - best) < 1e-12
    if any(
        abs(d - distances[expected_start]) < 1e-12
        for s, d in distances.items()
        if s < expected_start
    ):
        pytest.fail("a strictly smaller start index is equally good")


def test_odd_width_bias_is_deterministic_across_rows() -> None:
    """Odd widths keep the same fixed side on every row: no alternation."""
    gray = _cell_source(7, height=40)
    mask = stripe_mask(gray, P, 90.0)

    starts = {run[0] for y in range(mask.shape[0]) for run in _runs(mask[y, :P])}
    assert starts == {4}


# --------------------------------------------------------------------------
# 4. fractional phase at 45 / 30 / 60 degrees
# --------------------------------------------------------------------------


GRAY_ODD = 140  # W = 16 * (1 - 140/255) = 7.2157 -> N = 7 (odd)


@pytest.mark.parametrize("angle", [45.0, 30.0, 60.0])
def test_fractional_phase_odd_width_full_cells(angle: float) -> None:
    """Constant gray with an odd integer width, across many rows.

    Every full cell of every row must show exactly N consecutive black
    pixels whose block center is nearest to the continuous stripe center,
    with stable tie-left behaviour and no systematic 1 px drift.
    """
    height, width = 64, 64
    gray = _constant(GRAY_ODD, height, width)
    mask = stripe_mask(gray, P, angle)

    cot = _cot(angle)
    n_checked = 0
    for y in range(height):
        phase = (y * cot) % P
        left = phase - P
        while True:
            if left >= width:
                break
            right = left + P
            xs = math.ceil(_snap(left - 0.5))
            xe = math.ceil(_snap(right - 0.5))
            if xs >= 0 and xe <= width and xe - xs == P:
                # A full cell, fully inside the canvas.
                center = left + P * 0.5
                start = _block_start(center, 7)
                run = mask[y, start : start + 7]

                assert run.all(), (angle, y, left, start)
                # Everything else inside the cell stays white (the block of
                # a neighbouring cell can never intrude into this cell).
                cell = mask[y, xs:xe]
                assert int(cell.sum()) == 7, (angle, y, left)

                # Nearest-center property, checked independently by scanning
                # candidate starts around the optimum.  The continuous center
                # itself carries representation noise at oblique angles
                # (cot(45 deg) is 1.0000000000000002, not 1.0), so distances
                # within 1e-9 count as an exact tie -- and a tie must resolve
                # to the *smaller* index.
                d_impl = abs(start + 3.5 - center)
                for s in range(start - 2, start + 3):
                    d = abs(s + 3.5 - center)
                    assert d >= d_impl - 1e-9, (angle, y, left, s, d, d_impl)
                    if abs(d - d_impl) <= 1e-9:
                        assert s >= start, (angle, y, left, s)
                n_checked += 1
            left = right

    assert n_checked >= 100  # the scan really exercised many cells


def test_fractional_phase_no_systematic_drift() -> None:
    """Start indices must follow the geometry row by row, not accumulate."""
    height, width = 96, 64
    gray = _constant(GRAY_ODD, height, width)
    mask = stripe_mask(gray, P, 30.0)

    cot = _cot(30.0)
    for y in range(height):
        phase = (y * cot) % P
        left = phase - P
        while left < 0:
            left += P
        if left + P > width:
            continue  # no full cell in this row's first period window
        center = left + P * 0.5
        start = _block_start(center, 7)
        assert mask[y, start : start + 7].all(), (y, left, start)
        if start + 7 < width:
            assert not mask[y, start + 7], (y, left, start)


# --------------------------------------------------------------------------
# 5. edge cells: clip on the nominal canvas, never shift inward
# --------------------------------------------------------------------------


def test_right_edge_cell_is_clipped_not_shifted() -> None:
    """A 10px canvas, P = 16: the block [4, 12) must clip to [4, 10).

    Shifting the block inward to keep 8 visible pixels would move the stripe
    center, which is forbidden: only 6 pixels may be visible.
    """
    gray = _constant(128, 4, 10)  # W = 7.97 -> N = 8
    mask = stripe_mask(gray, P, 90.0)

    assert mask.shape == (4, 10)
    for y in range(4):
        row = mask[y]
        assert int(row.sum()) == 6  # 8 nominal, 2 clipped away
        assert row[:4].tolist() == [False] * 4
        assert row[4:].tolist() == [True] * 6


def test_left_edge_cell_is_clipped_not_shifted() -> None:
    """45 degrees: the k = -1 cell centres at x = 0 and clips at the left.

    Row 8 of a 45-degree pattern has phase 8, so the left-hand cell spans
    [-8, 8) with center 0; its N = 8 block is [-4, 4), clipped to [0, 4).
    Moving it right to show more pixels would move the center.
    """
    gray = _constant(128, 16, 16)
    mask = stripe_mask(gray, P, 45.0)

    row = mask[8]
    assert row[:4].tolist() == [True] * 4
    assert row[4:12].tolist() == [False] * 8
    assert row[12:].tolist() == [True] * 4


def test_edge_cells_never_exceed_nominal_width() -> None:
    """No clipped edge cell may show more black pixels than its N allows."""
    gray = _constant(140, 24, 40)  # N = 7
    mask = stripe_mask(gray, P, 45.0)

    for y in range(mask.shape[0]):
        for start, stop in _runs(mask[y]):
            assert stop - start <= 7


# --------------------------------------------------------------------------
# 6. pure white / pure black at all standard angles
# --------------------------------------------------------------------------


ALL_ANGLES = [0.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]


@pytest.mark.parametrize("angle", ALL_ANGLES)
def test_pure_white_is_empty_at_all_angles(angle: float) -> None:
    mask = stripe_mask(_constant(255, 37, 53), 8, angle)
    assert not mask.any()


@pytest.mark.parametrize("angle", ALL_ANGLES)
def test_pure_black_is_full_at_all_angles(angle: float) -> None:
    """W = P means N = P: every cell fills exactly, leaving no seam or gap.

    The 37x53 canvas is deliberately not a whole number of periods wide for
    any of the tested angles, so both edge columns exercise clipped cells.
    """
    mask = stripe_mask(_constant(0, 37, 53), 8, angle)
    assert mask.shape == (37, 53)
    assert mask.all(), f"uncovered pixels at {angle} degrees: " + str(
        [tuple(idx) for idx in np.argwhere(~mask)][:10]
    )


# --------------------------------------------------------------------------
# 7. whole-image oracle: every full cell matches floor(W + 0.5) exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [90.0, 45.0])
def test_every_full_cell_matches_floor_half_up(angle: float) -> None:
    """The Round 2 acceptance property: exact nearest-integer widths, always.

    For every full cell of a natural gradient image the measured black-pixel
    count must equal floor(W_continuous + 0.5) exactly -- 100 % of cells,
    not "mostly".  This is the automated twin of the quantitative check in
    the Round 2 review.
    """
    height, width = 48, 80
    ramp = np.linspace(255.0, 0.0, num=width)
    gray = np.tile(ramp, (height, 1)).astype(np.uint8)
    mask = stripe_mask(gray, P, angle)

    cot = _cot(angle)
    checked = 0
    for y in range(height):
        phase = (y * cot) % P
        left = phase - P
        while left < width:
            right = left + P
            xs = math.ceil(_snap(left - 0.5))
            xe = math.ceil(_snap(right - 0.5))
            if xs >= 0 and xe <= width and xe - xs == P:
                mean_gray = float(gray[y, xs:xe].mean())
                w_continuous = P * (1.0 - mean_gray / 255.0)
                expected = math.floor(w_continuous + 0.5)

                actual = int(mask[y, xs:xe].sum())
                assert actual == expected, (angle, y, left, w_continuous)
                checked += 1
            left = right

    assert checked >= 100


# --------------------------------------------------------------------------
# 8. non-divisible canvas sizes (partial trailing cell regression)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width,height,period", [(100, 8, 17), (50, 37, 9)])
def test_partial_cell_canvases_stay_in_bounds(width: int, height: int, period: int) -> None:
    """Non-divisible sizes: correct shape, pure black fills everything."""
    mask = stripe_mask(_constant(0, height, width), period, 90.0)
    assert mask.shape == (height, width)
    assert mask.all()

    white = stripe_mask(_constant(255, height, width), period, 90.0)
    assert not white.any()


def test_period_is_not_modified_for_partial_cells() -> None:
    """The nominal P stays 17 even on a 100px canvas (no auto-fit)."""
    gray = _constant(127, 8, 100)  # W = 17 * 128/255 = 8.53 -> N = 9
    mask = stripe_mask(gray, 17, 90.0)
    for y in range(8):
        runs = _runs(mask[y])
        for start, stop in runs:
            # Every run inside a full cell is exactly 9 px; only the two edge
            # remnants may be shorter.
            if 0 < start and stop < 100:
                assert stop - start == 9
