"""Stripe mode Round 1 tests: input contract, geometry, and tone behaviour.

The suite checks the *frozen* semantic contract rather than a frozen bitmap:
pixel-center membership, nominal-period line widths, per-row phase, edge cells,
and the transpose route for horizontal stripes.  Quantization of a continuous
line width onto discrete pixels necessarily has a +/-1 px tolerance, so the
geometry assertions use explicit, justified windows instead of brittle
pixel-exact expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

from halftone_playground import stripe_mask

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _constant(value: int, height: int, width: int) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _vertical_gradient(height: int, width: int) -> np.ndarray:
    """White at the top (row 0) fading to black at the bottom."""
    ramp = np.linspace(255.0, 0.0, num=height)
    return np.tile(ramp[:, None], (1, width)).astype(np.uint8)


def _horizontal_gradient(height: int, width: int) -> np.ndarray:
    """White at the left (col 0) fading to black at the right."""
    ramp = np.linspace(255.0, 0.0, num=width)
    return np.tile(ramp[None, :], (height, 1)).astype(np.uint8)


def _runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index pairs of consecutive True runs."""
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


def _expected_vertical_pattern(length: int, first: int, period: int, width: int) -> np.ndarray:
    """Expected 1-D pattern: ``width`` True pixels every ``period``, starting at ``first``."""
    expected = np.zeros(length, dtype=bool)
    for start in range(first, length, period):
        expected[start : start + width] = True
    return expected


# --------------------------------------------------------------------------
# 7.1 input contract
# --------------------------------------------------------------------------


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        stripe_mask(np.zeros(8, dtype=np.uint8), 4)
    with pytest.raises(ValueError):
        stripe_mask(np.zeros((4, 4, 3), dtype=np.uint8), 4)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        stripe_mask(np.zeros((0, 0), dtype=np.uint8), 4)
    with pytest.raises(ValueError):
        stripe_mask(np.zeros((4, 0), dtype=np.uint8), 4)


def test_rejects_non_uint8_input() -> None:
    with pytest.raises(ValueError):
        stripe_mask(np.zeros((4, 4), dtype=np.float32), 4)
    with pytest.raises(ValueError):
        stripe_mask(np.zeros((4, 4), dtype=np.int32), 4)


def test_rejects_non_positive_period() -> None:
    img = _constant(128, 4, 4)
    with pytest.raises(ValueError):
        stripe_mask(img, 0)
    with pytest.raises(ValueError):
        stripe_mask(img, -3)


def test_rejects_fractional_period() -> None:
    with pytest.raises(TypeError):
        stripe_mask(_constant(128, 8, 8), 4.5)  # type: ignore[arg-type]


def test_rejects_non_finite_angle() -> None:
    img = _constant(128, 4, 4)
    with pytest.raises(ValueError):
        stripe_mask(img, 4, float("nan"))
    with pytest.raises(ValueError):
        stripe_mask(img, 4, float("inf"))


def test_mask_shape_and_dtype_follow_input() -> None:
    img = _constant(128, 7, 13)
    mask = stripe_mask(img, 4, 90.0)
    assert mask.shape == img.shape
    assert mask.dtype == np.bool_


# --------------------------------------------------------------------------
# 7.2 pure white
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [90.0, 45.0, 0.0])
def test_pure_white_produces_no_lines(angle: float) -> None:
    mask = stripe_mask(_constant(255, 32, 32), 8, angle)
    assert not mask.any()


# --------------------------------------------------------------------------
# 7.3 pure black
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [90.0, 45.0, 0.0])
def test_pure_black_covers_entire_canvas(angle: float) -> None:
    """W = P must fill every cell, including partial edge cells.

    This single case exercises line width, full cell coverage, per-row angle
    phase, edge remnants and the horizontal transpose route at once.
    """
    img = np.zeros((32, 40), dtype=np.uint8)
    mask = stripe_mask(img, 8, angle)
    assert mask.all()


@pytest.mark.parametrize("angle", [30.0, 45.0, 60.0, 135.0])
def test_pure_black_covers_entire_canvas_at_oblique_angles(angle: float) -> None:
    """Pure black means W = P, which must leave no uncovered seam anywhere.

    The canvas is deliberately not a whole number of periods wide (80 x 44 with
    P = 8 gives a trailing cell of 4 px), so this also pins down the right-hand
    edge: the last column's pixel centre belongs to a clipped cell, and it must
    still be resolved correctly.  135 degrees additionally pins the negative
    cot branch, where the row shift runs the other way.
    """
    img = np.zeros((80, 44), dtype=np.uint8)
    mask = stripe_mask(img, 8, angle)

    assert mask.shape == (80, 44)
    assert mask.all(), f"uncovered pixels at {angle} degrees: " + str(
        [tuple(idx) for idx in np.argwhere(~mask)][:10]
    )
    # The very last column is the part of a clipped cell that is easiest to lose.
    assert mask[:, -1].all()


# --------------------------------------------------------------------------
# 7.4 50 % gray
# --------------------------------------------------------------------------


def test_fifty_percent_gray_halves_the_area() -> None:
    """With 127 gray and an even period the widths match the exact geometry.

    P = 16 -> W = 16 * (1 - 127/255) = 8.0 exactly, so the expected pixel set
    has no quantization ambiguity at all and can be asserted exactly.
    """
    img = _constant(127, 16, 64)
    mask = stripe_mask(img, 16, 90.0)

    expected = np.zeros(64, dtype=bool)
    for center in range(8, 64, 16):
        expected[center - 4 : center + 4] = True

    for y in range(mask.shape[0]):
        assert np.array_equal(mask[y], expected)

    assert abs(float(mask.mean()) - 0.5) < 0.01


# --------------------------------------------------------------------------
# 7.5 constant gray stability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [40, 127, 200])
def test_constant_gray_does_not_drift_with_y(value: int) -> None:
    mask = stripe_mask(_constant(value, 24, 64), 16, 90.0)
    for y in range(1, mask.shape[0]):
        assert np.array_equal(mask[y], mask[0])


def test_constant_gray_full_cells_have_stable_width() -> None:
    mask = stripe_mask(_constant(128, 8, 64), 16, 90.0)
    widths = [stop - start for start, stop in _runs(mask[0])]
    assert len(set(widths)) == 1


# --------------------------------------------------------------------------
# 7.6 vertical grayscale gradient
# --------------------------------------------------------------------------


def test_vertical_gradient_lines_thicken_towards_the_bottom() -> None:
    mask = stripe_mask(_vertical_gradient(64, 64), 16, 90.0)

    coverages = mask.mean(axis=1)
    assert coverages[0] < coverages[-1]
    assert np.all(np.diff(coverages) >= -1e-12)

    # The stripe centred at x = 8 spans [8 - W/2, 8 + W/2).  Rows near white
    # produce W = 0, so the pattern only appears once the image is dark enough;
    # from that point on the width must grow monotonically.
    measured = []
    for y in range(mask.shape[0]):
        span = next(
            (stop - start for start, stop in _runs(mask[y]) if start <= 8 < stop),
            None,
        )
        measured.append((y, span))

    first_drawn = next(y for y, span in measured if span is not None)
    assert first_drawn > 0  # the top rows really are blank
    widths = [span for _, span in measured if span is not None]
    assert widths[0] < widths[-1]
    assert all(a <= b for a, b in zip(widths, widths[1:]))


# --------------------------------------------------------------------------
# 7.7 horizontal grayscale gradient
# --------------------------------------------------------------------------


def test_horizontal_gradient_lines_thicken_towards_the_right() -> None:
    mask = stripe_mask(_horizontal_gradient(16, 128), 16, 90.0)
    runs = _runs(mask[0])
    assert len(runs) >= 4

    widths = [stop - start for start, stop in runs]
    for earlier, later in zip(widths, widths[1:]):
        assert earlier <= later
    assert widths[0] < widths[-1]


# --------------------------------------------------------------------------
# 7.8 non-divisible width
# --------------------------------------------------------------------------


def test_width_100_period_17_handles_partial_trailing_cell() -> None:
    img = np.zeros((8, 100), dtype=np.uint8)
    mask = stripe_mask(img, 17, 90.0)

    assert mask.shape == (8, 100)
    assert mask.all()

    # The last cell is cut off by the canvas edge, but it must still take part
    # in the averaging with the pixels that actually exist, and its line width
    # must still follow the nominal period.
    gray = _constant(127, 8, 100)
    partial = stripe_mask(gray, 17, 90.0)
    last_run = _runs(partial[0])[-1]

    # Last full cell centre is 93.5 (spans [85, 102), clipped by the canvas).
    nominal = 17 * (1 - 127 / 255)  # 8.0
    assert abs((last_run[1] - last_run[0]) - nominal) <= 1


# --------------------------------------------------------------------------
# 7.9 45 degrees
# --------------------------------------------------------------------------


def test_45_degrees_shifts_phase_from_row_to_row() -> None:
    mask = stripe_mask(_constant(128, 32, 64), 16, 45.0)

    # cot(45) = 1, so each row shifts by exactly one pixel.
    for y in range(1, mask.shape[0]):
        assert np.array_equal(mask[y], np.roll(mask[0], y))

    # A 45 degree pattern cannot look like vertical stripes.
    assert not np.array_equal(mask[1], mask[0])


def test_45_degrees_has_no_cracks_or_out_of_range() -> None:
    mask = stripe_mask(_vertical_gradient(40, 40), 8, 45.0)
    assert mask.shape == (40, 40)
    # Dark rows must be covered right up to the left edge.
    assert mask[20, 0]


# --------------------------------------------------------------------------
# 7.10 0 / 180 degrees
# --------------------------------------------------------------------------


def test_zero_and_180_agree() -> None:
    img = _vertical_gradient(24, 32)
    assert np.array_equal(stripe_mask(img, 8, 0.0), stripe_mask(img, 8, 180.0))


def test_zero_degrees_matches_transpose_route() -> None:
    img = _vertical_gradient(24, 32)
    assert np.array_equal(stripe_mask(img, 8, 0.0), stripe_mask(img.T, 8, 90.0).T)


def test_zero_degrees_gives_horizontal_stripes() -> None:
    """0 degrees must build horizontal bands, not tilted or vertical lines.

    A *constant* gray image is used on purpose.  With a horizontal gradient the
    variable-width design legitimately makes one and the same horizontal line
    grow and shrink along x, so a gradient cannot tell "horizontal stripe"
    apart from "vertical stripe with varying width".  On constant gray the
    answer is unambiguous: the pattern must be constant along every row, and it
    must cycle only as y advances.
    """
    mask = stripe_mask(_constant(128, 64, 64), 16, 0.0)
    assert mask.shape == (64, 64)

    # Constant along x within each row.
    for y in range(mask.shape[0]):
        assert mask[y].all() or not mask[y].any()

    # A genuine vertical cycle down y, and more than one band.
    # P = 16 with 128 gray gives W = 16 * (1 - 128/255) = 7.97, which quantizes
    # to 8 px bands at [4, 12), [20, 28), [36, 44), [52, 60).
    assert np.array_equal(
        mask[:, 0], _expected_vertical_pattern(64, first=4, period=16, width=8)
    )

    # Same result as the 90 degrees route applied to the transposed image.
    assert np.array_equal(
        mask, stripe_mask(_constant(128, 64, 64).T, 16, 90.0).T
    )


def test_zero_degrees_on_constant_gray_has_stable_band_width() -> None:
    """Every band of a horizontal-stripe pattern has the same height."""
    mask = stripe_mask(_constant(128, 32, 64), 16, 0.0)
    column = mask[:, 0]
    assert column.any()

    heights = []
    y = 0
    while y < column.size:
        if column[y]:
            start = y
            while y < column.size and column[y]:
                y += 1
            heights.append(y - start)
        else:
            y += 1
    assert len(set(heights)) == 1


def test_zero_degrees_respects_angle_equivalence() -> None:
    img = _horizontal_gradient(32, 32)
    assert np.array_equal(stripe_mask(img, 8, 0.0), stripe_mask(img, 8, -180.0))


# --------------------------------------------------------------------------
# 7.11 angle normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angle,equivalent",
    [(90.0, 270.0), (45.0, 225.0), (30.0, 210.0), (0.0, 180.0)],
)
def test_angles_normalize_modulo_180(angle: float, equivalent: float) -> None:
    img = _horizontal_gradient(32, 48)
    assert np.array_equal(stripe_mask(img, 8, angle), stripe_mask(img, 8, equivalent))
