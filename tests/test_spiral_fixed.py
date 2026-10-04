"""Fixed-width Spiral tests.

These pin the fixed-width spiral contract.  The *polar geometry* is the frozen
Spiral Round 1 geometry; the only change is that the line width on the polar
canvas is a constant instead of a grayscale-derived value.

Key properties:

* the very same frozen geometry values / chirality / sampling are reused;
* there is no forward grayscale polar unwrap (there is no grayscale at all);
* ``line_width == period`` makes the polar mask entirely ``True``, so after the
  inverse warp and the support clip the mask is exactly the circular support;
* arms ``1 / 2 / 3 / 12`` all run, the high-arm centre rosette is left alone;
* results are deterministic (no uninitialised outlier pixels).
"""

from __future__ import annotations

import numpy as np
import pytest

from halftone_playground.spiral import (
    MIN_SIDE,
    _angle_samples,
    _circular_support,
    _polar_center,
    _support_radius,
    _warp_radius,
    spiral_fixed_mask,
)

SIDE = 512
PERIOD = 16
ARMS_MATRIX = (1, 2, 3, 12)


def support(side: int) -> np.ndarray:
    """Reference circular support, computed independently of the module."""
    return _circular_support(side, _polar_center(side), _support_radius(side))


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_non_integer_side(self):
        with pytest.raises(TypeError, match="side must be an integer"):
            spiral_fixed_mask(64.0, PERIOD, 5, 1)

    def test_rejects_bool_side(self):
        with pytest.raises(TypeError, match="side must be an integer"):
            spiral_fixed_mask(True, PERIOD, 5, 1)

    @pytest.mark.parametrize("side", [0, 1, -4])
    def test_rejects_side_below_minimum(self, side):
        with pytest.raises(ValueError, match=f"side >= {MIN_SIDE}"):
            spiral_fixed_mask(side, PERIOD, 1, 1)

    def test_accepts_minimum_side(self):
        assert spiral_fixed_mask(MIN_SIDE, 2, 1, 1).shape == (MIN_SIDE, MIN_SIDE)

    @pytest.mark.parametrize("period", [0, -1, -16])
    def test_rejects_non_positive_period(self, period):
        with pytest.raises(ValueError, match="period must be a positive integer"):
            spiral_fixed_mask(64, period, 1, 1)

    @pytest.mark.parametrize("arms", [0, -1, -3])
    def test_rejects_non_positive_arms(self, arms):
        with pytest.raises(ValueError, match="arms must be a positive integer"):
            spiral_fixed_mask(64, PERIOD, 5, arms)

    @pytest.mark.parametrize("name,args", [("period", True), ("arms", False)])
    def test_rejects_bool_masquerading_as_int(self, name, args):
        period, arms = (args, 1) if name == "period" else (PERIOD, args)
        with pytest.raises(TypeError, match=f"{name} must be an integer"):
            spiral_fixed_mask(64, period, 5, arms)

    @pytest.mark.parametrize("name,args", [("period", 16.0), ("arms", 3.0)])
    def test_rejects_float(self, name, args):
        period, arms = (args, 1) if name == "period" else (PERIOD, args)
        with pytest.raises(TypeError, match=f"{name} must be an integer"):
            spiral_fixed_mask(64, period, 5, arms)


class TestLineWidthValidation:
    @pytest.mark.parametrize("line_width", [0, -1, -5])
    def test_rejects_non_positive_line_width(self, line_width):
        with pytest.raises(ValueError, match="line_width must be a positive integer"):
            spiral_fixed_mask(64, PERIOD, line_width, 1)

    @pytest.mark.parametrize("line_width", [17, 32, 100])
    def test_rejects_line_width_above_period(self, line_width):
        with pytest.raises(ValueError, match="must not exceed period"):
            spiral_fixed_mask(64, PERIOD, line_width, 1)

    @pytest.mark.parametrize("line_width", [5.0, True, "5"])
    def test_rejects_non_integer_line_width(self, line_width):
        with pytest.raises((TypeError, ValueError)):
            spiral_fixed_mask(64, PERIOD, line_width, 1)

    def test_does_not_clamp(self):
        with pytest.raises(ValueError):
            spiral_fixed_mask(64, PERIOD, PERIOD + 1, 1)
        # upper bound inclusive
        mask = spiral_fixed_mask(64, PERIOD, PERIOD, 1)
        assert mask.dtype == np.bool_


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------


class TestOutputContract:
    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_shape_dtype_and_binary(self, arms):
        mask = spiral_fixed_mask(64, PERIOD, 5, arms)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.bool_
        assert set(np.unique(mask).tolist()) <= {False, True}

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_corners_outside_the_disc(self, arms):
        mask = spiral_fixed_mask(SIDE, PERIOD, 5, arms)
        assert not mask[0, 0]
        assert not mask[0, -1]
        assert not mask[-1, 0]
        assert not mask[-1, -1]

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_nothing_outside_the_support(self, arms):
        mask = spiral_fixed_mask(SIDE, PERIOD, 5, arms)
        assert not (mask & ~support(SIDE)).any()

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_mid_width_actually_contains_lines(self, arms):
        mask = spiral_fixed_mask(SIDE, PERIOD, 5, arms)
        assert mask.any()

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_deterministic(self, arms):
        first = spiral_fixed_mask(128, PERIOD, 5, arms)
        second = spiral_fixed_mask(128, PERIOD, 5, arms)
        assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# hard property: line_width == period -> mask == circular support
# --------------------------------------------------------------------------


class TestLineWidthEqualsPeriod:
    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_mask_equals_circular_support(self, arms):
        mask = spiral_fixed_mask(SIDE, PERIOD, PERIOD, arms)
        assert np.array_equal(mask, support(SIDE))

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_strict_inside_outside_counts(self, arms):
        mask = spiral_fixed_mask(SIDE, PERIOD, PERIOD, arms)
        ref = support(SIDE)
        assert int((ref & ~mask).sum()) == 0, "inside_false must be 0"
        assert int((~ref & mask).sum()) == 0, "outside_true must be 0"

    @pytest.mark.parametrize("period", [4, 8, 16, 32, 64])
    def test_holds_for_many_periods(self, period):
        mask = spiral_fixed_mask(SIDE, period, period, 1)
        assert np.array_equal(mask, support(SIDE))

    @pytest.mark.parametrize("side", [2, 3, 16, 33, 64])
    def test_holds_for_many_sides(self, side):
        mask = spiral_fixed_mask(side, 4, 4, 3)
        assert np.array_equal(mask, support(side))


# --------------------------------------------------------------------------
# arms + partial widths
# --------------------------------------------------------------------------


class TestArms:
    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_partial_width_is_inside_support(self, arms):
        mask = spiral_fixed_mask(SIDE, PERIOD, 3, arms)
        assert not (mask & ~support(SIDE)).any()
        assert mask.any()

    @pytest.mark.parametrize("arms", [1, 2, 3, 5, 7, 12, 24, 100])
    def test_wide_arm_range_is_supported(self, arms):
        mask = spiral_fixed_mask(64, 8, 3, arms)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.bool_

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_ink_fraction_is_plausible(self, arms):
        # width/P = 5/16 of each polar period is ink -> roughly 5/16 of the disc.
        mask = spiral_fixed_mask(SIDE, PERIOD, 5, arms)
        disc = support(SIDE)
        inside = mask.sum() / disc.sum()
        assert 0.20 < inside < 0.45, (arms, inside)

    def test_high_arm_centre_structure_is_preserved(self):
        # The centre rosette is a construction feature, left as-is: a partial
        # width must not blow it away and must never mark outside the disc.
        mask = spiral_fixed_mask(SIDE, PERIOD, 3, 12)
        assert not (mask & ~support(SIDE)).any()
        assert mask[SIDE // 2 - 8 : SIDE // 2 + 8, SIDE // 2 - 8 : SIDE // 2 + 8].any()


# --------------------------------------------------------------------------
# scale / side independence and support behaviour of the geometry
# --------------------------------------------------------------------------


class TestGeometryReuse:
    def test_support_matches_the_variable_width_geometry(self):
        # The fixed-width path shares _polar_center / _support_radius.
        for side in (2, 3, 16, 64, 512):
            assert _support_radius(side) == _warp_radius(side) - 0.5
            assert _angle_samples(_warp_radius(side)) >= 1

    @pytest.mark.parametrize("side", [2, 3, 4, 5, 16, 17, 64])
    def test_small_squares_produce_correct_shape(self, side):
        mask = spiral_fixed_mask(side, 4, 2, 1)
        assert mask.shape == (side, side)
        assert mask.dtype == np.bool_
        assert not (mask & ~support(side)).any()
