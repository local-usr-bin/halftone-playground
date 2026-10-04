"""Spiral mode Round 1 tests.

These tests pin the *properties* that define the core -- input contract,
frozen geometry values, strict white/black behaviour, symmetry, seam
closure, determinism and the OpenCV dimension gate.  They deliberately do not
try to recognise spiral arms by computer vision, and they do not lean on large
golden images; the visual gate is a human review step.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from halftone_playground.spiral import (
    MIN_SIDE,
    OPENCV_MAX_DIMENSION,
    _angle_samples,
    _circular_support,
    _effective_stripe_angle,
    _polar_center,
    _radius_samples,
    _support_radius,
    _warp_radius,
    spiral_mask,
)

SIDE = 512
PERIOD = 16
ARMS_MATRIX = (1, 2, 3, 12)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def flat(value: int, side: int = SIDE) -> np.ndarray:
    """A flat square gray canvas."""
    return np.full((side, side), value, dtype=np.uint8)


def support(side: int) -> np.ndarray:
    """Reference circular support, computed independently of the module."""
    return _circular_support(side, _polar_center(side), _support_radius(side))


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            spiral_mask([[1, 2], [3, 4]], 16, 1)

    def test_rejects_rgb_3d(self):
        with pytest.raises(ValueError, match="2-D"):
            spiral_mask(np.zeros((8, 8, 3), dtype=np.uint8), 16, 1)

    def test_rejects_float_dtype(self):
        with pytest.raises(ValueError, match="uint8"):
            spiral_mask(np.zeros((8, 8), dtype=np.float32), 16, 1)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            spiral_mask(np.zeros((0, 0), dtype=np.uint8), 16, 1)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            spiral_mask(np.zeros((8, 16), dtype=np.uint8), 16, 1)

    def test_non_square_error_mentions_round_1_boundary(self):
        with pytest.raises(ValueError, match="not frozen yet"):
            spiral_mask(np.zeros((64, 32), dtype=np.uint8), 16, 1)

    @pytest.mark.parametrize("side", [0, 1])
    def test_rejects_side_below_minimum(self, side):
        if side == 0:
            with pytest.raises(ValueError):
                spiral_mask(np.zeros((0, 0), dtype=np.uint8), 16, 1)
        else:
            with pytest.raises(ValueError, match=f"side >= {MIN_SIDE}"):
                spiral_mask(np.zeros((1, 1), dtype=np.uint8), 16, 1)

    def test_accepts_minimum_side(self):
        assert spiral_mask(flat(128, MIN_SIDE), 2, 1).shape == (MIN_SIDE, MIN_SIDE)


# --------------------------------------------------------------------------
# parameter validation
# --------------------------------------------------------------------------


class TestParameterValidation:
    @pytest.mark.parametrize("period", [0, -1, -16])
    def test_rejects_non_positive_period(self, period):
        with pytest.raises(ValueError, match="period must be a positive integer"):
            spiral_mask(flat(128), period, 1)

    @pytest.mark.parametrize("arms", [0, -1, -3])
    def test_rejects_non_positive_arms(self, arms):
        with pytest.raises(ValueError, match="arms must be a positive integer"):
            spiral_mask(flat(128), 16, arms)

    @pytest.mark.parametrize("name,args", [("period", True), ("arms", False)])
    def test_rejects_bool_masquerading_as_int(self, name, args):
        period, arms = (args, 1) if name == "period" else (16, args)
        with pytest.raises(TypeError, match=f"{name} must be an integer"):
            spiral_mask(flat(128), period, arms)

    @pytest.mark.parametrize("name,args", [("period", 16.0), ("arms", 3.0)])
    def test_rejects_float(self, name, args):
        period, arms = (args, 1) if name == "period" else (16, args)
        with pytest.raises(TypeError, match=f"{name} must be an integer"):
            spiral_mask(flat(128), period, arms)

    def test_accepts_numpy_integers(self):
        mask = spiral_mask(flat(128), np.int64(16), np.int64(3))
        assert mask.dtype == np.bool_


# --------------------------------------------------------------------------
# geometry values
# --------------------------------------------------------------------------


class TestGeometryValues:
    def test_center_is_half_pixel(self):
        assert _polar_center(512) == (255.5, 255.5)
        assert _polar_center(2) == (0.5, 0.5)
        assert _polar_center(3) == (1.0, 1.0)

    def test_center_is_not_integer_division(self):
        # Regression guard for the (width / 2, height / 2) mistake.
        assert _polar_center(512) != (256.0, 256.0)

    def test_warp_and_support_radius_are_distinct(self):
        assert _warp_radius(512) == 256.0
        assert _support_radius(512) == 255.5
        assert _warp_radius(512) != _support_radius(512)
        for side in (2, 3, 16, 64, 512, 1024):
            assert _support_radius(side) == _warp_radius(side) - 0.5

    def test_radius_and_angle_samples(self):
        assert _radius_samples(256.0) == 256
        assert _angle_samples(256.0) == 1608

    def test_angle_samples_uses_half_up_not_bankers(self):
        # Choose a radius whose raw circumference lands exactly on ``n + 0.5``
        # so the two rounding rules genuinely disagree.  Picking a plain radius
        # instead would let half-up and banker's rounding return the same
        # value, which would leave the rule unverified.
        radius = 12.5 / (2.0 * math.pi)
        raw = 2.0 * math.pi * radius
        assert raw == 12.5, "the distinguishing input must land on x.5 exactly"

        # The frozen rule is floor(value + 0.5) -> 13.  Python's round() uses
        # banker's rounding and would pick the even neighbour -> 12.  Asserting
        # both keeps this a real discriminator rather than a coincidence.
        assert math.floor(raw + 0.5) == 13
        assert round(raw) == 12
        assert _angle_samples(radius) == 13

    @pytest.mark.parametrize("n", [2, 8, 12, 24])
    def test_angle_samples_never_falls_back_to_bankers_rounding(self, n):
        radius = (n + 0.5) / (2.0 * math.pi)
        assert 2.0 * math.pi * radius == n + 0.5
        assert round(2.0 * math.pi * radius) == n  # banker's picks the even one
        assert _angle_samples(radius) == n + 1     # frozen half-up picks up

    def test_radius_samples_is_ceil(self):
        assert _radius_samples(255.5) == 256
        assert _radius_samples(0.5) == 1

    def test_samples_are_never_scaled_up(self):
        # Guard against the 2x / 4x oversampling temptation.
        assert _radius_samples(256.0) == 256
        assert _angle_samples(256.0) == 1608

    @pytest.mark.parametrize("side", [2, 3, 16, 64, 512])
    def test_samples_are_positive_for_all_supported_sides(self, side):
        assert _radius_samples(_warp_radius(side)) >= 1
        assert _angle_samples(_warp_radius(side)) >= 1


class TestEffectiveStripeAngle:
    """The effective polar stripe angle, and with it the frozen chirality.

    The chirality of Spiral mode is *not* a separate flag: it is exactly the
    sign of the polar stripe slope.  The frozen value is a strictly positive
    slope, i.e. ``phase`` grows with the polar angle, which reads as a
    **clockwise spiral when followed outwards from the centre**
    (``positive slope`` / ``clockwise outward``).  These tests pin that sign
    so a future sign flip, a negated angle or an accidentally swapped
    ``atan2`` argument cannot slip through unnoticed.  No computer-vision arm
    tracing is used or needed: the sign of the slope is the whole contract.
    """

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_slope_reconstructs_arm_count(self, arms):
        angle_samples = _angle_samples(_warp_radius(SIDE))
        theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
        slope = 1.0 / math.tan(math.radians(theta))
        assert slope == pytest.approx(arms * PERIOD / angle_samples, rel=1e-12)

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_total_shift_is_whole_periods(self, arms):
        angle_samples = _angle_samples(_warp_radius(SIDE))
        theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
        slope = 1.0 / math.tan(math.radians(theta))
        assert angle_samples * slope == pytest.approx(arms * PERIOD, rel=1e-9)

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_phase_closes_at_the_seam(self, arms):
        angle_samples = _angle_samples(_warp_radius(SIDE))
        theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
        slope = 1.0 / math.tan(math.radians(theta))
        total_shift = angle_samples * slope
        closure = total_shift % PERIOD
        # Modulo of a float just under a multiple of P: accept either end.
        assert min(closure, PERIOD - closure) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_slope_is_strictly_positive_clockwise_outward(self, arms):
        """Chirality regression: the slope must never become negative.

        ``slope > 0`` is the frozen chirality (``positive slope`` ->
        ``clockwise outward``).  A negative slope would silently produce the
        opposite handedness, so the sign is asserted directly rather than only
        through the reconstructed magnitude.
        """
        angle_samples = _angle_samples(_warp_radius(SIDE))
        theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
        assert 0.0 < theta < 90.0
        # theta in (0, 90) implies cot(theta) > 0 in one step, and the slope
        # must also agree with the frozen formula rather than merely be signed.
        slope = 1.0 / math.tan(math.radians(theta))
        assert slope > 0.0
        assert slope == pytest.approx(arms * PERIOD / angle_samples, rel=1e-12)

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_effective_angle_is_never_of_the_opposite_handedness(self, arms):
        # atan2(angle_samples > 0, arms * period > 0) lives in the first
        # quadrant; the opposite handedness would need a negative numerator or
        # denominator and can therefore never be produced by the frozen call.
        angle_samples = _angle_samples(_warp_radius(SIDE))
        theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
        assert theta == pytest.approx(
            math.degrees(math.atan2(angle_samples, arms * PERIOD)), abs=1e-12
        )
        assert math.cos(math.radians(theta)) > 0.0
        assert math.sin(math.radians(theta)) > 0.0

    def test_reference_values_at_512(self):
        angle_samples = _angle_samples(_warp_radius(512))
        assert _effective_stripe_angle(16, 1, angle_samples) == pytest.approx(
            89.429912, abs=1e-6
        )
        assert _effective_stripe_angle(16, 12, angle_samples) == pytest.approx(
            83.190950, abs=1e-6
        )


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------


class TestOutputContract:
    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_shape_dtype_and_binary(self, arms):
        mask = spiral_mask(flat(128), PERIOD, arms)
        assert mask.shape == (SIDE, SIDE)
        assert mask.dtype == np.bool_
        assert set(np.unique(mask).tolist()) <= {False, True}

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_corners_are_outside_the_disc(self, arms):
        # A mid-gray spiral is *supposed* to contain line pixels, so this test
        # only pins what it actually promises: the four corners and everything
        # beyond the support disc stay empty, and the array contract holds.
        mask = spiral_mask(flat(128), PERIOD, arms)
        assert mask.shape == (SIDE, SIDE)
        assert mask.dtype == np.bool_
        assert not mask[0, 0]
        assert not mask[0, -1]
        assert not mask[-1, 0]
        assert not mask[-1, -1]
        assert not (mask & ~support(SIDE)).any()

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_mid_gray_actually_contains_lines(self, arms):
        # The counterpart of the corner check: mid-gray must produce ink.  The
        # pure-white no-ink property belongs to TestPureWhite and is asserted
        # there, never here.
        mask = spiral_mask(flat(128), PERIOD, arms)
        assert mask.any()

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_nothing_outside_the_support_is_marked(self, arms):
        mask = spiral_mask(flat(128), PERIOD, arms)
        assert not (mask & ~support(SIDE)).any()

    def test_deterministic_for_repeated_calls(self):
        gray = flat(100, 64)
        first = spiral_mask(gray, 8, 1)
        second = spiral_mask(gray, 8, 1)
        assert np.array_equal(first, second)

    def test_input_is_not_mutated(self):
        gray = flat(100, 64)
        before = gray.copy()
        spiral_mask(gray, 8, 1)
        assert np.array_equal(gray, before)

    def test_independent_from_renderer(self):
        from halftone_playground import render_black_on_white

        mask = spiral_mask(flat(128), PERIOD, 1)
        image = render_black_on_white(mask)
        assert image.dtype == np.uint8
        assert set(np.unique(image).tolist()) <= {0, 255}
        assert image[0, 0] == 255


# --------------------------------------------------------------------------
# hard properties: pure white / pure black
# --------------------------------------------------------------------------


class TestPureWhite:
    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_no_line_pixels_at_all(self, arms):
        mask = spiral_mask(flat(255), PERIOD, arms)
        assert not mask.any()

    @pytest.mark.parametrize("period", [8, 16, 32])
    @pytest.mark.parametrize("arms", (1, 3, 12))
    def test_no_line_pixels_for_any_reasonable_parameters(self, period, arms):
        assert not spiral_mask(flat(255), period, arms).any()


class TestPureBlack:
    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_mask_equals_circular_support(self, arms):
        mask = spiral_mask(flat(0), PERIOD, arms)
        assert np.array_equal(mask, support(SIDE))

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_strict_inside_outside_counts(self, arms):
        mask = spiral_mask(flat(0), PERIOD, arms)
        ref = support(SIDE)
        assert int((ref & ~mask).sum()) == 0
        assert int((~ref & mask).sum()) == 0

    @pytest.mark.parametrize("period", [4, 8, 16, 32, 64])
    def test_holds_for_many_periods(self, period):
        assert np.array_equal(spiral_mask(flat(0), period, 1), support(SIDE))


# --------------------------------------------------------------------------
# circular support and symmetry
# --------------------------------------------------------------------------


class TestCircularSupport:
    def test_support_is_strictly_inside_warp_disc(self):
        side = SIDE
        centre = _polar_center(side)
        coords = np.arange(side, dtype=np.float64)
        dy = coords - centre[1]
        dx = coords - centre[0]
        distance = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
        inside = support(side)
        assert float(distance[inside].max()) <= _support_radius(side)
        assert float(distance[~inside].min()) > _support_radius(side)

    def test_support_radius_stays_inside_warp_radius(self):
        # The frozen product contract, stated once and end-to-end.  How OpenCV
        # bins the final radial column is deliberately *not* re-derived here:
        # that boundary was settled by geometry probes 01b / 01c against real
        # OpenCV behaviour, and the unit suite only pins the contract and its
        # observable consequences.
        assert _warp_radius(SIDE) == 256.0
        assert _support_radius(SIDE) == 255.5
        assert _support_radius(SIDE) < _warp_radius(SIDE)

        # Observable consequence, rather than a restatement of the contract:
        # the disc is not empty, it is not the whole canvas, and every marked
        # pixel is inside it.
        disc = support(SIDE)
        assert disc.any()
        assert not disc.all()
        assert not (spiral_mask(flat(0), PERIOD, 1) & ~disc).any()

    def test_end_to_end_white_and_black_agree_with_the_support_disc(self):
        # Pure white -> nothing; pure black -> exactly the support disc.  These
        # two properties are what the probe-frozen radii actually buy, so they
        # are asserted here as the end-to-end counterpart of the ratio above.
        assert not spiral_mask(flat(255), PERIOD, 1).any()
        assert np.array_equal(spiral_mask(flat(0), PERIOD, 1), support(SIDE))

    def test_support_is_symmetric(self):
        ref = support(SIDE)
        assert np.array_equal(ref, ref[:, ::-1])
        assert np.array_equal(ref, ref[::-1, :])
        assert np.array_equal(ref, ref[::-1, ::-1])

    def test_support_area_matches_analytic_circle(self):
        area = int(support(SIDE).sum())
        analytic = math.pi * _support_radius(SIDE) ** 2
        assert area == pytest.approx(analytic, rel=2e-3)


class TestSymmetry:
    """Symmetry of the *circular support*, not of the spiral pattern.

    The frozen chirality is a strictly positive polar slope, i.e. a clockwise
    sweep away from the centre.  That makes a mid-gray spiral genuinely
    *asymmetric* under both mirrors -- mirroring flips the handedness of the
    arms -- so mirror/180-degree equality is deliberately **not** asserted for
    it.  A 180-degree rotation only shifts the phase by ``arms * P / 2``,
    which does not reproduce a constant-gray pattern either.

    What the construction does promise is that the disc itself is perfectly
    symmetric and centred, and that at the black extreme the mask *is* the
    disc.  Those are the contracts pinned here.
    """

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_support_disc_has_all_three_symmetries(self, arms):
        ref = support(SIDE)
        assert np.array_equal(ref, ref[:, ::-1])
        assert np.array_equal(ref, ref[::-1, :])
        assert np.array_equal(ref, ref[::-1, ::-1])

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_black_mask_inherits_every_symmetry_from_the_support(self, arms):
        # Pure black collapses the mask onto the support disc, so this is the
        # one mid/extreme case where symmetry is a real, expected property.
        mask = spiral_mask(flat(0), PERIOD, arms)
        assert np.array_equal(mask, mask[:, ::-1])
        assert np.array_equal(mask, mask[::-1, :])
        assert np.array_equal(mask, mask[::-1, ::-1])

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_white_mask_is_trivially_symmetric(self, arms):
        mask = spiral_mask(flat(255), PERIOD, arms)
        assert np.array_equal(mask, mask[:, ::-1])
        assert np.array_equal(mask, mask[::-1, :])
        assert np.array_equal(mask, mask[::-1, ::-1])

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_black_mask_has_no_center_shift(self, arms):
        mask = spiral_mask(flat(0), PERIOD, arms)
        # The centroid of a symmetric disc sits exactly on the image center;
        # any centre-shift bug would drag it off that point.
        ys, xs = np.nonzero(mask)
        assert float(xs.mean()) == pytest.approx((SIDE - 1) / 2.0, abs=0.05)
        assert float(ys.mean()) == pytest.approx((SIDE - 1) / 2.0, abs=0.05)

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_black_mask_covers_the_disc_symmetrically(self, arms):
        # A stronger centre check than the centroid: the marked/empty decision
        # is identical for every mirrored pixel pair inside the disc.
        mask = spiral_mask(flat(0), PERIOD, arms)
        assert np.array_equal(mask & support(SIDE), support(SIDE))

    def test_mid_gray_mirroring_flips_chirality_instead_of_matching(self):
        # Documents *why* the old symmetry contract was wrong: with a frozen
        # clockwise chirality a mid-gray spiral cannot equal its mirror image.
        mask = spiral_mask(flat(64), PERIOD, 1)
        assert not np.array_equal(mask, mask[:, ::-1])


# --------------------------------------------------------------------------
# seam properties
# --------------------------------------------------------------------------


class TestSeamProperties:
    """Seam behaviour of the polar stripe phase.

    ``phase`` lives on a circle of circumference ``period``, so the only
    meaningful distance between two phases is the *circular* distance.  A raw
    float comparison would fail purely because ``(x * slope) % P`` can land on
    ``P - 1e-13`` instead of ``0.0`` for a mathematically exact closure.
    """

    @staticmethod
    def _circular_distance(a: float, b: float, period: float) -> float:
        """Shortest distance between two phases on a period-sized circle."""
        delta = abs((a - b) % period)
        return min(delta, period - delta)

    def test_phase_closes_modulo_period(self):
        angle_samples = _angle_samples(_warp_radius(SIDE))
        for arms in ARMS_MATRIX:
            theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
            slope = 1.0 / math.tan(math.radians(theta))
            phase_start = (0 * slope) % PERIOD
            phase_wrap = (angle_samples * slope) % PERIOD
            distance = self._circular_distance(phase_start, phase_wrap, PERIOD)
            assert distance <= 1e-6, (
                f"arms={arms} does not close at the seam: "
                f"phase_start={phase_start!r} phase_wrap={phase_wrap!r}"
            )

    def test_tolerance_would_still_catch_a_real_phase_error(self):
        # Guard the guard: the tolerance must stay tight enough that a genuine
        # half-period or full-period drift is still reported as a failure.
        assert self._circular_distance(0.0, 0.5 * PERIOD, PERIOD) > 1e-6
        assert self._circular_distance(0.0, PERIOD, PERIOD) <= 1e-9
        assert self._circular_distance(0.0, 1.0, PERIOD) > 1e-6

    def test_wrap_phase_is_a_whole_number_of_periods_away_from_start(self):
        # The closure in its non-modular form: the total shift is an exact
        # multiple of the period, so the seam introduces no partial stripe.
        angle_samples = _angle_samples(_warp_radius(SIDE))
        for arms in ARMS_MATRIX:
            theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
            slope = 1.0 / math.tan(math.radians(theta))
            total_shift = angle_samples * slope
            periods = total_shift / PERIOD
            assert periods == pytest.approx(round(periods), abs=1e-6)
            assert round(periods) == arms

    def test_last_row_to_conceptual_next_row_step_matches_row_step(self):
        angle_samples = _angle_samples(_warp_radius(SIDE))
        for arms in ARMS_MATRIX:
            theta = _effective_stripe_angle(PERIOD, arms, angle_samples)
            slope = 1.0 / math.tan(math.radians(theta))
            ordinary = slope % PERIOD
            wrap = (angle_samples * slope - (angle_samples - 1) * slope) % PERIOD
            # The last->conceptual-next step is measured on the circle too.
            distance = self._circular_distance(ordinary, wrap, PERIOD)
            assert distance <= 1e-6, (
                f"arms={arms} seam step {wrap!r} differs from row step {ordinary!r}"
            )

    def test_last_row_is_not_required_to_equal_row_zero(self):
        # The rows are adjacent samples, so last != first; only their phase
        # difference has to match the ordinary step.
        angle_samples = _angle_samples(_warp_radius(SIDE))
        theta = _effective_stripe_angle(PERIOD, 1, angle_samples)
        slope = 1.0 / math.tan(math.radians(theta))
        assert ((angle_samples - 1) * slope) % PERIOD != pytest.approx(
            (angle_samples * slope) % PERIOD, abs=1e-9
        )

    def test_reference_angle_samples_is_the_rim_circumference(self):
        radius = _warp_radius(SIDE)
        assert _angle_samples(radius) == int(math.floor(2.0 * math.pi * radius + 0.5))
        assert _angle_samples(radius) == 1608


# --------------------------------------------------------------------------
# arm counts
# --------------------------------------------------------------------------


class TestArmCounts:
    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_runs_and_produces_ink(self, arms):
        mask = spiral_mask(flat(128), PERIOD, arms)
        assert mask.any(), f"arms={arms} produced an empty mask on 50% gray"

    @pytest.mark.parametrize("arms", ARMS_MATRIX)
    def test_ink_fraction_is_plausible_for_50_percent_gray(self, arms):
        # P*(1-G) with G = 128/255 gives a line covering ~half of each period,
        # so the black fraction of the disc should sit near 0.5.
        mask = spiral_mask(flat(128), PERIOD, arms)
        disc = support(SIDE)
        inside = mask.sum() / disc.sum()
        assert 0.30 < inside < 0.70

    @pytest.mark.parametrize("arms", (1, 2, 3, 12))
    def test_higher_arms_stay_well_defined(self, arms):
        mask = spiral_mask(flat(0), PERIOD, arms)
        assert mask.shape == (SIDE, SIDE)
        assert mask.dtype == np.bool_
        assert not (mask & ~support(SIDE)).any()

    def test_arms_twelve_does_not_crash_and_keeps_center_complex(self):
        # The complex rosette at the centre of a high-arm spiral is a design
        # feature: the test asserts it exists, it does not try to simplify it.
        mask = spiral_mask(flat(0), PERIOD, 12)
        assert np.array_equal(mask, support(SIDE))
        center_region = mask[
            SIDE // 2 - 8 : SIDE // 2 + 8, SIDE // 2 - 8 : SIDE // 2 + 8
        ]
        assert center_region.all()

    @pytest.mark.parametrize("arms", [1, 2, 3, 5, 7, 12, 24, 100])
    def test_wide_arm_range_is_supported(self, arms):
        mask = spiral_mask(flat(128, 64), 8, arms)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.bool_


# --------------------------------------------------------------------------
# gray response
# --------------------------------------------------------------------------


class TestGrayResponse:
    def test_line_width_grows_as_the_image_darkens(self):
        ink = [float(spiral_mask(flat(v), PERIOD, 1).mean()) for v in (230, 180, 128, 72, 24)]
        assert ink == sorted(ink)
        assert ink[0] < ink[-1]

    def test_image_scale_is_not_a_spiral_parameter(self):
        # resize_grayscale is the documented way to scale; the core takes the
        # scaled array directly and never rescales the period.
        from halftone_playground import resize_grayscale

        gray = flat(128, 128)
        scaled = resize_grayscale(gray, 2.0)
        assert scaled.shape == (256, 256)
        mask = spiral_mask(scaled, PERIOD, 1)
        assert mask.shape == (256, 256)


# --------------------------------------------------------------------------
# OpenCV hard dimension limit
# --------------------------------------------------------------------------


class TestOpenCVDimensionLimit:
    def test_limit_constant_documents_the_library_value(self):
        assert OPENCV_MAX_DIMENSION == 32767

    def test_rejects_too_wide_source(self):
        # Only the side has to be large; the array is a cheap broadcast view.
        huge = np.broadcast_to(np.uint8(128), (OPENCV_MAX_DIMENSION, OPENCV_MAX_DIMENSION))
        with pytest.raises(ValueError, match="OpenCV warpPolar/remap dimension limit"):
            spiral_mask(huge, PERIOD, 1)

    def test_error_message_mentions_the_library(self):
        huge = np.broadcast_to(np.uint8(128), (OPENCV_MAX_DIMENSION, OPENCV_MAX_DIMENSION))
        with pytest.raises(ValueError, match="OpenCV implementation limit"):
            spiral_mask(huge, PERIOD, 1)

    def test_just_below_the_limit_reaches_opencv(self):
        # 32766 is legal for the *source* check, but the polar image for such a
        # source is far past the limit too, so the geometric gate still fires.
        side = OPENCV_MAX_DIMENSION - 1
        huge = np.broadcast_to(np.uint8(128), (side, side))
        with pytest.raises(ValueError, match="OpenCV warpPolar/remap dimension limit"):
            spiral_mask(huge, PERIOD, 1)


# --------------------------------------------------------------------------
# small images
# --------------------------------------------------------------------------


class TestSmallImages:
    @pytest.mark.parametrize("side", [2, 3, 4, 5, 16, 17])
    def test_small_squares_produce_correct_shape(self, side):
        mask = spiral_mask(flat(128, side), 4, 1)
        assert mask.shape == (side, side)
        assert mask.dtype == np.bool_

    @pytest.mark.parametrize("side", [2, 3, 16])
    def test_small_squares_do_not_divide_by_zero(self, side):
        # implicit: the call would raise ZeroDivisionError otherwise
        assert spiral_mask(flat(0, side), 2, 1).shape == (side, side)

    @pytest.mark.parametrize("side", [2, 3, 16])
    def test_small_squares_have_positive_samples(self, side):
        assert _radius_samples(_warp_radius(side)) >= 1
        assert _angle_samples(_warp_radius(side)) >= 1

    @pytest.mark.parametrize("side", [2, 3, 16])
    def test_small_squares_respect_support(self, side):
        mask = spiral_mask(flat(0, side), 2, 1)
        assert np.array_equal(mask, support(side))
