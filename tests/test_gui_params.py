"""Tests for the pure GUI parameter layer (no display server required).

These cover the frozen GUI rules that can be decided without any Tk widget:
visibility mapping, hidden-value preservation, validation reusing the core
contracts, the Spiral automatic centered maximum-square semantics (effective
source size / crop box / crop-before-scale output projection) and the
output-resolution projection that must call the existing ``scaled_size``.
"""

from __future__ import annotations

import pytest

from halftone_playground.gui import params as gp
from halftone_playground.preprocess import scaled_size


# --------------------------------------------------------------------------
# visibility mapping
# --------------------------------------------------------------------------


class TestVisibleParameters:
    def test_stripe_variable_shows_angle_not_arms_not_line_width(self):
        visible = gp.visible_parameters(gp.MODE_STRIPE, gp.WIDTH_VARIABLE)
        assert gp.PARAM_PERIOD in visible
        assert gp.PARAM_SCALE in visible
        assert gp.PARAM_ANGLE in visible
        assert gp.PARAM_ARMS not in visible
        assert gp.PARAM_LINE_WIDTH not in visible

    def test_stripe_fixed_shows_angle_and_line_width_not_arms(self):
        visible = gp.visible_parameters(gp.MODE_STRIPE, gp.WIDTH_FIXED)
        assert gp.PARAM_ANGLE in visible
        assert gp.PARAM_LINE_WIDTH in visible
        assert gp.PARAM_ARMS not in visible

    def test_spiral_variable_shows_arms_not_angle_not_line_width(self):
        visible = gp.visible_parameters(gp.MODE_SPIRAL, gp.WIDTH_VARIABLE)
        assert gp.PARAM_ARMS in visible
        assert gp.PARAM_ANGLE not in visible
        assert gp.PARAM_LINE_WIDTH not in visible

    def test_spiral_fixed_shows_arms_and_line_width_not_angle(self):
        visible = gp.visible_parameters(gp.MODE_SPIRAL, gp.WIDTH_FIXED)
        assert gp.PARAM_ARMS in visible
        assert gp.PARAM_LINE_WIDTH in visible
        assert gp.PARAM_ANGLE not in visible

    def test_period_and_scale_always_visible(self):
        for mode in gp.MODES:
            for width in gp.WIDTH_MODES:
                visible = gp.visible_parameters(mode, width)
                assert gp.PARAM_PERIOD in visible
                assert gp.PARAM_SCALE in visible

    def test_angle_and_arms_are_mutually_exclusive(self):
        for width in gp.WIDTH_MODES:
            stripe = gp.visible_parameters(gp.MODE_STRIPE, width)
            spiral = gp.visible_parameters(gp.MODE_SPIRAL, width)
            assert (gp.PARAM_ANGLE in stripe) != (gp.PARAM_ANGLE in spiral)
            assert (gp.PARAM_ARMS in spiral) != (gp.PARAM_ARMS in stripe)

    def test_line_width_visibility_follows_width_mode(self):
        fixed = gp.visible_parameters(gp.MODE_STRIPE, gp.WIDTH_FIXED)
        variable = gp.visible_parameters(gp.MODE_STRIPE, gp.WIDTH_VARIABLE)
        assert (gp.PARAM_LINE_WIDTH in fixed) is True
        assert (gp.PARAM_LINE_WIDTH in variable) is False

    def test_render_choices_always_all_three(self):
        assert gp.render_choices_for() == gp.RENDER_CHOICES
        assert set(gp.RENDER_CHOICES) == {
            gp.RENDER_BLACK_ON_WHITE,
            gp.RENDER_WHITE_ON_BLACK,
            gp.RENDER_SOURCE_COLOR,
        }


# --------------------------------------------------------------------------
# hidden values are preserved
# --------------------------------------------------------------------------


class TestHiddenValuePreserved:
    def test_changing_mode_does_not_touch_values(self):
        p = gp.RenderParams()
        p.angle_text = "45"
        p.angle = 45.0

        p.mode = gp.MODE_SPIRAL  # Angle hidden
        assert p.angle_text == "45"
        assert p.angle == 45.0

        p.mode = gp.MODE_STRIPE  # Angle visible again
        assert p.angle_text == "45"
        assert p.angle == 45.0

    def test_changing_width_mode_does_not_touch_line_width(self):
        p = gp.RenderParams()
        p.line_width_text = "7"
        p.line_width = 7

        p.width_mode = gp.WIDTH_VARIABLE  # Line width hidden
        assert p.line_width_text == "7"
        assert p.line_width == 7

        p.width_mode = gp.WIDTH_FIXED  # visible again
        assert p.line_width_text == "7"
        assert p.line_width == 7

    def test_hidden_field_errors_do_not_block_visible_config(self):
        # A line width that is invalid for the current period must not matter
        # while the variable-width path hides it.
        p = gp.RenderParams(period_text="16", line_width_text="99")
        p.width_mode = gp.WIDTH_VARIABLE
        assert gp.validate(p) == {}
        # But it is reported once the fixed-width path shows it.
        p.width_mode = gp.WIDTH_FIXED
        assert gp.PARAM_LINE_WIDTH in gp.validate(p)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize("text,expected", [("3", 3), (" 3 ", 3), ("+3", 3), ("-2", -2)])
    def test_parse_int_accepts(self, text, expected):
        assert gp.parse_int(text) == expected

    @pytest.mark.parametrize("text", ["", "  ", "3.5", "abc", "1e3", "nan", None])
    def test_parse_int_rejects(self, text):
        assert gp.parse_int(text) is None

    @pytest.mark.parametrize("text,expected", [("1.3", 1.3), (" 2 ", 2.0), ("1e2", 100.0)])
    def test_parse_float_accepts(self, text, expected):
        assert gp.parse_float(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "abc", "nan", "inf", "-inf", None])
    def test_parse_float_rejects(self, text):
        assert gp.parse_float(text) is None


# --------------------------------------------------------------------------
# per-field validation (mirrors the core contracts)
# --------------------------------------------------------------------------


class TestFieldValidation:
    def test_default_params_are_valid(self):
        assert gp.validate(gp.RenderParams()) == {}

    @pytest.mark.parametrize("text", ["", "abc", "0", "-3", "2.5"])
    def test_period_invalid(self, text):
        p = gp.RenderParams(period_text=text)
        assert gp.field_error(p, gp.PARAM_PERIOD) is not None

    def test_period_valid(self):
        p = gp.RenderParams(period_text="16")
        assert gp.field_error(p, gp.PARAM_PERIOD) is None

    @pytest.mark.parametrize("text", ["", "abc", "0", "-1", "nan", "inf"])
    def test_scale_invalid(self, text):
        p = gp.RenderParams(scale_text=text)
        assert gp.field_error(p, gp.PARAM_SCALE) is not None

    @pytest.mark.parametrize("text", ["0.1", "1", "1.3", "100.0"])
    def test_scale_valid(self, text):
        p = gp.RenderParams(scale_text=text)
        assert gp.field_error(p, gp.PARAM_SCALE) is None

    @pytest.mark.parametrize("text", ["", "abc", "nan", "inf"])
    def test_angle_invalid(self, text):
        p = gp.RenderParams(angle_text=text)
        assert gp.field_error(p, gp.PARAM_ANGLE) is not None

    @pytest.mark.parametrize("text", ["0", "45", "90", "-30", "359.9"])
    def test_angle_valid(self, text):
        p = gp.RenderParams(angle_text=text)
        assert gp.field_error(p, gp.PARAM_ANGLE) is None

    @pytest.mark.parametrize("text", ["", "abc", "0", "-1", "1.5"])
    def test_arms_invalid(self, text):
        p = gp.RenderParams(arms_text=text)
        assert gp.field_error(p, gp.PARAM_ARMS) is not None

    def test_arms_valid(self):
        p = gp.RenderParams(arms_text="3")
        assert gp.field_error(p, gp.PARAM_ARMS) is None

    @pytest.mark.parametrize("period,lw", [("16", "1"), ("16", "16"), ("16", "8")])
    def test_line_width_valid_within_range(self, period, lw):
        p = gp.RenderParams(period_text=period, line_width_text=lw)
        assert gp.field_error(p, gp.PARAM_LINE_WIDTH) is None

    @pytest.mark.parametrize("period,lw", [("16", "17"), ("16", "0"), ("16", "-1"), ("16", "abc")])
    def test_line_width_invalid(self, period, lw):
        p = gp.RenderParams(period_text=period, line_width_text=lw)
        assert gp.field_error(p, gp.PARAM_LINE_WIDTH) is not None

    def test_line_width_above_period_message_mentions_period(self):
        p = gp.RenderParams(period_text="16", line_width_text="20")
        message = gp.field_error(p, gp.PARAM_LINE_WIDTH)
        assert message is not None
        assert "period" in message.lower()

    def test_unknown_field_raises(self):
        with pytest.raises(KeyError):
            gp.field_error(gp.RenderParams(), "nope")


# --------------------------------------------------------------------------
# output resolution must reuse scaled_size
# --------------------------------------------------------------------------


class TestOutputResolution:
    def test_none_source_returns_none(self):
        assert gp.output_resolution(gp.MODE_STRIPE, None, "1.0") is None
        assert gp.output_resolution(gp.MODE_SPIRAL, None, "1.0") is None

    def test_matches_scaled_size_exactly(self):
        for scale in ("1.0", "1.3", "2.0", "0.5", "100.0"):
            expected = scaled_size(512, 512, float(scale))
            assert gp.output_resolution(gp.MODE_STRIPE, (512, 512), scale) == expected

    def test_non_square_source_matches_scaled_size_for_stripe(self):
        expected = scaled_size(640, 480, 1.25)
        assert gp.output_resolution(gp.MODE_STRIPE, (640, 480), "1.25") == expected

    @pytest.mark.parametrize("text", ["abc", "0", "-1", "", "nan", "inf"])
    def test_invalid_scale_returns_none(self, text):
        assert gp.output_resolution(gp.MODE_STRIPE, (512, 512), text) is None
        assert gp.output_resolution(gp.MODE_SPIRAL, (512, 512), text) is None

    def test_format_resolution_none_is_em_dash(self):
        assert gp.format_resolution(None) == "\u2014"

    def test_format_resolution_uses_dimension_sign(self):
        assert gp.format_resolution((666, 666)) == "666 \u00d7 666"


# --------------------------------------------------------------------------
# Spiral effective source: automatic centered maximum square
# --------------------------------------------------------------------------


class TestCenterSquare:
    def test_side_is_min_dimension(self):
        assert gp.center_square_side((1492, 2558)) == 1492
        assert gp.center_square_side((2558, 1492)) == 1492
        assert gp.center_square_side((1492, 1492)) == 1492

    def test_portrait_crop_box_is_centered(self):
        # 1492 x 2558 portrait -> square 1492, vertical margins 0 / 1066.
        assert gp.center_square_crop_box((1492, 2558)) == (0, 533, 1492, 2025)

    def test_landscape_crop_box_is_centered(self):
        assert gp.center_square_crop_box((2558, 1492)) == (533, 0, 2025, 1492)

    def test_square_crop_box_is_whole_image(self):
        assert gp.center_square_crop_box((1492, 1492)) == (0, 0, 1492, 1492)

    def test_odd_difference_floors_and_extra_pixel_stays_rightbottom(self):
        # width 101, height 100 -> side 100; left = (101-100)//2 = 0 so the
        # extra pixel stays on the RIGHT (right == 100, not 101).
        assert gp.center_square_crop_box((101, 100)) == (0, 0, 100, 100)
        # height 101, width 100 -> top = (101-100)//2 = 0, extra pixel on the
        # BOTTOM.
        assert gp.center_square_crop_box((100, 101)) == (0, 0, 100, 100)
        # Larger even difference: 1502 x 1492 -> margin 10, floor(10/2)=5.
        assert gp.center_square_crop_box((1502, 1492)) == (5, 0, 1497, 1492)
        # Odd total difference: 1503 x 1492 -> diff 11, left = 11//2 = 5,
        # right = 5 + 1492 = 1497 (the 11th pixel stays on the right).
        assert gp.center_square_crop_box((1503, 1492)) == (5, 0, 1497, 1492)

    def test_crop_box_is_always_square_and_inside_source(self):
        for size in [(1492, 2558), (2558, 1492), (101, 100), (100, 101), (7, 7)]:
            left, top, right, bottom = gp.center_square_crop_box(size)
            assert right - left == bottom - top
            assert 0 <= left < right <= size[0]
            assert 0 <= top < bottom <= size[1]


class TestEffectiveSourceSize:
    def test_none_source_returns_none(self):
        assert gp.effective_source_size(gp.MODE_STRIPE, None) is None
        assert gp.effective_source_size(gp.MODE_SPIRAL, None) is None

    def test_stripe_keeps_original_aspect(self):
        assert gp.effective_source_size(gp.MODE_STRIPE, (1492, 2558)) == (1492, 2558)
        assert gp.effective_source_size(gp.MODE_STRIPE, (2558, 1492)) == (2558, 1492)

    @pytest.mark.parametrize(
        "size",
        [(1492, 2558), (2558, 1492), (1492, 1492), (101, 100)],
    )
    def test_spiral_is_always_square_min_dimension(self, size):
        side = min(size)
        assert gp.effective_source_size(gp.MODE_SPIRAL, size) == (side, side)


# --------------------------------------------------------------------------
# Spiral output resolution (crop-before-scale, reuses scaled_size)
# --------------------------------------------------------------------------


class TestSpiralOutputResolution:
    def test_portrait_spiral_is_square(self):
        # A) 1492 x 2558 source -> Spiral effective 1492 x 1492.
        assert gp.output_resolution(gp.MODE_SPIRAL, (1492, 2558), "1") == (1492, 1492)

    def test_landscape_spiral_is_square(self):
        # B) 2558 x 1492 source -> Spiral effective 1492 x 1492.
        assert gp.output_resolution(gp.MODE_SPIRAL, (2558, 1492), "1") == (1492, 1492)

    def test_square_spiral_stays_square(self):
        # C) already square stays 1492 x 1492.
        assert gp.output_resolution(gp.MODE_SPIRAL, (1492, 1492), "1") == (1492, 1492)

    def test_spiral_scale_three(self):
        # D) 1492 square at scale 3 -> 4476 x 4476.
        assert gp.output_resolution(gp.MODE_SPIRAL, (1492, 2558), "3") == (4476, 4476)

    @pytest.mark.parametrize("scale", ["1.3", "0.5", "1.25", "2.7"])
    def test_spiral_non_integer_scale_uses_scaled_size(self, scale):
        # E) non-integer scale must reuse the existing scaled_size rounding;
        # never re-implement the half-up rule.
        side = gp.center_square_side((1492, 2558))
        expected = scaled_size(side, side, float(scale))
        assert gp.output_resolution(gp.MODE_SPIRAL, (1492, 2558), scale) == expected

    def test_spiral_effective_equals_crop_then_scale(self):
        # The effective size fed to scaled_size is the centered square, i.e.
        # the crop happens *before* the scale, not after.
        side = gp.center_square_side((3000, 1000))
        assert gp.output_resolution(gp.MODE_SPIRAL, (3000, 1000), "2") == scaled_size(
            side, side, 2.0
        )


# --------------------------------------------------------------------------
# mode switching: Stripe keeps aspect, Spiral is square, and back again
# --------------------------------------------------------------------------


class TestModeSwitchOutput:
    def test_switch_stripe_spiral_stripe(self):
        # G) same rectangular source.
        size = (1492, 2558)
        assert gp.output_resolution(gp.MODE_STRIPE, size, "1") == (1492, 2558)
        assert gp.output_resolution(gp.MODE_SPIRAL, size, "1") == (1492, 1492)
        # ...and back to Stripe restores the original aspect ratio.
        assert gp.output_resolution(gp.MODE_STRIPE, size, "1") == (1492, 2558)


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------


class TestDefaults:
    def test_defaults_match_cli_contract(self):
        from halftone_playground import cli

        assert gp.DEFAULT_PERIOD == cli.DEFAULT_PERIOD
        assert gp.DEFAULT_SCALE == cli.DEFAULT_SCALE
        assert gp.DEFAULT_ANGLE == cli.DEFAULT_ANGLE
        assert gp.DEFAULT_ARMS == cli.DEFAULT_ARMS

    def test_default_line_width_is_one(self):
        # GUI-only value, chosen as the smallest always-legal width.
        assert gp.DEFAULT_LINE_WIDTH == 1

    def test_render_params_copy_is_independent(self):
        p = gp.RenderParams()
        clone = p.copy()
        clone.angle_text = "123"
        assert p.angle_text == "90"
        assert clone.angle_text == "123"
