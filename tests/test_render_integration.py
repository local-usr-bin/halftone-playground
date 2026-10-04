"""Integration tests: one geometry, three renderers, byte-identical layouts.

This is the round's central claim.  A Stripe mask and a Spiral mask are each
computed exactly once and then fed to all three output variants:

* ``render_black_on_white``,
* ``render_white_on_black``,
* ``render_source_color_on_white``.

Whatever differs between the outputs must be *pixel values only*.  Where the
mask is ``False`` the source-colour image must be pure white; where the mask is
``True`` it must be a byte-exact copy of the Cartesian source.  For Spiral this
also proves that the source RGB never went through the polar pipeline: the
copied pixels are the ones at the final Cartesian coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from halftone_playground import (
    render_black_on_white,
    render_source_color_on_white,
    render_white_on_black,
    resize_rgb,
    spiral_mask,
    stripe_mask,
)

_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def _ramp_gray(height: int, width: int) -> np.ndarray:
    """A deterministic horizontal grayscale ramp (no randomness anywhere)."""
    return np.tile(
        np.linspace(0, 255, width, dtype=np.uint8)[None, :], (height, 1)
    )


def _color_field(height: int, width: int) -> np.ndarray:
    """A deterministic RGB field in which every region reads differently.

    ``R`` grows with ``x``, ``G`` grows with ``y`` and ``B`` is a third fixed
    function of both -- three orthogonal-looking ramps so a misplaced copy is
    immediately visible in a single channel.
    """
    ys = np.arange(height, dtype=np.uint16)[:, None]
    xs = np.arange(width, dtype=np.uint16)[None, :]
    b8 = np.uint16(255)
    r = ((xs * b8) // max(width - 1, 1)).astype(np.uint8)
    g = ((ys * b8) // max(height - 1, 1)).astype(np.uint8)
    b = ((xs * np.uint16(7) + ys * np.uint16(11)) % np.uint16(256)).astype(np.uint8)
    return np.stack(
        [np.broadcast_to(r, (height, width)),
         np.broadcast_to(g, (height, width)),
         b],
        axis=-1,
    )


# --------------------------------------------------------------------------
# Stripe integration
# --------------------------------------------------------------------------


def test_stripe_three_variants_share_one_mask() -> None:
    gray = _ramp_gray(128, 128)
    rgb = _color_field(128, 128)
    mask = stripe_mask(gray, 16, 90.0)

    bw = render_black_on_white(mask)
    wb = render_white_on_black(mask)
    color = render_source_color_on_white(mask, rgb)

    # Binary complementarity over the full Stripe canvas.
    assert (
        bw.astype(np.uint16) + wb.astype(np.uint16) == 255
    ).all(), "black_on_white and white_on_black are not complements"

    # All three pay attention to exactly the same mask.
    assert (bw[mask] == 0).all() and (bw[~mask] == 255).all()
    assert (wb[mask] == 255).all() and (wb[~mask] == 0).all()
    assert (color[~mask] == _WHITE).all(), "background is not pure white"
    assert np.array_equal(color[mask], rgb[mask]), "line colour is not exact"

    # And the geometry itself is untouched: every line pixel in the colour
    # output coincides with a black pixel in the black-on-white output.
    assert ((color != _WHITE).any(axis=-1) == mask).all()


def test_stripe_color_line_pixels_are_exact_source_bytes() -> None:
    gray = _ramp_gray(64, 96)
    rgb = _color_field(64, 96)
    mask = stripe_mask(gray, 12, 90.0)
    out = render_source_color_on_white(mask, rgb)

    ys, xs = np.nonzero(mask)
    assert ys.size > 50, "fixture produced too few line pixels to be meaningful"
    for y, x in zip(ys.tolist(), xs.tolist()):
        assert out[y, x].tolist() == rgb[y, x].tolist()


def test_stripe_scaled_rgb_aligns_with_mask() -> None:
    """scale=1.3 on both grayscale and RGB keeps the compositor happy."""
    gray = _ramp_gray(73, 101)
    rgb = _color_field(73, 101)

    from halftone_playground import resize_grayscale

    gray_scaled = resize_grayscale(gray, 1.3)
    rgb_scaled = resize_rgb(rgb, 1.3)
    mask = stripe_mask(gray_scaled, 16, 90.0)

    assert rgb_scaled.shape[:2] == mask.shape
    out = render_source_color_on_white(mask, rgb_scaled)
    assert np.array_equal(out[mask], rgb_scaled[mask])
    assert (out[~mask] == _WHITE).all()


# --------------------------------------------------------------------------
# Spiral integration
# --------------------------------------------------------------------------


def test_spiral_three_variants_share_one_mask() -> None:
    gray = _ramp_gray(256, 256)
    rgb = _color_field(256, 256)
    mask = spiral_mask(gray, 16, arms=3)

    bw = render_black_on_white(mask)
    wb = render_white_on_black(mask)
    color = render_source_color_on_white(mask, rgb)

    assert (bw.astype(np.uint16) + wb.astype(np.uint16) == 255).all()
    assert (color[~mask] == _WHITE).all()
    assert np.array_equal(color[mask], rgb[mask])


def test_spiral_color_uses_final_cartesian_coordinates() -> None:
    """Colour is copied at the final Cartesian coordinates, not polar ones.

    If the RGB source had been pushed through the polar transform there would
    be interpolation residues; the copied pixels would then not match exactly.
    The test samples well over 100 line pixels and demands byte equality.
    """
    gray = _ramp_gray(256, 256)
    rgb = _color_field(256, 256)
    mask = spiral_mask(gray, 16, arms=3)
    out = render_source_color_on_white(mask, rgb)

    ys, xs = np.nonzero(mask)
    assert ys.size > 100, "need at least 100 line pixels for a meaningful check"

    # Deterministic spread across the whole disc (not a random draw).
    sample = np.linspace(0, ys.size - 1, num=min(ys.size, 512)).astype(int)
    for index in sample.tolist():
        y, x = int(ys[index]), int(xs[index])
        assert out[y, x].tolist() == rgb[y, x].tolist(), (y, x)

    # Whole-mask equality is the strongest form of the same statement.
    assert np.array_equal(out[mask], rgb[mask])


def test_spiral_outside_circle_is_pure_white_in_color_output() -> None:
    side = 256
    rgb = _color_field(side, side)
    mask = spiral_mask(_ramp_gray(side, side), 16, arms=3)
    out = render_source_color_on_white(mask, rgb)

    corners = [(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)]
    for y, x in corners:
        assert mask[y, x] is np.False_ or not mask[y, x]
        assert out[y, x].tolist() == [255, 255, 255], (y, x)

    # The whole outside-of-support region must be white.
    assert (out[~mask] == _WHITE).all()


def test_spiral_binary_invert_makes_the_whole_canvas_black() -> None:
    """``render_white_on_black`` turns the *entire* background black.

    That includes the corners outside the circular support, where the mask is
    ``False``: "white on black" means a black canvas, not a white surround.
    """
    side = 256
    mask = spiral_mask(_ramp_gray(side, side), 16, arms=3)
    wb = render_white_on_black(mask)

    assert (wb[~mask] == 0).all(), "outside-support background is not black"
    assert wb[0, 0] == 0
    assert wb[0, side - 1] == 0
    assert wb[side - 1, 0] == 0
    assert wb[side - 1, side - 1] == 0
    assert (wb[mask] == 255).all()


def test_spiral_high_arm_centre_rosette_is_preserved() -> None:
    """The dense centre rosette is a construction feature, not a defect.

    The renderers must reproduce the mask verbatim there -- no smoothing, no
    extra geometry -- so all three variants stay mutually consistent.
    """
    side = 256
    mask = spiral_mask(_ramp_gray(side, side), 16, arms=12)
    rgb = _color_field(side, side)

    bw = render_black_on_white(mask)
    wb = render_white_on_black(mask)
    color = render_source_color_on_white(mask, rgb)

    assert (bw.astype(np.uint16) + wb.astype(np.uint16) == 255).all()
    assert np.array_equal(color[mask], rgb[mask])
    # The centre really does contain structure to preserve.
    assert mask[110:146, 110:146].any()


@pytest.mark.parametrize("arms", [1, 2, 3, 12])
def test_spiral_arms_do_not_change_color_semantics(arms: int) -> None:
    side = 128
    mask = spiral_mask(_ramp_gray(side, side), 16, arms=arms)
    rgb = _color_field(side, side)
    out = render_source_color_on_white(mask, rgb)

    assert (out[~mask] == _WHITE).all()
    assert np.array_equal(out[mask], rgb[mask])
