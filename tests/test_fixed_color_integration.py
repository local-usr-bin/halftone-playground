"""Fixed-width color integration tests.

Fixed-width geometry reuses the existing renderer layer unchanged.  These
tests pin the end-to-end claim of this round:

* fixed Stripe mask + Cartesian source RGB -> ``render_source_color_on_white``;
* fixed Spiral mask + Cartesian source RGB -> ``render_source_color_on_white``;

the background is pure white, line pixels are byte-exact copies of the source,
the RGB source is *never* compared through the polar pipeline, and the
fixed-width geometry is independent of any grayscale modulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from halftone_playground import (
    render_source_color_on_white,
    resize_grayscale,
    resize_rgb,
    spiral_fixed_mask,
    stripe_fixed_mask,
)

_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def _color_field(height: int, width: int) -> np.ndarray:
    ys = np.arange(height, dtype=np.uint16)[:, None]
    xs = np.arange(width, dtype=np.uint16)[None, :]
    r = ((xs * np.uint16(255)) // max(width - 1, 1)).astype(np.uint8)
    g = ((ys * np.uint16(255)) // max(height - 1, 1)).astype(np.uint8)
    b = ((xs * np.uint16(7) + ys * np.uint16(11)) % np.uint16(256)).astype(np.uint8)
    return np.stack(
        [np.broadcast_to(r, (height, width)),
         np.broadcast_to(g, (height, width)),
         b],
        axis=-1,
    )


# --------------------------------------------------------------------------
# Stripe fixed color
# --------------------------------------------------------------------------


def test_stripe_fixed_color_exact_copy() -> None:
    mask = stripe_fixed_mask((96, 128), 16, 5, 90.0)
    rgb = _color_field(96, 128)
    out = render_source_color_on_white(mask, rgb)

    assert out.shape == (96, 128, 3)
    assert out.dtype == np.uint8
    assert (out[~mask] == _WHITE).all(), "background is not pure white"
    assert np.array_equal(out[mask], rgb[mask]), "line colour is not exact"


def test_stripe_fixed_geometry_is_independent_of_grayscale() -> None:
    """The fixed mask never reads grayscale, so two 'sources' give one mask."""
    a = stripe_fixed_mask((64, 64), 16, 7, 45.0)
    b = stripe_fixed_mask((64, 64), 16, 7, 45.0)
    assert np.array_equal(a, b)

    rgb = _color_field(64, 64)
    out_a = render_source_color_on_white(a, rgb)
    out_b = render_source_color_on_white(b, rgb)
    assert np.array_equal(out_a, out_b)


# --------------------------------------------------------------------------
# Spiral fixed color
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arms", [1, 2, 3, 12])
def test_spiral_fixed_color_exact_copy(arms: int) -> None:
    side = 128
    mask = spiral_fixed_mask(side, 16, 5, arms)
    rgb = _color_field(side, side)
    out = render_source_color_on_white(mask, rgb)

    assert out.shape == (side, side, 3)
    assert (out[~mask] == _WHITE).all()
    assert np.array_equal(out[mask], rgb[mask])


def test_spiral_fixed_outside_circle_is_pure_white() -> None:
    side = 128
    mask = spiral_fixed_mask(side, 16, 5, 3)
    rgb = _color_field(side, side)
    out = render_source_color_on_white(mask, rgb)

    for y, x in ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)):
        assert not mask[y, x]
        assert out[y, x].tolist() == [255, 255, 255]


def test_spiral_fixed_rgb_never_enters_polar() -> None:
    """A polar resample of RGB would leave interpolation residue.

    The copied line pixels must be byte-exact at the final Cartesian
    coordinates, which is only possible if the RGB never went through
    ``warpPolar``.
    """
    side = 128
    mask = spiral_fixed_mask(side, 16, 5, 3)
    rgb = _color_field(side, side)
    out = render_source_color_on_white(mask, rgb)

    ys, xs = np.nonzero(mask)
    assert ys.size > 100
    sample = np.linspace(0, ys.size - 1, num=512).astype(np.int64)
    for index in sample.tolist():
        y, x = int(ys[index]), int(xs[index])
        assert out[y, x].tolist() == rgb[y, x].tolist()
    assert np.array_equal(out[mask], rgb[mask])


# --------------------------------------------------------------------------
# scale / period / width independence
# --------------------------------------------------------------------------


def test_scale_and_fixed_width_are_independent() -> None:
    """scale=2 with P=16, width=5 -> a 2x canvas keeps P=16 and width=5."""
    gray = np.full((64, 64), 128, dtype=np.uint8)
    rgb = _color_field(64, 64)

    gray_s = resize_grayscale(gray, 2.0)
    rgb_s = resize_rgb(rgb, 2.0)
    assert gray_s.shape == (128, 128)

    mask = stripe_fixed_mask((128, 128), 16, 5, 90.0)
    out = render_source_color_on_white(mask, rgb_s)
    assert out.shape == (128, 128, 3)
    assert np.array_equal(out[mask], rgb_s[mask])


def test_fixed_width_mask_does_not_depend_on_a_gray_array_at_all() -> None:
    # The public API accepts a shape, so there is no way to pass grayscale in.
    import inspect

    sig = inspect.signature(stripe_fixed_mask)
    assert list(sig.parameters) == ["shape", "period", "line_width", "angle_deg"]
