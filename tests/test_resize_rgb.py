"""``resize_rgb`` tests: the RGB half of the preprocessing layer.

``resize_rgb`` exists so that the source-colour renderer can be handed an RGB
image that lines up pixel-for-pixel with the mask the geometry produced.  It
therefore has to obey the *exact same* rules as ``resize_grayscale``:

* identical target size (the shared half-up ``scaled_size`` rule);
* identical fixed BICUBIC filter -- never a user choice;
* a pixel-identical no-op at ``image_scale == 1.0``;
* the ``period`` stays independent of the scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from halftone_playground import (
    render_black_on_white,
    render_source_color_on_white,
    resize_grayscale,
    resize_rgb,
    scaled_size,
    stripe_mask,
)


def _rgb(height: int, width: int) -> np.ndarray:
    grid = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    channels = np.stack(
        [
            (grid * 3 % 256).astype(np.uint8),
            (grid * 5 % 256).astype(np.uint8),
            (grid * 7 % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    return channels


def _gray(height: int, width: int) -> np.ndarray:
    return np.tile(np.arange(width, dtype=np.uint8)[None, :], (height, 1))


# --------------------------------------------------------------------------
# size semantics: shared with the grayscale resizer
# --------------------------------------------------------------------------


def test_resize_rgb_identity_preserves_every_channel() -> None:
    rgb = _rgb(40, 53)
    out = resize_rgb(rgb, 1.0)

    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)
    # A copy, not a view.
    assert out is not rgb
    out[0, 0, 0] = (int(rgb[0, 0, 0]) + 1) % 256
    assert not np.array_equal(out, rgb)


def test_resize_rgb_identity_accepts_integer_one() -> None:
    rgb = _rgb(9, 13)
    assert np.array_equal(resize_rgb(rgb, 1), rgb)


def test_resize_rgb_1p3_matches_scaled_size() -> None:
    rgb = _rgb(73, 101)
    out = resize_rgb(rgb, 1.3)

    target_w, target_h = scaled_size(101, 73, 1.3)
    assert (target_w, target_h) == (131, 95)
    assert out.shape == (target_h, target_w, 3)
    assert out.dtype == np.uint8


def test_resize_rgb_doubles_dimensions() -> None:
    out = resize_rgb(_rgb(48, 64), 2.0)
    assert out.shape == (96, 128, 3)
    assert out.dtype == np.uint8


def test_resize_rgb_halves_dimensions_with_half_up() -> None:
    out = resize_rgb(_rgb(73, 101), 0.5)
    assert out.shape == (37, 51, 3)


def test_resize_rgb_agrees_with_resize_grayscale_dimensions() -> None:
    """The two resizers must never drift apart on the same scale."""
    gray = _gray(73, 101)
    rgb = _rgb(73, 101)
    for scale in (0.5, 1.0, 1.3, 2.0, 2.75):
        g_out = resize_grayscale(gray, scale)
        c_out = resize_rgb(rgb, scale)
        assert g_out.shape == c_out.shape[:2], scale
        assert c_out.shape[:2] == tuple(reversed(scaled_size(101, 73, scale)))


def test_resize_rgb_hundred_times_uses_scaled_size_only() -> None:
    """The huge-scale case is checked arithmetically, never allocated."""
    rgb = _rgb(3, 4)
    assert scaled_size(4, 3, 100.0) == (400, 300)
    # No huge RGB allocation is performed: only the size rule is asserted.
    assert rgb.shape == (3, 4, 3)


def test_resize_rgb_constant_image_stays_constant() -> None:
    """BICUBIC on a flat colour must reproduce that colour exactly."""
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[:] = (12, 200, 77)
    out = resize_rgb(rgb, 3.0)
    assert out.shape == (60, 60, 3)
    assert (out == np.array([12, 200, 77], dtype=np.uint8)).all()


def test_resize_rgb_output_is_always_uint8_3d() -> None:
    out = resize_rgb(_rgb(30, 40), 1.7)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape[-1] == 3
    assert out.shape[:2] == tuple(reversed(scaled_size(40, 30, 1.7)))


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, -2.0, float("nan"), float("inf"), float("-inf")]
)
def test_resize_rgb_rejects_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError):
        resize_rgb(_rgb(8, 8), scale)


@pytest.mark.parametrize("scale", ["2", None, True, [2]])
def test_resize_rgb_rejects_non_numeric_scales(scale) -> None:
    with pytest.raises(TypeError):
        resize_rgb(_rgb(8, 8), scale)


def test_resize_rgb_rejects_collapsing_dimensions() -> None:
    with pytest.raises(ValueError, match="at least 1x1"):
        resize_rgb(_rgb(3, 3), 0.1)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros(8, dtype=np.uint8),
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((0, 8, 3), dtype=np.uint8),
    ],
)
def test_resize_rgb_rejects_bad_shapes(rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        resize_rgb(rgb, 1.5)


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.uint16])
def test_resize_rgb_rejects_non_uint8(dtype) -> None:
    with pytest.raises(ValueError):
        resize_rgb(np.zeros((8, 8, 3), dtype=dtype), 1.5)


def test_resize_rgb_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        resize_rgb([[[1, 2, 3]]], 1.5)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# period and image_scale stay independent with an RGB source in play
# --------------------------------------------------------------------------


def _period_of_row(row: np.ndarray) -> int:
    n = row.size
    for p in range(1, n):
        if np.array_equal(row[: n - p], row[p:]):
            return p
    return n


def test_period_unchanged_when_scale_and_rgb_are_used() -> None:
    """512x512, scale=2, P=16 -> 1024x1024 canvas and still P=16.

    Both the grayscale geometry input and the RGB companion are doubled by the
    same scale, but the Stripe core receives the user's ``period`` verbatim.
    """
    gray = np.full((512, 512), 128, dtype=np.uint8)
    rgb = _rgb(512, 512)

    gray_scaled = resize_grayscale(gray, 2.0)
    rgb_scaled = resize_rgb(rgb, 2.0)
    assert gray_scaled.shape == (1024, 1024)
    assert rgb_scaled.shape == (1024, 1024, 3)

    mask = stripe_mask(gray_scaled, 16, 90.0)
    assert _period_of_row(mask[0]) == 16

    # The scaled RGB source aligns with the mask, so the compositor accepts it.
    out = render_source_color_on_white(mask, rgb_scaled)
    assert out.shape == (1024, 1024, 3)
    assert np.array_equal(out[mask], rgb_scaled[mask])


def test_black_on_white_pipeline_unchanged_by_rgb_preprocessing() -> None:
    """Adding ``resize_rgb`` must not touch the existing binary pipeline."""
    gray = np.full((64, 64), 90, dtype=np.uint8)
    mask = stripe_mask(gray, 16, 90.0)
    out = render_black_on_white(mask)
    assert set(np.unique(out).tolist()) <= {0, 255}
