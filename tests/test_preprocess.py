"""Preprocessing tests: ``image_scale`` semantics and BICUBIC resize.

Frozen product rules under test:

* any positive finite float is a legal scale (no arbitrary upper limit:
  ``100.0`` is as valid as ``1.3``);
* target dimensions use half-up rounding, never banker's rounding;
* ``image_scale == 1.0`` is a guaranteed pixel-identical no-op;
* the resample filter is fixed (Pillow BICUBIC) and not exposed as a choice;
* ``scaled_size`` is pure arithmetic and computes huge outputs without
  allocating anything;
* dimensions below 1x1 are rejected with a clear error, never clamped;
* non-finite/overflowing products are rejected as invalid input;
* the stripe ``period`` is completely independent of ``image_scale``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from halftone_playground import resize_grayscale, scaled_size, stripe_mask


def _constant(value: int, height: int, width: int) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


# --------------------------------------------------------------------------
# scaled_size: the half-up arithmetic
# --------------------------------------------------------------------------


def test_scaled_size_identity_at_one() -> None:
    assert scaled_size(512, 512, 1.0) == (512, 512)
    assert scaled_size(4000, 3000, 1) == (4000, 3000)  # int 1 is fine too


def test_scaled_size_1p3_non_integer_example() -> None:
    """101x73 at 1.3 -> 131x95 (half-up per axis, not banker's rounding)."""
    assert scaled_size(101, 73, 1.3) == (131, 95)
    # GUI-preview examples from the spec.
    assert scaled_size(512, 512, 1.3) == (666, 666)
    assert scaled_size(256, 256, 1.3) == (333, 333)
    assert scaled_size(4000, 3000, 1.3) == (5200, 3900)


def test_scaled_size_doubles_exactly_at_two() -> None:
    assert scaled_size(512, 512, 2.0) == (1024, 1024)
    assert scaled_size(101, 73, 2.0) == (202, 146)


def test_scaled_size_halves_with_half_up() -> None:
    """0.5 on odd dimensions hits exact .5 values: half-up must round them up.

    101 * 0.5 = 50.5 -> 51 and 73 * 0.5 = 36.5 -> 37 (banker's rounding
    would give 50 and 36; the spec forbids that).
    """
    assert scaled_size(101, 73, 0.5) == (51, 37)
    assert scaled_size(100, 60, 0.5) == (50, 30)


def test_scaled_size_hundred_times_without_any_cap() -> None:
    """4000x3000 at scale 100 must compute 400000x300000, not refuse.

    This is pure arithmetic: no image is allocated and no arbitrary maximum
    scale is applied.
    """
    assert scaled_size(4000, 3000, 100.0) == (400000, 300000)


def test_scaled_size_accepts_very_large_but_finite_scales() -> None:
    # There is deliberately no upper bound; only non-finite results fail.
    assert scaled_size(4000, 3000, 1e6) == (4_000_000_000, 3_000_000_000)


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, -0.5, float("nan"), float("inf"), float("-inf")]
)
def test_scaled_size_rejects_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError):
        scaled_size(100, 100, scale)


@pytest.mark.parametrize("scale", ["2.0", None, True, [2]])
def test_scaled_size_rejects_non_numeric_scales(scale) -> None:
    with pytest.raises(TypeError):
        scaled_size(100, 100, scale)


def test_scaled_size_rejects_collapsing_dimensions() -> None:
    """A tiny scale that rounds a dimension below 1 must be refused loudly.

    Clamping silently to 1x1 is forbidden by the contract.
    """
    with pytest.raises(ValueError, match="at least 1x1"):
        scaled_size(3, 3, 0.1)
    with pytest.raises(ValueError, match="at least 1x1"):
        scaled_size(1, 1, 0.4)


def test_scaled_size_rejects_overflowing_products() -> None:
    """A finite scale whose product overflows is invalid input, not a resource
    problem: it must raise instead of producing a bogus dimension."""
    with pytest.raises(ValueError):
        scaled_size(4000, 3000, 1e308)


@pytest.mark.parametrize("bad_dimension", [0, -5])
def test_scaled_size_rejects_non_positive_dimensions(bad_dimension: int) -> None:
    with pytest.raises(ValueError):
        scaled_size(bad_dimension, 100, 1.0)
    with pytest.raises(ValueError):
        scaled_size(100, bad_dimension, 1.0)


@pytest.mark.parametrize("bad_dimension", [10.5, "10", True, None])
def test_scaled_size_rejects_non_integer_dimensions(bad_dimension) -> None:
    with pytest.raises(TypeError):
        scaled_size(bad_dimension, 100, 1.0)
    with pytest.raises(TypeError):
        scaled_size(100, bad_dimension, 1.0)


# --------------------------------------------------------------------------
# resize_grayscale: hard output properties
# --------------------------------------------------------------------------


def test_resize_identity_returns_identical_pixels() -> None:
    """scale = 1 must not change a single pixel value."""
    gray = np.tile(np.arange(256, dtype=np.uint8)[None, :], (40, 1))
    out = resize_grayscale(gray, 1.0)

    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, gray)
    # And it is a copy, not a view of the input.
    assert out is not gray
    out[0, 0] = 255 if gray[0, 0] != 255 else 0
    assert gray[0, 0] != out[0, 0]


def test_resize_identity_with_integer_one() -> None:
    gray = _constant(77, 9, 13)
    out = resize_grayscale(gray, 1)
    assert np.array_equal(out, gray)


def test_resize_1p3_matches_scaled_size() -> None:
    gray = _constant(128, 73, 101)  # (height, width) = (73, 101)
    out = resize_grayscale(gray, 1.3)

    target_w, target_h = scaled_size(101, 73, 1.3)
    assert (target_w, target_h) == (131, 95)
    assert out.shape == (target_h, target_w)  # (95, 131)


def test_resize_doubles_dimensions() -> None:
    gray = _constant(200, 48, 64)
    out = resize_grayscale(gray, 2.0)
    assert out.shape == (96, 128)
    assert out.dtype == np.uint8
    assert out.ndim == 2


def test_resize_halves_dimensions() -> None:
    gray = _constant(200, 73, 101)
    out = resize_grayscale(gray, 0.5)
    assert out.shape == (37, 51)


def test_resize_output_is_always_uint8_2d() -> None:
    """No RGB, no float, no alpha channel can ever appear."""
    out = resize_grayscale(_constant(90, 30, 40), 1.7)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == tuple(reversed(scaled_size(40, 30, 1.7)))


def test_resize_constant_image_stays_constant() -> None:
    """BICUBIC on a flat image must reproduce the flat value exactly."""
    for value in (0, 64, 128, 200, 255):
        out = resize_grayscale(_constant(value, 20, 20), 3.0)
        assert out.shape == (60, 60)
        assert (out == value).all(), value


@pytest.mark.parametrize(
    "scale", [0.0, -2.0, float("nan"), float("inf"), float("-inf")]
)
def test_resize_rejects_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError):
        resize_grayscale(_constant(128, 8, 8), scale)


@pytest.mark.parametrize("scale", ["2", None, True])
def test_resize_rejects_non_numeric_scales(scale) -> None:
    with pytest.raises(TypeError):
        resize_grayscale(_constant(128, 8, 8), scale)


@pytest.mark.parametrize(
    "gray",
    [
        np.zeros(8, dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((0, 4), dtype=np.uint8),
    ],
)
def test_resize_rejects_bad_gray_inputs(gray: np.ndarray) -> None:
    with pytest.raises(ValueError):
        resize_grayscale(gray, 1.5)


def test_resize_rejects_non_uint8_gray() -> None:
    with pytest.raises(ValueError):
        resize_grayscale(np.full((8, 8), 128, dtype=np.float32), 1.5)


def test_resize_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        resize_grayscale([[1, 2], [3, 4]], 1.5)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# period and image_scale are fully independent
# --------------------------------------------------------------------------


def _period_of_row(row: np.ndarray) -> int:
    """Smallest positive p such that row[x] == row[x + p] wherever defined."""
    n = row.size
    for p in range(1, n):
        if np.array_equal(row[: n - p], row[p:]):
            return p
    return n


def test_period_is_not_rescaled_by_image_scale() -> None:
    """The core must receive exactly the period the user asked for.

    Case A (scale=1, P=16), Case B (scale=2, P=16) and Case C (scale=2, P=32)
    on a constant 128-gray source.  If the software secretly multiplied the
    period by the scale, Case B would show a period of 32 and this test would
    fail.  The user who wants the same stripe density at 2x must set P=32
    themselves (Case C).
    """
    source = _constant(128, 128, 128)

    # Case A: no scaling, P = 16.
    mask_a = stripe_mask(source, 16, 90.0)
    assert _period_of_row(mask_a[0]) == 16

    # Case B: scale = 2, still P = 16 -- the period must stay 16 pixels on
    # the 256x256 scaled image, i.e. 16 stripes across, not 8.
    scaled_b = resize_grayscale(source, 2.0)
    assert scaled_b.shape == (256, 256)
    mask_b = stripe_mask(scaled_b, 16, 90.0)
    assert _period_of_row(mask_b[0]) == 16
    # 128 gray -> W = 16 * 127/255 = 7.97 -> N = 8 black pixels per period.
    runs_b = np.flatnonzero(np.diff(np.concatenate(([False], mask_b[0], [False]))))
    widths = np.diff(runs_b)
    assert set(widths.tolist()) == {8}

    # Case C: scale = 2, P = 32 -- the user's explicit choice for the same
    # visual density, and now the period really is 32.
    mask_c = stripe_mask(scaled_b, 32, 90.0)
    assert _period_of_row(mask_c[0]) == 32
    runs_c = np.flatnonzero(np.diff(np.concatenate(([False], mask_c[0], [False]))))
    assert set(np.diff(runs_c).tolist()) == {16}  # W = 32 * 127/255 = 15.96 -> 16

    # B and C are genuinely different patterns.
    assert not np.array_equal(mask_b, mask_c)


def test_scaled_pipeline_is_resize_then_core() -> None:
    """The pipeline order is fixed: resize first, stripes on the scaled image.

    A 512-wide source at scale 2 with P=16 must show 64 stripes across the
    1024-wide output, exactly as if the user had supplied the 1024 image
    directly.
    """
    source = _constant(100, 8, 512)
    scaled = resize_grayscale(source, 2.0)
    assert scaled.shape == (16, 1024)

    via_pipeline = stripe_mask(scaled, 16, 90.0)
    direct = stripe_mask(_constant(100, 16, 1024), 16, 90.0)
    assert np.array_equal(via_pipeline, direct)
