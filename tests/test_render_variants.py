"""Renderer / compositor tests: same mask, three looks, zero geometry change.

Round 1 adds two output variants on top of the frozen geometry cores:

* ``render_white_on_black`` -- binary inversion;
* ``render_source_color_on_white`` -- variable-width lines painted with the
  *Cartesian* source RGB.

The tests below pin down the properties the work order cares about: strict
bi-level output, exact per-pixel complementarity, byte-exact colour copying,
an explicitly white background, strict input validation, and -- crucially --
that no renderer ever mutates its inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from halftone_playground import (
    render_black_on_white,
    render_source_color_on_white,
    render_white_on_black,
)


def _mixed_mask() -> np.ndarray:
    mask = np.zeros((6, 7), dtype=bool)
    mask[0, 0] = True
    mask[1, 3] = True
    mask[2, :] = True
    mask[5, 6] = True
    return mask


def _distinct_rgb(height: int, width: int) -> np.ndarray:
    """An RGB image in which every pixel differs (deterministic, no random)."""
    grid = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    r = (grid * 3 % 256).astype(np.uint8)
    g = (grid * 5 + 17 % 256).astype(np.uint8)
    b = (grid * 7 + 91 % 256).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


# --------------------------------------------------------------------------
# render_white_on_black: the binary inversion variant
# --------------------------------------------------------------------------


def test_white_on_black_all_false_is_all_black() -> None:
    out = render_white_on_black(np.zeros((4, 5), dtype=bool))
    assert np.unique(out).tolist() == [0]


def test_white_on_black_all_true_is_all_white() -> None:
    out = render_white_on_black(np.ones((4, 5), dtype=bool))
    assert np.unique(out).tolist() == [255]


def test_white_on_black_mixed_mask_maps_true_to_255_false_to_0() -> None:
    mask = _mixed_mask()
    out = render_white_on_black(mask)
    assert out[mask].tolist() == [255] * int(mask.sum())
    assert out[~mask].tolist() == [0] * int((~mask).sum())


def test_white_on_black_is_exact_complement_of_black_on_white() -> None:
    """The two binary renderers must be per-pixel complements, always."""
    for mask in (
        np.zeros((3, 3), dtype=bool),
        np.ones((3, 3), dtype=bool),
        _mixed_mask(),
        np.eye(9, dtype=bool),
    ):
        bw = render_black_on_white(mask)
        wb = render_white_on_black(mask)
        total = bw.astype(np.uint16) + wb.astype(np.uint16)
        assert (total == 255).all(), mask


def test_white_on_black_output_dtype_and_shape() -> None:
    mask = _mixed_mask()
    out = render_white_on_black(mask)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == mask.shape


def test_white_on_black_output_is_strictly_bi_level() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[::3, ::5] = True
    out = render_white_on_black(mask)
    assert set(np.unique(out).tolist()) <= {0, 255}


def test_white_on_black_does_not_mutate_or_share_memory_with_mask() -> None:
    mask = _mixed_mask()
    before = mask.copy()
    out = render_white_on_black(mask)
    assert np.array_equal(mask, before)
    # Editing the output must never reach back into the mask.
    out[0, 0] = 0
    assert np.array_equal(mask, before)


# --------------------------------------------------------------------------
# render_white_on_black: validation (mirrors render_black_on_white)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.int32),
        np.zeros((4, 4), dtype=float),
    ],
)
def test_white_on_black_rejects_non_bool_mask(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        render_white_on_black(mask)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(4, dtype=bool),
        np.zeros((4, 4, 1), dtype=bool),
    ],
)
def test_white_on_black_rejects_non_2d_mask(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        render_white_on_black(mask)


def test_white_on_black_rejects_empty_mask() -> None:
    with pytest.raises(ValueError):
        render_white_on_black(np.zeros((0, 4), dtype=bool))


def test_white_on_black_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        render_white_on_black([[True, False]])  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# render_source_color_on_white: byte-exact colour copying
# --------------------------------------------------------------------------


def test_source_color_copies_line_pixels_exactly() -> None:
    mask = _mixed_mask()
    rgb = _distinct_rgb(*mask.shape)
    out = render_source_color_on_white(mask, rgb)

    # mask=False -> pure white.
    assert (out[~mask] == np.array([255, 255, 255], dtype=np.uint8)).all()
    # mask=True -> byte-exact copy of the source pixel.
    assert np.array_equal(out[mask], rgb[mask])


def test_source_color_output_contract() -> None:
    mask = _mixed_mask()
    rgb = _distinct_rgb(*mask.shape)
    out = render_source_color_on_white(mask, rgb)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape == (*mask.shape, 3)


def test_source_color_all_false_is_pure_white() -> None:
    mask = np.zeros((5, 4), dtype=bool)
    rgb = _distinct_rgb(5, 4)
    out = render_source_color_on_white(mask, rgb)
    assert (out == np.array([255, 255, 255], dtype=np.uint8)).all()


def test_source_color_all_true_is_exact_source() -> None:
    mask = np.ones((5, 4), dtype=bool)
    rgb = _distinct_rgb(5, 4)
    out = render_source_color_on_white(mask, rgb)
    assert np.array_equal(out, rgb)


def test_distinct_source_pixels_really_differ() -> None:
    """Guard the fixture itself: a broken fixture would hide copy bugs."""
    rgb = _distinct_rgb(4, 5)
    flat = rgb.reshape(-1, 3)
    assert len({tuple(px) for px in flat.tolist()}) == flat.shape[0]


def test_source_color_does_not_mutate_inputs() -> None:
    mask = _mixed_mask()
    rgb = _distinct_rgb(*mask.shape)
    mask_before = mask.copy()
    rgb_before = rgb.copy()

    out = render_source_color_on_white(mask, rgb)

    assert np.array_equal(mask, mask_before)
    assert np.array_equal(rgb, rgb_before)
    # The output must not be a view that aliases the source buffer.
    out[0, 0] = [0, 0, 0]
    assert np.array_equal(rgb, rgb_before)


def test_source_color_never_introduces_a_fourth_channel() -> None:
    mask = _mixed_mask()
    rgb = _distinct_rgb(*mask.shape)
    out = render_source_color_on_white(mask, rgb)
    assert out.shape[-1] == 3
    assert out.dtype == np.uint8


# --------------------------------------------------------------------------
# render_source_color_on_white: validation
# --------------------------------------------------------------------------


def _good_rgb() -> np.ndarray:
    return _distinct_rgb(6, 7)


@pytest.mark.parametrize("bad_mask", ["not an array", None, [[True]]])
def test_source_color_rejects_non_array_mask(bad_mask) -> None:
    with pytest.raises(TypeError):
        render_source_color_on_white(bad_mask, _good_rgb())


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.zeros(6, dtype=bool),  # 1-D
        np.zeros((6, 7, 1), dtype=bool),  # 3-D
        np.zeros((4, 4), dtype=np.uint8),  # wrong dtype
        np.zeros((0, 7), dtype=bool),  # empty
    ],
)
def test_source_color_rejects_bad_mask(bad_mask: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        render_source_color_on_white(bad_mask, _good_rgb())


def test_source_color_rejects_non_uint8_rgb() -> None:
    with pytest.raises(ValueError):
        render_source_color_on_white(
            _mixed_mask(), _distinct_rgb(6, 7).astype(np.int32)
        )
    with pytest.raises(ValueError):
        render_source_color_on_white(
            _mixed_mask(), _distinct_rgb(6, 7).astype(np.float32)
        )


def test_source_color_rejects_2d_rgb() -> None:
    with pytest.raises(ValueError):
        render_source_color_on_white(_mixed_mask(), np.zeros((6, 7), dtype=np.uint8))


def test_source_color_rejects_4_channel_rgb() -> None:
    with pytest.raises(ValueError):
        render_source_color_on_white(
            _mixed_mask(), np.zeros((6, 7, 4), dtype=np.uint8)
        )


def test_source_color_rejects_empty_rgb() -> None:
    with pytest.raises(ValueError):
        render_source_color_on_white(
            _mixed_mask(), np.zeros((0, 7, 3), dtype=np.uint8)
        )


@pytest.mark.parametrize(
    "shape",
    [(6, 8, 3), (5, 7, 3), (7, 7, 3), (1, 1, 3)],
)
def test_source_color_rejects_spatial_mismatch(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="match mask"):
        render_source_color_on_white(
            _mixed_mask(), np.zeros(shape, dtype=np.uint8)
        )


def test_source_color_rejects_non_array_rgb() -> None:
    with pytest.raises(TypeError):
        render_source_color_on_white(_mixed_mask(), [[[0, 0, 0]]])  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# render_source_color_on_white: never resizes the RGB source
# --------------------------------------------------------------------------


def test_source_color_never_resizes_rgb() -> None:
    """A resize would produce valid pixels instead of a rejection.

    The compositor is explicitly forbidden from resizing (spatial alignment is
    a preprocessing concern), so an off-by-one source must raise rather than
    be quietly stretched.
    """
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, :] = True
    rgb_near_miss = _distinct_rgb(10, 11)
    with pytest.raises(ValueError):
        render_source_color_on_white(mask, rgb_near_miss)
    # And the correctly-sized one is accepted unchanged.
    out = render_source_color_on_white(mask, _distinct_rgb(10, 10))
    assert out.shape == (10, 10, 3)
