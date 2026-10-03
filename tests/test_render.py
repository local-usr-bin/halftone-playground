"""Renderer tests: the mask -> image layer is strict and bi-level."""

from __future__ import annotations

import numpy as np
import pytest

from halftone_playground import render_black_on_white


def test_true_becomes_black_and_false_becomes_white() -> None:
    mask = np.array([[True, False], [False, True]], dtype=bool)
    out = render_black_on_white(mask)
    assert out.dtype == np.uint8
    assert out.shape == mask.shape
    assert out[0, 0] == 0
    assert out[0, 1] == 255
    assert out[1, 0] == 255
    assert out[1, 1] == 0


def test_output_is_strictly_bi_level() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[::3, ::5] = True
    out = render_black_on_white(mask)
    assert set(np.unique(out).tolist()) <= {0, 255}


def test_all_true_renders_fully_black() -> None:
    out = render_black_on_white(np.ones((4, 4), dtype=bool))
    assert np.unique(out).tolist() == [0]


def test_all_false_renders_fully_white() -> None:
    out = render_black_on_white(np.zeros((4, 4), dtype=bool))
    assert np.unique(out).tolist() == [255]


def test_rejects_non_bool_mask() -> None:
    with pytest.raises(ValueError):
        render_black_on_white(np.zeros((4, 4), dtype=np.uint8))


def test_rejects_non_2d_mask() -> None:
    with pytest.raises(ValueError):
        render_black_on_white(np.zeros((4,), dtype=bool))
