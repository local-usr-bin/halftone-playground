"""Phase 0 smoke tests: package import, array sanity, PNG round-trip.

These tests deliberately avoid any GUI toolkit and contain no halftoning
logic.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

import halftone_playground
from halftone_playground import GRADIENT_SIZE, make_gradient, save_gray_png


def test_package_import() -> None:
    assert halftone_playground.__version__


def test_gradient_shape_dtype_and_range() -> None:
    arr = make_gradient()
    assert arr.shape == (GRADIENT_SIZE, GRADIENT_SIZE)
    assert arr.dtype == np.uint8
    assert int(arr.min()) == 0
    assert int(arr.max()) == 255


def test_png_roundtrip(tmp_path) -> None:
    arr = make_gradient()
    out = tmp_path / "gradient.png"
    save_gray_png(arr, out)

    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.mode == "L"
        assert img.size == (GRADIENT_SIZE, GRADIENT_SIZE)
        reloaded = np.asarray(img)

    assert reloaded.shape == arr.shape
    assert np.array_equal(reloaded, arr)


def test_saved_png_is_256x256(tmp_path) -> None:
    out = tmp_path / "size.png"
    save_gray_png(make_gradient(), out)
    assert Image.open(out).size == (256, 256)


def test_no_gui_toolkits_loaded(tmp_path) -> None:
    """The smoke pipeline is pure file I/O; no GUI toolkit may be pulled in."""
    save_gray_png(make_gradient(), tmp_path / "g.png")
    for gui in ("tkinter", "PyQt5", "PySide2", "PyQt6", "PySide6"):
        assert gui not in sys.modules
