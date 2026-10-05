"""Tests for the pure PNG-saving helpers (no display server required).

These cover the decisions GUI-002B can make without Tk: the suggested file
name, the "PNG only, case-insensitive, default extension" rule, and the
byte-exact encode of the stored full-resolution result.  The shell-level
behaviour (button lifecycle, dialog wiring, cancel, failure) is exercised with
a real Tk widget tree in ``tests/test_gui_app.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halftone_playground.gui import saving


# --------------------------------------------------------------------------
# suggested name
# --------------------------------------------------------------------------


class TestDefaultSaveName:
    def test_stem_gets_the_suffix(self):
        assert saving.default_save_name("photo.jpg") == "photo_halftone.png"

    def test_original_extension_is_dropped_not_kept(self):
        assert saving.default_save_name("photo.JPG") == "photo_halftone.png"
        # ...and never turns into "photo.jpg_halftone.png" or similar.
        assert saving.default_save_name(Path("dir/photo.jpeg")) == "photo_halftone.png"

    def test_dots_in_the_stem_are_preserved(self):
        assert saving.default_save_name("my.photo.png") == "my.photo_halftone.png"

    def test_no_stem_falls_back_to_the_default(self):
        assert saving.default_save_name("") == saving.DEFAULT_SAVE_NAME


# --------------------------------------------------------------------------
# extension rule
# --------------------------------------------------------------------------


class TestResolvePngPath:
    def test_plain_png_is_accepted(self):
        assert saving.resolve_png_path("out.png") == Path("out.png")

    @pytest.mark.parametrize("name", ["out.PNG", "out.Png", "OUT.pNg"])
    def test_png_suffix_is_case_insensitive(self, name):
        assert saving.resolve_png_path(name) == Path(name)

    def test_missing_suffix_gets_the_default_png(self):
        assert saving.resolve_png_path("out") == Path("out.png")

    @pytest.mark.parametrize("name", ["out.jpg", "out.jpeg", "out.bmp", "out.tif"])
    def test_explicit_other_suffix_is_rejected(self, name):
        with pytest.raises(saving.NotPngError):
            saving.resolve_png_path(name)

    def test_rejection_is_a_save_error(self):
        # So the shell can catch one exception type for every failure.
        assert issubclass(saving.NotPngError, saving.SaveError)

    def test_resolution_touches_no_file(self, tmp_path):
        target = tmp_path / "nothing-here.png"
        saving.resolve_png_path(target)
        assert not target.exists()


# --------------------------------------------------------------------------
# byte-exact encode of the stored result
# --------------------------------------------------------------------------


def _rgb_array(height: int = 7, width: int = 5) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    return np.dstack(
        [
            (xs * 255 // max(1, width - 1)).astype(np.uint8),
            (ys * 255 // max(1, height - 1)).astype(np.uint8),
            ((xs + ys) % 256).astype(np.uint8),
        ]
    )


def _rgba_array(height: int = 6, width: int = 6) -> np.ndarray:
    rgb = _rgb_array(height, width)
    alpha = np.zeros((height, width), np.uint8)
    # A checkerboard-ish 0/255 alpha so both values are present.
    alpha[(np.add.outer(np.arange(height), np.arange(width)) % 2) == 0] = 255
    return np.dstack([rgb, alpha])


class TestSavePngRoundTrip:
    def test_rgb_is_written_as_rgb_and_reads_back_identically(self, tmp_path):
        pixels = _rgb_array()
        target = tmp_path / "stripe.png"
        written = saving.save_png(pixels, "RGB", target)
        assert written == target

        with Image.open(target) as reopened:
            # Check the *original* mode first: converting before comparing
            # would hide a wrongly-encoded file (e.g. an alpha channel added).
            assert reopened.mode == "RGB"
            assert reopened.size == (pixels.shape[1], pixels.shape[0])
            assert np.array_equal(np.array(reopened), pixels)

    def test_rgb_has_no_alpha_channel(self, tmp_path):
        target = tmp_path / "stripe.png"
        saving.save_png(_rgb_array(), "RGB", target)
        with Image.open(target) as reopened:
            assert reopened.mode == "RGB"
            assert len(reopened.getbands()) == 3

    def test_rgba_is_written_as_rgba_and_reads_back_identically(self, tmp_path):
        pixels = _rgba_array()
        target = tmp_path / "spiral.png"
        saving.save_png(pixels, "RGBA", target)

        with Image.open(target) as reopened:
            assert reopened.mode == "RGBA"
            assert reopened.size == (pixels.shape[1], pixels.shape[0])
            assert np.array_equal(np.array(reopened), pixels)

    def test_rgba_alpha_keeps_only_zero_and_full(self, tmp_path):
        pixels = _rgba_array()
        target = tmp_path / "spiral.png"
        saving.save_png(pixels, "RGBA", target)
        with Image.open(target) as reopened:
            alpha = np.array(reopened)[..., 3]
            assert set(np.unique(alpha).tolist()) <= {0, 255}

    def test_bilevel_rgb_is_not_collapsed_to_a_palette(self, tmp_path):
        # A pure black/white RGB result must stay a 3-channel RGB PNG; a
        # palette ("P") encoding would be a silent format change.
        pixels = np.zeros((8, 8, 3), np.uint8)
        pixels[::2] = 255
        target = tmp_path / "bilevel.png"
        saving.save_png(pixels, "RGB", target)
        with Image.open(target) as reopened:
            assert reopened.mode == "RGB"
            assert np.array_equal(np.array(reopened), pixels)

    def test_the_file_really_is_a_png(self, tmp_path):
        target = tmp_path / "magic.png"
        saving.save_png(_rgb_array(), "RGB", target)
        with open(target, "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


class TestSavePngFailures:
    def test_non_png_suffix_raises_and_writes_nothing(self, tmp_path):
        target = tmp_path / "result.jpg"
        with pytest.raises(saving.NotPngError):
            saving.save_png(_rgb_array(), "RGB", target)
        assert not target.exists()
        assert not (tmp_path / "result.jpg.png").exists()

    def test_unwritable_destination_raises_a_friendly_save_error(self, tmp_path):
        missing_dir = tmp_path / "does-not-exist" / "out.png"
        with pytest.raises(saving.SaveError) as excinfo:
            saving.save_png(_rgb_array(), "RGB", missing_dir)
        message = str(excinfo.value)
        assert "Traceback" not in message
        assert "out.png" in message

    def test_failure_leaves_no_partial_file(self, tmp_path):
        missing_dir = tmp_path / "nope" / "out.png"
        with pytest.raises(saving.SaveError):
            saving.save_png(_rgb_array(), "RGB", missing_dir)
        assert not missing_dir.exists()
