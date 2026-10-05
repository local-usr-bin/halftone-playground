"""Tests for the pure preview-image helpers (no display server required).

Covers aspect-preserving fit arithmetic (no crop / no stretch), single-frame
loading, multi-frame rejection and the guarantee that building a preview never
mutates the source image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halftone_playground.gui import imageutil


# --------------------------------------------------------------------------
# fit_size (aspect-preserving "contain")
# --------------------------------------------------------------------------


class TestFitSize:
    def test_wide_image_fits_by_width(self):
        # 400x200 into 320x320 -> scale 0.8 -> 320x160
        assert imageutil.fit_size((400, 200), (320, 320)) == (320, 160)

    def test_tall_image_fits_by_height(self):
        # 200x400 into 320x320 -> scale 0.8 -> 160x320
        assert imageutil.fit_size((200, 400), (320, 320)) == (160, 320)

    def test_square_stays_square(self):
        assert imageutil.fit_size((100, 100), (256, 256)) == (256, 256)

    def test_small_image_scales_up(self):
        assert imageutil.fit_size((10, 10), (100, 100)) == (100, 100)

    def test_aspect_ratio_preserved(self):
        width, height = imageutil.fit_size((1000, 500), (400, 400))
        assert width / height == pytest.approx(2.0, rel=1e-3)

    def test_result_never_exceeds_box(self):
        for source in [(1000, 333), (333, 1000), (7, 7), (1920, 1080)]:
            for box in [(320, 320), (200, 100), (50, 400)]:
                width, height = imageutil.fit_size(source, box)
                # Allow a 1px rounding tolerance.
                assert width <= box[0] + 1
                assert height <= box[1] + 1

    def test_degenerate_box_returns_one(self):
        assert imageutil.fit_size((100, 100), (0, 0)) == (1, 1)
        assert imageutil.fit_size((100, 100), (-5, 10)) == (1, 1)

    def test_degenerate_source_returns_one(self):
        assert imageutil.fit_size((0, 0), (100, 100)) == (1, 1)


# --------------------------------------------------------------------------
# scaled_preview
# --------------------------------------------------------------------------


class TestScaledPreview:
    def test_preview_does_not_mutate_source(self):
        source = Image.fromarray(np.full((80, 40, 3), 128, np.uint8), "RGB")
        original = source.copy()
        preview = imageutil.scaled_preview(source, (320, 320))
        assert preview is not None
        # The source must be untouched.
        assert np.array_equal(np.array(source), np.array(original))
        # The preview is a different object.
        assert preview is not source

    def test_preview_fits_box_preserving_aspect(self):
        # (height=100, width=200) -> a wide 200x100 image.
        source = Image.fromarray(np.full((100, 200, 3), 200, np.uint8), "RGB")
        preview = imageutil.scaled_preview(source, (320, 320))
        assert preview is not None
        assert preview.size == (320, 160)

    def test_preview_converts_to_rgb(self):
        gray = Image.fromarray(np.full((40, 40), 128, np.uint8), "L")
        preview = imageutil.scaled_preview(gray, (100, 100))
        assert preview is not None
        assert preview.mode == "RGB"

    def test_degenerate_box_returns_none(self):
        source = Image.fromarray(np.full((40, 40, 3), 0, np.uint8), "RGB")
        assert imageutil.scaled_preview(source, (0, 100)) is None


# --------------------------------------------------------------------------
# load_single_frame
# --------------------------------------------------------------------------


class TestLoadSingleFrame:
    def test_loads_png(self, tmp_path):
        path = tmp_path / "one.png"
        Image.fromarray(np.full((32, 24), 100, np.uint8), "L").save(path)
        image = imageutil.load_single_frame(path)
        assert image.size == (24, 32)
        assert image.mode == "RGB"

    def test_loads_jpeg(self, tmp_path):
        path = tmp_path / "one.jpg"
        Image.fromarray(np.full((16, 16, 3), 200, np.uint8), "RGB").save(path)
        image = imageutil.load_single_frame(path)
        assert image.size == (16, 16)
        assert image.mode == "RGB"

    def test_rejects_multiframe_gif(self, tmp_path):
        path = tmp_path / "anim.gif"
        frames = [
            Image.fromarray(np.full((16, 16), v, np.uint8), "L") for v in (0, 128, 255)
        ]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100)
        with pytest.raises(imageutil.MultiFrameImageError):
            imageutil.load_single_frame(path)

    def test_multiframe_error_is_a_load_error(self):
        # MultiFrameImageError must be catchable as a generic load error too.
        assert issubclass(imageutil.MultiFrameImageError, imageutil.ImageLoadError)

    def test_rejects_non_image(self, tmp_path):
        path = tmp_path / "not_an_image.txt"
        path.write_text("hello")
        with pytest.raises(imageutil.ImageLoadError):
            imageutil.load_single_frame(path)

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(imageutil.ImageLoadError):
            imageutil.load_single_frame(tmp_path / "absent.png")

    def test_returns_detached_image(self, tmp_path):
        # The returned image must survive after the file handle is closed.
        path = tmp_path / "one.png"
        Image.fromarray(np.full((10, 10), 50, np.uint8), "L").save(path)
        image = imageutil.load_single_frame(path)
        assert image.getpixel((0, 0)) == (50, 50, 50)
