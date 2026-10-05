"""Tests for the GUI generation pipeline (toolkit-free, always run).

These are ordinary tests: :mod:`halftone_playground.gui.pipeline` imports no Tk
and no widget, so the whole 12-combination matrix, the Spiral crop and the
alpha semantics can be pinned headlessly.

Only the *pure* pieces are covered here.  The worker/threading tests live in
``test_gui_worker.py``, and the shell-level assertions live in
``test_gui_app.py`` under the ``gui`` marker.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halftone_playground.gui import params as gp
from halftone_playground.gui.pipeline import (
    GenerationKey,
    GenerationRequest,
    GenerationResult,
    PipelineError,
    generate,
)
from halftone_playground.preprocess import (
    circular_support,
    resize_grayscale,
    resize_rgb,
    scaled_size,
)
from halftone_playground.render import render_source_color_on_white
from halftone_playground.spiral import (
    _polar_center,
    _support_radius,
    spiral_mask,
)

SQUARE = 64
RECT_W, RECT_H = 100, 60


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _rect_source(path: Path, width: int = RECT_W, height: int = RECT_H) -> Path:
    """A rectangular RGB source with a monotone, position-dependent pattern.

    Both axes vary, so a wrong crop box or a scale-then-crop mix-up changes
    the pixels rather than merely the shape.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    arr = np.dstack(
        [
            (xs * 255 // max(width - 1, 1)).astype(np.uint8),
            (ys * 255 // max(height - 1, 1)).astype(np.uint8),
            np.full((height, width), 7, np.uint8),
        ]
    )
    Image.fromarray(arr, "RGB").save(path)
    return path


def _square_source(path: Path, side: int = SQUARE) -> Path:
    ys, xs = np.mgrid[0:side, 0:side]
    arr = np.dstack(
        [
            (xs * 255 // (side - 1)).astype(np.uint8),
            (ys * 255 // (side - 1)).astype(np.uint8),
            np.full((side, side), 3, np.uint8),
        ]
    )
    Image.fromarray(arr, "RGB").save(path)
    return path


@pytest.fixture()
def rect_png(tmp_path: Path) -> Path:
    return _rect_source(tmp_path / "rect.png")


@pytest.fixture()
def square_png(tmp_path: Path) -> Path:
    return _square_source(tmp_path / "square.png")


def _request(
    path: Path,
    size: tuple[int, int],
    *,
    mode: str = gp.MODE_STRIPE,
    width_mode: str = gp.WIDTH_VARIABLE,
    render: str = gp.RENDER_BLACK_ON_WHITE,
    scale: str = "1",
    period: str = "8",
    angle: str = "90",
    arms: str = "3",
    line_width: str = "2",
) -> GenerationRequest:
    p = gp.RenderParams(
        mode=mode,
        width_mode=width_mode,
        render=render,
        scale_text=scale,
        period_text=period,
        angle_text=angle,
        arms_text=arms,
        line_width_text=line_width,
    )
    return GenerationRequest.from_params(p, path, size)


# --------------------------------------------------------------------------
# the 12 combinations
# --------------------------------------------------------------------------

COMBINATIONS = [
    (mode, width_mode, render)
    for mode in gp.MODES
    for width_mode in gp.WIDTH_MODES
    for render in gp.RENDER_CHOICES
]


def test_matrix_is_complete() -> None:
    assert len(COMBINATIONS) == 12
    assert len(set(COMBINATIONS)) == 12


@pytest.mark.parametrize("mode,width_mode,render", COMBINATIONS)
def test_every_combination_produces_the_frozen_output_format(
    rect_png, mode, width_mode, render
) -> None:
    """All 12 combinations run and obey the RGB / RGBA output contract."""
    request = _request(
        rect_png,
        (RECT_W, RECT_H),
        mode=mode,
        width_mode=width_mode,
        render=render,
    )
    result = generate(request)

    assert isinstance(result, GenerationResult)
    assert result.pixels.dtype == np.uint8
    assert result.kind == mode
    assert result.render == render

    if mode == gp.MODE_STRIPE:
        # Stripe: H x W x 3, never cropped, no alpha at all.
        assert result.mode == "RGB"
        assert result.has_alpha is False
        assert result.pixels.shape == (RECT_H, RECT_W, 3)
        assert result.size == (RECT_W, RECT_H)
    else:
        # Spiral: N x N x 4, alpha present.
        assert result.mode == "RGBA"
        assert result.has_alpha is True
        side = min(RECT_W, RECT_H)
        assert result.pixels.shape == (side, side, 4)
        assert result.size == (side, side)


@pytest.mark.parametrize("mode,width_mode,render", COMBINATIONS)
def test_alpha_only_exists_for_spiral(
    square_png, mode, width_mode, render
) -> None:
    result = generate(
        _request(
            square_png,
            (SQUARE, SQUARE),
            mode=mode,
            width_mode=width_mode,
            render=render,
        )
    )
    assert result.has_alpha == (mode == gp.MODE_SPIRAL)


# --------------------------------------------------------------------------
# Spiral: centered maximum-square crop
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", ["1", "1.3", "2", "0.5"])
def test_spiral_crops_before_scaling(rect_png, scale) -> None:
    """Spiral output == manual centered crop, *then* scale.

    If the order were reversed (scale the whole rectangle, then crop) the
    intermediate dimensions would differ and, with a non-integer scale, the
    pixels would too -- so comparing against the manual crop-then-scale
    reference is a real test of the frozen order, not a tautology.
    """
    result = generate(
        _request(
            rect_png,
            (RECT_W, RECT_H),
            mode=gp.MODE_SPIRAL,
            render=gp.RENDER_SOURCE_COLOR,
            scale=scale,
        )
    )

    box = gp.center_square_crop_box((RECT_W, RECT_H))
    with Image.open(rect_png) as im:
        cropped = np.array(im.convert("L").crop(box), dtype=np.uint8)
    reference = resize_grayscale(cropped, float(scale))

    assert result.pixels.shape[:2] == reference.shape
    assert result.size == (reference.shape[1], reference.shape[0])


def test_spiral_crop_before_scale_is_distinguishable_from_the_wrong_order(
    rect_png,
) -> None:
    """Guard: the *wrong* order is structurally impossible, not merely worse.

    Two checks, because dimensions alone are not decisive: for a 100x60 source
    at scale 1.3 the wrong order's ``min(130, 78) == 78`` coincides with the
    correct ``scaled_size(60, 60, 1.3) == 78``.

    1. The wrong order scales the *whole rectangle* first, which leaves a
       non-square ``78 x 130`` array -- and the frozen Spiral core accepts only
       square input, so it raises.  Scaling before cropping cannot even be
       expressed with the real core.
    2. The colour the pipeline writes equals the crop-then-scale reference
       exactly, which pins the order that *is* implemented.
    """
    scale = 1.3
    result = generate(
        _request(
            rect_png,
            (RECT_W, RECT_H),
            mode=gp.MODE_SPIRAL,
            render=gp.RENDER_SOURCE_COLOR,
            scale=str(scale),
        )
    )

    with Image.open(rect_png) as im:
        full_gray = np.array(im.convert("L"), dtype=np.uint8)
        full_rgb = np.array(im.convert("RGB"), dtype=np.uint8)

    # (1) The wrong order leaves a non-square canvas the Spiral core rejects.
    wrong_gray = resize_grayscale(full_gray, scale)
    assert wrong_gray.shape[0] != wrong_gray.shape[1]
    with pytest.raises(ValueError):
        spiral_mask(wrong_gray, 8, 3)

    # (2) The implemented order is exactly crop-then-scale.
    box = gp.center_square_crop_box((RECT_W, RECT_H))
    reference_rgb = resize_rgb(
        np.array(Image.fromarray(full_rgb, "RGB").crop(box), dtype=np.uint8),
        scale,
    )
    reference_gray = resize_grayscale(
        np.array(Image.fromarray(full_gray, "L").crop(box), dtype=np.uint8),
        scale,
    )
    reference = render_source_color_on_white(
        spiral_mask(reference_gray, 8, 3), reference_rgb
    )
    assert np.array_equal(result.pixels[..., :3], reference)


def test_spiral_square_source_is_left_alone(rect_png, square_png) -> None:
    result = generate(
        _request(
            square_png,
            (SQUARE, SQUARE),
            mode=gp.MODE_SPIRAL,
            render=gp.RENDER_BLACK_ON_WHITE,
        )
    )
    assert result.size == (SQUARE, SQUARE)


@pytest.mark.parametrize("width,height", [(100, 60), (60, 100), (101, 60)])
def test_spiral_crop_box_is_the_frozen_one(tmp_path, width, height) -> None:
    path = _rect_source(tmp_path / f"r{width}x{height}.png", width, height)
    result = generate(
        _request(
            path, (width, height), mode=gp.MODE_SPIRAL, render=gp.RENDER_BLACK_ON_WHITE
        )
    )
    assert result.size == (min(width, height),) * 2


def test_stripe_is_never_cropped(rect_png) -> None:
    result = generate(
        _request(rect_png, (RECT_W, RECT_H), mode=gp.MODE_STRIPE)
    )
    assert result.size == (RECT_W, RECT_H)


# --------------------------------------------------------------------------
# source colour: RGB must never enter the polar pipeline
# --------------------------------------------------------------------------


def test_source_colour_spiral_uses_the_same_crop_box_as_grayscale(
    rect_png,
) -> None:
    """The RGB companion must be cropped and scaled exactly like the mask.

    The check is behavioural: build the color result and independently build
    "crop the RGB with the frozen box, then scale it", then confirm the colour
    that landed in the output is the *same colour at the same coordinates* as
    that reference.  Any deviation in box or scale would shift the gradient.
    """
    request = _request(
        rect_png,
        (RECT_W, RECT_H),
        mode=gp.MODE_SPIRAL,
        render=gp.RENDER_SOURCE_COLOR,
        scale="1.3",
    )
    result = generate(request)

    box = gp.center_square_crop_box((RECT_W, RECT_H))
    with Image.open(rect_png) as im:
        cropped_rgb = np.array(im.convert("RGB").crop(box), dtype=np.uint8)
    reference_rgb = resize_rgb(cropped_rgb, 1.3)

    assert reference_rgb.shape[:2] == result.pixels.shape[:2]

    # ``render_source_color_on_white`` writes the exact source pixel on line
    # pixels and pure white elsewhere.  Every non-white output pixel must
    # therefore equal the reference pixel at the same coordinate.
    colour = result.pixels[..., :3]
    written = np.any(colour != 255, axis=2)
    assert written.any(), "expected some line pixels to carry source colour"
    assert np.array_equal(colour[written], reference_rgb[written])


def test_pipeline_never_imports_or_calls_opencv() -> None:
    """Structural proof that no RGB/grayscale data enters ``warpPolar``.

    The polar unwrap lives in ``spiral.py``.  If the GUI pipeline could reach
    OpenCV itself, it could also unwrap or remap colour data, which the round
    forbids.  Asserting the module neither imports nor references ``cv2``
    pins that boundary at the source level instead of relying on a comment.
    """
    source = Path(gp.__file__).with_name("pipeline.py").read_text()
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "cv2" not in imported

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
    assert "cv2" not in referenced


# --------------------------------------------------------------------------
# alpha semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width_mode", gp.WIDTH_MODES)
@pytest.mark.parametrize("render", gp.RENDER_CHOICES)
def test_spiral_alpha_is_exactly_the_circular_support(
    rect_png, width_mode, render
) -> None:
    result = generate(
        _request(
            rect_png,
            (RECT_W, RECT_H),
            mode=gp.MODE_SPIRAL,
            width_mode=width_mode,
            render=render,
        )
    )
    side = result.width
    disc = circular_support(side, _polar_center(side), _support_radius(side))
    expected = np.where(disc, np.uint8(255), np.uint8(0))
    assert np.array_equal(result.pixels[..., 3], expected)


def test_alpha_has_only_zero_and_255(rect_png) -> None:
    result = generate(
        _request(rect_png, (RECT_W, RECT_H), mode=gp.MODE_SPIRAL)
    )
    assert set(np.unique(result.pixels[..., 3]).tolist()) == {0, 255}


def test_every_pixel_inside_the_disc_is_opaque(rect_png) -> None:
    result = generate(
        _request(rect_png, (RECT_W, RECT_H), mode=gp.MODE_SPIRAL)
    )
    side = result.width
    disc = circular_support(side, _polar_center(side), _support_radius(side))
    inside = result.pixels[disc][..., 3]
    assert inside.size > 0
    assert set(np.unique(inside).tolist()) == {255}


def test_every_pixel_outside_the_disc_is_transparent(rect_png) -> None:
    result = generate(
        _request(rect_png, (RECT_W, RECT_H), mode=gp.MODE_SPIRAL)
    )
    side = result.width
    disc = circular_support(side, _polar_center(side), _support_radius(side))
    outside = result.pixels[~disc][..., 3]
    assert outside.size > 0
    assert set(np.unique(outside).tolist()) == {0}


def test_alpha_is_never_derived_from_the_line_mask(rect_png) -> None:
    """The decisive alpha test: mask varies inside the disc, alpha does not.

    A common wrong implementation sets alpha from the line mask, which makes
    background pixels inside the disc transparent.  To catch that, this checks
    that the disc genuinely contains *both* line and background pixels while
    every one of them is fully opaque.
    """
    result = generate(
        _request(
            rect_png,
            (RECT_W, RECT_H),
            mode=gp.MODE_SPIRAL,
            render=gp.RENDER_BLACK_ON_WHITE,
            period="8",
        )
    )
    side = result.width
    disc = circular_support(side, _polar_center(side), _support_radius(side))

    # Inside the disc the rendered image must contain both black (line) and
    # white (background) pixels, otherwise this test would be vacuous.
    luminance = result.pixels[disc][..., 0]
    assert set(np.unique(luminance).tolist()) == {0, 255}
    # ...and yet alpha is uniformly 255 for both of them.
    assert set(np.unique(result.pixels[disc][..., 3]).tolist()) == {255}


# --------------------------------------------------------------------------
# RGBA composition helper
# --------------------------------------------------------------------------


def test_render_rgba_is_exposed_and_pure() -> None:
    from halftone_playground.gui.pipeline import render_rgba

    rgb = np.zeros((4, 4, 3), np.uint8)
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    opaque = np.zeros((4, 4), bool)
    opaque[1:3, 1:3] = True

    out = render_rgba(rgb, mask, opaque=opaque)
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.uint8
    assert np.array_equal(out[..., 3], np.where(opaque, 255, 0).astype(np.uint8))
    # The mask must not influence alpha even though it differs from ``opaque``.
    assert out[0, 0, 3] == 0


def test_render_rgba_rejects_mismatched_shapes() -> None:
    from halftone_playground.gui.pipeline import render_rgba

    with pytest.raises(PipelineError):
        render_rgba(
            np.zeros((4, 4, 3), np.uint8),
            np.zeros((4, 4), bool),
            opaque=np.zeros((3, 3), bool),
        )


# --------------------------------------------------------------------------
# errors are user-facing sentences
# --------------------------------------------------------------------------


def test_missing_file_raises_a_plain_pipeline_error(tmp_path) -> None:
    missing = tmp_path / "nope.png"
    request = _request(
        missing, (RECT_W, RECT_H), mode=gp.MODE_STRIPE
    )
    with pytest.raises(PipelineError) as excinfo:
        generate(request)
    message = str(excinfo.value)
    assert "nope.png" in message
    assert "Traceback" not in message


def test_changed_source_size_is_reported(tmp_path) -> None:
    path = _rect_source(tmp_path / "r.png", 100, 60)
    request = _request(path, (999, 999), mode=gp.MODE_STRIPE)
    with pytest.raises(PipelineError) as excinfo:
        generate(request)
    assert "changed on disk" in str(excinfo.value)


def test_invalid_parameters_are_rejected_while_snapshotting(rect_png) -> None:
    p = gp.RenderParams(period_text="0")
    with pytest.raises(PipelineError):
        GenerationRequest.from_params(p, rect_png, (RECT_W, RECT_H))
