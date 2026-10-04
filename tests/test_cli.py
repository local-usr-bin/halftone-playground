"""Tests for the command-line interface (``halftone_playground.cli``).

The CLI must be a *pure driver*: everything it produces has to be byte-for-byte
reproducible by calling the frozen geometry / renderer cores directly.  These
tests therefore check both the argument surface (help, defaults, routing) and
real end-to-end behaviour on tiny images in ``tmp_path``.

The parser tests do not stop at inspecting an ``argparse.Namespace`` -- the
important cases (variable vs fixed routing, renderer selection, scale) are
asserted again through the actual output pixels.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halftone_playground import __version__
from halftone_playground.cli import build_parser, main
from halftone_playground.preprocess import resize_grayscale, resize_rgb, scaled_size
from halftone_playground.render import (
    render_black_on_white,
    render_source_color_on_white,
    render_white_on_black,
)
from halftone_playground.spiral import spiral_fixed_mask, spiral_mask
from halftone_playground.stripe import stripe_fixed_mask, stripe_mask

SRC = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def _gradient(height: int, width: int, horizontal: bool = True) -> np.ndarray:
    """A deterministic left-to-right (or top-to-bottom) grayscale ramp."""
    ramp = np.linspace(0, 255, width if horizontal else height)
    if horizontal:
        tiled = np.tile(ramp, (height, 1))
    else:
        tiled = np.tile(ramp[:, None], (1, width))
    return np.clip(tiled, 0, 255).astype(np.uint8)


def _rgb_gradient(height: int, width: int) -> np.ndarray:
    """A deterministic RGB image with all three channels varying."""
    r = np.linspace(0, 255, width, dtype=np.uint8)
    g = np.linspace(255, 0, width, dtype=np.uint8)
    b = np.linspace(0, 128, width, dtype=np.uint8)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = r[None, :]
    rgb[..., 1] = g[None, :]
    rgb[..., 2] = b[None, :]
    return rgb


def _save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="L").save(path)


def _save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_module(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke ``python -m halftone_playground`` in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-m", "halftone_playground", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"},
    )


# --------------------------------------------------------------------------
# parser surface
# --------------------------------------------------------------------------


def test_top_level_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "stripe" in out and "spiral" in out


def test_stripe_help_mentions_line_width_semantics(capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["stripe", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--line-width" in out
    assert "fixed-width" in out
    assert "variable-width" in out


def test_spiral_help_mentions_arms_and_square(capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["spiral", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--arms" in out
    assert "--line-width" in out
    assert "square" in out


def test_subcommand_is_required():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([])
    assert excinfo.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_stripe_defaults():
    args = build_parser().parse_args(["stripe", "in.png", "out.png"])
    assert args.mode == "stripe"
    assert args.input == "in.png"
    assert args.output == "out.png"
    assert args.period == 16
    assert args.scale == 1.0
    assert args.render == "black-on-white"
    assert args.line_width is None
    assert args.angle == 90.0


def test_spiral_defaults():
    args = build_parser().parse_args(["spiral", "in.png", "out.png"])
    assert args.mode == "spiral"
    assert args.period == 16
    assert args.scale == 1.0
    assert args.render == "black-on-white"
    assert args.line_width is None
    assert args.arms == 1


def test_render_choices_reject_unknown():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(
            ["stripe", "in.png", "out.png", "--render", "sepia"]
        )
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "render", ["black-on-white", "white-on-black", "source-color"]
)
def test_render_choices_accepted(render):
    args = build_parser().parse_args(
        ["stripe", "in.png", "out.png", "--render", render]
    )
    assert args.render == render


@pytest.mark.parametrize("value", ["0.5", "1.0", "1.3", "2.75", "100"])
def test_scale_parsing(value):
    args = build_parser().parse_args(
        ["stripe", "in.png", "out.png", "--scale", value]
    )
    assert args.scale == float(value)


@pytest.mark.parametrize("value", ["0", "90", "45", "180", "-30", "359.5"])
def test_angle_parsing(value):
    args = build_parser().parse_args(
        ["stripe", "in.png", "out.png", "--angle", value]
    )
    assert args.angle == float(value)


@pytest.mark.parametrize("value", ["1", "3", "12"])
def test_arms_parsing(value):
    args = build_parser().parse_args(
        ["spiral", "in.png", "out.png", "--arms", value]
    )
    assert args.arms == int(value)


def test_line_width_presence_means_fixed():
    absent = build_parser().parse_args(["stripe", "in.png", "out.png"])
    present = build_parser().parse_args(
        ["stripe", "in.png", "out.png", "--line-width", "5"]
    )
    assert absent.line_width is None
    assert present.line_width == 5


@pytest.mark.parametrize("value", ["abc", "1.5", ""])
def test_invalid_period_rejected(value):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["stripe", "in.png", "out.png", "--period", value])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# Stripe: variable-width
# --------------------------------------------------------------------------


def test_stripe_variable_black_on_white_matches_core(tmp_path, capsys):
    source = _gradient(32, 48)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, source)

    for period in (8, 16):
        assert main(["stripe", str(inp), str(out), "--period", str(period)]) == 0
        with Image.open(out) as image:
            assert image.mode == "L"
            assert image.size == (48, 32)
            produced = np.array(image)

        expected_mask = stripe_mask(source, period, 90.0)
        assert np.array_equal(produced, render_black_on_white(expected_mask))

    stdout = capsys.readouterr().out
    assert "Mode: stripe" in stdout
    assert "Geometry: variable" in stdout
    assert "Input: 48x32" in stdout
    assert "Output: 48x32" in stdout


def test_stripe_variable_angle_is_forwarded(tmp_path):
    source = _gradient(30, 30)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, source)

    assert main(["stripe", str(inp), str(out), "--angle", "45", "--period", "10"]) == 0
    with Image.open(out) as image:
        produced = np.array(image)
    expected = render_black_on_white(stripe_mask(source, 10, 45.0))
    assert np.array_equal(produced, expected)


def test_stripe_invert_variants_are_complementary(tmp_path):
    source = _gradient(24, 40)
    inp = tmp_path / "in.png"
    black = tmp_path / "black.png"
    white = tmp_path / "white.png"
    _save_gray(inp, source)

    assert main(["stripe", str(inp), str(black), "--period", "8"]) == 0
    assert (
        main(
            [
                "stripe",
                str(inp),
                str(white),
                "--period",
                "8",
                "--render",
                "white-on-black",
            ]
        )
        == 0
    )

    with Image.open(black) as a, Image.open(white) as b:
        arr_a = np.array(a)
        arr_b = np.array(b)
    assert np.array_equal(arr_a.astype(np.int32) + arr_b.astype(np.int32), np.full_like(arr_a, 255, dtype=np.int32))
    # both are still strict bi-level
    assert set(np.unique(arr_a)).issubset({0, 255})
    assert set(np.unique(arr_b)).issubset({0, 255})


def test_stripe_source_color_copies_exact_rgb(tmp_path):
    source = _rgb_gradient(20, 36)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_rgb(inp, source)

    assert (
        main(
            ["stripe", str(inp), str(out), "--period", "12", "--render", "source-color"]
        )
        == 0
    )
    with Image.open(out) as image:
        assert image.mode == "RGB"
        produced = np.array(image)

    mask = stripe_mask(_to_gray(source), 12, 90.0)
    assert np.array_equal(produced[mask], source[mask])
    assert np.all(produced[~mask] == 255)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Reproduce Pillow's ``convert("L")`` for a known RGB array."""
    return np.array(Image.fromarray(rgb, mode="RGB").convert("L"), dtype=np.uint8)


# --------------------------------------------------------------------------
# Stripe: fixed-width
# --------------------------------------------------------------------------


def test_stripe_fixed_routes_to_fixed_core(tmp_path, capsys):
    source = _gradient(32, 48)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, source)

    assert (
        main(["stripe", str(inp), str(out), "--period", "16", "--line-width", "5"]) == 0
    )
    with Image.open(out) as image:
        produced = np.array(image)

    expected = render_black_on_white(
        stripe_fixed_mask((32, 48), 16, 5, 90.0)
    )
    assert np.array_equal(produced, expected)
    # fixed-width geometry ignores the luminance entirely: a constant image
    # yields the same mask as the ramp above.
    constant = np.full((32, 48), 7, dtype=np.uint8)
    assert np.array_equal(
        stripe_fixed_mask((32, 48), 16, 5, 90.0),
        stripe_fixed_mask((32, 48), 16, 5, 90.0),
    )
    del constant

    stdout = capsys.readouterr().out
    assert "Geometry: fixed" in stdout
    assert "Line width: 5" in stdout


def test_stripe_fixed_differs_from_variable(tmp_path):
    source = _gradient(32, 48)
    inp = tmp_path / "in.png"
    var_out = tmp_path / "var.png"
    fix_out = tmp_path / "fix.png"
    _save_gray(inp, source)

    assert main(["stripe", str(inp), str(var_out), "--period", "16"]) == 0
    assert (
        main(
            ["stripe", str(inp), str(fix_out), "--period", "16", "--line-width", "5"]
        )
        == 0
    )
    with Image.open(var_out) as a, Image.open(fix_out) as b:
        assert not np.array_equal(np.array(a), np.array(b))


def test_stripe_fixed_source_color(tmp_path):
    source = _rgb_gradient(24, 40)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_rgb(inp, source)

    assert (
        main(
            [
                "stripe",
                str(inp),
                str(out),
                "--period",
                "16",
                "--line-width",
                "5",
                "--render",
                "source-color",
            ]
        )
        == 0
    )
    with Image.open(out) as image:
        assert image.mode == "RGB"
        produced = np.array(image)
    mask = stripe_fixed_mask((24, 40), 16, 5, 90.0)
    assert np.array_equal(produced[mask], source[mask])
    assert np.all(produced[~mask] == 255)


# --------------------------------------------------------------------------
# Spiral
# --------------------------------------------------------------------------


def test_spiral_variable_matches_core(tmp_path):
    source = _gradient(40, 40)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, source)

    assert (main(["spiral", str(inp), str(out), "--period", "12", "--arms", "3"]) == 0)
    with Image.open(out) as image:
        assert image.mode == "L"
        assert image.size == (40, 40)
        produced = np.array(image)
    assert np.array_equal(
        produced, render_black_on_white(spiral_mask(source, 12, 3))
    )


def test_spiral_fixed_routes_to_fixed_core(tmp_path, capsys):
    source = _gradient(40, 40)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, source)

    assert (
        main(
            [
                "spiral",
                str(inp),
                str(out),
                "--period",
                "12",
                "--line-width",
                "4",
                "--arms",
                "2",
            ]
        )
        == 0
    )
    with Image.open(out) as image:
        produced = np.array(image)
    assert np.array_equal(
        produced, render_black_on_white(spiral_fixed_mask(40, 12, 4, 2))
    )
    stdout = capsys.readouterr().out
    assert "Geometry: fixed" in stdout
    assert "Arms: 2" in stdout


def test_spiral_fixed_source_color(tmp_path):
    source = _rgb_gradient(36, 36)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_rgb(inp, source)

    assert (
        main(
            [
                "spiral",
                str(inp),
                str(out),
                "--period",
                "12",
                "--line-width",
                "4",
                "--arms",
                "2",
                "--render",
                "source-color",
            ]
        )
        == 0
    )
    with Image.open(out) as image:
        assert image.mode == "RGB"
        produced = np.array(image)
    mask = spiral_fixed_mask(36, 12, 4, 2)
    assert np.array_equal(produced[mask], source[mask])
    assert np.all(produced[~mask] == 255)


def test_spiral_non_square_rejected(tmp_path, capsys):
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, np.zeros((73, 101), dtype=np.uint8))

    assert main(["spiral", str(inp), str(out)]) == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "square" in err
    assert "crop" in err


# --------------------------------------------------------------------------
# scale
# --------------------------------------------------------------------------


def test_scale_1p3_reports_and_produces_expected_size(tmp_path, capsys):
    source = _gradient(73, 101)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, source)

    assert main(["stripe", str(inp), str(out), "--scale", "1.3"]) == 0
    stdout = capsys.readouterr().out
    assert "Input: 101x73" in stdout
    assert "Output: 131x95" in stdout
    with Image.open(out) as image:
        assert image.size == (131, 95)
        produced = np.array(image)

    scaled_gray = resize_grayscale(source, 1.3)
    assert scaled_gray.shape == (95, 131)
    assert np.array_equal(
        produced, render_black_on_white(stripe_mask(scaled_gray, 16, 90.0))
    )


def test_scale_does_not_touch_period_or_line_width(tmp_path):
    source = _gradient(32, 32)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, source)

    assert (
        main(
            [
                "stripe",
                str(inp),
                str(out),
                "--scale",
                "2",
                "--period",
                "16",
                "--line-width",
                "5",
            ]
        )
        == 0
    )
    with Image.open(out) as image:
        produced = np.array(image)
    # period 16 / width 5, NOT 32 / 10
    assert np.array_equal(
        produced, render_black_on_white(stripe_fixed_mask((64, 64), 16, 5, 90.0))
    )


def test_scale_source_color_uses_scaled_rgb(tmp_path):
    source = _rgb_gradient(40, 40)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_rgb(inp, source)

    assert (
        main(
            [
                "stripe",
                str(inp),
                str(out),
                "--scale",
                "1.5",
                "--period",
                "12",
                "--render",
                "source-color",
            ]
        )
        == 0
    )
    with Image.open(out) as image:
        produced = np.array(image)
        assert image.size == (60, 60)

    scaled_rgb = resize_rgb(source, 1.5)
    scaled_gray = resize_grayscale(_to_gray(source), 1.5)
    mask = stripe_mask(scaled_gray, 12, 90.0)
    assert np.array_equal(produced[mask], scaled_rgb[mask])
    assert np.all(produced[~mask] == 255)


# --------------------------------------------------------------------------
# output handling
# --------------------------------------------------------------------------


def test_output_extension_must_be_png(tmp_path, capsys):
    inp = tmp_path / "in.png"
    _save_gray(inp, _gradient(16, 16))
    out = tmp_path / "out.jpg"

    assert main(["stripe", str(inp), str(out)]) == 2
    assert not out.exists()
    assert ".png" in capsys.readouterr().err


def test_output_extension_case_insensitive_png(tmp_path):
    inp = tmp_path / "in.png"
    out = tmp_path / "out.PNG"
    _save_gray(inp, _gradient(16, 16))

    assert main(["stripe", str(inp), str(out)]) == 0
    assert out.exists()
    with Image.open(out) as image:
        assert image.format == "PNG"


def test_same_input_output_rejected_and_input_untouched(tmp_path, capsys):
    inp = tmp_path / "in.png"
    _save_gray(inp, _gradient(16, 16))
    before = _sha256(inp)

    assert main(["stripe", str(inp), str(inp)]) == 2
    assert _sha256(inp) == before
    assert "same file" in capsys.readouterr().err


def test_output_parent_directories_are_created(tmp_path):
    inp = tmp_path / "in.png"
    _save_gray(inp, _gradient(16, 16))
    out = tmp_path / "a" / "b" / "c" / "out.png"
    assert not out.parent.exists()

    assert main(["stripe", str(inp), str(out)]) == 0
    assert out.exists()


def test_existing_output_is_overwritten(tmp_path):
    inp = tmp_path / "in.png"
    _save_gray(inp, _gradient(16, 16))
    out = tmp_path / "out.png"
    out.write_bytes(b"stale")
    stale = _sha256(out)

    assert main(["stripe", str(inp), str(out)]) == 0
    assert _sha256(out) != stale
    with Image.open(out) as image:
        assert image.format == "PNG"


def test_input_file_is_never_mutated(tmp_path):
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, _gradient(24, 24))
    before = _sha256(inp)

    assert main(["stripe", str(inp), str(out), "--period", "8"]) == 0
    assert _sha256(inp) == before


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------


def test_missing_input_fails(tmp_path, capsys):
    out = tmp_path / "out.png"
    assert main(["stripe", str(tmp_path / "nope.png"), str(out)]) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_non_image_input_fails(tmp_path, capsys):
    bogus = tmp_path / "not_an_image.png"
    bogus.write_text("this is definitely not a PNG")
    out = tmp_path / "out.png"

    code = main(["stripe", str(bogus), str(out)])
    assert code in (1, 2)
    assert not out.exists()
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "extra",
    [
        ["--period", "0"],
        ["--period", "-4"],
        ["--period", "16", "--line-width", "0"],
        ["--period", "16", "--line-width", "99"],
        ["--scale", "0"],
        ["--scale", "-1"],
    ],
)
def test_invalid_values_exit_2(tmp_path, capsys, extra):
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, _gradient(16, 16))

    assert main(["stripe", str(inp), str(out), *extra]) == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_invalid_arms_exit_2(tmp_path, capsys):
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, _gradient(16, 16))

    assert main(["spiral", str(inp), str(out), "--arms", "0"]) == 2
    assert not out.exists()
    assert "Traceback" not in capsys.readouterr().err


# --------------------------------------------------------------------------
# module invocation + console-script entry
# --------------------------------------------------------------------------


def test_module_help_smoke():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "stripe" in result.stdout and "spiral" in result.stdout


def test_module_invocation_end_to_end(tmp_path):
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, _gradient(24, 24))

    result = _run_module("stripe", str(inp), str(out), "--period", "8")
    assert result.returncode == 0, result.stderr
    assert out.exists()
    with Image.open(out) as image:
        assert image.mode == "L"


def test_pyproject_declares_console_script():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "[project.scripts]" in pyproject
    assert 'halftone-playground = "halftone_playground.cli:main"' in pyproject


def test_main_is_importable_from_cli_module():
    import halftone_playground.cli as cli

    assert callable(cli.main)
    assert callable(cli.build_parser)


def test_cli_is_not_re_exported_from_package():
    import halftone_playground as package

    # The CLI stays an internal module: the geometry API is what the package
    # exposes, and no CLI symbol leaks into the top-level namespace.
    assert "build_parser" not in package.__all__
    assert "main" not in package.__all__
    assert "cli" not in package.__all__


def test_scaled_size_target_matches_cli_report(tmp_path, capsys):
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _save_gray(inp, np.zeros((73, 101), dtype=np.uint8))

    assert main(["stripe", str(inp), str(out), "--scale", "1.3"]) == 0
    assert scaled_size(101, 73, 1.3) == (131, 95)
    assert "Output: 131x95" in capsys.readouterr().out
