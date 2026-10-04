"""Command-line interface for halftone-playground.

This module is a *thin* driver on top of the frozen geometry and renderer
cores.  It performs no image algorithm of its own: it loads an input with
Pillow, picks one of the existing cores, hands the resulting mask to one of the
existing renderers and saves the result as a PNG.

Two subcommands exist, mirroring the two halftone modes::

    halftone-playground stripe INPUT OUTPUT [options]
    halftone-playground spiral INPUT OUTPUT [options]

The same behaviour is available as a module::

    python -m halftone_playground stripe INPUT OUTPUT [options]

Variable-width vs fixed-width
-----------------------------

There is deliberately no ``--width-mode`` flag.  A single, blunt rule decides
which geometry core runs:

===========================  ===========================================
``--line-width`` absent      variable-width geometry (``stripe_mask`` /
                             ``spiral_mask``); the local grayscale drives
                             the width of every line.
``--line-width N`` present   fixed-width geometry (``stripe_fixed_mask`` /
                             ``spiral_fixed_mask``); every full cell draws
                             exactly ``N`` pixels and the source luminance
                             is **not** consulted for the width at all.
===========================  ===========================================

With ``--line-width`` the input image content only still decides the canvas
size -- and, for ``--render source-color``, the RGB values painted inside the
lines.  Fixed-width plus ``black-on-white`` / ``white-on-black`` therefore
produces pure geometry: the same shape regardless of what the picture holds.

Boundaries of this round
------------------------

* output is **PNG only** (strict bi-level and exact source-RGB pixels must
  survive a round trip, so lossy formats are rejected);
* ``spiral`` needs a **square** input; no automatic crop / fit / pad exists;
* no batch processing, no directories, no GUI;
* ``image_scale`` is passed through untouched: it is never capped and never
  adjusts ``--period`` or ``--line-width``.

Exit codes
----------

``0``
    Success.
``1``
    Runtime failure (reading the input, decoding, writing the output).
``2``
    Usage / validation failure (bad arguments, non-PNG output, non-square
    spiral input, an output that would overwrite the input, ...).  argparse's
    own parse errors also exit ``2``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import __version__
from .preprocess import resize_grayscale, resize_rgb, scaled_size
from .render import (
    render_black_on_white,
    render_source_color_on_white,
    render_white_on_black,
)
from .spiral import spiral_fixed_mask, spiral_mask
from .stripe import stripe_fixed_mask, stripe_mask

__all__ = ["build_parser", "main"]

PROG = "halftone-playground"

#: ``--render`` value -> renderer callable.  ``source-color`` is the only
#: variant that additionally needs the scaled Cartesian RGB source.
_RENDERERS = {
    "black-on-white": render_black_on_white,
    "white-on-black": render_white_on_black,
    "source-color": None,  # handled explicitly: needs the RGB source
}

_RENDER_CHOICES = tuple(_RENDERERS)

DEFAULT_PERIOD = 16
DEFAULT_SCALE = 1.0
DEFAULT_RENDER = "black-on-white"
DEFAULT_ANGLE = 90.0
DEFAULT_ARMS = 1

#: Errors the CLI turns into a friendly ``error: ...`` message instead of a
#: traceback.  A programming bug (anything else) is deliberately *not* caught,
#: so it still surfaces with its stack trace.
_USER_ERRORS = (ValueError, TypeError, OSError, UnidentifiedImageError)


class _CliError(Exception):
    """A validation / usage problem: message on stderr, exit code 2."""


class _RuntimeError(Exception):
    """An I/O or image failure: message on stderr, exit code 1."""


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with its two subcommands."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Stripe and Spiral line halftoning with variable- or fixed-width "
            "geometry and multiple render variants.  Geometry produces a "
            "boolean mask; a renderer turns that mask into pixels."
        ),
        epilog=(
            "examples:\n"
            f"  {PROG} stripe input.jpg output.png\n"
            f"  {PROG} stripe input.jpg output.png --period 16 --angle 45 "
            "--scale 1.3 --render source-color\n"
            f"  {PROG} stripe input.jpg output.png --period 16 --line-width 5 "
            "--angle 90 --render source-color\n"
            f"  {PROG} spiral square.jpg output.png --period 16 --arms 3\n"
            f"  {PROG} spiral square.jpg output.png --period 16 --line-width 5 "
            "--arms 3 --render source-color\n"
            f"  python -m halftone_playground stripe input.jpg output.png\n"
            "\n"
            "output is always PNG; spiral mode requires a square input"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROG} {__version__}",
    )

    subparsers = parser.add_subparsers(dest="mode", metavar="{stripe,spiral}")
    subparsers.required = True

    subparsers.add_parser(
        "stripe",
        help="draw variable- or fixed-width straight stripes",
        description=(
            "Draw straight stripes.  Without --line-width the local luminance "
            "sets each line's width (variable-width); with --line-width N every "
            "full cell draws exactly N pixels and the luminance no longer "
            "affects the width (fixed-width)."
        ),
        parents=[_common_parent(), _stripe_parent()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers.add_parser(
        "spiral",
        help="draw variable- or fixed-width spiral arms (square input)",
        description=(
            "Draw spiral arms from a square input.  Without --line-width the "
            "local luminance sets each line's width (variable-width); with "
            "--line-width N every full cell draws exactly N pixels and the "
            "luminance no longer affects the width (fixed-width).  No "
            "automatic crop / fit is implemented, so a non-square input is "
            "rejected."
        ),
        parents=[_common_parent(), _spiral_parent()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser


def _common_parent() -> argparse.ArgumentParser:
    """Options shared by ``stripe`` and ``spiral``."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("input", metavar="INPUT", help="input image (e.g. PNG, JPEG)")
    parent.add_argument("output", metavar="OUTPUT", help="output PNG path")
    parent.add_argument(
        "--period",
        type=int,
        default=DEFAULT_PERIOD,
        metavar="INT",
        help="distance between adjacent line centres, in pixels (default: 16)",
    )
    parent.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        metavar="FLOAT",
        help=(
            "rescale the source image by this factor before geometry runs; "
            "any positive finite value, no upper limit (default: 1.0).  The "
            "scale never adjusts --period or --line-width"
        ),
    )
    parent.add_argument(
        "--render",
        choices=_RENDER_CHOICES,
        default=DEFAULT_RENDER,
        metavar="CHOICE",
        help=(
            "output variant: "
            + ", ".join(_RENDER_CHOICES)
            + " (default: black-on-white)"
        ),
    )
    parent.add_argument(
        "--line-width",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "constant line width in pixels.  When given, geometry switches to "
            "fixed-width: every full cell draws exactly INT pixels and the "
            "source luminance no longer affects the line width.  When omitted, "
            "geometry is variable-width and the luminance drives the width"
        ),
    )
    return parent


def _stripe_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--angle",
        type=float,
        default=DEFAULT_ANGLE,
        metavar="FLOAT",
        help=(
            "stripe orientation in degrees: 90 = vertical, 0/180 = horizontal "
            "(default: 90)"
        ),
    )
    return parent


def _spiral_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--arms",
        type=int,
        default=DEFAULT_ARMS,
        metavar="INT",
        help="number of spiral arms (default: 1)",
    )
    return parent


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.  ``argv`` defaults to ``sys.argv[1:]``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _run(args)
    except _CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except _RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except _USER_ERRORS as exc:
        # A core validation guard (bad period / angle / arms / scale / line
        # width / size) is a user-input problem, not a crash.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    _validate_paths(input_path, output_path)

    try:
        with Image.open(input_path) as image:
            image.load()
            n_frames = getattr(image, "n_frames", 1)
            if n_frames != 1:
                raise _CliError(
                    f"input has {n_frames} frames; multi-frame images are not "
                    "supported in this round -- please supply a single-frame "
                    "image"
                )
            width, height = image.size
            if args.mode == "spiral" and width != height:
                raise _CliError(
                    "Spiral mode currently requires a square input image; "
                    "automatic crop/fit is not implemented."
                )
            target_width, target_height = _target_size(width, height, args.scale)
            source = image.copy()
    except _CliError:
        raise
    except UnidentifiedImageError as exc:
        raise _RuntimeError(
            f"cannot read input image '{input_path}': {exc}"
        ) from exc
    except OSError as exc:
        raise _RuntimeError(
            f"cannot read input image '{input_path}': {exc}"
        ) from exc

    mask, scaled_rgb = _build_mask(args, source, (target_width, target_height))

    if args.render == "source-color":
        assert scaled_rgb is not None
        rendered = render_source_color_on_white(mask, scaled_rgb)
        mode = "RGB"
    else:
        rendered = _RENDERERS[args.render](mask)  # type: ignore[operator]
        mode = "L"

    _print_summary(args, (width, height), (target_width, target_height))
    _save_png(output_path, rendered, mode)
    print(f"Saved: {output_path}")


# --------------------------------------------------------------------------
# geometry / renderer routing (all calls go to the frozen cores)
# --------------------------------------------------------------------------


def _build_mask(
    args: argparse.Namespace,
    source: Image.Image,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pick the geometry core and return ``(mask, scaled_rgb_or_None)``."""
    target_width, target_height = target_size
    needs_color = args.render == "source-color"

    if args.line_width is not None:
        # Fixed-width geometry never reads the grayscale: the canvas size is
        # all it needs, so no pointless gray conversion / resize happens.
        assert args.mode == "stripe" or target_width == target_height
        if args.mode == "stripe":
            mask = stripe_fixed_mask(
                (target_height, target_width),
                args.period,
                args.line_width,
                args.angle,
            )
        else:
            mask = spiral_fixed_mask(
                target_width, args.period, args.line_width, args.arms
            )
    else:
        gray = np.array(source.convert("L"), dtype=np.uint8)
        gray = resize_grayscale(gray, args.scale)
        if args.mode == "stripe":
            mask = stripe_mask(gray, args.period, args.angle)
        else:
            mask = spiral_mask(gray, args.period, args.arms)

    scaled_rgb = None
    if needs_color:
        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        scaled_rgb = resize_rgb(rgb, args.scale)
    return mask, scaled_rgb


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise _RuntimeError(f"input file does not exist: {input_path}")
    if not input_path.is_file():
        raise _RuntimeError(f"input is not a file: {input_path}")
    if output_path.suffix.lower() != ".png":
        raise _CliError(
            "output must be a .png file (strict bi-level and exact source-RGB "
            f"pixels require a lossless format); got '{output_path}'"
        )
    try:
        same = input_path.resolve() == output_path.resolve()
    except OSError:
        same = False
    if same:
        raise _CliError(
            f"input and output resolve to the same file: {input_path}; "
            "refusing to overwrite the input"
        )


def _target_size(width: int, height: int, scale: float) -> tuple[int, int]:
    try:
        return scaled_size(width, height, scale)
    except _USER_ERRORS as exc:
        raise _CliError(str(exc)) from exc


def _save_png(path: Path, array: np.ndarray, mode: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _RuntimeError(f"cannot create output directory for '{path}': {exc}")

    try:
        Image.fromarray(array, mode=mode).save(path, format="PNG")
    except OSError as exc:
        raise _RuntimeError(f"cannot write output '{path}': {exc}") from exc


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _print_summary(
    args: argparse.Namespace,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> None:
    geometry = "fixed" if args.line_width is not None else "variable"
    print(f"Mode: {args.mode}")
    print(f"Geometry: {geometry}")
    if args.line_width is not None:
        print(f"Line width: {args.line_width}")
    print(f"Period: {args.period}")
    if args.mode == "stripe":
        print(f"Angle: {_format_number(args.angle)}")
    else:
        print(f"Arms: {args.arms}")
    print(f"Scale: {_format_number(args.scale)}")
    print(f"Render: {args.render}")
    print(f"Input: {source_size[0]}x{source_size[1]}")
    print(f"Output: {target_size[0]}x{target_size[1]}")


def _format_number(value: float) -> str:
    """Render a number without a trailing ``.0`` for integral values."""
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))
