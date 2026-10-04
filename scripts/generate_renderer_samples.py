#!/usr/bin/env python3
"""Generate the representative renderer / compositor sample images.

These PNGs are the *tracked* visual artifacts for Renderer / Compositor Round
1.  They show the same geometry rendered three ways:

1. white background + black line   (``render_black_on_white``)
2. black background + white line   (``render_white_on_black``)
3. white background + source RGB   (``render_source_color_on_white``)

for both geometry modes (Stripe and Spiral).

Everything here is deterministic: the synthetic RGB source is a fixed
programmatic colour field with no randomness, so the outputs are stable byte
for byte.  Regenerating twice must leave ``git diff -- samples/render`` empty.

Run from the repository root:

    python scripts/generate_renderer_samples.py

Outputs land in ``samples/render/``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from halftone_playground import (  # noqa: E402
    render_black_on_white,
    render_source_color_on_white,
    render_white_on_black,
    spiral_mask,
    stripe_mask,
)

STRIPE_SIZE = 256
STRIPE_PERIOD = 16
STRIPE_ANGLE = 90.0

SPIRAL_SIZE = 512
SPIRAL_PERIOD = 16
SPIRAL_ARMS = 3

OUTPUT_DIR = REPO_ROOT / "samples" / "render"

_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def stripe_gray(size: int = STRIPE_SIZE) -> np.ndarray:
    """A deterministic horizontal grayscale ramp for Stripe mode."""
    ramp = np.linspace(0, 255, size, dtype=np.uint8)
    return np.tile(ramp[None, :], (size, 1))


def spiral_gray(size: int = SPIRAL_SIZE) -> np.ndarray:
    """A deterministic radial grayscale ramp for Spiral mode.

    White at the centre, black towards the rim, so the gray -> line-width
    response shows up as a continuous widening of the arms.
    """
    centre = (size - 1) / 2.0
    coords = np.arange(size, dtype=np.float64)
    dx = coords - centre
    dy = coords - centre
    radius = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    ramp = 255.0 * (1.0 - radius / radius.max())
    return np.clip(ramp, 0.0, 255.0).astype(np.uint8)


def color_field(size: int) -> np.ndarray:
    """A deterministic RGB source in which every region reads differently.

    ``R`` grows with ``x``, ``G`` grows with ``y`` and ``B`` is a third fixed
    function of both.  No randomness is used, so the tracked samples are fully
    reproducible.
    """
    ys = np.arange(size, dtype=np.int64)[:, None]
    xs = np.arange(size, dtype=np.int64)[None, :]
    r = (xs * 255 // (size - 1)).astype(np.uint8)
    g = (ys * 255 // (size - 1)).astype(np.uint8)
    b = ((xs * 7 + ys * 11) % 256).astype(np.uint8)
    return np.stack(
        [
            np.broadcast_to(r, (size, size)),
            np.broadcast_to(g, (size, size)),
            b,
        ],
        axis=-1,
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_binary(image: np.ndarray, label: str) -> None:
    unique = set(np.unique(image).tolist())
    assert unique <= {0, 255}, f"{label} is not strictly bi-level: {sorted(unique)}"
    assert image.dtype == np.uint8, f"{label} is not uint8"


def _check_color(image: np.ndarray, mask: np.ndarray, rgb: np.ndarray, label: str) -> None:
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    assert (image[~mask] == _WHITE).all(), f"{label} background is not pure white"
    assert np.array_equal(image[mask], rgb[mask]), f"{label} line colour is not exact"


def write_pair(
    stem: str,
    mask: np.ndarray,
    rgb: np.ndarray,
    report: list[tuple[str, str, str]],
) -> None:
    """Write the three variants for one geometry and record their hashes."""
    binary = {
        "black_on_white": render_black_on_white(mask),
        "white_on_black": render_white_on_black(mask),
    }
    color = render_source_color_on_white(mask, rgb)

    _check_binary(binary["black_on_white"], f"{stem} black_on_white")
    _check_binary(binary["white_on_black"], f"{stem} white_on_black")
    _check_color(color, mask, rgb, f"{stem} source_color_on_white")

    for suffix, image in binary.items():
        path = OUTPUT_DIR / f"{stem}_{suffix}.png"
        Image.fromarray(image, mode="L").save(path, format="PNG")
        report.append((path.name, "L", sha256_of(path)))

    path = OUTPUT_DIR / f"{stem}_source_color_on_white.png"
    Image.fromarray(color, mode="RGB").save(path, format="PNG")
    report.append((path.name, "RGB", sha256_of(path)))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report: list[tuple[str, str, str]] = []

    stripe_mask_arr = stripe_mask(
        stripe_gray(), STRIPE_PERIOD, STRIPE_ANGLE
    )
    write_pair(
        "stripe", stripe_mask_arr, color_field(STRIPE_SIZE), report
    )

    spiral_mask_arr = spiral_mask(spiral_gray(), SPIRAL_PERIOD, SPIRAL_ARMS)
    write_pair(
        "spiral", spiral_mask_arr, color_field(SPIRAL_SIZE), report
    )

    for name, mode, digest in report:
        path = OUTPUT_DIR / name
        print(f"{path.relative_to(REPO_ROOT)}  mode={mode}  sha256={digest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
