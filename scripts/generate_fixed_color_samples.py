#!/usr/bin/env python3
"""Generate the tracked fixed-width / source-colour sample images.

These PNGs are the *tracked* visual artifacts for the Fixed-Width Color Round.
They show fixed-width geometry (line width chosen by the user, never derived
from grayscale) filled with the source image's own colour:

1. Stripe fixed colour        -- P=16, w=3, angle=90
2. Stripe fixed colour        -- P=16, w=7, angle=45
3. Spiral fixed colour arms1  -- P=16, w=3
4. Spiral fixed colour arms3  -- P=16, w=5
5. Spiral fixed colour arms12 -- P=16, w=7

The synthetic RGB source is a fixed programmatic colour field with no
randomness, so the outputs are stable byte for byte.  Regenerating twice must
leave ``git diff -- samples/fixed_color`` empty.

Deliberately kept separate and simple: this generator does not merge the
existing stripe / spiral / renderer generators into a single framework.

Run from the repository root:

    python scripts/generate_fixed_color_samples.py

Outputs land in ``samples/fixed_color/``.
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
    render_source_color_on_white,
    spiral_fixed_mask,
    stripe_fixed_mask,
)

PERIOD = 16
SIZE = 256
OUTPUT_DIR = REPO_ROOT / "samples" / "fixed_color"

_WHITE = np.array([255, 255, 255], dtype=np.uint8)


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


def _check_color(image: np.ndarray, mask: np.ndarray, rgb: np.ndarray, label: str) -> None:
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    assert (image[~mask] == _WHITE).all(), f"{label} background is not pure white"
    assert np.array_equal(image[mask], rgb[mask]), f"{label} line colour is not exact"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb = color_field(SIZE)

    report: list[tuple[str, str]] = []

    stripe_samples = [
        ("stripe_fixed_color_p16_w3_a90.png", 3, 90.0),
        ("stripe_fixed_color_p16_w7_a45.png", 7, 45.0),
    ]
    for name, line_width, angle in stripe_samples:
        mask = stripe_fixed_mask((SIZE, SIZE), PERIOD, line_width, angle)
        image = render_source_color_on_white(mask, rgb)
        _check_color(image, mask, rgb, name)
        path = OUTPUT_DIR / name
        Image.fromarray(image, mode="RGB").save(path, format="PNG")
        report.append((name, sha256_of(path)))

    spiral_samples = [
        ("spiral_fixed_color_p16_w3_arms1.png", 3, 1),
        ("spiral_fixed_color_p16_w5_arms3.png", 5, 3),
        ("spiral_fixed_color_p16_w7_arms12.png", 7, 12),
    ]
    for name, line_width, arms in spiral_samples:
        mask = spiral_fixed_mask(SIZE, PERIOD, line_width, arms)
        image = render_source_color_on_white(mask, rgb)
        _check_color(image, mask, rgb, name)
        path = OUTPUT_DIR / name
        Image.fromarray(image, mode="RGB").save(path, format="PNG")
        report.append((name, sha256_of(path)))

    for name, digest in report:
        path = OUTPUT_DIR / name
        print(f"{path.relative_to(REPO_ROOT)}  mode=RGB  sha256={digest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
