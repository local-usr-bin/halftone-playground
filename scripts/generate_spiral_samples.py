#!/usr/bin/env python3
"""Generate the representative Spiral mode sample images.

These PNGs are the *human visual acceptance* artifacts for Spiral Round 1.
They are deliberately plain: strict black/white, no antialiasing, no
post-processing, and no touching-up of the dense centre rosette that high
arm counts produce -- that pattern is a property of the construction and is
kept verbatim as a regression artifact.

Run from the repository root:

    python scripts/generate_spiral_samples.py

Outputs land in ``samples/spiral/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from halftone_playground import (  # noqa: E402
    render_black_on_white,
    spiral_mask,
)

SIZE = 512
PERIOD = 16
OUTPUT_DIR = REPO_ROOT / "samples" / "spiral"


def constant_gray(value: int, size: int = SIZE) -> np.ndarray:
    """A flat gray canvas."""
    return np.full((size, size), value, dtype=np.uint8)


def radial_gradient(size: int = SIZE) -> np.ndarray:
    """White at the centre, black towards the rim.

    A radial ramp is the natural companion to a radial construction: it makes
    the gray -> line-width response visible as a continuous widening of the
    spiral as the radius grows.  Coordinates reuse the same ``(N - 1) / 2``
    centre as the geometry so the ramp and the disc stay concentric.
    """
    centre = (size - 1) / 2.0
    coords = np.arange(size, dtype=np.float64)
    dx = coords - centre
    dy = coords - centre
    radius = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    ramp = 255.0 * (1.0 - radius / radius.max())
    return np.clip(ramp, 0.0, 255.0).astype(np.uint8)


SAMPLES = [
    ("spiral_constant_50_arms1_p16.png", constant_gray(128), 1),
    ("spiral_constant_50_arms2_p16.png", constant_gray(128), 2),
    ("spiral_constant_50_arms3_p16.png", constant_gray(128), 3),
    ("spiral_constant_50_arms12_p16.png", constant_gray(128), 12),
    ("spiral_radial_gradient_arms1_p16.png", radial_gradient(), 1),
    ("spiral_radial_gradient_arms3_p16.png", radial_gradient(), 3),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, gray, arms in SAMPLES:
        mask = spiral_mask(gray, PERIOD, arms)
        image = render_black_on_white(mask)
        path = OUTPUT_DIR / name
        Image.fromarray(image, mode="L").save(path, format="PNG")
        unique = np.unique(image).tolist()
        print(
            f"{path.relative_to(REPO_ROOT)}  "
            f"size={image.shape[1]}x{image.shape[0]}  "
            f"arms={arms}  "
            f"P={PERIOD}  "
            f"ink={float(mask.mean()):.3f}  "
            f"values={unique}  "
            f"corner={int(image[0, 0])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
