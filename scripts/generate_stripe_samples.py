#!/usr/bin/env python3
"""Generate the representative Stripe mode Round 1 sample images.

These PNGs are the *human visual acceptance* artifacts for the first Stripe
round.  They are deliberately plain: strict black/white, no antialiasing, no
resizing, no post-processing.  What you see is exactly the boolean mask.

Run from the repository root:

    python scripts/generate_stripe_samples.py

Outputs land in ``samples/stripe/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from halftone_playground import render_black_on_white, stripe_mask  # noqa: E402

SIZE = 256
PERIOD = 16
OUTPUT_DIR = REPO_ROOT / "samples" / "stripe"


def constant_gray(value: int, size: int = SIZE) -> np.ndarray:
    """A flat gray canvas."""
    return np.full((size, size), value, dtype=np.uint8)


def vertical_gradient(size: int = SIZE) -> np.ndarray:
    """White at the top, black at the bottom."""
    ramp = np.linspace(255.0, 0.0, num=size)
    return np.tile(ramp[:, None], (1, size)).astype(np.uint8)


def horizontal_gradient(size: int = SIZE) -> np.ndarray:
    """White at the left, black at the right."""
    ramp = np.linspace(255.0, 0.0, num=size)
    return np.tile(ramp[None, :], (size, 1)).astype(np.uint8)


SAMPLES = [
    ("stripe_constant_50_a90_p16.png", constant_gray(128), 90.0),
    ("stripe_vertical_gradient_a90_p16.png", vertical_gradient(), 90.0),
    ("stripe_vertical_gradient_a45_p16.png", vertical_gradient(), 45.0),
    ("stripe_horizontal_gradient_a90_p16.png", horizontal_gradient(), 90.0),
    ("stripe_vertical_gradient_a0_p16.png", vertical_gradient(), 0.0),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, gray, angle_deg in SAMPLES:
        mask = stripe_mask(gray, PERIOD, angle_deg)
        image = render_black_on_white(mask)
        path = OUTPUT_DIR / name
        Image.fromarray(image, mode="L").save(path, format="PNG")
        print(
            f"{path.relative_to(REPO_ROOT)}  "
            f"size={image.shape[1]}x{image.shape[0]}  "
            f"angle={angle_deg}  "
            f"ink={float(mask.mean()):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
