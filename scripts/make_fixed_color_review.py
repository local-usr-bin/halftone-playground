#!/usr/bin/env python3
"""Generate the Fixed-Width Color Round 1 real-image review artifacts.

This script intentionally lives *outside* the tracked pipeline: it consumes the
scikit-image sample photograph ``astronaut.png`` (Eileen Collins) as a source
image and writes a review set next to it.  Neither the RGB source nor these
review outputs are committed to Git -- they exist for the human visual
acceptance gate only.

Source: ``skimage/data/astronaut.png`` from scikit-image v0.25.2, fetched from
GitHub raw.  scikit-image itself is **not** installed or imported by this
script; the downloaded PNG is read with Pillow.

Usage:

    python scripts/make_fixed_color_review.py <src_png> <out_dir>

The source must be a 512x512 RGB PNG.  It is used as the **Cartesian** colour
source only: fixed-width geometry never reads grayscale, so there is no
grayscale conversion at all here.

Outputs (all 512x512 RGB, source colour on pure white):

* 3 fixed-width Stripe variants  (P=16): w3/a90, w7/a90, w5/a45
* 3 fixed-width Spiral variants  (P=16): w3/arms1, w5/arms3, w7/arms12
* one mobile-friendly 2-column review sheet

The RGB source is copied at the final Cartesian coordinates; it never enters
``warpPolar``.
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

SIDE = 512
PERIOD = 16

_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rgb(src_png: Path) -> np.ndarray:
    """Load the source photograph as a 512x512 ``uint8`` RGB array."""
    with Image.open(src_png) as image:
        if image.size != (SIDE, SIDE):
            raise SystemExit(f"expected a {SIDE}x{SIDE} source, got {image.size}")
        return np.array(image.convert("RGB"), dtype=np.uint8)


def check_color(image: np.ndarray, mask: np.ndarray, rgb: np.ndarray, label: str) -> None:
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape == (SIDE, SIDE, 3)
    assert (image[~mask] == _WHITE).all(), f"{label} background is not pure white"
    assert np.array_equal(image[mask], rgb[mask]), f"{label} line colour is not exact"


def build_sheet(panels, columns: int, path: Path) -> None:
    """Compose a labelled, mobile-friendly review sheet (2 columns).

    RGB previews may be resized with BICUBIC for display only; the underlying
    review outputs are saved untouched at full 512x512.
    """
    from PIL import ImageDraw

    tiles = []
    for title, image in panels:
        pil = Image.fromarray(image)
        if pil.width != 512:
            resample = Image.NEAREST if not _is_color_array(image) else Image.BICUBIC
            pil = pil.resize((512, 512), resample=resample)
        tiles.append((title, pil.convert("RGB")))

    caption = 26
    cols = columns
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new(
        "RGB", (cols * 512, rows * (512 + caption)), color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, tile) in enumerate(tiles):
        row, col = divmod(index, cols)
        x = col * 512
        y = row * (512 + caption)
        sheet.paste(tile, (x, y))
        draw.text((x + 6, y + 512 + 6), title, fill=(0, 0, 0))
    sheet.save(path, format="PNG")


def _is_color_array(image: np.ndarray) -> bool:
    return image.ndim == 3


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_png = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb(src_png)
    print(f"source: {src_png.name}  sha256={sha256_of(src_png)}")
    print(f"rgb dtype={rgb.dtype} shape={rgb.shape}")

    outputs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    # ---- Fixed Stripe: 3 variants --------------------------------------
    stripe_variants = [
        ("fixed_r1_stripe_p16_w3_a90.png", 3, 90.0),
        ("fixed_r1_stripe_p16_w7_a90.png", 7, 90.0),
        ("fixed_r1_stripe_p16_w5_a45.png", 5, 45.0),
    ]
    for name, line_width, angle in stripe_variants:
        mask = stripe_fixed_mask((SIDE, SIDE), PERIOD, line_width, angle)
        image = render_source_color_on_white(mask, rgb)
        check_color(image, mask, rgb, name)
        outputs[name] = image
        masks[name] = mask
        path = out_dir / name
        Image.fromarray(image, mode="RGB").save(path, format="PNG")
        print(f"{name}  P={PERIOD} w={line_width} a={angle}  sha256={sha256_of(path)}")

    # ---- Fixed Spiral: 3 variants --------------------------------------
    spiral_variants = [
        ("fixed_r1_spiral_p16_w3_arms1.png", 3, 1),
        ("fixed_r1_spiral_p16_w5_arms3.png", 5, 3),
        ("fixed_r1_spiral_p16_w7_arms12.png", 7, 12),
    ]
    for name, line_width, arms in spiral_variants:
        mask = spiral_fixed_mask(SIDE, PERIOD, line_width, arms)
        image = render_source_color_on_white(mask, rgb)
        check_color(image, mask, rgb, name)
        # Circle outside must be pure white.
        assert (image[~mask] == _WHITE).all()
        outputs[name] = image
        masks[name] = mask
        path = out_dir / name
        Image.fromarray(image, mode="RGB").save(path, format="PNG")
        print(
            f"{name}  P={PERIOD} w={line_width} arms={arms}  "
            f"sha256={sha256_of(path)}"
        )

    # ---- exact Cartesian RGB copy, hard check --------------------------
    for name, mask in masks.items():
        assert np.array_equal(outputs[name][mask], rgb[mask]), name
    print("all six outputs: line pixels are byte-exact Cartesian source copies")

    # ---- review sheet: 2 columns, mobile friendly ----------------------
    face = rgb[60:260, 60:260]
    suit = rgb[300:460, 150:330]
    centre = outputs["fixed_r1_spiral_p16_w5_arms3.png"][176:336, 176:336]

    panels = [
        ("source RGB", rgb),
        ("fixed Stripe  P=16 W=3 a=90", outputs["fixed_r1_stripe_p16_w3_a90.png"]),
        ("fixed Stripe  P=16 W=7 a=90", outputs["fixed_r1_stripe_p16_w7_a90.png"]),
        ("fixed Stripe  P=16 W=5 a=45", outputs["fixed_r1_stripe_p16_w5_a45.png"]),
        ("fixed Spiral  P=16 W=3 arms=1", outputs["fixed_r1_spiral_p16_w3_arms1.png"]),
        ("fixed Spiral  P=16 W=5 arms=3", outputs["fixed_r1_spiral_p16_w5_arms3.png"]),
        ("fixed Spiral  P=16 W=7 arms=12", outputs["fixed_r1_spiral_p16_w7_arms12.png"]),
        ("face crop (source RGB)", face),
        ("suit crop (source RGB)", suit),
        ("fixed Spiral centre crop (W=5 arms=3)", centre),
    ]
    sheet_path = out_dir / "fixed_color_r1_real_review_sheet.png"
    build_sheet(panels, columns=2, path=sheet_path)
    print(f"fixed_color_r1_real_review_sheet.png  sha256={sha256_of(sheet_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
