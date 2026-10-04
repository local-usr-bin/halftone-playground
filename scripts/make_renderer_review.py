#!/usr/bin/env python3
"""Generate the Renderer / Compositor Round 1 real-image review artifacts.

This script intentionally lives *outside* the tracked pipeline: it consumes the
scikit-image sample photograph ``astronaut.png`` (Eileen Collins) as a source
image and writes a review set next to it.  Neither the RGB source nor these
review outputs are committed to Git -- they exist for the human visual
acceptance gate only.

Source: ``skimage/data/astronaut.png`` from scikit-image v0.25.2, fetched from
GitHub raw.  scikit-image itself is **not** installed or imported.

Usage:

    python scripts/make_renderer_review.py <src_png> <out_dir>

The source must be a 512x512 RGB PNG.  It is used **twice, from the same
original**:

* ``convert("L")`` for the geometry cores (never resized, retoned or
  contrast-adjusted);
* the raw RGB as the Cartesian colour source for the source-colour compositor.

Crucially the RGB data never enters a polar transform: for Spiral the colour is
copied at the final Cartesian coordinates, so there is no extra colour
resampling of any kind.
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

SIDE = 512
PERIOD = 16
STRIPE_ANGLE = 90.0
SPIRAL_ARMS = 3

_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(src_png: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gray, rgb)`` for a 512x512 RGB source PNG."""
    with Image.open(src_png) as image:
        if image.size != (SIDE, SIDE):
            raise SystemExit(f"expected a {SIDE}x{SIDE} source, got {image.size}")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        gray = np.array(image.convert("L"), dtype=np.uint8)
    return gray, rgb


def check_binary(image: np.ndarray, label: str) -> None:
    assert image.dtype == np.uint8 and image.ndim == 2 and image.shape == (SIDE, SIDE)
    unique = set(np.unique(image).tolist())
    assert unique <= {0, 255}, f"{label} is not strictly bi-level: {sorted(unique)}"


def check_color(image: np.ndarray, mask: np.ndarray, rgb: np.ndarray, label: str) -> None:
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape == (SIDE, SIDE, 3)
    assert (image[~mask] == _WHITE).all(), f"{label} background is not pure white"
    assert np.array_equal(image[mask], rgb[mask]), f"{label} line colour is not exact"


def build_sheet(panels, columns: int, path: Path) -> None:
    """Compose a labelled, mobile-friendly review sheet.

    Binary previews are always resized with ``Image.NEAREST`` so the sheet can
    never invent anti-aliased grey the core never produced.  RGB previews may
    use BICUBIC for display only; the underlying review outputs are saved
    untouched at full 512x512.
    """
    from PIL import ImageDraw

    tiles = []
    for title, image, is_color in panels:
        pil = Image.fromarray(image)
        if pil.width != 512:
            resample = Image.BICUBIC if is_color else Image.NEAREST
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


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_png = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gray, rgb = load_source(src_png)
    print(f"source: {src_png.name}  sha256={sha256_of(src_png)}")
    print(f"rgb dtype={rgb.dtype} shape={rgb.shape}")

    # ---- Stripe: one mask, three renderers ------------------------------
    stripe = stripe_mask(gray, PERIOD, STRIPE_ANGLE)
    stripe_bw = render_black_on_white(stripe)
    stripe_wb = render_white_on_black(stripe)
    stripe_color = render_source_color_on_white(stripe, rgb)

    check_binary(stripe_bw, "stripe black_on_white")
    check_binary(stripe_wb, "stripe white_on_black")
    check_color(stripe_color, stripe, rgb, "stripe source_color_on_white")
    assert (stripe_bw.astype(np.uint16) + stripe_wb.astype(np.uint16) == 255).all()

    outputs = {
        "renderer_r1_stripe_black_on_white.png": (stripe_bw, "L"),
        "renderer_r1_stripe_white_on_black.png": (stripe_wb, "L"),
        "renderer_r1_stripe_source_color_on_white.png": (stripe_color, "RGB"),
    }

    # ---- Spiral: one mask, three renderers ------------------------------
    spiral = spiral_mask(gray, PERIOD, SPIRAL_ARMS)
    spiral_bw = render_black_on_white(spiral)
    spiral_wb = render_white_on_black(spiral)
    spiral_color = render_source_color_on_white(spiral, rgb)

    check_binary(spiral_bw, "spiral black_on_white")
    check_binary(spiral_wb, "spiral white_on_black")
    check_color(spiral_color, spiral, rgb, "spiral source_color_on_white")
    assert (spiral_bw.astype(np.uint16) + spiral_wb.astype(np.uint16) == 255).all()

    # ---- Spiral: exact Cartesian RGB match over >= 100 line pixels ------
    ys, xs = np.nonzero(spiral)
    assert ys.size > 100, "not enough spiral line pixels for the exactness check"
    sample = np.linspace(0, ys.size - 1, num=200).astype(np.int64)
    checked = 0
    for index in sample.tolist():
        y, x = int(ys[index]), int(xs[index])
        assert spiral_color[y, x].tolist() == rgb[y, x].tolist(), (y, x)
        checked += 1
    # Whole-disc equality is the strongest form of the same statement.
    assert np.array_equal(spiral_color[spiral], rgb[spiral])
    print(
        f"spiral source-colour exact Cartesian match: {checked} sampled pixels "
        f"+ entire mask ({int(spiral.sum())} px) byte-exact; no polar resample"
    )

    # ---- Circle outside must be pure white in the colour variant --------
    corners = [(0, 0), (0, SIDE - 1), (SIDE - 1, 0), (SIDE - 1, SIDE - 1)]
    for y, x in corners:
        assert spiral_color[y, x].tolist() == [255, 255, 255], (y, x)
    assert (spiral_color[~spiral] == _WHITE).all()
    print("spiral source-colour: outside the circular support is pure white")

    outputs["renderer_r1_spiral_black_on_white.png"] = (spiral_bw, "L")
    outputs["renderer_r1_spiral_white_on_black.png"] = (spiral_wb, "L")
    outputs["renderer_r1_spiral_source_color_on_white.png"] = (spiral_color, "RGB")

    for name, (image, mode) in outputs.items():
        path = out_dir / name
        Image.fromarray(image, mode=mode).save(path, format="PNG")
        print(f"{name}  mode={mode}  sha256={sha256_of(path)}")

    # ---- Real review sheet: 2 columns, mobile friendly ------------------
    face = rgb[60:260, 60:260]
    suit = rgb[300:460, 150:330]
    centre = spiral_color[176:336, 176:336]

    panels = [
        ("source RGB", rgb, True),
        ("Stripe black on white  P=16 a=90", stripe_bw, False),
        ("Stripe white on black", stripe_wb, False),
        ("Stripe source colour on white", stripe_color, True),
        ("Spiral black on white  P=16 arms=3", spiral_bw, False),
        ("Spiral white on black", spiral_wb, False),
        ("Spiral source colour on white", spiral_color, True),
        ("face / hair crop (source RGB)", face, True),
        ("coloured suit crop (source RGB)", suit, True),
        ("spiral centre crop (source colour)", centre, True),
    ]
    sheet_path = out_dir / "renderer_r1_real_review_sheet.png"
    build_sheet(panels, columns=2, path=sheet_path)
    print(f"renderer_r1_real_review_sheet.png  sha256={sha256_of(sheet_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
