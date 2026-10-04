#!/usr/bin/env python3
"""Generate the Spiral Round 1 real-image review artifacts (repository-external).

This script intentionally lives *outside* the tracked pipeline: it consumes the
scikit-image sample photograph ``astronaut.png`` (Eileen Collins) as a source
image and writes a review set next to it.  Neither the RGB source nor these
review outputs are committed to Git -- they exist for the human visual
acceptance gate only.

Source: ``skimage/data/astronaut.png`` from scikit-image v0.25.2, fetched from
GitHub raw.  scikit-image itself is **not** installed or imported.

Usage:

    python scripts/make_spiral_review.py <src_png> <out_dir>

The source must be a 512x512 RGB PNG.  It is converted with Pillow
``convert("L")`` and is never resized, retoned or contrast-adjusted.

Chirality note: the frozen Spiral chirality is ``positive slope``, i.e. a
**clockwise spiral when followed outwards from the centre**
(``clockwise outward``).  Any label written onto a sheet must use that
wording.
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
    spiral_mask,
)

PERIOD = 16
ARMS_SET = (1, 2, 3, 12)
SIDE = 512


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gray(src_png: Path) -> np.ndarray:
    """Load the source photograph as a 512x512 ``uint8`` grayscale array."""
    with Image.open(src_png) as image:
        if image.size != (SIDE, SIDE):
            raise SystemExit(f"expected a {SIDE}x{SIDE} source, got {image.size}")
        gray = image.convert("L")
        return np.array(gray, dtype=np.uint8)


def circular_support(side: int) -> np.ndarray:
    """Reference support disc, used only for the inside-disk ink fraction."""
    centre = (side - 1) / 2.0
    support_radius = (side - 1) / 2.0
    coords = np.arange(side, dtype=np.float64)
    dx = coords - centre
    dy = coords - centre
    return (dy[:, None] ** 2 + dx[None, :] ** 2) <= support_radius * support_radius


def check_hard_properties(image: np.ndarray) -> None:
    """Assert the Round 1 output contract for every review image."""
    unique = set(np.unique(image).tolist())
    assert unique <= {0, 255}, f"image is not strictly bi-level: {sorted(unique)}"
    for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert image[y, x] == 255, f"corner ({y},{x}) is not white"


def build_sheet(panels, columns: int, scale: int, path: Path) -> None:
    """Compose a labelled, nearest-neighbour-scaled review sheet.

    ``panels`` is a sequence of ``(title, uint8 array)`` pairs.  Binary
    previews are always resized with ``Image.NEAREST`` so the review sheet
    cannot invent anti-aliased grey that the core never produced.
    """
    from PIL import ImageDraw

    tiles = []
    for title, image in panels:
        tile = Image.fromarray(image, mode="L")
        if scale != 1:
            tile = tile.resize(
                (tile.width * scale, tile.height * scale), resample=Image.NEAREST
            )
        tiles.append((title, tile.convert("RGB")))

    tile_w = max(t.width for _, t in tiles)
    tile_h = max(t.height for _, t in tiles)
    caption = 26
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_w, rows * (tile_h + caption)), color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(sheet)

    for index, (title, tile) in enumerate(tiles):
        row, col = divmod(index, columns)
        x = col * tile_w
        y = row * (tile_h + caption)
        sheet.paste(tile, (x, y))
        draw.text((x + 6, y + tile_h + 6), title, fill=(0, 0, 0))

    sheet.save(path, format="PNG")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_png = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = load_gray(src_png)
    disc = circular_support(SIDE)

    print(f"source: {src_png.name}  sha256={sha256_of(src_png)}")
    print(f"gray dtype={gray.dtype} shape={gray.shape}")

    masks = {}
    images = {}
    for arms in ARMS_SET:
        mask = spiral_mask(gray, PERIOD, arms)
        image = render_black_on_white(mask)
        check_hard_properties(image)
        masks[arms] = mask
        images[arms] = image

        name = f"spiral_r1_eileen_arms{arms}_p16.png"
        path = out_dir / name
        Image.fromarray(image, mode="L").save(path, format="PNG")

        whole = float((image == 0).mean())
        inside = float((image == 0)[disc].mean())
        print(
            f"{name}  size={image.shape[1]}x{image.shape[0]}  "
            f"black_whole={whole:.6f}  black_inside_disk={inside:.6f}  "
            f"corner={int(image[0, 0])}  sha256={sha256_of(path)}"
        )

    # ---- real review sheet: 2 columns, mobile friendly -------------------
    face = images[1][60:260, 60:260]
    centre_crop = images[1][176:336, 176:336]
    rim_crop = images[1][196:316, 396:512]

    panels = [("source grayscale (Eileen Collins)", gray)]
    panels += [
        (f"arms{arms} P=16  clockwise outward", images[arms]) for arms in ARMS_SET
    ]
    panels += [
        ("face crop (arms1)", face),
        ("centre crop (arms1)", centre_crop),
        ("seam / outer-rim crop (arms1)", rim_crop),
    ]
    build_sheet(panels, columns=2, scale=1, path=out_dir / "spiral_r1_real_review_sheet.png")
    print("wrote spiral_r1_real_review_sheet.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
