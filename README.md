# halftone-playground

Experiments in variable-width line halftoning: **Stripe mode** (straight
variable-width stripes) and **Spiral mode** (polar-unwrapped spiral stripes).
Both modes share one core idea: line width is driven by the local grayscale of
the source image, so the black/white area ratio reproduces the visual tone.

**Working package name**: `halftone_playground` (internal name only, not the
final product name).

## Development order

1. **Stripe mode** first — must be completed and accepted before anything else.
2. **Spiral mode** afterwards.

## Current status

**Stripe mode Round 2 core is implemented.** Grayscale-driven geometry
produces a boolean line mask with **integer line widths** (each full cell
draws exactly `N = floor(W + 0.5)` consecutive black pixels, placed so the
block center is nearest to the stripe center, exact ties to the smaller
index), and a separate renderer turns that mask into a strict black/white
image. An optional preprocessing step rescales the source image
(`image_scale`) before the geometry runs. There is no CLI and no GUI.

Not implemented yet: **Spiral mode**, colored or inverted output, transparency.

## Stripe parameters

- **`period`** — distance between adjacent stripe **centers**, in pixels of
  the image entering the core (i.e. the *scaled* image). Positive integer.
  Each stripe owns roughly half a period either side of its center, and the
  local mean gray of that region drives the line width. The period is
  nominal: it is never rescaled to fill the canvas, so the last cell of a row
  may be cut off by the edge.
- **`angle_deg`** — stripe orientation, where **`90` = vertical** and
  **`0` / `180` = horizontal**. Any finite float; the orientation repeats
  every 180 degrees.
- **`image_scale`** — spatial scale factor applied to the source image
  **before** the Stripe core runs. Any positive finite float; non-integer
  values are fine (`1.3`, `2.75`, `100.0`). The resize uses a fixed internal
  BICUBIC filter (not a user choice). Target dimensions round half-up:
  `floor(original * scale + 0.5)`.

`image_scale` and `period` are **fully independent** parameters. The period
always refers to pixels of the scaled image, and the software never adjusts
it: `image_scale=2.0, period=16` stays `P=16` (so stripes look denser
relative to the original). If you want to double the resolution while
keeping roughly the same stripe density, set `image_scale=2.0, period=32`
yourself — nothing is linked automatically.

## Stripe mode

`stripe_mask(gray, period, angle_deg=90.0)` returns a `bool` mask where `True`
marks a line pixel. Optional preprocessing lives in
`resize_grayscale(gray, image_scale)` / `scaled_size(width, height,
image_scale)`; the pipeline order is `source → grayscale → BICUBIC resize →
stripe_mask → renderer`.

Rendering is a separate step, `render_black_on_white(mask)`, which returns a
`uint8` image with lines as `0` and background as `255`.

```python
import numpy as np
from halftone_playground import (
    resize_grayscale, scaled_size, stripe_mask, render_black_on_white,
)
from PIL import Image

gray = np.full((256, 256), 128, dtype=np.uint8)   # 50 % gray
scaled = resize_grayscale(gray, image_scale=1.3)  # -> 333x333
mask = stripe_mask(scaled, period=16, angle_deg=90.0)
Image.fromarray(render_black_on_white(mask), mode="L").save("stripes.png")

scaled_size(512, 512, 1.3)                        # (666, 666), no allocation
```

## Layout

- `src/halftone_playground/` — package code (`stripe.py` geometry, `render.py` output)
- `tests/` — pytest suite
- `scripts/` — small runnable helpers
- `samples/` — generated example images
- `docs/` — future design notes

## Dependencies

- Python >= 3.9
- [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/)
- [pytest](https://pytest.org) for tests

## Test commands

Run from the repository root:

```bash
pip install pytest   # once, for the dev environment
pytest               # full test suite
```

## Sample images

Regenerate the Stripe acceptance images into `samples/stripe/`:

```bash
python scripts/generate_stripe_samples.py
```
