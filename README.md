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
(`image_scale`) before the geometry runs.

**Spiral mode Round 1 core is implemented.** It reuses the very same Stripe
core inside a linear polar unwrap; see the Spiral section below.

There is no CLI and no GUI. Not implemented yet: colored or inverted output,
transparency, non-square Spiral input.

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

## Spiral mode

Spiral mode does **not** draw a spiral from a parametric equation. It changes
coordinate system instead and reuses the shared Stripe core:

```
Cartesian grayscale
  -> linear polar unwrap            (OpenCV warpPolar)
  -> shared variable-width Stripe core
  -> inverse linear polar           (WARP_INVERSE_MAP)
  -> circular support mask
  -> bool mask
```

On the polar canvas the radius is the horizontal axis and the angle is the
vertical axis, so a straight tilted stripe becomes a constant-pitch family of
turns; unwrapping it back to Cartesian yields a multi-arm spiral whose line
width is driven by the local grayscale, exactly like the stripes.

```python
import numpy as np
from halftone_playground import spiral_mask, render_black_on_white
from PIL import Image

gray = np.full((512, 512), 128, dtype=np.uint8)   # square, uint8, 2-D
mask = spiral_mask(gray, period=16, arms=1)
Image.fromarray(render_black_on_white(mask), mode="L").save("spiral.png")
```

Spiral parameters:

- **`period`** — distance between adjacent lines in the polar image, measured
  in pixels of the Cartesian output image. Positive integer, independent of
  `image_scale` in the same way as Stripe.
- **`arms`** — number of spiral arms. Positive integer; any value from `1`
  upwards is valid. At high arm counts the centre develops a dense rosette —
  that pattern is a property of the construction and is deliberately left
  as-is.

`spiral_mask(gray, period, arms)` returns a `bool` mask where `True` marks a
spiral line pixel; everything outside the circular support (the four corners
included) is always `False`.

**Round 1 boundaries:**

- **square grayscale input only** (`shape == (n, n)`, 2-D `uint8`). The
  fit / crop / letterbox policy for non-square sources is not frozen yet, so
  non-square input is rejected rather than silently handled. This is a Round 1
  product boundary, not a claim that a polar unwrap is inherently square-only.
- the default — and only — **chirality is `clockwise outward`**: following an
  arm away from the centre sweeps clockwise. There is no chirality parameter.
- `image_scale` is **not** a Spiral parameter; call `resize_grayscale(...)`
  before `spiral_mask(...)` if a scale is wanted.

Spiral mode uses OpenCV's `warpPolar`, whose underlying `remap` currently caps
image dimensions at less than 32767 pixels; larger sources raise a clear
`ValueError` instead of being tiled or silently downscaled.

## Layout

- `src/halftone_playground/` — package code (`stripe.py` and `spiral.py` geometry, `render.py` output)
- `tests/` — pytest suite
- `scripts/` — small runnable helpers
- `samples/` — generated example images
- `docs/` — future design notes

## Dependencies

- Python >= 3.9
- [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/)
- [OpenCV](https://opencv.org/) (`opencv-python-headless`) — used by Spiral mode
  for the polar transforms
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

Regenerate the Spiral acceptance images into `samples/spiral/`:

```bash
python scripts/generate_spiral_samples.py
```
