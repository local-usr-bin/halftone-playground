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

**Stripe mode Round 1 core is implemented.** Grayscale-driven geometry produces a
boolean line mask, and a separate renderer turns that mask into a strict
black/white image. There is no CLI and no GUI.

Not implemented yet: **Spiral mode**, colored or inverted output, transparency.

## Stripe mode

`stripe_mask(gray, period, angle_deg=90.0)` returns a `bool` mask where `True`
marks a line pixel. Two parameters control the geometry:

- **`period`** — distance between adjacent stripe **centers**, in output pixels
  (positive integer). Each stripe owns roughly half a period either side of its
  center, and the local mean gray of that region becomes the line width. The
  period is nominal: it is never rescaled to fill the canvas, so the last cell of
  a row may be cut off by the edge.
- **`angle_deg`** — stripe orientation, where **`90` = vertical** and
  **`0` / `180` = horizontal**. The orientation repeats every 180 degrees.

Rendering is a separate step, `render_black_on_white(mask)`, which returns a
`uint8` image with lines as `0` and background as `255`.

```python
import numpy as np
from halftone_playground import stripe_mask, render_black_on_white
from PIL import Image

gray = np.full((256, 256), 128, dtype=np.uint8)   # 50 % gray
mask = stripe_mask(gray, period=16, angle_deg=90.0)
Image.fromarray(render_black_on_white(mask), mode="L").save("stripes.png")
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
