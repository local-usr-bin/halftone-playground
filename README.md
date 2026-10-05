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

**Renderer / Compositor Round 1 is implemented.** The geometry core and the
pixels are fully separated: Stripe and Spiral both emit only a `bool` mask
(`True` = line), and independent renderers decide what a line pixel looks
like. Three output variants ship today — white background + black line, black
background + white line, and white background + the source image's own colour
on the line.

**Fixed-width geometry variants are implemented.** On top of the
variable-width cores, `stripe_fixed_mask` and `spiral_fixed_mask` draw lines
whose width is chosen directly by the user (`line_width`, `1..period`). The
width no longer depends on grayscale at all — only the colour still comes from
the Cartesian source image, via the existing
`render_source_color_on_white` compositor. See the Fixed-width section below.

**A first command-line interface is implemented.** `halftone-playground`
(and `python -m halftone_playground`) wires the existing cores and renderers
into a single-image tool. See the [CLI](#cli) section below.

**A first GUI shell is implemented (GUI-001), a real generation pipeline now
runs in it (GUI-002A), and its results can be saved (GUI-002B).** A plain
Tkinter/ttk window (Controls on the left, Source / Result previews on the
right) can open an image, show a source preview, validate parameters,
**Generate** the real full-resolution result on a background worker, and
**Save PNG…** it. The GUI accepts a rectangular source for Spiral and performs
the centered maximum-square crop itself. See the [GUI](#gui) section below and
[`docs/GUI_ROUND0_DECISIONS.md`](docs/GUI_ROUND0_DECISIONS.md) for the frozen
GUI decisions.

Not implemented yet: transparency editing beyond the Spiral disc, custom RGB
backgrounds, batch processing.

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
`uint8` image with lines as `0` and background as `255`.```python
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

## Fixed-width geometry variants

Fixed-width geometry is the counterpart of the variable-width cores. The
*raster placement* is exactly the same frozen geometry (period phase, angle
semantics, nearest block center, tie-left, edge clipping); only the source of
the line width changes:

| core | per-cell line width |
|---|---|
| `stripe_mask` / `spiral_mask` | derived from the local grayscale: `N = floor(P * (1 - G) + 0.5)` |
| `stripe_fixed_mask` / `spiral_fixed_mask` | the constant `line_width` you pass in |

- **`line_width`** — constant line width, in pixels of the **scaled** image.
  A positive integer in the inclusive range **`1..period`**. Out-of-range
  values are **rejected, not clamped**: `line_width` above `period`, `0`,
  negative values, floats and booleans all raise. `line_width = period` fills
  every covered cell, so a fully tiled canvas becomes entirely `True`.
- **No grayscale in the geometry.** Fixed-width geometry never reads the
  source grayscale — that is the whole point. The width is applied directly as
  an integer, so it is exact for *any* `period` (a synthetic uint8 grayscale
  route would only offer 256 levels and would lose width levels for
  `period > 255`).
- **Colour still comes from the source.** The fixed-width source-colour
  variant renders through the very same `render_source_color_on_white`
  compositor, so a line pixel is still a **byte-exact copy of the Cartesian
  source RGB** at the same `(y, x)`; everything else is pure white. For fixed
  Spiral the polar round trip carries the **binary mask only** — the RGB never
  enters `warpPolar`.

`stripe_fixed_mask(shape, period, line_width, angle_deg=90.0)` accepts a
**shape** `(height, width)` rather than a grayscale array, because fixed-width
geometry has no source image to read. `spiral_fixed_mask(side, period,
line_width, arms)` likewise takes a square `side`. Both return a `bool` mask
(`True` = line); the Spiral mask is always `False` outside the circular
support.

```python
import numpy as np
from PIL import Image
from halftone_playground import (
    stripe_fixed_mask, spiral_fixed_mask, render_source_color_on_white,
)

rgb = np.array(Image.open("photo.png").convert("RGB"), dtype=np.uint8)

# Constant 5 px stripes at 90 degrees, coloured with the source pixels.
mask = stripe_fixed_mask(rgb.shape[:2], period=16, line_width=5, angle_deg=90.0)
out = render_source_color_on_white(mask, rgb)

# Constant 5 px, 3-arm spiral; geometry is square-only in this round.
sp = spiral_fixed_mask(512, period=16, line_width=5, arms=3)
out2 = render_source_color_on_white(sp, rgb)
```

`image_scale`, `period` and `line_width` stay fully independent — the software
never links them. Scaling the canvas by `2.0` leaves `period` and `line_width`
untouched, so if you want thickness to look identical after a scale you change
`period` / `line_width` yourself.

## Renderers and compositors

Both geometry modes produce exactly one thing: a `bool` mask where `True`
marks a line pixel. What that pixel *looks like* is decided by a separate,
pure layer. Because the renderer only ever reads the mask, the very same mask
can be rendered several ways with **identical geometry** — only the pixel
values change. No renderer recomputes gray, recomputes line width, or smooths
/ resizes / blurs the mask.

| function | `mask=True` | `mask=False` | output |
|---|---|---|---|
| `render_black_on_white(mask)` | `0` | `255` | 2-D `uint8` |
| `render_white_on_black(mask)` | `255` | `0` | 2-D `uint8` |
| `render_source_color_on_white(mask, source_rgb)` | `source_rgb[y, x]` | `(255, 255, 255)` | `H×W×3` `uint8` |

- **White on black is a true renderer inversion.** The mask is consumed
  unchanged; there is no `invert` parameter in the Stripe or Spiral cores. For
  one mask the two binary renderers are per-pixel complements
  (`black_on_white + white_on_black == 255`). The whole canvas background
  becomes black, corners outside a shape's support included.
- **Source colour is still variable width.** The existing width mask drives
  the geometry; only the line pixel value changes from black to the matching
  source pixel. This is *not* a fixed-width coloured-line variant.

#### White on black in the GUI keeps the source positive

At the **renderer** level white-on-black is an inversion of one mask, as above.
At the **GUI / application** level the product rule is different: the white line
must grow with the source's *brightness* (`W_white = P · G`), so a white area
stays white instead of reading as a photographic negative. The GUI therefore
inverts the already cropped and scaled grayscale for the Variable +
White-on-black combination before it reaches the geometry
(`preprocess.invert_luminance`), leaving the frozen `W = P · (1 − G)` formula,
the cores' API and the renderers untouched. Fixed width and Source colour are
never inverted, Black on white is unchanged, and the Spiral alpha channel is
still decided only by the circular support.

### Source colour and Cartesian space

`render_source_color_on_white` copies colour from the **Cartesian** RGB source
at the final Cartesian coordinates:

```
Cartesian RGB source ──┐
                       ├─► Cartesian grayscale ─► Stripe / Spiral core ─► Cartesian bool mask ─┐
                       │                                                                       │
                       └───────────────────────────────────────────────────────────────────────┴─► RGB output
```

For Spiral in particular the colour **never enters the polar pipeline**: there
is no forward unwrap and no inverse unwrap for RGB, so no extra colour
resampling is introduced. A line pixel is a byte-for-byte copy of the source
pixel — no re-tinting, quantization, gamma or alpha blending. Line pixels keep
the source's own tones, so the colour variant reads as a textured tone study
rather than a literal photo.

The background is fixed to white; there is no `background_color` parameter and
no alpha channel in this round.

### Aligning the RGB source (`image_scale`)

When `image_scale != 1` and the source-colour variant is used, the RGB source
must land on the same canvas the geometry ran on. `resize_rgb(rgb,
image_scale)` is the RGB companion of `resize_grayscale`: it shares the exact
same `scaled_size` target and the same fixed BICUBIC filter, so the two never
drift apart.

```python
import numpy as np
from PIL import Image
from halftone_playground import (
    resize_grayscale, resize_rgb, stripe_mask, render_source_color_on_white,
)

src = Image.open("photo.png").convert("RGB")
rgb = np.array(src, dtype=np.uint8)          # Cartesian RGB source
gray = np.array(Image.open("photo.png").convert("L"), dtype=np.uint8)

gray_s = resize_grayscale(gray, 1.3)         # both use the same target size
rgb_s = resize_rgb(rgb, 1.3)                 # and the same BICUBIC filter
mask = stripe_mask(gray_s, period=16, angle_deg=90.0)

out = render_source_color_on_white(mask, rgb_s)   # H×W×3 uint8, white background
```

The compositor never resizes: if `source_rgb` does not match `mask` exactly it
raises `ValueError` rather than silently stretching. Spatial alignment is a
preprocessing concern. As everywhere else, `image_scale` and `period` stay
independent — scaling the canvas does not touch the period.

## CLI

Install the package and a `halftone-playground` command becomes available:

```bash
pip install .
halftone-playground --help
```

The very same behaviour is reachable without installing anything, via the
module entry point (handy from a source checkout with `pythonpath = ["src"]`
already configured, or with `PYTHONPATH=src`):

```bash
python -m halftone_playground --help
```

### Syntax

```
halftone-playground stripe INPUT OUTPUT [options]
halftone-playground spiral INPUT OUTPUT [options]
python -m halftone_playground stripe INPUT OUTPUT [options]
```

`stripe` and `spiral` are required subcommands; `INPUT` and `OUTPUT` are
positional. `OUTPUT` must end in `.png` (case-insensitive).

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--period INT` | `16` | Distance between adjacent line centres, in pixels of the output. |
| `--scale FLOAT` | `1.0` | Rescale the source before geometry runs. Any positive finite value; no upper limit. |
| `--render CHOICE` | `black-on-white` | One of `black-on-white`, `white-on-black`, `source-color`. |
| `--line-width INT` | *(unset)* | **Presence switches the geometry to fixed-width** (see below). |
| `--angle FLOAT` | `90.0` | *(stripe)* Stripe orientation: 90 = vertical, 0/180 = horizontal. |
| `--arms INT` | `1` | *(spiral)* Number of spiral arms. |

### Variable-width vs fixed-width

There is no `--width-mode` flag — one blunt rule decides:

- **`--line-width` omitted** → variable-width geometry (`stripe_mask` /
  `spiral_mask`). The local luminance of the source drives each line's width.
- **`--line-width N`** → fixed-width geometry (`stripe_fixed_mask` /
  `spiral_fixed_mask`). Every full cell draws exactly `N` pixels and the
  source luminance **no longer affects the line width at all**.

  With `--render black-on-white` / `white-on-black` the input image content
  therefore only still decides the canvas size (and, through `--scale`, the
  scaled canvas size) — the drawn geometry is identical whatever the picture
  holds. With `--render source-color` the source RGB still colours the lines,
  exactly as in the variable-width case, because colour and geometry are
  separate layers.

### `image_scale` and the period are independent

`--scale 2 --period 16 --line-width 5` means: double the image dimensions,
then run period `16` / width `5`. Nothing is auto-scaled — `--scale` never
touches `--period` or `--line-width`. If you want the visual density to stay
proportional, adjust those by hand.

### Examples

```bash
# Variable stripe, black on white
halftone-playground stripe input.jpg output.png

# Variable stripe, source colour, rotated, scaled 1.3x
halftone-playground stripe input.jpg output.png --period 16 --angle 45 --scale 1.3 --render source-color

# Fixed stripe, source colour
halftone-playground stripe input.jpg output.png --period 16 --line-width 5 --angle 90 --render source-color

# Variable spiral (square input), 3 arms
halftone-playground spiral square.jpg output.png --period 16 --arms 3

# Fixed spiral, source colour
halftone-playground spiral square.jpg output.png --period 16 --line-width 5 --arms 3 --render source-color

# Module form
python -m halftone_playground stripe input.jpg output.png
```

Before processing, the CLI prints the resolved resolutions, e.g.:

```
Input: 512x512
Output: 666x666
```

### Round 1 boundaries

- **Output is PNG only.** Strict bi-level pixels and exact source-RGB pixels
  must survive a round trip, so lossy formats (JPEG, WebP, ...) are rejected
  as output.
- **Spiral requires a square input.** No automatic crop / fit / letterbox /
  pad is implemented; a non-square input is rejected with a clear error.
- **Input** is any single-frame raster image Pillow can read (PNG and JPEG are
  the primary targets). Multi-frame images are rejected rather than partially
  processed.
- **No batch processing**, no folder traversal, no GUI.
- If `INPUT` and `OUTPUT` resolve to the same file the run is refused.
- Missing output parent directories are created automatically; an existing
  output file is overwritten.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Runtime failure (reading / decoding the input, writing the output). |
| `2` | Usage or validation failure (bad arguments, non-PNG output, non-square spiral input, output would overwrite the input, ...). argparse's own parse errors also exit `2`. |

## GUI

A GUI shell ships as **GUI-001**, extended with a real generation pipeline in
**GUI-002A**. It is a thin entry point **parallel** to the CLI and calls the
same Python APIs directly — it never shells out to the CLI and never
reimplements geometry.

Launch it as a module (from a source checkout with `PYTHONPATH=src`, or after
`pip install .`):

```bash
python -m halftone_playground.gui
```

What GUI-001 does:

- a plain ttk window: **Controls** on the left, a horizontal **Source Preview |
  Result Preview** pair on the right, an inline status strip at the bottom;
- `Open Image…` loads a single-frame raster (PNG / JPEG and anything else
  Pillow can read), shows the filename, the source `W × H`, a source preview
  (aspect-preserving, never cropped or stretched) and the projected output
  resolution;
- mode / width-mode / render controls, with Stripe↔Spiral and Variable↔Fixed
  parameter visibility that **never resets hidden values**;
- inline validation reusing the core contracts (`Scale` positive finite,
  `Period` / `Arms` positive integers, `1 ≤ Line width ≤ Period`);
- a **rectangular source is legal in Spiral mode**. The GUI takes the centered
  maximum square (`side = min(W, H)`) automatically, before any scale. The
  Source Preview still shows the complete original image, and the Output
  readout projects a square for Spiral while Stripe keeps the source aspect
  ratio. There is no manual crop UI and no crop editor.

What **GUI-002A** adds:

- a real **`Generate`** button. It is enabled when a source is loaded, the
  parameters validate and no job is running; it runs the whole preprocess /
  geometry / render chain at **full resolution**;
- the centered maximum-square crop for Spiral is now **actually performed**, in
  the frozen order — crop first, then scale. Stripe is never cropped;
- a real **Result Preview** showing the actual output (a display-only scaled
  copy; the stored result stays full-resolution). No geometry is ever re-run at
  preview resolution;
- the **`current` / `stale`** lifecycle. Changing an output-affecting parameter
  marks the result stale; returning to the same *semantic* configuration makes
  it current again (`Scale = 1` and `Scale = 1.0` are the same configuration).
  A stale result is still displayed, and a new source clears it outright;
- correct output semantics for all 12 combinations:
  **Stripe → `H × W × 3` RGB with no alpha**, **Spiral → `N × N × 4` RGBA**
  whose alpha is exactly the circular support (`255` inside the disc, `0`
  outside, only those two values) and never derived from the line mask;
- a **single background worker** (no queue, no job manager, no Cancel). The Tk
  main thread is the only thread that touches widgets, the hand-off is a
  `queue.Queue` drained from an `after` poll, and the worker thread is a daemon
  so closing the window mid-job is safe.

What **GUI-002B** adds:

- a real **`Save PNG…`** button. It is enabled exactly when a result is
  `current` and no job is running (`none`, `stale` or a running generation
  leaves it disabled), and writes the **stored full-resolution result** to a
  PNG through the native Save As dialog (defaulting to
  `<source_stem>_halftone.png` in the source's directory);
- saving **never re-runs** the pipeline — no preprocess, geometry or renderer,
  and never the scaled-down Result Preview; it encodes the pixels that already
  exist;
- **PNG only** (the suffix is case-insensitive). A `Stripe` result is written as
  a 3-channel **RGB** PNG with no alpha; a `Spiral` result is written as a
  4-channel **RGBA** PNG whose alpha is the circular support, so the four
  corners outside the disc stay transparent. A cancelled dialog is a strict
  no-op, and the current result is left intact even if a write fails.

What this round deliberately does **not** do (later rounds):

- no `Cancel`, no job queue, no second worker, no batch processing, no history,
  no preset manager, no crop UI, no zoom / pan, no theming;
- no packaging, no installer, no portable build.

The frozen GUI decisions (layout, state model, boundaries, stage order) live in
[`docs/GUI_ROUND0_DECISIONS.md`](docs/GUI_ROUND0_DECISIONS.md).

## Layout

- `src/halftone_playground/` — package code (`stripe.py` and `spiral.py`
  geometry, `preprocess.py` scaling, `render.py` output, `cli.py` command line)
- `src/halftone_playground/gui/` — the Tk/ttk GUI (`params.py` / `state.py` /
  `imageutil.py` / `pipeline.py` / `worker.py` are toolkit-free logic; `app.py`
  is the widgets)
- `tests/` — pytest suite
- `scripts/` — small runnable helpers
- `samples/` — generated example images
- `docs/` — decision records and design notes

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
pytest               # full headless test suite (GUI shell tests deselected)
```

The default run is toolkit-free: it never imports `tkinter`. The GUI *shell*
tests drive a real Tk widget tree, so they are opt-in and need a display (or
`Xvfb`):

```bash
pytest -m gui                    # GUI shell tests, needs DISPLAY
xvfb-run -a pytest -m gui        # ...or run them under a virtual display
```

The toolkit-free GUI *logic* tests (`params` / `state` / `imageutil` /
`pipeline` / `worker`) are ordinary tests and always run — including the full
12-combination generation matrix, the Spiral crop-before-scale order and the
alpha semantics, none of which need a display.

## Sample images

Regenerate the Stripe acceptance images into `samples/stripe/`:

```bash
python scripts/generate_stripe_samples.py
```

Regenerate the Spiral acceptance images into `samples/spiral/`:

```bash
python scripts/generate_spiral_samples.py
```

Regenerate the renderer / compositor samples into `samples/render/` (the same
Stripe and Spiral geometry rendered black-on-white, white-on-black and in
source colour):

```bash
python scripts/generate_renderer_samples.py
```

The real-image review set (six renderer outputs plus a mobile-friendly review
sheet) is generated by `scripts/make_renderer_review.py`. It consumes the
public-domain scikit-image sample photograph outside the repository and its
outputs are deliberately not committed.

Regenerate the fixed-width source-colour samples into `samples/fixed_color/`:

```bash
python scripts/generate_fixed_color_samples.py
```

The fixed-width real-image review set (three fixed Stripe and three fixed
Spiral source-colour outputs plus a review sheet) is generated by
`scripts/make_fixed_color_review.py`; like the other review scripts it consumes
the sample photograph outside the repository and is not committed.
