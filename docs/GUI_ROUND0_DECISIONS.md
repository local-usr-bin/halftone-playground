# GUI Round 0 — Decision Record

Scope: the decisions frozen for the GUI v1 shell, kept short so a future task
can recover the intent without reading the whole history. This is **not** a UI
specification; it only records what was decided and why.

Status: GUI-001 (shell) implements the structure below. The Generate / Save
pipelines and the stale-result rules are **GUI-002+** and are not implemented
yet.

## Toolkit and style

- **Tkinter + ttk**, stdlib only. No extra UI dependency.
- A plain, restrained ttk look. Deliberately **no** card design, shadows,
  gradients, animation, custom theme, skin system or complex menu bar.
- Product positioning: a small, direct, easy-to-maintain halftone playground,
  not a commercial UI.

## Layout

- **Left:** Controls.
- **Right:** `Source Preview | Result Preview`, the two previews **side by
  side**, horizontally.
- **Bottom:** a single inline status / validation strip.
- Top bar: `Open Image…` and the current source filename.
- Not required to be pixel-perfect; ordinary ttk layout only, no UI framework.

## Controls

- **Mode:** Stripe / Spiral.
- **Width:** Variable / Fixed.
- **Render:** Black on white / White on black / Source color — **always all
  three, in every mode**.
- **Parameters:** Period, Scale, Angle, Arms, Line width.
- **Actions:** `Generate`, `Save PNG…` (explicitly *not* "Generate Preview" —
  `Generate` produces the real **full-resolution** result; the preview is only
  a display layer).

## Parameter visibility (and hidden values)

- **Mode = Stripe** → show `Angle`, hide `Arms`.
- **Mode = Spiral** → show `Arms`, hide `Angle`.
- **Width = Variable** → hide `Line width`.
- **Width = Fixed** → show `Line width`.
- **Hiding a parameter never resets it.** `Angle = 45` survives
  Stripe → Spiral → Stripe; `Line width` behaves the same. Visibility is a
  presentation decision only, implemented as an independent, testable pure
  function (`gui/params.py: visible_parameters`).

## Actions and geometry

- `Generate` **never auto-runs** geometry on a parameter change. Changing
  `Scale` only *projects* the output resolution; it never resizes an image,
  runs geometry or does a Spiral polar transform.
- `Generate` produces the real full-resolution result. The display preview and
  the real result are **separate**: resizing a preview only affects the GUI and
  must never modify the source image, the preprocess input or the output
  semantics.
- In GUI-001 both `Generate` and `Save PNG…` are **disabled**; no real
  pipeline exists.

## State model (concepts, not a giant enum)

- **Source:** `none` | `loaded`.
- **Job:** `idle` | `running`.
- **Result:** `none` | `current` | `stale`.
- A single explicit object with a few fields plus small pure helpers — no
  `READY_WITH_...` combination state machine.
- **`current` / `stale`:** a result is `current` after a successful generate;
  an output-affecting parameter change makes it `stale`. A stale result may
  still be **displayed**, but **Save is disabled**. If the parameters return to
  the same semantic configuration, the result may become `current` again.
- **A new source clears the result outright** — it is never kept as `stale`.
- **Inputs lock while a job runs:** `Open Image`, `Generate`, `Save PNG` and
  all parameter controls are disabled during a running job.
- **Single worker job:** at most one generate at a time.
- **No Cancel in GUI v1.**
- The worker (GUI-002) does preprocess / geometry / rendering and never touches
  Tk widgets; the Tk main thread receives the finished result and updates the
  UI.

## Spiral rectangular source (revised after Windows manual acceptance)

**Superseded decision.** An earlier Round 0 note said the GUI *rejects* a
non-square source for Spiral (*"Spiral currently requires a square image."*).
Windows manual acceptance (GUI-001 correction) replaced it: the GUI now
**accepts a rectangular source for Spiral**.

The frozen behaviour is:

- **GUI accepts a rectangular source for Spiral.** A non-square source with
  `Mode = Spiral` is a legal configuration, never a validation error.
- **Automatic centered maximum-square crop.** The user is responsible for
  placing the key subject near the centre of the picture; the GUI takes the
  largest possible square from the centre as the effective Spiral input:
  `side = min(width, height)`, `left = (width - side) // 2`,
  `top = (height - side) // 2`. A 1 px remainder (odd difference) stays on the
  right / bottom. There is no sub-pixel crop and no other rule.
- **Crop before Scale.** The order is fixed: `source → centered square crop →
  scale → Spiral geometry`. The whole rectangle is never scaled first and then
  cropped.
- **Source Preview always shows the complete source.** Even with
  `Mode = Spiral`, the Source Preview is the full original image; it is never
  cropped, masked or boxed.
- **Result Preview represents the actual output.** A Stripe result keeps the
  source aspect ratio; a Spiral result is square.
- **No manual crop UI.** No crop editor, no crop box, no dimming mask, no
  drag-to-recompose, no zoom / pan crop UI.
- **Spiral core remains square-only.** `spiral_mask` keeps its square grayscale
  input contract; it is never taught to accept a rectangular image.
- **GUI / application preprocess owns rectangular → square.** The centre-crop
  lives in the GUI / application layer, before the core is called, so the core
  never learns the original image was rectangular:
  `rectangular source → GUI/application preprocess → centered square → scale →
  spiral_mask(square)`.
- **CLI R1 unchanged.** The CLI keeps its current non-square Spiral rejection;
  whether the GUI and CLI eventually share the auto-crop is a later discussion.

The output-resolution readout follows the same split (a projection only — no
image is cropped, resized or transformed in GUI-001): Stripe projects from the
original `W × H`, Spiral projects from `scaled_size(side, side, scale)` with
`side = min(W, H)`, still reusing the existing `scaled_size` rounding rule.

For a future **Source-color** Spiral result, the grayscale and RGB paths must
use the **same** centered square crop box and the **same** Scale; the RGB is
only the final Cartesian source colour and still never enters the polar
pipeline.

## Integration boundary

- The GUI calls the **Python APIs directly**, exactly like the CLI. It is a
  thin entry point **parallel** to the CLI.
- **No** `GUI → subprocess CLI → core`. No shell commands from the GUI. No
  second copy of geometry. No changes to core algorithm semantics or to the CLI
  contract.
- `Save PNG…` is the only output format in GUI v1.

## Explicitly out of scope for GUI v1

No complex menu bar, recent files, drag-and-drop, theme selector, skin system,
settings page, history, batch queue, preset manager, export profiles, zoom /
pan / crop editor, before/after slider, interactive canvas editor, undo / redo,
project files, plugin architecture, arbitrary background colour, alpha /
transparent background, installer or portable EXE.

## Stage order

1. **GUI Shell** (this round) — layout, controls, state model, Open Image,
   Source Preview, validation, disabled actions.
2. **Core integration** (GUI-002) — real Generate / Save, worker, stale result.
3. **Windows visual / manual acceptance** (GUI-003) — fonts, DPI, window size,
   file dialog feel, preview proportions.
4. **Packaging.**

Headless / Linux / cloud testing cannot substitute for the Windows manual
acceptance stage.
