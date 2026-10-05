# GUI Round 0 — Decision Record

Scope: the decisions frozen for the GUI v1 shell, kept short so a future task
can recover the intent without reading the whole history. This is **not** a UI
specification; it only records what was decided and why.

Status: GUI-001 (shell) implements the structure below. GUI-002A implements the
real Generate pipeline, the worker, the Result Preview, the `current` / `stale`
lifecycle and the positive White-on-black polarity. `Save PNG…` and the
packaging rounds remain **not implemented**.

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
- In GUI-001 both `Generate` and `Save PNG…` were **disabled**. In GUI-002A
  `Generate` is live; `Save PNG…` is still disabled (GUI-002B).

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
  GUI-002A implements this with a **normalized semantic key**
  (`gui/pipeline.py: GenerationKey`): the comparison is on canonical values,
  so `Scale = 1` and `Scale = 1.0` are the same configuration, and parameters
  that cannot affect the output (a hidden `Angle` in Spiral, a hidden `Arms`
  in Stripe, a hidden `Line width` in Variable width) never invalidate a
  result.
- **A new source clears the result outright** — it is never kept as `stale`.
  GUI-002A drops the stored full-resolution pixels as well as the metadata.
- **Inputs lock while a job runs:** `Open Image`, `Generate`, `Save PNG` and
  all parameter controls are disabled during a running job.
- **Single worker job:** at most one generate at a time.
- **No Cancel in GUI v1.**
- The worker (GUI-002A) does preprocess / geometry / rendering and never
  touches Tk widgets; the Tk main thread receives the finished result and
  updates the UI. Concretely, the hand-off is a `queue.Queue` drained from a
  `root.after` poll, the single worker thread is a **daemon** (so closing the
  window mid-job needs no `join` and no cancellation), and a runtime failure
  is reported as a plain sentence in a message box — never a raw traceback.
- **Output format frozen in GUI-002A:** a Stripe result is `H × W × 3` RGB
  with **no alpha**; a Spiral result is `N × N × 4` RGBA whose alpha is
  exactly the circular support (`255` inside the disc, `0` outside, only those
  two values), built from the *same* `circular_support` helper the Spiral
  geometry uses and **never** from the line mask, so every pixel inside the
  disc — line or background — is fully opaque.

## Render polarity (GUI-002A correction)

- **White on black keeps the source positive.** The white line must grow as the
  source gets *brighter* (`W_white = P · G`), so a white area stays white and a
  black area stays black. It is **not** a photographic negative of the
  black-on-white result.
- This is implemented at the **application** level, not in the geometry. The
  variable-width cores keep their single frozen formula `W = P · (1 − G)` and
  their API; the pipeline simply feeds them the **inverted** grayscale
  (`preprocess.invert_luminance`) on the Variable + White-on-black path.
- Applied **after** the crop and the Scale, so the Spiral centred crop, the
  RGB/mask alignment and the output sizes are untouched. Stripe and Spiral are
  treated identically.
- **Black on white is unchanged** (`W = P · (1 − G)` on the raw luminance), and
  only the grayscale is ever inverted — a `Source-color` result still copies the
  original Cartesian source pixels.
- **Fixed width is not affected:** it derives no width from luminance, so a
  "negative" has no meaning there and both polarities produce identical output.
- **Alpha is not affected:** it is still decided only by `circular_support`.


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
image is cropped, resized or transformed there): Stripe projects from the
original `W × H`, Spiral projects from `scaled_size(side, side, scale)` with
`side = min(W, H)`, still reusing the existing `scaled_size` rounding rule.

GUI-002A performs that crop for real, in exactly this order:
`source → centered square crop → scale → Spiral geometry`. The crop is applied
to the grayscale *and* to the Cartesian RGB companion with the **same** box
and the **same** scale.

For a **Source-color** Spiral result, the grayscale and RGB paths use the
**same** centered square crop box and the **same** Scale; the RGB is only the
final Cartesian source colour and still never enters the polar pipeline. In
GUI-002A this is enforced structurally: `gui/pipeline.py` does not import
OpenCV, so it cannot unwrap or remap colour even by accident.


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

1. **GUI Shell** (GUI-001) — layout, controls, state model, Open Image,
   Source Preview, validation, disabled actions.
2. **Core integration** (GUI-002A / GUI-002B) — GUI-002A delivered the real
   Generate pipeline, the single worker, the Result Preview and the
   `current` / `stale` lifecycle. GUI-002B delivers the Save PNG workflow.
3. **Windows visual / manual acceptance** (GUI-003) — fonts, DPI, window size,
   file dialog feel, preview proportions.
4. **Packaging.**

Headless / Linux / cloud testing cannot substitute for the Windows manual
acceptance stage.
