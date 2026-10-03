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

## Current status: Phase 0 environment baseline

This commit only establishes a portable Python project skeleton: packaging,
image I/O smoke checks, and automated tests. **It contains no halftoning,
stripe, threshold, or spiral algorithms yet.**

## Layout

- `src/halftone_playground/` — package code
- `tests/` — pytest suite
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
