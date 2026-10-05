"""PNG saving for the GUI shell (pure logic, no toolkit import).

GUI-002B's only job is to write the **already generated, full-resolution**
result to a PNG.  Every decision that can be made without a widget lives here,
so it can be exercised headless -- the same split ``params`` / ``state`` /
``pipeline`` already use:

* :func:`default_save_name` builds the suggested file name from the source;
* :func:`resolve_png_path` applies the "PNG only, case-insensitive, default
  extension" rule to whatever the native dialog returned;
* :func:`save_png` encodes the stored result array as-is.

What this module deliberately does **not** do
---------------------------------------------

* it never re-runs any preprocess / geometry / renderer step and never calls
  :func:`halftone_playground.gui.pipeline.generate` -- the array it is handed
  *is* the finished result;
* it never resizes, colour-manages, palette-maps, flattens alpha or adds a
  background fill: the pixels are wrapped with :func:`PIL.Image.fromarray` at
  the result's own mode (``"RGB"`` or ``"RGBA"``) -- a wrap, not a ``convert``
  -- and written with an explicit ``format="PNG"``, so the encoder is chosen by
  the caller and never inferred from a file extension;
* it invents no overwrite/backup/versioning scheme: the native Save dialog
  already asks before overwriting, and a second, GUI-level rule would only be a
  worse duplicate.

Consequences the shell relies on: a Stripe result stays a 3-channel ``RGB``
PNG, and a Spiral result stays a 4-channel ``RGBA`` PNG whose alpha is the
circular support (``255`` inside the disc, ``0`` outside).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

__all__ = [
    "DEFAULT_SAVE_NAME",
    "NotPngError",
    "PNG_GLOB",
    "PNG_SUFFIX",
    "SaveError",
    "default_save_name",
    "resolve_png_path",
    "save_png",
]

#: The single output format of GUI v1.
PNG_SUFFIX = ".png"
PNG_GLOB = "*.png"

#: Fallback suggested name used only when no source is known (which the shell
#: never reaches, because Save is disabled without a current result).  Kept so
#: :func:`default_save_name` has a total, testable contract.
DEFAULT_SAVE_NAME = "halftone.png"


class SaveError(Exception):
    """A save failure phrased for a normal user, never a traceback."""


class NotPngError(SaveError):
    """The chosen file name is explicitly not a PNG.

    Raised for a *declared* non-PNG suffix such as ``result.jpg``.  The shell
    turns this into a plain "only PNG is supported" message and writes nothing;
    it never silently substitutes or appends another extension.
    """


def default_save_name(source_path: Union[str, Path]) -> str:
    """Return the suggested Save name for ``source_path``.

    ``<source_stem>_halftone.png`` -- e.g. ``photo.JPG`` -> ``photo_halftone.png``.
    The original extension is dropped, never appended, so the suggestion can
    never look like a JPEG that is really a PNG.
    """
    stem = Path(source_path).stem
    if not stem:
        return DEFAULT_SAVE_NAME
    return f"{stem}_halftone.png"


def resolve_png_path(path: Union[str, Path]) -> Path:
    """Normalise a dialog result to a ``.png`` path, or raise :class:`NotPngError`.

    Rules (frozen for GUI-002B):

    * a **missing** suffix gets the default ``.png`` appended -- this mirrors
      the dialog's own ``defaultextension`` behaviour and is *not* the banned
      ``.jpg`` -> ``.jpg.png`` rewrite, which applies to an *explicit*
      other-format suffix;
    * ``.png`` in any case (``.png`` / ``.PNG`` / ``.Png``) is accepted as-is;
    * any other explicit suffix raises :class:`NotPngError`.

    No file is touched here; this is pure path arithmetic.
    """
    path = Path(path)
    suffix = path.suffix
    if not suffix:
        return Path(f"{path}{PNG_SUFFIX}")
    if suffix.lower() == PNG_SUFFIX:
        return path
    raise NotPngError(
        "Only PNG files are supported. Please choose a name ending in '.png'."
    )


def save_png(
    pixels: np.ndarray,
    mode: str,
    path: Union[str, Path],
) -> Path:
    """Encode ``pixels`` (the full-resolution result) to ``path`` as a PNG.

    ``mode`` is the result's own Pillow mode and is passed straight to
    :func:`PIL.Image.fromarray`, so an ``"RGB"`` result is written as a
    3-channel PNG with no alpha and an ``"RGBA"`` result keeps its 4 channels
    and its exact alpha values.  Nothing is converted, resized or filled.

    Returns the resolved :class:`~pathlib.Path` that was written.

    Raises
    ------
    NotPngError
        If ``path`` has an explicit non-PNG suffix.
    SaveError
        If the file cannot be written (permissions, a missing directory, a full
        disk, ...).  The message is a plain sentence for the user.
    """
    target = resolve_png_path(path)
    try:
        # ``fromarray`` *wraps* the buffer at the given mode; it is not a
        # colour conversion, so the stored pixels reach the encoder unchanged.
        image = Image.fromarray(pixels, mode=mode)
        # An explicit format keeps the encoder independent of the file name.
        image.save(target, format="PNG")
    except OSError as exc:
        raise SaveError(f"Could not save '{target.name}': {exc}") from exc
    return target
