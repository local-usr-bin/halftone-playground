"""Preview image helpers (pure, no toolkit import).

The GUI shows a *display preview* of the source.  That preview is a scaled
down copy for on-screen display only; it must never touch the source image,
the preprocess input or the full-resolution output semantics.  Keeping the
scaling arithmetic here, free of any Tk import, makes it testable headless:

* :func:`fit_size` computes the aspect-preserving "contain" size for a preview
  box (no crop, no stretch);
* :func:`load_single_frame` opens an image, rejects animated / multi-frame
  input with a normal-user message, and returns a detached RGB copy;
* :func:`scaled_preview` produces the display-sized RGB array.

The module never writes the source, never mutates the caller's image and never
runs any halftone geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

__all__ = [
    "ImageLoadError",
    "MultiFrameImageError",
    "fit_size",
    "load_single_frame",
    "scaled_preview",
]


class ImageLoadError(Exception):
    """A user-facing image-open failure (never a traceback for the user).

    ``message`` is phrased for a normal person; the underlying technical
    detail is kept on ``__cause__`` for stderr / debug logging.
    """


class MultiFrameImageError(ImageLoadError):
    """The input is animated / multi-frame and is deliberately not accepted."""


def fit_size(
    source_size: tuple[int, int],
    box_size: tuple[int, int],
) -> tuple[int, int]:
    """Return the aspect-preserving size that fits ``source_size`` in a box.

    "Contain" behaviour: scale down (or up) uniformly so the whole image fits
    *inside* ``box_size`` while preserving the aspect ratio.  The result never
    crops and never stretches -- it is exactly the classic
    ``min(box_w / w, box_h / h)`` scale applied to both axes.

    Both inputs are ``(width, height)`` in pixels.  Degenerate boxes (a side
    ``<= 0``) or degenerate sources return a ``1x1`` floor so the caller can
    never ask Pillow to build a zero-sized image.
    """
    src_w, src_h = source_size
    box_w, box_h = box_size
    if src_w <= 0 or src_h <= 0 or box_w <= 0 or box_h <= 0:
        return (1, 1)

    scale = min(box_w / src_w, box_h / src_h)
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))
    return (width, height)


def load_single_frame(path: Path) -> Image.Image:
    """Open ``path`` and return a detached RGB copy of its single frame.

    Parameters
    ----------
    path:
        The image file to open.

    Returns
    -------
    PIL.Image.Image
        A fully loaded, detached ``RGB`` image.  The file handle is closed by
        the time this returns, so callers may safely keep the image around.

    Raises
    ------
    MultiFrameImageError
        If the file is animated / multi-frame.  The first frame is *not*
        silently taken as the source -- the whole file is refused with a plain
        explanation.
    ImageLoadError
        If the file cannot be read or decoded at all.
    """
    try:
        with Image.open(path) as image:
            n_frames = getattr(image, "n_frames", 1)
            if n_frames != 1:
                raise MultiFrameImageError(
                    f"The image has {n_frames} frames. Animated or "
                    "multi-frame images are not supported -- please open a "
                    "single-frame image."
                )
            image.load()
            return image.convert("RGB")
    except MultiFrameImageError:
        raise
    except UnidentifiedImageError as exc:
        raise ImageLoadError(
            f"Cannot read '{Path(path).name}': it is not a recognised image "
            "file."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ImageLoadError(
            f"Cannot read '{Path(path).name}': {exc}"
        ) from exc


def scaled_preview(
    image: Image.Image,
    box_size: tuple[int, int],
) -> Optional[Image.Image]:
    """Return a display-sized RGB copy of ``image`` that fits ``box_size``.

    The preview is a separate, cheap object for on-screen display.  This never
    resizes ``image`` in place and never returns the original: the caller's
    full-resolution source is untouched, so the display preview can never leak
    into the preprocess / geometry / output pipeline.

    Returns ``None`` for a degenerate box (a side ``<= 0``) so the caller can
    simply show nothing rather than build a zero-sized image.
    """
    if box_size[0] <= 0 or box_size[1] <= 0:
        return None
    width, height = fit_size(image.size, box_size)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if (width, height) == image.size:
        return image.copy()
    # NEAREST is deliberate: the preview is a preview, and nearest keeps the
    # per-pixel relationship crisp without implying any output quality.  This
    # is display-only and never used for the real result.
    return image.resize((width, height), resample=Image.NEAREST)
