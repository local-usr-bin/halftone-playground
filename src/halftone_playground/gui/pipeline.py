"""GUI generate pipeline: request snapshot, semantic key and result (pure).

This module is the *whole* of GUI-002A's image work, and it deliberately
contains **no Tk import and no widget knowledge**.  The worker thread and the
Tk shell both talk to it; it never talks back.

What lives here
---------------

* :class:`GenerationRequest` -- an immutable snapshot of everything that can
  affect the output.  Built on the Tk main thread, then handed to the worker.
  Because it is a frozen dataclass holding only plain values (plus a path and
  a size), it cannot be mutated by a later user edit: "what the user asked
  for" and "what the worker is computing" can never drift apart.
* :class:`GenerationKey` -- the *normalized semantic* identity of a request,
  derived only from parameters that can change the pixels.  It is what makes
  ``current`` / ``stale`` decidable: the live parameters are re-keyed and
  compared to the key stored with the result.
* :func:`generate` -- the pure pipeline for all 12 combinations
  (Stripe|Spiral x Variable|Fixed x BlackOnWhite|WhiteOnBlack|SourceColor).
* :class:`GenerationResult` -- the real, **full-resolution** result plus its
  provenance (mode, size, whether it carries alpha, render variant).

Frozen semantics this module implements
---------------------------------------

Stripe
    never crops.  The source grayscale / RGB is scaled with the existing
    :func:`halftone_playground.preprocess.resize_grayscale` /
    :func:`resize_rgb` -- there is no second resize implementation here.

Spiral
    crops **first**, then scales: ``source -> centered maximum square crop ->
    scale -> Spiral geometry``.  The crop box is the frozen
    ``side = min(W, H)`` / floor-centred box from
    :mod:`halftone_playground.gui.params`, and it is applied with Pillow's own
    integer crop, so it is exact.  Scaling the full rectangle first and
    cropping afterwards is explicitly *not* what happens.

Source colour
    For Spiral, the RGB companion is cropped with the **exact same** centred
    square box and scaled by the **same** factor as the grayscale, so the mask
    and the colour pixels line up pixel-for-pixel.  The RGB array then goes
    straight to
    :func:`halftone_playground.render.render_source_color_on_white` at the
    final Cartesian coordinates: **no RGB array ever enters ``warpPolar``**,
    there is no forward or inverse polar unwrap of colour, and therefore no
    extra colour resampling.

Output format
    * Stripe result: ``H x W x 3`` ``uint8`` RGB, **no alpha channel**.
    * Spiral result: ``N x N x 4`` ``uint8`` RGBA, where the alpha channel is
      ``255`` strictly inside the circular support and ``0`` outside, **only
      those two values**, and is **never** derived from ``line_mask``.  Every
      pixel inside the disc is fully opaque, including background pixels.

White-on-black polarity
    ``white_on_black`` keeps the source **positive**: the *white* line grows as
    the source gets brighter, so a white area stays white and a black area
    stays black.  The variable-width cores express the opposite, black-on-white
    convention (``W = P * (1 - G)``), so this module inverts the
    **already cropped and scaled** grayscale before handing it to the geometry
    (see :func:`halftone_playground.preprocess.invert_luminance`).  The cores --
    and their width formula and API -- are untouched, and both Stripe and
    Spiral get the same treatment.  Fixed-width geometry derives no width from
    the source, so it is never inverted, and ``source_color`` is left alone.

The alpha channel comes from :func:`halftone_playground.preprocess.circular_support`
-- the same single definition the Spiral geometry uses -- so the opaque region
and the visible disc can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..preprocess import (
    circular_support,
    invert_luminance,
    resize_grayscale,
    resize_rgb,
    scaled_size,
)
from ..render import (
    render_black_on_white,
    render_source_color_on_white,
    render_white_on_black,
)
from ..spiral import spiral_fixed_mask, spiral_mask, _polar_center, _support_radius
from ..stripe import stripe_fixed_mask, stripe_mask
from . import params as gui_params

__all__ = [
    "GenerationKey",
    "GenerationRequest",
    "GenerationResult",
    "PipelineError",
    "STALE_RELEVANT_FIELDS",
    "generate",
    "render_rgba",
]


class PipelineError(Exception):
    """A generation failure phrased for a normal user, never a traceback.

    The worker catches this and hands the message to the main thread; anything
    else escaping the worker is a programming bug and is reported separately
    with a generic message so no raw traceback ever reaches the GUI.
    """


# --------------------------------------------------------------------------
# generation key: the normalized semantic identity of a request
# --------------------------------------------------------------------------


def _canonical_text(text: str) -> Optional[str]:
    """Canonical form of a raw numeric field, or ``None`` when unparseable.

    ``int`` fields are canonicalised as plain integers; float fields have any
    trailing ``.0`` dropped.  That is what makes ``"1"`` and ``"1.0"`` the
    same semantic configuration -- the requirement that a user who types
    ``1.0`` instead of ``1`` does not get a bogus ``stale`` result.
    """
    value = gui_params.parse_float(text)
    if value is None:
        return None
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _canonical_int_text(text: str) -> Optional[str]:
    value = gui_params.parse_int(text)
    if value is None:
        return None
    return str(value)


#: The only parameter fields that can change the pixels.
#:
#: Deliberately a named, closed set rather than "everything on
#: :class:`RenderParams`": the raw ``*_text`` mirrors are view state, and a
#: future field that does *not* affect output must be added here explicitly
#: rather than silently invalidating results.  ``line_width`` is included
#: because the fixed-width path uses it, but it is only *keyed* when it is
#: actually visible.
STALE_RELEVANT_FIELDS = (
    "mode",
    "width_mode",
    "render",
    "period",
    "scale",
    "angle",
    "arms",
    "line_width",
)


@dataclass(frozen=True)
class GenerationKey:
    """Normalized semantic identity of a :class:`GenerationRequest`.

    Two requests that would produce byte-identical output have equal keys; two
    requests that could produce different output have different keys.  All
    values are canonical strings so equality is plain and stable::

        GenerationKey.from_params(p) == GenerationKey.from_params(p2)
        # True when p and p2 mean the same thing, e.g. scale "1" vs "1.0"
    """

    mode: str
    width_mode: str
    render: str
    period: str
    scale: str
    angle: Optional[str]
    arms: Optional[str]
    line_width: Optional[str]

    @classmethod
    def from_params(cls, p: gui_params.RenderParams) -> "GenerationKey":
        """Derive the key from the live parameters.

        Only *visible* parameters are keyed, exactly as validation works: a
        hidden field (``Angle`` in Spiral, ``Arms`` in Stripe, ``Line width``
        in Variable width) cannot influence the output, so changing it must
        never mark a result stale.  A field that is visible but currently
        unparseable keys as ``None``; the result is then never considered
        current, which is correct -- that configuration cannot be generated.
        """
        visible = gui_params.visible_parameters(p.mode, p.width_mode)
        return cls(
            mode=p.mode,
            width_mode=p.width_mode,
            render=p.render,
            period=_canonical_int_text(p.period_text) or "",
            scale=_canonical_text(p.scale_text) or "",
            angle=(
                _canonical_text(p.angle_text)
                if gui_params.PARAM_ANGLE in visible
                else None
            ),
            arms=(
                _canonical_int_text(p.arms_text)
                if gui_params.PARAM_ARMS in visible
                else None
            ),
            line_width=(
                _canonical_int_text(p.line_width_text)
                if gui_params.PARAM_LINE_WIDTH in visible
                else None
            ),
        )


# --------------------------------------------------------------------------
# immutable request snapshot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    """Everything the worker needs, frozen at the moment ``Generate`` ran.

    Built on the Tk main thread from the live parameters and the loaded
    source, then handed to the worker.  It holds **only plain immutable
    values** (``str`` / ``int`` / ``float`` / ``Path`` / a size tuple), so no
    later widget edit can reach into the running job.

    ``source_path`` is a path rather than a decoded image on purpose: the
    worker re-reads the file, so the full-resolution source never has to be
    shared across threads.
    """

    source_path: Path
    source_size: tuple[int, int]

    mode: str
    width_mode: str
    render: str

    period: int
    scale: float
    angle: float
    arms: int
    line_width: int

    key: GenerationKey

    @classmethod
    def from_params(
        cls,
        p: gui_params.RenderParams,
        source_path: Path,
        source_size: tuple[int, int],
    ) -> "GenerationRequest":
        """Snapshot ``p`` for ``source_path``.

        Raises :class:`PipelineError` when a visible field is not a value this
        round can actually run -- the caller (the shell) additionally checks
        :func:`halftone_playground.gui.params.validate` first, so this is a
        belt-and-braces guard rather than the primary user feedback.
        """
        errors = gui_params.validate(p)
        if errors:
            first = next(iter(errors.values()))
            raise PipelineError(first)

        period = gui_params.parse_int(p.period_text)
        scale = gui_params.parse_float(p.scale_text)
        angle = gui_params.parse_float(p.angle_text)
        arms = gui_params.parse_int(p.arms_text)
        line_width = gui_params.parse_int(p.line_width_text)

        if period is None or scale is None or angle is None or arms is None:
            raise PipelineError("Some parameters are not valid numbers.")
        if line_width is None:
            raise PipelineError("Line width must be a whole number.")

        return cls(
            source_path=Path(source_path),
            source_size=(int(source_size[0]), int(source_size[1])),
            mode=p.mode,
            width_mode=p.width_mode,
            render=p.render,
            period=int(period),
            scale=float(scale),
            angle=float(angle),
            arms=int(arms),
            line_width=int(line_width),
            key=GenerationKey.from_params(p),
        )


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationResult:
    """The real, full-resolution output plus everything needed to place it.

    ``pixels`` is the *actual* output at full resolution -- never a preview.
    ``mode`` is the Pillow mode string (``"RGB"`` or ``"RGBA"``) and
    ``has_alpha`` is the single place the shell looks to decide whether alpha
    exists at all.  ``key`` is the provenance: the semantic configuration that
    produced these pixels, which is what ``current`` / ``stale`` compares
    against.
    """

    pixels: np.ndarray
    mode: str
    width: int
    height: int
    render: str
    kind: str
    key: GenerationKey

    @property
    def has_alpha(self) -> bool:
        return self.mode == "RGBA"

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


def _spiral_side(source_size: tuple[int, int]) -> int:
    """Centered maximum-square side, straight from the frozen GUI contract."""
    return gui_params.center_square_side(source_size)


def _crop_rgb(
    rgb: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    """Apply the frozen centred square crop to a Cartesian RGB array.

    Done with Pillow's ``Image.crop`` on the *array* rather than by numpy
    slicing so the crop follows exactly the same half-open ``(left, top,
    right, bottom)`` convention as the source preview path, and so a future
    change to the crop rule has one obvious place to land.

    ``box`` comes from :func:`halftone_playground.gui.params.center_square_crop_box`,
    so the RGB crop and the grayscale crop are the same rectangle by
    construction.
    """
    left, top, right, bottom = box
    if (left, top, right, bottom) == (0, 0, rgb.shape[1], rgb.shape[0]):
        # Already the full frame: skip the copy so the common square case
        # cannot pay for an extra allocation.
        return rgb
    return np.array(
        Image.fromarray(rgb, mode="RGB").crop((left, top, right, bottom)),
        dtype=np.uint8,
    )


def _crop_gray(
    gray: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    """Grayscale twin of :func:`_crop_rgb`, using the identical box."""
    left, top, right, bottom = box
    if (left, top, right, bottom) == (0, 0, gray.shape[1], gray.shape[0]):
        return gray
    return np.array(
        Image.fromarray(gray, mode="L").crop((left, top, right, bottom)),
        dtype=np.uint8,
    )


def _load_source_arrays(
    request: GenerationRequest,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the source file once and return ``(gray, rgb)`` full-frame arrays."""
    try:
        with Image.open(request.source_path) as image:
            image.load()
            n_frames = getattr(image, "n_frames", 1)
            if n_frames != 1:
                raise PipelineError(
                    "The image has more than one frame; only single-frame "
                    "images can be generated."
                )
            gray = np.array(image.convert("L"), dtype=np.uint8)
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    except PipelineError:
        raise
    except OSError as exc:
        raise PipelineError(
            f"Cannot read '{request.source_path.name}': {exc}"
        ) from exc
    return gray, rgb


def _spiral_canvas(
    gray: np.ndarray,
    rgb: np.ndarray,
    request: GenerationRequest,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop-then-scale both grayscale and RGB for Spiral.

    Returns ``(scaled_gray, scaled_rgb, box)``.  The **same** box and the
    **same** scale are applied to both arrays, which is what keeps the mask
    and the colour pixels aligned.
    """
    box = gui_params.center_square_crop_box(request.source_size)

    cropped_gray = _crop_gray(gray, box)
    cropped_rgb = _crop_rgb(rgb, box)

    # Crop first, scale second -- never the other way round.
    scaled_gray = resize_grayscale(cropped_gray, request.scale)
    scaled_rgb = resize_rgb(cropped_rgb, request.scale)

    if scaled_gray.shape != scaled_rgb.shape[:2]:
        # Both resizers share scaled_size() and the fixed filter, so this can
        # only mean the contract was broken somewhere else.  Fail loudly
        # rather than render misaligned colour.
        raise PipelineError(
            "Internal error: the grayscale and colour canvases disagree "
            f"({scaled_gray.shape} vs {scaled_rgb.shape[:2]})."
        )
    return scaled_gray, scaled_rgb, box


def _spiral_mask(
    scaled_gray: np.ndarray, request: GenerationRequest
) -> np.ndarray:
    """Run the frozen Spiral geometry on the cropped+scaled square."""
    side = int(scaled_gray.shape[0])
    if request.width_mode == gui_params.WIDTH_FIXED:
        return spiral_fixed_mask(
            side, request.period, request.line_width, request.arms
        )
    return spiral_mask(scaled_gray, request.period, request.arms)


def _stripe_mask(
    scaled_gray: np.ndarray, request: GenerationRequest
) -> np.ndarray:
    """Run the frozen Stripe geometry on the (uncropped) canvas."""
    height, width = scaled_gray.shape
    if request.width_mode == gui_params.WIDTH_FIXED:
        return stripe_fixed_mask(
            (height, width), request.period, request.line_width, request.angle
        )
    return stripe_mask(scaled_gray, request.period, request.angle)


def _geometry_grayscale(
    scaled_gray: np.ndarray, request: GenerationRequest
) -> np.ndarray:
    """Return the luminance the *variable-width* geometry should see.

    The frozen cores map luminance to line width in the black-on-white sense
    (``W = P * (1 - G)``), alongside ``render_black_on_white``.  For the
    ``white_on_black`` variant the line is white, so its width has to grow with
    the source's brightness instead of shrinking -- ``W_white = P * G``.  The
    cheapest exact way to get that out of the unchanged cores is to hand them
    the inverted luminance, so an application-level decision never leaks into
    the geometry contract.

    Scope of the correction (deliberately narrow):

    * only ``render == white_on_black`` **and** ``width_mode == variable`` is
      inverted -- fixed-width geometry reads no grayscale at all, so a
      "negative" has no meaning there;
    * ``black_on_white`` keeps its existing ``W = P * (1 - G)`` behaviour
      untouched, and ``source_color`` never gets here;
    * the inversion is applied to the grayscale **after** the crop and scale
      steps, so the Spiral centred crop and the RGB/mask alignment are
      unaffected;
    * only the grayscale is inverted.  The RGB companion -- used solely to pick
      a line pixel's colour in the ``source_color`` path -- keeps the original
      source values, which is why this returns a grayscale-only array.
    """
    if (
        request.width_mode == gui_params.WIDTH_VARIABLE
        and request.render == gui_params.RENDER_WHITE_ON_BLACK
    ):
        return invert_luminance(scaled_gray)
    return scaled_gray


def render_rgba(
    rgb: np.ndarray, mask: np.ndarray, *, opaque: np.ndarray
) -> np.ndarray:
    """Compose a Cartesian RGB, a line mask and an opacity mask into RGBA.

    Kept public (and pure) so the alpha semantics can be tested directly.

    ``opaque`` is the circular support.  The alpha channel is exactly
    ``where(opaque, 255, 0)``: only ``0`` / ``255`` can appear, every pixel
    inside the disc is fully opaque, and the line mask **never** influences
    alpha.  Colour is the renderer's ordinary Cartesian output, so colour data
    still never passes through any polar transform.
    """
    if rgb.shape[:2] != mask.shape or rgb.shape[:2] != opaque.shape:
        raise PipelineError(
            "Internal error: colour, mask and support shapes disagree "
            f"({rgb.shape[:2]}, {mask.shape}, {opaque.shape})."
        )
    height, width = mask.shape
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = np.where(opaque, np.uint8(255), np.uint8(0))
    return out


def generate(request: GenerationRequest) -> GenerationResult:
    """Run the whole pipeline for ``request`` and return the real result.

    This is the single entry point the worker calls.  It is a pure function of
    ``request`` plus the file at ``request.source_path``: no globals, no
    widget access, no shared mutable state.

    Returns a :class:`GenerationResult` at full resolution:

    * Stripe -> ``H x W x 3`` ``uint8`` RGB (no alpha);
    * Spiral -> ``N x N x 4`` ``uint8`` RGBA with the circular-support alpha.

    Raises :class:`PipelineError` for anything a user should see as a plain
    sentence (unreadable file, a parameter combination the core rejects, an
    allocation the machine cannot satisfy).
    """
    gray, rgb = _load_source_arrays(request)

    # ``source_size`` is ``(width, height)``; the arrays are ``(height, width)``.
    height, width = gray.shape[:2]
    if (width, height) != tuple(request.source_size):
        # The file changed under us between load and Generate.  Using the
        # snapshot size would silently crop the wrong rectangle, so say so.
        raise PipelineError(
            "The source image changed on disk. Please open it again."
        )

    if request.mode == gui_params.MODE_SPIRAL:
        scaled_gray, scaled_rgb, _box = _spiral_canvas(gray, rgb, request)
        # White-on-black keeps the source positive: the geometry sees inverted
        # luminance on the variable-width path only.  RGB is never inverted.
        mask = _spiral_mask(_geometry_grayscale(scaled_gray, request), request)

        side = int(mask.shape[0])
        center = _polar_center(side)
        support_radius = _support_radius(side)
        opaque = circular_support(side, center, support_radius)

        if request.render == gui_params.RENDER_SOURCE_COLOR:
            colored = render_source_color_on_white(mask, scaled_rgb)
        else:
            colored = _render_flat(mask, request)
        pixels = render_rgba(colored, mask, opaque=opaque)
        return GenerationResult(
            pixels=pixels,
            mode="RGBA",
            width=side,
            height=side,
            render=request.render,
            kind=request.mode,
            key=request.key,
        )

    # ---- Stripe: never cropped, aspect ratio preserved -------------------
    scaled_gray = resize_grayscale(gray, request.scale)
    # Same positive white-on-black correction as the Spiral path above.
    mask = _stripe_mask(_geometry_grayscale(scaled_gray, request), request)
    scaled_rgb = resize_rgb(rgb, request.scale)

    if request.render == gui_params.RENDER_SOURCE_COLOR:
        pixels = render_source_color_on_white(mask, scaled_rgb)
    else:
        pixels = _render_flat(mask, request)

    height, width = mask.shape
    return GenerationResult(
        pixels=pixels,
        mode="RGB",
        width=width,
        height=height,
        render=request.render,
        kind=request.mode,
        key=request.key,
    )


def _render_flat(mask: np.ndarray, request: GenerationRequest) -> np.ndarray:
    """Render a bi-level variant and lift it to RGB.

    ``black-on-white`` / ``white-on-black`` are inherently single-channel
    images.  The GUI result for Stripe is required to be ``H x W x 3`` RGB
    (no alpha), so the single channel is replicated across three channels
    *after* the renderer ran -- the renderer itself is untouched and no colour
    decision is made here.
    """
    if request.render == gui_params.RENDER_BLACK_ON_WHITE:
        flat = render_black_on_white(mask)
    elif request.render == gui_params.RENDER_WHITE_ON_BLACK:
        flat = render_white_on_black(mask)
    else:  # pragma: no cover - guarded by the caller
        raise PipelineError(f"Unknown render variant {request.render!r}.")
    return np.repeat(flat[:, :, None], 3, axis=2).astype(np.uint8)
