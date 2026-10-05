"""GUI parameters: modes, renderers, visibility and validation (pure logic).

This module is deliberately **free of any GUI toolkit import**.  It holds the
small, testable decisions that the Tk shell needs, so they can be exercised
without a display server:

* the fixed choice sets (mode / width mode / render variant);
* which parameter widgets are *visible* for a given mode / width mode;
* the rule that hiding a parameter must **never reset its value**;
* per-field validation that reuses the existing core contracts instead of
  inventing a second set of business rules;
* the Spiral *effective source* semantics: a rectangular source is centered-
  cropped to its largest square (see :func:`effective_source_size` /
  :func:`center_square_crop_box`) **before** any scale is applied, so the
  projected output size for Spiral is square while Stripe keeps the source's
  aspect ratio;
* the output-resolution projection, which calls the *existing*
  :func:`halftone_playground.preprocess.scaled_size` -- there is no second
  rounding implementation here.

Everything in this module is a pure function of its arguments.  Nothing here
knows about widgets, images or the file system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

from ..preprocess import scaled_size

__all__ = [
    "DEFAULT_ARMS",
    "DEFAULT_ANGLE",
    "DEFAULT_LINE_WIDTH",
    "DEFAULT_PERIOD",
    "DEFAULT_RENDER",
    "DEFAULT_SCALE",
    "MODE_SPIRAL",
    "MODE_STRIPE",
    "MODES",
    "RENDER_BLACK_ON_WHITE",
    "RENDER_CHOICES",
    "RENDER_SOURCE_COLOR",
    "RENDER_WHITE_ON_BLACK",
    "RenderParams",
    "WIDTH_FIXED",
    "WIDTH_VARIABLE",
    "WIDTH_MODES",
    "center_square_crop_box",
    "center_square_side",
    "effective_source_size",
    "field_error",
    "format_resolution",
    "output_resolution",
    "parse_float",
    "parse_int",
    "render_choices_for",
    "validate",
    "visible_parameters",
]
# --------------------------------------------------------------------------
# frozen choice sets (mirror the CLI where an equivalent exists)
# --------------------------------------------------------------------------

MODE_STRIPE = "stripe"
MODE_SPIRAL = "spiral"
MODES = (MODE_STRIPE, MODE_SPIRAL)

WIDTH_VARIABLE = "variable"
WIDTH_FIXED = "fixed"
WIDTH_MODES = (WIDTH_VARIABLE, WIDTH_FIXED)

#: Render variant identifiers.  These are plain internal keys; the Tk layer
#: maps them to the frozen ``cli._RENDERERS`` render callables.  Kept as
#: strings so this module never imports render functions.
RENDER_BLACK_ON_WHITE = "black_on_white"
RENDER_WHITE_ON_BLACK = "white_on_black"
RENDER_SOURCE_COLOR = "source_color"
RENDER_CHOICES = (
    RENDER_BLACK_ON_WHITE,
    RENDER_WHITE_ON_BLACK,
    RENDER_SOURCE_COLOR,
)

#: The render variants always shown, regardless of mode / width mode.  Frozen
#: in GUI Round 0: all three options are always present.
_ALWAYS_VISIBLE_RENDER = RENDER_CHOICES

# --------------------------------------------------------------------------
# defaults reused from the existing CLI / project contract
# --------------------------------------------------------------------------
#
# These are imported-by-value from the CLI contract rather than re-derived, so
# the GUI opens on exactly the same numbers the CLI uses.  ``DEFAULT_SCALE`` is
# 1.0 and ``DEFAULT_LINE_WIDTH`` is the *GUI-only* starting value for the
# fixed-width spinbox; there is no CLI default for line width because the CLI
# treats an absent ``--line-width`` as "variable-width".  GUI Round 0 chose the
# simplest, most conservative value, ``DEFAULT_LINE_WIDTH = 1``: it is always
# valid for any ``period >= 1`` (the frozen contract is ``1 <= line_width <=
# period``) and never silently implies a wider line than the user drew.

DEFAULT_PERIOD = 16
DEFAULT_SCALE = 1.0
DEFAULT_ANGLE = 90.0
DEFAULT_ARMS = 1
DEFAULT_RENDER = RENDER_BLACK_ON_WHITE

#: GUI-only initial value for the fixed-width line-width field.  Chosen as the
#: smallest legal width so it is valid against every legal period.
DEFAULT_LINE_WIDTH = 1

#: Parameter identifiers used for visibility, in display order.
PARAM_PERIOD = "period"
PARAM_SCALE = "scale"
PARAM_ANGLE = "angle"
PARAM_ARMS = "arms"
PARAM_LINE_WIDTH = "line_width"

_ALL_PARAMETERS = (
    PARAM_PERIOD,
    PARAM_SCALE,
    PARAM_ANGLE,
    PARAM_ARMS,
    PARAM_LINE_WIDTH,
)


# --------------------------------------------------------------------------
# the parameter value container
# --------------------------------------------------------------------------


@dataclass
class RenderParams:
    """The full set of GUI parameter values.

    A single flat container is used (rather than one class per mode) so that
    switching mode / width mode can only ever *change which fields are shown*
    -- the values themselves live here untouched.  That is what makes
    "hiding a parameter never resets it" trivially true: visibility is derived
    from the mode flags, and the values are never rewritten on a mode switch.

    ``line_width`` is ``None`` until the fixed-width path needs it; the Tk
    layer keeps its own text buffer so a user can type freely.

    ``scale_text`` / ``period_text`` etc. are *not* stored here: raw entry text
    is a view concern.  This container holds parsed, validated values plus the
    raw text needed to round-trip invalid input back to the widget.
    """

    mode: str = MODE_STRIPE
    width_mode: str = WIDTH_VARIABLE
    render: str = DEFAULT_RENDER

    period: int = DEFAULT_PERIOD
    scale: float = DEFAULT_SCALE
    angle: float = DEFAULT_ANGLE
    arms: int = DEFAULT_ARMS
    line_width: int = DEFAULT_LINE_WIDTH

    #: Raw text mirrors.  The Tk entry widgets bind to these so that an
    #: in-progress edit (e.g. ``"1."`` or ``""``) can be represented faithfully
    #: and validated later instead of being coerced on every keystroke.
    period_text: str = field(default_factory=lambda: str(DEFAULT_PERIOD))
    scale_text: str = field(default_factory=lambda: _format_default(DEFAULT_SCALE))
    angle_text: str = field(default_factory=lambda: _format_default(DEFAULT_ANGLE))
    arms_text: str = field(default_factory=lambda: str(DEFAULT_ARMS))
    line_width_text: str = field(
        default_factory=lambda: str(DEFAULT_LINE_WIDTH)
    )

    def copy(self) -> "RenderParams":
        """Return an independent copy (dataclasses are mutable containers)."""
        return replace(self)


def _format_default(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


# --------------------------------------------------------------------------
# visibility mapping
# --------------------------------------------------------------------------


def visible_parameters(mode: str, width_mode: str) -> frozenset[str]:
    """Return the set of parameter identifiers that should be visible.

    Frozen GUI Round 0 rules:

    ====================  ==================  ==================================
    control               visible when        hidden otherwise
    ====================  ==================  ==================================
    ``Angle``             ``mode=stripe``      hidden for spiral
    ``Arms``              ``mode=spiral``      hidden for stripe
    ``Line width``        ``width_mode=fixed`` hidden for variable
    ====================  ==================  ==================================

    ``Period`` and ``Scale`` are always visible in both modes.  The render
    variants are a separate always-present choice and are *not* part of this
    set.

    Hiding a parameter is a *presentation* decision only: this function never
    touches values.
    """
    visible = {PARAM_PERIOD, PARAM_SCALE}

    if mode == MODE_STRIPE:
        visible.add(PARAM_ANGLE)
    # else: spiral -> Angle hidden

    if mode == MODE_SPIRAL:
        visible.add(PARAM_ARMS)
    # else: stripe -> Arms hidden

    if width_mode == WIDTH_FIXED:
        visible.add(PARAM_LINE_WIDTH)
    # else: variable -> Line width hidden

    return frozenset(visible)


def render_choices_for() -> tuple[str, ...]:
    """Return the render variants to offer.  Always all three (frozen).

    A function rather than a bare constant so the Tk layer has one obvious
    place to call, and so a future round can extend it without a signature
    change at the call site.
    """
    return _ALWAYS_VISIBLE_RENDER


# --------------------------------------------------------------------------
# parsing helpers (raw text -> typed value, never raising)
# --------------------------------------------------------------------------


def parse_float(text: str) -> Optional[float]:
    """Parse a float from user text; return ``None`` when it is not a number.

    Accepts surrounding whitespace.  Rejects the empty string, ``nan`` and
    ``inf`` (they are not *finite* numbers, which every GUI numeric field
    requires) and Python-only literals such as ``1_0`` are accepted exactly as
    ``float()`` accepts them -- the GUI does not second-guess ``float``.
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_int(text: str) -> Optional[int]:
    """Parse an integer from user text; return ``None`` when invalid.

    Only plain integers are accepted: a value with a fractional part (``"3.5"``)
    or an exponent form that is not integral is rejected, matching the core's
    strict "plain integer" contract rather than silently truncating.
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        # ``int("3.5")`` raises; ``int(" 3 ")`` works.  A leading ``+`` is fine.
        return int(stripped, 10)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# per-field validation (reuses the core contracts)
# --------------------------------------------------------------------------


def field_error(params: RenderParams, name: str) -> Optional[str]:
    """Return a user-facing error for one field, or ``None`` when valid.

    The checks deliberately mirror the frozen core contracts instead of
    inventing new ones:

    * ``Scale``  -> positive finite real (``preprocess`` contract);
    * ``Period`` -> plain integer ``>= 1`` (stripe / spiral contract);
    * ``Angle``  -> finite real (stripe contract);
    * ``Arms``   -> plain integer ``>= 1`` (spiral contract);
    * ``Line width`` -> plain integer with ``1 <= line_width <= period`` when
      the fixed-width path is active (``stripe_fixed_mask`` /
      ``spiral_fixed_mask`` contract).

    The messages are phrased for a normal user, not a developer.
    """
    if name == PARAM_PERIOD:
        value = parse_int(params.period_text)
        if value is None:
            return "Period must be a whole number."
        if value <= 0:
            return "Period must be at least 1."
        return None

    if name == PARAM_SCALE:
        value = parse_float(params.scale_text)
        if value is None:
            return "Scale must be a number."
        if value <= 0.0:
            return "Scale must be greater than 0."
        return None

    if name == PARAM_ANGLE:
        value = parse_float(params.angle_text)
        if value is None:
            return "Angle must be a number."
        return None

    if name == PARAM_ARMS:
        value = parse_int(params.arms_text)
        if value is None:
            return "Arms must be a whole number."
        if value <= 0:
            return "Arms must be at least 1."
        return None

    if name == PARAM_LINE_WIDTH:
        value = parse_int(params.line_width_text)
        if value is None:
            return "Line width must be a whole number."
        if value <= 0:
            return "Line width must be at least 1."
        period = parse_int(params.period_text)
        if period is not None and period > 0 and value > period:
            return "Line width must not exceed the period."
        return None

    raise KeyError(f"unknown parameter {name!r}")


def validate(params: RenderParams) -> dict[str, str]:
    """Return ``{field: message}`` for every currently *relevant* field error.

    Only fields that are actually visible for the current mode / width mode are
    validated: an error the user cannot see would be confusing, and a hidden
    ``line_width`` must not block a variable-width job.  Unknown fields are
    ignored.
    """
    errors: dict[str, str] = {}
    for name in visible_parameters(params.mode, params.width_mode):
        message = field_error(params, name)
        if message is not None:
            errors[name] = message
    return errors


# --------------------------------------------------------------------------
# Spiral effective source: automatic centered maximum square
# --------------------------------------------------------------------------
#
# GUI product semantics frozen after Windows manual acceptance (GUI-001
# correction): the GUI accepts a *rectangular* source for Spiral.  The user is
# responsible for placing the important subject near the image centre; the GUI
# takes the largest possible square from the centre of the source as the
# effective Spiral input.  There is no manual crop UI and no crop editor, and
# the Source Preview keeps showing the complete original image.
#
# The pipeline order is fixed:  source -> centered square crop -> scale ->
# Spiral geometry.  Only the *dimensions* and the integer crop box are computed
# here; no image is ever cropped, resized or transformed in this pure module.
#
# The Spiral core (`spiral_mask`) deliberately stays square-only: converting a
# rectangular source to a square is the GUI / application preprocess layer's
# job, so the core never has to know the original image was rectangular.


def center_square_side(source_size: tuple[int, int]) -> int:
    """Return the side of the largest square that fits inside the source.

    ``side = min(width, height)``.  For an already-square source this is just
    its side; for a rectangular one it is the shorter dimension.
    """
    width, height = source_size
    return min(width, height)


def center_square_crop_box(
    source_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the integer ``(left, top, right, bottom)`` centred square crop.

    The square is taken from the **centre** of the source using floor-centred
    offsets::

        side  = min(width, height)
        left  = (width  - side) // 2
        top   = (height - side) // 2
        right = left + side
        bottom = top + side

    When the width / height difference is odd the extra single pixel stays on
    the **right** (for width) or **bottom** (for height) because the offsets
    are floored.  There is deliberately no sub-pixel crop and no other rule.

    The box is expressed as half-open pixel coordinates (``right`` and
    ``bottom`` are exclusive), matching Pillow's crop convention.
    """
    width, height = source_size
    side = center_square_side(source_size)
    left = (width - side) // 2
    top = (height - side) // 2
    return (left, top, left + side, top + side)


def effective_source_size(
    mode: str,
    source_size: Optional[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    """Return the ``(width, height)`` that actually feeds the geometry.

    * ``source_size is None`` -> ``None`` (nothing to project yet);
    * ``mode = spiral``       -> ``(side, side)`` with ``side = min(w, h)``
      (the centered maximum-square crop the GUI performs before scaling);
    * ``mode = stripe``       -> the source's ``(w, h)`` unchanged (Stripe
      keeps the original aspect ratio).

    This is pure arithmetic: it never crops, resizes or allocates an image.
    """
    if source_size is None:
        return None
    if mode == MODE_SPIRAL:
        side = center_square_side(source_size)
        return (side, side)
    return (source_size[0], source_size[1])


# --------------------------------------------------------------------------
# output resolution (reuses the existing scaled_size)
# --------------------------------------------------------------------------


def output_resolution(
    mode: str,
    source_size: Optional[tuple[int, int]],
    scale_text: str,
) -> Optional[tuple[int, int]]:
    """Return the projected ``(width, height)`` after scaling, or ``None``.

    This is a *pure projection*: it calls the existing
    :func:`halftone_playground.preprocess.scaled_size` so the GUI can never
    drift from the core's half-up rounding rule.  It never allocates an image,
    never resizes, never runs geometry.

    The size handed to ``scaled_size`` is the **effective source** size for the
    mode (see :func:`effective_source_size`), so the crop-before-scale order is
    respected: for ``mode=spiral`` a rectangular source is first reduced to its
    centered maximum square, then scaled; for ``mode=stripe`` the original
    ``(w, h)`` is scaled directly.  No second rounding rule is introduced --
    ``scaled_size`` remains the single source of truth.

    Returns ``None`` when there is no source, the scale text is not a valid
    positive finite number, or ``scaled_size`` itself rejects the combination.
    """
    effective = effective_source_size(mode, source_size)
    if effective is None:
        return None
    scale = parse_float(scale_text)
    if scale is None or scale <= 0.0:
        return None
    width, height = effective
    try:
        return scaled_size(width, height, scale)
    except (TypeError, ValueError):
        return None


def format_resolution(size: Optional[tuple[int, int]]) -> str:
    """Format a resolution for display, or an em dash when unavailable."""
    if size is None:
        return "\u2014"  # em dash
    width, height = size
    return f"{width} \u00d7 {height}"
