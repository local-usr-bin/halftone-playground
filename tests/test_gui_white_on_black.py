"""White-on-black must keep the source *positive* (GUI-002A correction).

The variable-width geometry was written for the black-on-white convention

    W = P * (1 - G)

so a dark source paints a **wide** line.  In the ``white_on_black`` variant the
line is white, and the product decision is that it must grow with the source's
brightness instead:

    W_white = P * G

i.e. a white area stays white and a black area stays black -- the source is
kept positive rather than being turned into a photographic negative.

The implementation does this by inverting the *already cropped and scaled*
grayscale on the variable-width path only, leaving the frozen
``stripe_mask`` / ``spiral_mask`` width formula and API untouched.  These tests
pin the observable consequences of that decision.

The tests are pure and toolkit-free (``gui.pipeline`` imports no Tk), so they
run on the default ``pytest`` pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halftone_playground.gui import params as gp
from halftone_playground.gui.pipeline import (
    GenerationRequest,
    generate,
)
from halftone_playground.preprocess import (
    circular_support,
    invert_luminance,
    resize_grayscale,
)
from halftone_playground.render import (
    render_black_on_white,
    render_white_on_black,
)
from halftone_playground.spiral import (
    _polar_center,
    _support_radius,
    spiral_mask,
)
from halftone_playground.stripe import stripe_mask

SIDE = 64
PERIOD = 16
ANGLE = "90"
ARMS = "3"
LINE_WIDTH = "4"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _flat(path: Path, value: int, side: int = SIDE) -> Path:
    """A uniform mid-...um, uniform grayscale source at ``value``."""
    Image.fromarray(np.full((side, side), value, np.uint8), "L").save(path)
    return path


def _gradient(path: Path, side: int = SIDE) -> Path:
    """A left-to-right luminance ramp (0 -> 255)."""
    xs = (np.arange(side) * 255 // (side - 1)).astype(np.uint8)
    arr = np.tile(xs, (side, 1)).astype(np.uint8)
    Image.fromarray(arr, "L").save(path)
    return path


def _request(
    path: Path,
    size: tuple[int, int] = (SIDE, SIDE),
    *,
    mode: str = gp.MODE_STRIPE,
    width_mode: str = gp.WIDTH_VARIABLE,
    render: str = gp.RENDER_WHITE_ON_BLACK,
    period: str = str(PERIOD),
    scale: str = "1",
) -> GenerationRequest:
    p = gp.RenderParams(
        mode=mode,
        width_mode=width_mode,
        render=render,
        period_text=period,
        scale_text=scale,
        angle_text=ANGLE,
        arms_text=ARMS,
        line_width_text=LINE_WIDTH,
    )
    return GenerationRequest.from_params(p, path, size)


def _white_line_count(result) -> int:
    """Number of pure-white (line) pixels in a white-on-black result.

    Every non-disc pixel of a Spiral result is transparent black, and the
    background of a white-on-black render is black, so counting ``== 255``
    over the first channel counts exactly the white lines.
    """
    return int((result.pixels[..., 0] == 255).sum())


def _black_line_count(result) -> int:
    """Number of pure-black (line) pixels in a black-on-white result."""
    return int((result.pixels[..., 0] == 0).sum())


def _disc(result) -> np.ndarray:
    side = result.width
    return circular_support(side, _polar_center(side), _support_radius(side))


# --------------------------------------------------------------------------
# 0. the shared helper itself
# --------------------------------------------------------------------------


def test_invert_luminance_is_the_exact_complement() -> None:
    gray = np.array([[0, 1, 127, 128, 254, 255]], dtype=np.uint8)
    out = invert_luminance(gray)
    assert np.array_equal(out, np.array([[255, 254, 128, 127, 1, 0]], np.uint8))
    assert out.dtype == np.uint8


def test_invert_luminance_is_pure_and_identity_when_applied_twice() -> None:
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    original = gray.copy()
    once = invert_luminance(gray)
    twice = invert_luminance(once)
    assert np.array_equal(gray, original), "input must not be mutated"
    assert np.array_equal(twice, original)
    assert once is not gray


# --------------------------------------------------------------------------
# 1. black source -> minimal white; white source -> maximal white
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", gp.MODES)
def test_black_source_gives_the_minimum_white_line(tmp_path, mode) -> None:
    """A black source must produce (almost) no white line.

    ``white_on_black`` is the positive rendering of the same luminance, so a
    black area is black.  With ``W_white = P * G`` and ``G == 0`` every cell
    width is ``0``, so no white pixel may be painted at all.
    """
    path = _flat(tmp_path / "black.png", 0)
    result = generate(_request(path, mode=mode, render=gp.RENDER_WHITE_ON_BLACK))

    assert _white_line_count(result) == 0
    # Nothing is white: inside the disc the picture is uniformly black.
    pixels = result.pixels[_disc(result)] if mode == gp.MODE_SPIRAL else result.pixels
    assert set(np.unique(pixels[..., 0]).tolist()) == {0}


@pytest.mark.parametrize("mode", gp.MODES)
def test_white_source_gives_the_maximum_white_line(tmp_path, mode) -> None:
    """A white source must produce the widest possible white line.

    ``W_white = P * G`` with ``G == 1`` saturates the cell at the full period,
    so the disc is filled with white.
    """
    path = _flat(tmp_path / "white.png", 255)
    result = generate(_request(path, mode=mode, render=gp.RENDER_WHITE_ON_BLACK))

    pixels = result.pixels[_disc(result)] if mode == gp.MODE_SPIRAL else result.pixels
    assert set(np.unique(pixels[..., 0]).tolist()) == {255}
    assert _white_line_count(result) > 0


@pytest.mark.parametrize("mode", gp.MODES)
def test_white_on_black_is_monotone_in_source_brightness(tmp_path, mode) -> None:
    """Brighter source -> strictly more white, for both geometries.

    This is the property the correction is *for*: the white line width grows
    with the source luminance.  Tested at four levels so a mere
    "black and white are right" coincidence cannot pass it.
    """
    counts = []
    for value in (0, 85, 170, 255):
        path = _flat(tmp_path / f"v{value}.png", value)
        result = generate(
            _request(path, mode=mode, render=gp.RENDER_WHITE_ON_BLACK)
        )
        counts.append(_white_line_count(result))

    assert counts == sorted(counts), counts
    assert counts[0] < counts[-1], counts


@pytest.mark.parametrize("mode", gp.MODES)
def test_white_on_black_is_not_the_negative_of_black_on_white(
    tmp_path, mode
) -> None:
    """Guard against the old, inverted behaviour.

    Before the correction, ``white_on_black`` was an exact per-pixel complement
    of ``black_on_white`` (it reused the same mask and merely swapped the
    line / background colours).  Keeping the source positive requires a
    *different mask*, so the two must no longer be complements: a black source
    is black in both variants rather than black-in-one / white-in-the-other.
    """
    path = _flat(tmp_path / "black.png", 0)
    white_on_black = generate(
        _request(path, mode=mode, render=gp.RENDER_WHITE_ON_BLACK)
    )
    black_on_white = generate(
        _request(path, mode=mode, render=gp.RENDER_BLACK_ON_WHITE)
    )

    if mode == gp.MODE_SPIRAL:
        disc = _disc(white_on_black)
        a = white_on_black.pixels[disc][..., 0].astype(np.int16)
        b = black_on_white.pixels[disc][..., 0].astype(np.int16)
    else:
        a = white_on_black.pixels[..., 0].astype(np.int16)
        b = black_on_white.pixels[..., 0].astype(np.int16)

    # Complement would mean a + b == 255 everywhere; a black source breaks it.
    assert not np.all(a + b == 255)


# --------------------------------------------------------------------------
# 2. mid-grey tone must not be inverted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", gp.MODES)
def test_mid_grey_tone_is_not_inverted(tmp_path, mode) -> None:
    """A mid-grey source must land near half coverage, not at its opposite.

    ``G = 128`` gives ``W_white = P * 128/255 ~ P/2``, so roughly half the disc
    is white.  The inverted implementation would give ``W = P * (1 - 128/255)``
    -- i.e. *also* about half -- which is why coverage alone is not decisive;
    the polarity check below is what distinguishes the two.
    """
    path = _flat(tmp_path / "mid.png", 128)
    result = generate(_request(path, mode=mode, render=gp.RENDER_WHITE_ON_BLACK))

    if mode == gp.MODE_SPIRAL:
        disc = _disc(result)
        white = int((result.pixels[disc][..., 0] == 255).sum())
        total = int(disc.sum())
    else:
        white = _white_line_count(result)
        total = int(result.pixels[..., 0].size)

    fraction = white / total
    assert 0.35 < fraction < 0.65, fraction


@pytest.mark.parametrize("mode", gp.MODES)
def test_gradient_bright_half_is_whiter_than_dark_half(tmp_path, mode) -> None:
    """Spatial polarity: the bright side of a ramp carries more white.

    A per-column comparison is immune to any global coverage coincidence, so it
    is the strongest single statement of "positive, not inverted".
    """
    path = _gradient(tmp_path / "ramp.png")
    result = generate(_request(path, mode=mode, render=gp.RENDER_WHITE_ON_BLACK))

    pixels = result.pixels
    if mode == gp.MODE_SPIRAL:
        disc = _disc(result)
        left_half = np.zeros_like(disc)
        left_half[:, : SIDE // 4] = disc[:, : SIDE // 4]
        right_half = np.zeros_like(disc)
        right_half[:, -SIDE // 4 :] = disc[:, -SIDE // 4 :]
        dark = int((pixels[left_half][..., 0] == 255).sum())
        bright = int((pixels[right_half][..., 0] == 255).sum())
        dark_area = int(left_half.sum())
        bright_area = int(right_half.sum())
    else:
        dark = int((pixels[:, : SIDE // 4, 0] == 255).sum())
        bright = int((pixels[:, -SIDE // 4 :, 0] == 255).sum())
        dark_area = SIDE * (SIDE // 4)
        bright_area = SIDE * (SIDE // 4)

    assert dark / dark_area < bright / bright_area


# --------------------------------------------------------------------------
# 3. Stripe and Spiral are both covered by the same correction
# --------------------------------------------------------------------------


def test_stripe_and_spiral_use_the_same_inverted_luminance(tmp_path) -> None:
    """Both geometries must be driven by the inverted grayscale.

    The mask each mode produces must equal the mask obtained by feeding the
    frozen core the explicitly inverted source -- for Stripe on the full frame,
    for Spiral on the (already square) frame.
    """
    path = _gradient(tmp_path / "ramp.png")
    with Image.open(path) as im:
        gray = np.array(im.convert("L"), dtype=np.uint8)
    inverted = invert_luminance(gray)

    stripe = generate(_request(path, mode=gp.MODE_STRIPE))
    assert np.array_equal(
        stripe.pixels[..., 0] == 255,
        render_white_on_black(stripe_mask(inverted, PERIOD, 90.0)) == 255,
    )

    spiral = generate(_request(path, mode=gp.MODE_SPIRAL))
    disc = _disc(spiral)
    assert np.array_equal(
        spiral.pixels[..., 0] == 255,
        (render_white_on_black(spiral_mask(inverted, PERIOD, int(ARMS))) == 255)
        & disc,
    )


# --------------------------------------------------------------------------
# 4. black-on-white output is unchanged
# --------------------------------------------------------------------------


def test_black_on_white_matches_the_untouched_reference(tmp_path) -> None:
    """``black_on_white`` must still be ``W = P * (1 - G)`` on the raw source.

    The correction must not have leaked into the black-on-white path, so the
    output has to equal the frozen reference built straight from the original
    (non-inverted) grayscale.
    """
    path = _gradient(tmp_path / "ramp.png")
    with Image.open(path) as im:
        gray = np.array(im.convert("L"), dtype=np.uint8)

    result = generate(_request(path, render=gp.RENDER_BLACK_ON_WHITE))
    reference = render_black_on_white(stripe_mask(gray, PERIOD, 90.0))
    assert np.array_equal(result.pixels[..., 0], reference)


def test_black_on_white_source_polarity_is_dark_line_on_light(tmp_path) -> None:
    """A black source still paints the widest **black** line."""
    path = _flat(tmp_path / "black.png", 0)
    result = generate(_request(path, render=gp.RENDER_BLACK_ON_WHITE))
    assert set(np.unique(result.pixels[..., 0]).tolist()) == {0}
    assert _black_line_count(result) == result.pixels[..., 0].size


# --------------------------------------------------------------------------
# 5. fixed width and source colour are unaffected
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", gp.MODES)
def test_fixed_width_is_never_inverted(tmp_path, mode) -> None:
    """Fixed-width geometry reads no grayscale, so both polarities coincide.

    Whatever the source luminance, the fixed-width mask is identical -- and
    therefore so is the rendering, white-on-black included.
    """
    black = _flat(tmp_path / "black.png", 0)
    white = _flat(tmp_path / "white.png", 255)

    from_black = generate(
        _request(
            black,
            mode=mode,
            width_mode=gp.WIDTH_FIXED,
            render=gp.RENDER_WHITE_ON_BLACK,
        )
    )
    from_white = generate(
        _request(
            white,
            mode=mode,
            width_mode=gp.WIDTH_FIXED,
            render=gp.RENDER_WHITE_ON_BLACK,
        )
    )
    assert np.array_equal(from_black.pixels, from_white.pixels)


@pytest.mark.parametrize("mode", gp.MODES)
def test_source_colour_is_unaffected_by_the_correction(tmp_path, mode) -> None:
    """``source_color`` still copies the Cartesian source; RGB is never inverted.

    Every non-white output pixel must equal the source pixel at the same
    coordinate -- on the full frame for Stripe, on the centred crop for Spiral.
    """
    xs = (np.arange(SIDE) * 255 // (SIDE - 1)).astype(np.uint8)
    rgb = np.dstack(
        [np.tile(xs, (SIDE, 1)), np.full((SIDE, SIDE), 40, np.uint8),
         np.full((SIDE, SIDE), 200, np.uint8)]
    ).astype(np.uint8)
    path = tmp_path / "colour.png"
    Image.fromarray(rgb, "RGB").save(path)

    result = generate(_request(path, mode=mode, render=gp.RENDER_SOURCE_COLOR))

    colour = result.pixels[..., :3]
    written = np.any(colour != 255, axis=2)
    assert written.any()
    if mode == gp.MODE_STRIPE:
        assert np.array_equal(colour[written], rgb[written])
    else:
        disc = _disc(result)
        assert np.array_equal(colour[written], rgb[written])


# --------------------------------------------------------------------------
# 6. Spiral alpha is still decided only by the circular support
# --------------------------------------------------------------------------


@pytest.mark.parametrize("render", gp.RENDER_CHOICES)
def test_spiral_alpha_is_still_only_the_circular_support(tmp_path, render) -> None:
    """The correction must not touch alpha.

    For every render variant the alpha channel is exactly
    ``where(circular_support, 255, 0)``, with no other value anywhere.
    """
    path = _flat(tmp_path / "mid.png", 100)
    result = generate(
        _request(path, mode=gp.MODE_SPIRAL, render=render)
    )
    side = result.width
    disc = circular_support(side, _polar_center(side), _support_radius(side))
    assert np.array_equal(
        result.pixels[..., 3], np.where(disc, np.uint8(255), np.uint8(0))
    )
    assert set(np.unique(result.pixels[..., 3]).tolist()) <= {0, 255}


def test_white_on_black_alpha_is_uniform_inside_the_disc(tmp_path) -> None:
    """Even when the disc is solid white, alpha stays uniformly 255.

    This is the alpha / line-mask independence check in the new polarity: a
    saturated white-on-black disc has no black background pixels at all, yet
    alpha must still be ``255`` everywhere inside and ``0`` outside.
    """
    path = _flat(tmp_path / "white.png", 255)
    result = generate(
        _request(path, mode=gp.MODE_SPIRAL, render=gp.RENDER_WHITE_ON_BLACK)
    )
    disc = _disc(result)
    assert set(np.unique(result.pixels[disc][..., 3]).tolist()) == {255}
    assert set(np.unique(result.pixels[~disc][..., 3]).tolist()) == {0}


# --------------------------------------------------------------------------
# 7. geometry is still applied to the *scaled* grayscale
# --------------------------------------------------------------------------


def test_inversion_happens_after_scaling(tmp_path) -> None:
    """The inversion is applied after the crop/scale, not before.

    Inverting commutes with a linear resize only up to rounding, so the
    implementation detail matters: the contract is "preprocess, then invert".
    The output must therefore equal the frozen core fed
    ``invert_luminance(resize_grayscale(gray, scale))`` exactly.
    """
    path = _gradient(tmp_path / "ramp.png")
    with Image.open(path) as im:
        gray = np.array(im.convert("L"), dtype=np.uint8)

    scale = 1.3
    result = generate(_request(path, scale=str(scale)))
    scaled = resize_grayscale(gray, scale)
    reference = render_white_on_black(
        stripe_mask(invert_luminance(scaled), PERIOD, 90.0)
    )
    assert np.array_equal(result.pixels[..., 0], reference)
