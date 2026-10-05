"""Tests for the normalized generation key and byte-stability (toolkit-free).

Two things are pinned here:

1. :class:`~halftone_playground.gui.pipeline.GenerationKey` is *semantic*: it
   ignores how a number was typed and ignores parameters that cannot affect
   the output, so the ``current`` / ``stale`` decision is never confused by
   view state.
2. ``generate`` is deterministic -- the same request yields byte-identical
   pixels.  That is what makes ``current`` a meaningful claim about a stored
   result rather than a guess.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from halftone_playground.gui import params as gp
from halftone_playground.gui.pipeline import (
    GenerationKey,
    GenerationRequest,
    generate,
)


def _png(path, width=64, height=64):
    ys, xs = np.mgrid[0:height, 0:width]
    arr = np.dstack(
        [
            (xs * 255 // max(width - 1, 1)).astype(np.uint8),
            (ys * 255 // max(height - 1, 1)).astype(np.uint8),
            np.full((height, width), 9, np.uint8),
        ]
    )
    Image.fromarray(arr, "RGB").save(path)
    return path


@pytest.fixture()
def png(tmp_path):
    return _png(tmp_path / "s.png")


# --------------------------------------------------------------------------
# normalization: "1" == "1.0"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("1", "1.0"),
        ("1", "1.00"),
        ("2", "2.0"),
        ("1.5", "1.50"),
        ("0.5", "0.500"),
        ("10", "10.0"),
    ],
)
def test_scale_text_is_normalized(a, b) -> None:
    pa = gp.RenderParams(scale_text=a)
    pb = gp.RenderParams(scale_text=b)
    assert GenerationKey.from_params(pa) == GenerationKey.from_params(pb)


def test_period_text_is_normalized() -> None:
    a = gp.RenderParams(period_text="16")
    b = gp.RenderParams(period_text=" 16 ")
    assert GenerationKey.from_params(a) == GenerationKey.from_params(b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("16", "17"),
        ("1", "1.3"),
        ("1", "2"),
        ("0.5", "0.75"),
    ],
)
def test_different_values_do_not_collide(a, b) -> None:
    pa = gp.RenderParams(scale_text=a)
    pb = gp.RenderParams(scale_text=b)
    assert GenerationKey.from_params(pa) != GenerationKey.from_params(pb)


def test_one_and_one_point_zero_share_a_key_end_to_end(png) -> None:
    """The requirement, expressed the way the shell experiences it."""
    p1 = gp.RenderParams(mode=gp.MODE_SPIRAL, render=gp.RENDER_SOURCE_COLOR, scale_text="1")
    p2 = gp.RenderParams(mode=gp.MODE_SPIRAL, render=gp.RENDER_SOURCE_COLOR, scale_text="1.0")

    r1 = generate(GenerationRequest.from_params(p1, png, (64, 64)))
    r2 = generate(GenerationRequest.from_params(p2, png, (64, 64)))

    assert r1.key == r2.key
    assert np.array_equal(r1.pixels, r2.pixels)


# --------------------------------------------------------------------------
# irrelevant parameters must not invalidate a result
# --------------------------------------------------------------------------


def test_hidden_angle_does_not_affect_the_key() -> None:
    """``Angle`` is hidden in Spiral, so it can never make a result stale."""
    a = gp.RenderParams(mode=gp.MODE_SPIRAL, angle_text="30")
    b = gp.RenderParams(mode=gp.MODE_SPIRAL, angle_text="45")
    assert GenerationKey.from_params(a) == GenerationKey.from_params(b)


def test_hidden_arms_does_not_affect_the_key() -> None:
    """``Arms`` is hidden in Stripe."""
    a = gp.RenderParams(mode=gp.MODE_STRIPE, arms_text="1")
    b = gp.RenderParams(mode=gp.MODE_STRIPE, arms_text="12")
    assert GenerationKey.from_params(a) == GenerationKey.from_params(b)


def test_hidden_line_width_does_not_affect_the_key() -> None:
    """``Line width`` is hidden in Variable width."""
    a = gp.RenderParams(width_mode=gp.WIDTH_VARIABLE, line_width_text="1")
    b = gp.RenderParams(width_mode=gp.WIDTH_VARIABLE, line_width_text="7")
    assert GenerationKey.from_params(a) == GenerationKey.from_params(b)


def test_visible_angle_does_affect_the_key() -> None:
    a = gp.RenderParams(mode=gp.MODE_STRIPE, angle_text="30")
    b = gp.RenderParams(mode=gp.MODE_STRIPE, angle_text="45")
    assert GenerationKey.from_params(a) != GenerationKey.from_params(b)


def test_visible_line_width_does_affect_the_key() -> None:
    a = gp.RenderParams(width_mode=gp.WIDTH_FIXED, line_width_text="2")
    b = gp.RenderParams(width_mode=gp.WIDTH_FIXED, line_width_text="5")
    assert GenerationKey.from_params(a) != GenerationKey.from_params(b)


def test_mode_width_and_render_are_part_of_the_key() -> None:
    base = gp.RenderParams()
    for field, values in (
        ("mode", gp.MODES),
        ("width_mode", gp.WIDTH_MODES),
        ("render", gp.RENDER_CHOICES),
    ):
        keys = set()
        for value in values:
            p = gp.RenderParams(**{field: value})
            keys.add(GenerationKey.from_params(p))
        assert len(keys) == len(values)


def test_unparseable_field_keys_as_none_and_never_matches() -> None:
    good = gp.RenderParams(scale_text="1")
    broken = gp.RenderParams(scale_text="")
    assert GenerationKey.from_params(broken) != GenerationKey.from_params(good)
    assert GenerationKey.from_params(good) == GenerationKey.from_params(good)


def test_key_is_hashable_and_frozen() -> None:
    key = GenerationKey.from_params(gp.RenderParams())
    assert hash(key) == hash(GenerationKey.from_params(gp.RenderParams()))
    with pytest.raises(Exception):
        key.scale = "9"  # type: ignore[misc]


# --------------------------------------------------------------------------
# request snapshot immutability
# --------------------------------------------------------------------------


def test_request_is_an_immutable_snapshot(png) -> None:
    params = gp.RenderParams(mode=gp.MODE_SPIRAL, scale_text="1.3")
    request = GenerationRequest.from_params(params, png, (64, 64))

    # A later UI edit must not reach into the running job's request.
    params.scale_text = "2"
    params.mode = gp.MODE_STRIPE

    assert request.scale == 1.3
    assert request.mode == gp.MODE_SPIRAL


def test_request_key_matches_its_own_parameters(png) -> None:
    params = gp.RenderParams(mode=gp.MODE_SPIRAL, width_mode=gp.WIDTH_FIXED)
    request = GenerationRequest.from_params(params, png, (64, 64))
    assert request.key == GenerationKey.from_params(params)


# --------------------------------------------------------------------------
# byte stability
# --------------------------------------------------------------------------

COMBOS = [
    (mode, width_mode, render)
    for mode in gp.MODES
    for width_mode in gp.WIDTH_MODES
    for render in gp.RENDER_CHOICES
]


def _digest(result) -> str:
    header = f"{result.mode}:{result.width}x{result.height}:".encode()
    return hashlib.sha256(header + result.pixels.tobytes()).hexdigest()


@pytest.mark.parametrize("mode,width_mode,render", COMBOS)
def test_generate_is_byte_stable(png, mode, width_mode, render) -> None:
    """The same request must always produce the same bytes.

    Determinism is what lets ``current`` mean "these pixels still describe
    these parameters".  It also means the sample-generation scripts and any
    future Save round can rely on identical output.
    """
    params = gp.RenderParams(
        mode=mode, width_mode=width_mode, render=render, scale_text="1.5"
    )
    request = GenerationRequest.from_params(params, png, (64, 64))
    first = _digest(generate(request))
    second = _digest(generate(request))
    assert first == second


def test_reading_the_source_twice_is_stable(tmp_path) -> None:
    """Re-encoding the source losslessly must not shift the result."""
    a = _png(tmp_path / "a.png")
    b = _png(tmp_path / "b.png")
    params = gp.RenderParams(mode=gp.MODE_SPIRAL, render=gp.RENDER_SOURCE_COLOR)
    ra = generate(GenerationRequest.from_params(params, a, (64, 64)))
    rb = generate(GenerationRequest.from_params(params, b, (64, 64)))
    assert np.array_equal(ra.pixels, rb.pixels)
