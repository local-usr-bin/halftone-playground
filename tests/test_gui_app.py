"""Tests for the Tk/ttk GUI shell (opt-in: require a display server).

The shell owns widgets, so exercising it needs a real Tk connection.  These
tests are marked ``gui`` and are **deselected by the default pytest run**
(``addopts = -m 'not gui'``): they need a display (or ``Xvfb``) and importing
``tkinter`` would violate the headless smoke guard in
``tests/test_smoke.py``.  Run them explicitly with:

    pytest -m gui              # needs DISPLAY, or: xvfb-run pytest -m gui

They assert the visible behaviour the frozen spec calls out, via the real
widget callbacks -- no screenshot pixel comparison.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from PIL import Image

from halftone_playground.gui import params as gp
from halftone_playground.gui import state as gs

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def tk_root():
    """Create one Tcl/Tk interpreter for this test module."""
    if importlib.util.find_spec("tkinter") is None:
        pytest.skip("tkinter is not available")

    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        message = str(exc).lower()
        if "no display name" in message or "couldn't connect to display" in message:
            pytest.skip("no usable Tk display; run with a DISPLAY or xvfb-run")
        raise

    root.withdraw()
    root.update_idletasks()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture()
def app(tk_root):
    import tkinter as tk

    from halftone_playground.gui.app import HalftoneApp

    window = tk.Toplevel(tk_root)
    application = HalftoneApp(window)
    window.update_idletasks()
    try:
        yield application
    finally:
        window.destroy()
        tk_root.update_idletasks()


@pytest.fixture()
def image_factory(tmp_path):
    def _make(name: str, width: int, height: int):
        path = tmp_path / name
        Image.fromarray(np.full((height, width), 128, np.uint8), "L").save(path)
        return path

    return _make


# --------------------------------------------------------------------------
# initial shell
# --------------------------------------------------------------------------


class TestInitialShell:
    def test_defaults_are_stripe_variable_black_on_white(self, app):
        assert app.params.mode == gp.MODE_STRIPE
        assert app.params.width_mode == gp.WIDTH_VARIABLE
        assert app.params.render == gp.RENDER_BLACK_ON_WHITE

    def test_generate_and_save_disabled(self, app):
        assert str(app._generate_btn["state"]) == "disabled"
        assert str(app._save_btn["state"]) == "disabled"

    def test_no_source_shows_neutral_status(self, app):
        assert app._source_res_var.get() == "Source: \u2014"
        assert app._output_res_var.get() == "Output: \u2014"

    def test_result_preview_is_placeholder(self, app):
        # GUI-001 must not fake a result.
        assert app._result_image_label.cget("text") == "No result yet"
        assert app._result_photo is None


# --------------------------------------------------------------------------
# open image
# --------------------------------------------------------------------------


class TestOpenImage:
    def test_load_updates_filename_and_resolution(self, app, image_factory):
        path = image_factory("square.png", 64, 48)
        app._load_source(path)
        app.root.update_idletasks()
        assert app.state.source.filename == "square.png"
        assert app._source_res_var.get() == "Source: 64 \u00d7 48"
        assert app._output_res_var.get() == "Output: 64 \u00d7 48"

    def test_load_builds_source_preview_photo(self, app, image_factory):
        path = image_factory("photo.png", 40, 40)
        app._load_source(path)
        app.root.update_idletasks()
        assert app._source_photo is not None

    def test_load_clears_previous_result(self, app, image_factory):
        path = image_factory("a.png", 32, 32)
        app._load_source(path)
        app.root.update_idletasks()
        app.state.set_result(gs.ResultInfo(32, 32))
        app._load_source(image_factory("b.png", 16, 16))
        app.root.update_idletasks()
        assert app.state.result_status == gs.RESULT_NONE

    def test_failed_load_keeps_existing_source(self, app, image_factory, tmp_path):
        good = image_factory("good.png", 32, 32)
        app._load_source(good)
        app.root.update_idletasks()

        # Silence the error dialog and record that it was shown.
        shown = {}
        app._show_error = lambda title, message: shown.update(title=title, message=message)

        bad = tmp_path / "bad.txt"
        bad.write_text("not an image")
        app._load_source(bad)
        app.root.update_idletasks()

        assert app.state.source.filename == "good.png"
        assert "title" in shown  # an error dialog was raised, not a traceback


# --------------------------------------------------------------------------
# visibility through real widgets
# --------------------------------------------------------------------------


class TestVisibilityWidgets:
    def _mapped(self, app):
        return {n for n, w in app._param_rows.items() if w["entry"].winfo_manager() == "grid"}

    def test_stripe_shows_angle_hides_arms(self, app):
        app.root.update_idletasks()
        mapped = self._mapped(app)
        assert gp.PARAM_ANGLE in mapped
        assert gp.PARAM_ARMS not in mapped

    def test_spiral_shows_arms_hides_angle(self, app):
        app._mode_var.set("Spiral")
        app._on_mode_changed()
        app.root.update_idletasks()
        mapped = self._mapped(app)
        assert gp.PARAM_ARMS in mapped
        assert gp.PARAM_ANGLE not in mapped

    def test_fixed_shows_line_width(self, app):
        app._width_var.set("Fixed")
        app._on_width_changed()
        app.root.update_idletasks()
        assert gp.PARAM_LINE_WIDTH in self._mapped(app)

    def test_variable_hides_line_width(self, app):
        app.root.update_idletasks()
        assert gp.PARAM_LINE_WIDTH not in self._mapped(app)


# --------------------------------------------------------------------------
# hidden values are preserved through the widgets
# --------------------------------------------------------------------------


class TestHiddenValuesPreserved:
    def test_angle_survives_stripe_spiral_stripe(self, app):
        app._param_rows["angle"]["var"].set("45")
        app.root.update_idletasks()
        assert app.params.angle_text == "45"

        app._mode_var.set("Spiral")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app.params.angle_text == "45"  # hidden but remembered

        app._mode_var.set("Stripe")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app.params.angle_text == "45"
        assert app._param_rows["angle"]["var"].get() == "45"

    def test_line_width_survives_variable_fixed_variable(self, app):
        app._width_var.set("Fixed")
        app._on_width_changed()
        app.root.update_idletasks()
        app._param_rows["line_width"]["var"].set("7")
        app.root.update_idletasks()
        assert app.params.line_width_text == "7"

        app._width_var.set("Variable")
        app._on_width_changed()
        app.root.update_idletasks()
        assert app.params.line_width_text == "7"

        app._width_var.set("Fixed")
        app._on_width_changed()
        app.root.update_idletasks()
        assert app._param_rows["line_width"]["var"].get() == "7"


# --------------------------------------------------------------------------
# inline validation
# --------------------------------------------------------------------------


class TestInlineValidation:
    def test_invalid_scale_shows_inline_and_dash_output(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        app._param_rows["scale"]["var"].set("abc")
        app.root.update_idletasks()
        assert app._output_res_var.get() == "Output: \u2014"
        assert "number" in app._status_var.get().lower()

    def test_valid_scale_restores_output_and_status(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        app._param_rows["scale"]["var"].set("1.3")
        app.root.update_idletasks()
        assert app._output_res_var.get() == "Output: 83 \u00d7 83"
        assert app._status_var.get() == "Ready."

    def test_line_width_above_period_reports_inline(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        app._width_var.set("Fixed")
        app._on_width_changed()
        app._param_rows["line_width"]["var"].set("999")
        app.root.update_idletasks()
        assert "period" in app._status_var.get().lower()


# --------------------------------------------------------------------------
# spiral rectangular source compatibility (GUI-001 product correction)
# --------------------------------------------------------------------------


class TestSpiralRectangularSource:
    def test_spiral_with_non_square_source_is_ready(self, app, image_factory):
        # A rectangular source with Spiral is a *legal* configuration now:
        # the GUI takes the centered maximum square automatically, so no
        # square-only error is shown.
        app._load_source(image_factory("wide.png", 80, 40))
        app.root.update_idletasks()
        app._mode_var.set("Spiral")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app._status_var.get() == "Ready."

    def test_spiral_with_square_source_is_ready(self, app, image_factory):
        app._load_source(image_factory("sq.png", 64, 64))
        app.root.update_idletasks()
        app._mode_var.set("Spiral")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app._status_var.get() == "Ready."

    def test_rectangular_spiral_produces_no_square_only_error(self, app, image_factory):
        # H) the retired message must never appear again.
        app._load_source(image_factory("portrait.png", 40, 80))
        app.root.update_idletasks()
        app._mode_var.set("Spiral")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert "square" not in app._status_var.get().lower()

    def test_output_resolution_follows_mode(self, app, image_factory):
        # 1492 x 2558 source: Stripe keeps the aspect ratio, Spiral projects
        # the centered maximum square, and switching back restores Stripe.
        app._load_source(image_factory("big.png", 1492, 2558))
        app.root.update_idletasks()
        assert app._output_res_var.get() == "Output: 1492 \u00d7 2558"

        app._mode_var.set("Spiral")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app._output_res_var.get() == "Output: 1492 \u00d7 1492"

        app._mode_var.set("Stripe")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app._output_res_var.get() == "Output: 1492 \u00d7 2558"

    def test_switching_to_spiral_keeps_full_source_state(self, app, image_factory):
        # I) switching mode must not modify the source size / source preview:
        # the preview always shows the complete original image.
        app._load_source(image_factory("wide.png", 80, 40))
        app.root.update_idletasks()
        assert app.state.source_size == (80, 40)
        assert app._source_image.size == (80, 40)
        assert app._source_photo is not None

        app._mode_var.set("Spiral")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app.state.source_size == (80, 40)
        assert app._source_image.size == (80, 40)
        assert app._source_photo is not None

        app._mode_var.set("Stripe")
        app._on_mode_changed()
        app.root.update_idletasks()
        assert app.state.source_size == (80, 40)


# --------------------------------------------------------------------------
# generate / save remain inert
# --------------------------------------------------------------------------


class TestActionsInert:
    def test_generate_and_save_stay_disabled_with_source(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        assert str(app._generate_btn["state"]) == "disabled"
        assert str(app._save_btn["state"]) == "disabled"

    def test_no_result_is_ever_produced(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        assert app.state.result is None
        assert app.state.result_status == gs.RESULT_NONE
