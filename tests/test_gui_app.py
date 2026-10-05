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
from halftone_playground.preprocess import circular_support
from halftone_playground.spiral import _polar_center, _support_radius

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
# generate eligibility / save remains inert (GUI-002A)
# --------------------------------------------------------------------------


class TestGenerateEligibility:
    def test_generate_enabled_once_a_source_is_loaded(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        assert str(app._generate_btn["state"]) == "normal"

    def test_save_disabled_without_a_result(self, app, image_factory):
        # A loaded source with no generated result is not saveable: there is
        # simply nothing to write, so the button stays honestly disabled.
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        assert str(app._save_btn["state"]) == "disabled"

    def test_generate_disabled_without_a_source(self, app):
        assert str(app._generate_btn["state"]) == "disabled"

    def test_generate_disabled_while_a_job_runs(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        app.state.begin_job()
        app.refresh()
        assert str(app._generate_btn["state"]) == "disabled"
        # ...and every other input locks too.
        assert str(app._mode_box["state"]) == "disabled"
        assert str(app._open_btn["state"]) == "disabled"

    def test_no_result_until_generate_runs(self, app, image_factory):
        app._load_source(image_factory("s.png", 64, 64))
        app.root.update_idletasks()
        assert app.state.result is None
        assert app.state.result_status == gs.RESULT_NONE
        assert app._result_image_label.cget("text") == "No result yet"


# --------------------------------------------------------------------------
# the real Generate pipeline (GUI-002A)
# --------------------------------------------------------------------------


def _pump_until_done(app, timeout: float = 20.0) -> None:
    """Run the Tk event loop until the worker has delivered its outcome.

    ``root.update()`` dispatches the ``after`` poll callback that drains the
    worker queue -- the same path a real user's idle event loop takes.

    The loop is driven by the *subject's own* signal: it stops once the worker
    is no longer busy AND, for a successful job, a result has been installed.
    A failure leaves ``result`` as ``None``, so the caller can also wait for a
    failure by checking ``state.is_running`` instead.
    """
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.root.update()
        if not app._worker.busy and not app.state.is_running:
            # One more dispatch so a just-queued outcome is handled.
            app.root.update()
            if not app._worker.busy and not app.state.is_running:
                return
        time.sleep(0.01)
    raise AssertionError("generation did not finish")


@pytest.fixture()
def rect_source(tmp_path):
    """A rectangular RGB source; every channel varies with position."""
    ys, xs = np.mgrid[0:60, 0:100]
    arr = np.dstack(
        [
            (xs * 255 // 99).astype(np.uint8),
            (ys * 255 // 59).astype(np.uint8),
            np.full((60, 100), 7, np.uint8),
        ]
    )
    path = tmp_path / "rect.png"
    Image.fromarray(arr, "RGB").save(path)
    return path


class TestGeneratePipeline:
    def test_stripe_stores_a_full_resolution_rgb_result(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)

        result = app.state.result
        assert result is not None
        assert result.mode == "RGB"
        assert result.size == (100, 60)
        # The stored pixels are the real output at full resolution...
        assert app._result_pixels.shape == (60, 100, 3)
        assert app._result_pixels.dtype == np.uint8

    def test_spiral_stores_a_square_rgba_result(self, app, rect_source):
        app.params.mode = gp.MODE_SPIRAL
        app._sync_mode_state()
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)

        result = app.state.result
        assert result is not None
        assert result.mode == "RGBA"
        assert result.size == (60, 60)
        assert app._result_pixels.shape == (60, 60, 4)

    def test_result_preview_shows_a_scaled_copy_not_the_result(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app.root.update()

        # A preview bitmap exists, and the stored result is untouched by it.
        assert app._result_photo is not None
        assert app._result_pixels.shape == (60, 100, 3)
        assert app._result_image_label.cget("text") == ""

    def test_preview_never_replaces_the_full_resolution_result(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        before = app._result_pixels.copy()
        app._schedule_preview_refresh()
        app.root.update()
        assert np.array_equal(app._result_pixels, before)

    def test_result_is_current_right_after_generating(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert app.state.result_status == gs.RESULT_CURRENT

    def test_save_enabled_after_a_current_result(self, app, rect_source):
        # GUI-002B: a freshly generated (current) result makes Save available.
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert app.state.result_status == gs.RESULT_CURRENT
        assert str(app._save_btn["state"]) == "normal"

    def test_inputs_unlock_after_the_job_finishes(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert str(app._mode_box["state"]) == "readonly"
        assert str(app._generate_btn["state"]) == "normal"


class TestResultLifecycleInShell:
    def test_changing_scale_marks_the_result_stale(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert app.state.result_status == gs.RESULT_CURRENT

        app._param_rows["scale"]["var"].set("1.3")
        app.root.update()
        assert app.state.result_status == gs.RESULT_STALE

    def test_returning_to_the_same_value_restores_current(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app._param_rows["scale"]["var"].set("1.3")
        app.root.update()
        assert app.state.result_status == gs.RESULT_STALE
        app._param_rows["scale"]["var"].set("1")
        app.root.update()
        assert app.state.result_status == gs.RESULT_CURRENT

    def test_typing_one_point_zero_keeps_the_result_current(self, app, rect_source):
        """The semantic-key requirement, through the real widget path."""
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app._param_rows["scale"]["var"].set("1.0")
        app.root.update()
        assert app.state.result_status == gs.RESULT_CURRENT

    def test_hidden_parameter_change_keeps_the_result_current(self, app, rect_source):
        app.params.mode = gp.MODE_SPIRAL
        app._sync_mode_state()
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        # Angle is hidden in Spiral, so editing it must not invalidate results.
        app._param_rows["angle"]["var"].set("45")
        app.root.update()
        assert app.state.result_status == gs.RESULT_CURRENT

    def test_stale_result_keeps_being_displayed(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        stored = app._result_pixels.copy()
        app._param_rows["scale"]["var"].set("1.3")
        app.root.update()
        app._schedule_preview_refresh()
        app.root.update()
        assert app.state.result_status == gs.RESULT_STALE
        assert app._result_pixels is not None
        assert np.array_equal(app._result_pixels, stored)

    def test_loading_a_new_source_clears_the_result(self, app, rect_source, image_factory):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert app.state.result is not None

        app._load_source(image_factory("other.png", 32, 32))
        app.root.update()
        assert app.state.result is None
        assert app.state.result_status == gs.RESULT_NONE
        assert app._result_pixels is None
        assert app._result_image_label.cget("text") == "No result yet"


class TestGenerateFailure:
    def test_failure_shows_a_message_and_no_result(self, app, tmp_path, monkeypatch):
        """A runtime failure is reported on the main thread, without a result."""
        shown = {}

        def fake_error(title, message, **kwargs):
            shown["title"] = title
            shown["message"] = message

        monkeypatch.setattr(
            "halftone_playground.gui.app.messagebox.showerror", fake_error
        )

        # Load a real source, then delete it so the worker cannot read it.
        path = tmp_path / "gone.png"
        Image.fromarray(np.full((32, 32, 3), 100, np.uint8), "RGB").save(path)
        app._load_source(path)
        path.unlink()

        app._on_generate()
        _pump_until_done(app)

        assert app.state.result is None
        assert shown.get("title") == "Cannot generate"
        assert "gone.png" in shown.get("message", "")
        assert "Traceback" not in shown.get("message", "")
        # ...and the app is usable again.
        assert app.state.is_running is False


# --------------------------------------------------------------------------
# Save PNG (GUI-002B)
# --------------------------------------------------------------------------


def _install_save_dialog(monkeypatch, result_path, capture=None):
    """Replace the native Save dialog and record the kwargs it was called with.

    ``result_path`` of ``None`` or ``""`` stands for the user cancelling.
    """
    def _fake(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return "" if result_path is None else str(result_path)

    monkeypatch.setattr(
        "halftone_playground.gui.app.filedialog.asksaveasfilename", _fake
    )


def _silence_errors(monkeypatch) -> dict:
    """Capture ``messagebox.showerror`` calls instead of showing a dialog."""
    seen: dict = {}

    def _fake(title, message, **kwargs):
        seen["title"] = title
        seen["message"] = message

    monkeypatch.setattr(
        "halftone_playground.gui.app.messagebox.showerror", _fake
    )
    return seen


class TestSaveEnablement:
    def test_save_enabled_once_a_current_result_exists(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert app.state.result_status == gs.RESULT_CURRENT
        assert str(app._save_btn["state"]) == "normal"

    def test_save_disabled_when_the_result_is_stale(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app._param_rows["scale"]["var"].set("1.3")
        app.root.update()
        assert app.state.result_status == gs.RESULT_STALE
        assert str(app._save_btn["state"]) == "disabled"

    def test_save_reenabled_when_the_key_returns(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app._param_rows["scale"]["var"].set("1.3")
        app.root.update()
        assert str(app._save_btn["state"]) == "disabled"

        app._param_rows["scale"]["var"].set("1")
        app.root.update()
        assert app.state.result_status == gs.RESULT_CURRENT
        assert str(app._save_btn["state"]) == "normal"

    def test_save_disabled_while_a_job_runs(self, app, rect_source):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert str(app._save_btn["state"]) == "normal"

        app.state.begin_job()
        app.refresh()
        assert str(app._save_btn["state"]) == "disabled"

        app.state.end_job()
        app.refresh()
        assert str(app._save_btn["state"]) == "normal"

    def test_save_disabled_after_a_new_source(self, app, rect_source, image_factory):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        assert str(app._save_btn["state"]) == "normal"

        app._load_source(image_factory("other.png", 16, 16))
        app.root.update()
        assert app.state.result is None
        assert str(app._save_btn["state"]) == "disabled"

    def test_callback_blocks_saving_when_not_current(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        # Reaching the callback with a stale result must still write nothing.
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app._param_rows["scale"]["var"].set("1.3")
        app.root.update()

        capture: dict = {}
        _install_save_dialog(monkeypatch, tmp_path / "x.png", capture)
        app._on_save()

        assert capture == {}  # the dialog was never even opened
        assert not (tmp_path / "x.png").exists()


class TestSaveDialog:
    def test_dialog_defaults(self, app, rect_source, monkeypatch, tmp_path):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)

        capture: dict = {}
        _install_save_dialog(monkeypatch, tmp_path / "out.png", capture)
        app._on_save()

        assert capture["parent"] is app.root
        assert capture["title"] == "Save PNG"
        assert capture["initialdir"] == str(rect_source.parent)
        assert capture["initialfile"] == "rect_halftone.png"
        assert capture["defaultextension"] == ".png"
        assert ("PNG", "*.png") in capture["filetypes"]

    def test_initial_name_drops_the_original_extension(
        self, app, image_factory, monkeypatch, tmp_path
    ):
        source = image_factory("shot.jpg", 40, 40)
        app._load_source(source)
        app.root.update_idletasks()

        capture: dict = {}
        _install_save_dialog(monkeypatch, tmp_path / "out.png", capture)
        # No result yet -> the callback no-ops, so drive the name helper
        # through a real generation instead.
        app._on_generate()
        _pump_until_done(app)
        app._on_save()
        assert capture["initialfile"] == "shot_halftone.png"


class TestSaveCancel:
    @pytest.mark.parametrize("cancelled", [None, ""])
    def test_cancel_is_a_strict_no_op(self, app, rect_source, monkeypatch, cancelled):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app.root.update()

        before_pixels = app._result_pixels.copy()
        before_result = app.state.result
        before_status = app.state.result_status
        before_key = app.state.result.generation_key
        before_preview = app._result_photo
        before_message = app._status_var.get()
        before_source = app.state.source

        seen = _silence_errors(monkeypatch)
        _install_save_dialog(monkeypatch, cancelled)
        app._on_save()
        app.root.update()

        assert app.state.result is before_result
        assert app.state.result_status == before_status
        assert app.state.result.generation_key == before_key
        assert app.state.source is before_source
        assert np.array_equal(app._result_pixels, before_pixels)
        assert app._result_photo is before_preview
        assert app._status_var.get() == before_message
        assert seen == {}  # no error box, and no "save cancelled" box
        assert str(app._save_btn["state"]) == "normal"


class TestSaveDoesNotRegenerate:
    def test_save_uses_the_stored_result_without_running_the_pipeline(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        stored = app._result_pixels.copy()

        import halftone_playground.gui.pipeline as pipeline

        def _boom(*_args, **_kwargs):
            raise AssertionError("the pipeline must not run during Save")

        for name in (
            "generate",
            "resize_grayscale",
            "resize_rgb",
            "stripe_mask",
            "spiral_mask",
            "render_rgba",
        ):
            monkeypatch.setattr(pipeline, name, _boom)

        started = {"count": 0}
        real_start = app._worker.start

        def _counting_start(job):
            started["count"] += 1
            return real_start(job)

        monkeypatch.setattr(app._worker, "start", _counting_start)

        target = tmp_path / "saved.png"
        _install_save_dialog(monkeypatch, target)
        app._on_save()

        assert started["count"] == 0  # no second worker job was started
        with Image.open(target) as reopened:
            assert reopened.mode == "RGB"
            assert np.array_equal(np.array(reopened), stored)

    def test_saved_file_is_full_resolution_not_the_preview(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        app.root.update()

        assert app._result_photo is not None
        result_size = app.state.result.size  # (100, 60)
        preview_size = (app._result_photo.width(), app._result_photo.height())
        assert preview_size != result_size  # the preview really is scaled

        target = tmp_path / "full.png"
        _install_save_dialog(monkeypatch, target)
        app._on_save()

        with Image.open(target) as reopened:
            assert reopened.size == result_size
            assert reopened.size != preview_size


class TestSaveRoundTripInShell:
    def test_stripe_saves_a_rgb_png_identical_to_the_result(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        stored = app._result_pixels.copy()
        assert app.state.result.mode == "RGB"

        target = tmp_path / "stripe.png"
        _install_save_dialog(monkeypatch, target)
        app._on_save()

        with Image.open(target) as reopened:
            assert reopened.mode == "RGB"
            assert reopened.size == (stored.shape[1], stored.shape[0])
            assert np.array_equal(np.array(reopened), stored)

    def test_spiral_saves_an_rgba_png_with_circular_support_alpha(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app.params.mode = gp.MODE_SPIRAL
        app._sync_mode_state()
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)

        stored = app._result_pixels.copy()
        assert app.state.result.mode == "RGBA"
        side = stored.shape[0]

        target = tmp_path / "spiral.png"
        _install_save_dialog(monkeypatch, target)
        app._on_save()

        with Image.open(target) as reopened:
            assert reopened.mode == "RGBA"
            data = np.array(reopened)

        assert data.shape == stored.shape
        assert np.array_equal(data, stored)

        alpha = data[..., 3]
        assert set(np.unique(alpha).tolist()) <= {0, 255}

        opaque = circular_support(side, _polar_center(side), _support_radius(side))
        assert np.array_equal(alpha == 255, opaque)
        assert np.array_equal(alpha == 0, ~opaque)
        # Inside the disc everything is opaque -- the gaps between the lines
        # included.  Prove both kinds of pixel are really present, so "all
        # opaque" is not vacuously true.
        inside = data[..., :3][opaque]
        assert np.any(np.all(inside == 255, axis=1))  # opaque background gap
        assert np.any(np.all(inside == 0, axis=1))    # opaque line


class TestSaveSuccessAndFailure:
    def test_success_shows_inline_status_and_no_message_box(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        key_before = app.state.result.generation_key

        seen = _silence_errors(monkeypatch)
        target = tmp_path / "ok.png"
        _install_save_dialog(monkeypatch, target)
        app._on_save()
        app.root.update()

        assert target.exists()
        assert seen == {}  # neither a success box nor an error box
        assert "saved" in app._status_var.get().lower()
        # result / key / current status all survive, Save stays usable
        assert app.state.result is not None
        assert app.state.result.generation_key == key_before
        assert app.state.result_status == gs.RESULT_CURRENT
        assert str(app._save_btn["state"]) == "normal"

    def test_failure_reports_friendly_error_and_preserves_state(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        key_before = app.state.result.generation_key
        stored = app._result_pixels.copy()
        status_before = app._status_var.get()

        seen = _silence_errors(monkeypatch)
        # A directory that does not exist -> a real OSError from Pillow.
        bad = tmp_path / "missing-dir" / "out.png"
        _install_save_dialog(monkeypatch, bad)
        app._on_save()
        app.root.update()

        assert seen.get("title") == "Cannot save PNG"
        assert "Traceback" not in seen.get("message", "")
        assert not bad.exists()
        # A save failure must not be dressed up as a generate failure nor as a
        # success, and the current result must survive so Save can be retried.
        assert app.state.result is not None
        assert app.state.result.generation_key == key_before
        assert app.state.result_status == gs.RESULT_CURRENT
        assert np.array_equal(app._result_pixels, stored)
        assert app._status_var.get() == status_before
        assert "saved" not in app._status_var.get().lower()
        assert str(app._save_btn["state"]) == "normal"

    def test_save_can_be_retried_after_a_failure(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)
        _silence_errors(monkeypatch)

        bad = tmp_path / "nope" / "out.png"
        _install_save_dialog(monkeypatch, bad)
        app._on_save()
        assert not bad.exists()

        good = tmp_path / "good.png"
        _install_save_dialog(monkeypatch, good)
        app._on_save()
        assert good.exists()
        assert app.state.result_status == gs.RESULT_CURRENT


class TestSaveExtension:
    def test_explicit_non_png_suffix_is_rejected_without_writing(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)

        seen = _silence_errors(monkeypatch)
        target = tmp_path / "result.jpg"
        _install_save_dialog(monkeypatch, target)
        app._on_save()

        assert seen.get("title") == "Cannot save PNG"
        assert "png" in seen.get("message", "").lower()
        assert not target.exists()
        # No sneaky ".jpg.png" second write either.
        assert not (tmp_path / "result.jpg.png").exists()
        assert app.state.result_status == gs.RESULT_CURRENT

    def test_uppercase_png_suffix_is_accepted(
        self, app, rect_source, monkeypatch, tmp_path
    ):
        app._load_source(rect_source)
        app._on_generate()
        _pump_until_done(app)

        target = tmp_path / "OUT.PNG"
        _install_save_dialog(monkeypatch, target)
        app._on_save()

        assert target.exists()
        with Image.open(target) as reopened:
            assert reopened.mode == "RGB"
