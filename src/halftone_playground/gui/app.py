"""Tkinter / ttk GUI shell for halftone-playground (GUI-001).

This is a *thin* entry point, exactly like :mod:`halftone_playground.cli`.  It
owns widgets and event wiring only; every decision that can be made without a
toolkit lives in :mod:`halftone_playground.gui.params` and
:mod:`halftone_playground.gui.state`, and the image display scaling lives in
:mod:`halftone_playground.gui.imageutil`.

What GUI-001 does
-----------------

* a plain, restrained ttk layout: **Controls on the left**, a horizontal
  **Source Preview | Result Preview** pair on the right, an inline
  status / validation strip along the bottom;
* ``Open Image...`` loads a single-frame raster (PNG / JPEG and anything else
  Pillow can read), shows the filename, the source ``W x H``, the source
  preview and the projected output resolution;
* mode / width-mode / render controls, with Stripe-Spiral and
  Variable-Fixed parameter visibility that **never resets hidden values**;
* inline validation reusing the core contracts, and a projected output
  resolution that follows the mode: Stripe keeps the source aspect ratio while
  Spiral projects the **centered maximum square** the GUI will use (see
  :mod:`halftone_playground.gui.params`) -- a projection only, no image is
  cropped or resized in GUI-001.

What GUI-001 deliberately does **not** do
-----------------------------------------

* no real ``Generate`` pipeline (the button is present but disabled);
* no real ``Save PNG...`` implementation (disabled);
* no worker thread, no cancel, no stale-result pipeline, no packaging.

The layout is intentionally pre-wired for GUI-002: the action-row, result-area
and status helpers already exist so the future pipeline has an obvious home,
without building a speculative framework now.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk

from . import imageutil, params as gui_params
from .state import GuiState, SourceInfo

__all__ = ["HalftoneApp", "run"]

#: File-dialog filter.  PNG / JPEG are the primary targets (the spec's
#: emphasis); the "All supported" entry lets any Pillow-readable raster through
#: without pretending the GUI only understands two extensions.
_FILE_TYPES = [
    ("Supported images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("All files", "*.*"),
]

_WINDOW_TITLE = "Halftone Playground"

#: Preview box side, in pixels.  A single conservative square so the two
#: previews line up; the actual displayed image is fitted inside it with its
#: aspect ratio preserved.
_PREVIEW_BOX = 320

#: Human labels <-> internal render keys.  The internal keys match the CLI's
#: ``--render`` concepts; the Tk layer maps them to the frozen render callables.
_RENDER_LABELS = {
    gui_params.RENDER_BLACK_ON_WHITE: "Black on white",
    gui_params.RENDER_WHITE_ON_BLACK: "White on black",
    gui_params.RENDER_SOURCE_COLOR: "Source color",
}
_RENDER_BY_LABEL = {label: key for key, label in _RENDER_LABELS.items()}

_MODE_LABELS = {
    gui_params.MODE_STRIPE: "Stripe",
    gui_params.MODE_SPIRAL: "Spiral",
}
_WIDTH_LABELS = {
    gui_params.WIDTH_VARIABLE: "Variable",
    gui_params.WIDTH_FIXED: "Fixed",
}


class HalftoneApp:
    """The single Tk application object.

    Owns the root window, the widget tree and the :class:`GuiState`.  All
    widget callbacks funnel through the small ``_on_*`` methods, which update
    the state and then call :meth:`refresh` to reconcile the view.  Keeping a
    single reconcile step means the pure state model stays authoritative and
    the widgets never hold business rules.
    """

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.state = GuiState()
        self.params = gui_params.RenderParams()

        #: The full-resolution decoded source (display only; never mutated by
        #: the preview code and never handed to geometry in GUI-001).
        self._source_image: Optional[Image.Image] = None
        #: Strong references to the on-screen PhotoImages.  Tk keeps only a
        #: weak reference, so without these the images are garbage collected
        #: and the labels go blank -- the classic Tkinter image-lifetime trap.
        self._source_photo: Optional[ImageTk.PhotoImage] = None
        self._result_photo: Optional[ImageTk.PhotoImage] = None

        #: The last (box_w, box_h, source_id) the source preview was built for.
        #: Reconfiguring the preview label changes its size and therefore fires
        #: another ``<Configure>`` event; without this guard that would be an
        #: infinite rebuild loop.  We only rebuild when the box or the source
        #: actually changed.
        self._preview_key: Optional[tuple[int, int, int]] = None
        #: Whether a source-preview rebuild is already scheduled for idle time.
        self._preview_pending: bool = False

        self._build_window()
        self._build_layout()
        self.refresh()

    # ------------------------------------------------------------------
    # window / layout construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        self.root.title(_WINDOW_TITLE)
        self.root.minsize(760, 460)
        style = ttk.Style(self.root)
        # Respect the platform's own theme; only fall back to "clam" when the
        # default look has no usable theme at all.  No custom skin system.
        if "clam" in style.theme_names():
            try:
                style.theme_use("clam")
            except tk.TclError:  # pragma: no cover - defensive
                pass

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_top_bar()
        self._build_body()
        self._build_status_bar()

    def _build_top_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)

        ttk.Button(
            bar, text="Open Image\u2026", command=self._on_open_image
        ).grid(row=0, column=0, sticky="w")

        self._filename_var = tk.StringVar(value="No image loaded")
        ttk.Label(
            bar, textvariable=self._filename_var, anchor="w"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _build_body(self) -> None:
        body = ttk.Frame(self.root, padding=(8, 4, 8, 4))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_controls(body)
        self._build_previews(body)

    # -- left column: controls -----------------------------------------

    def _build_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Controls", padding=(8, 6))
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        for col in range(2):
            controls.columnconfigure(col, weight=0)

        row = 0

        # Mode -----------------------------------------------------------
        ttk.Label(controls, text="Mode").grid(row=row, column=0, sticky="w")
        self._mode_var = tk.StringVar(
            value=_MODE_LABELS[self.params.mode]
        )
        mode_box = ttk.Combobox(
            controls,
            textvariable=self._mode_var,
            values=[_MODE_LABELS[m] for m in gui_params.MODES],
            state="readonly",
            width=12,
        )
        mode_box.grid(row=row, column=1, sticky="ew", pady=2)
        mode_box.bind("<<ComboboxSelected>>", self._on_mode_changed)
        self._mode_box = mode_box
        row += 1

        # Width ----------------------------------------------------------
        ttk.Label(controls, text="Width").grid(row=row, column=0, sticky="w")
        self._width_var = tk.StringVar(
            value=_WIDTH_LABELS[self.params.width_mode]
        )
        width_box = ttk.Combobox(
            controls,
            textvariable=self._width_var,
            values=[_WIDTH_LABELS[w] for w in gui_params.WIDTH_MODES],
            state="readonly",
            width=12,
        )
        width_box.grid(row=row, column=1, sticky="ew", pady=2)
        width_box.bind("<<ComboboxSelected>>", self._on_width_changed)
        self._width_box = width_box
        row += 1

        # Render ---------------------------------------------------------
        ttk.Label(controls, text="Render").grid(row=row, column=0, sticky="w")
        self._render_var = tk.StringVar(
            value=_RENDER_LABELS[self.params.render]
        )
        render_box = ttk.Combobox(
            controls,
            textvariable=self._render_var,
            values=[_RENDER_LABELS[r] for r in gui_params.RENDER_CHOICES],
            state="readonly",
            width=12,
        )
        render_box.grid(row=row, column=1, sticky="ew", pady=2)
        render_box.bind("<<ComboboxSelected>>", self._on_render_changed)
        self._render_box = render_box
        row += 1

        ttk.Separator(controls, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6
        )
        row += 1

        ttk.Label(controls, text="Parameters").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        # Parameter rows.  Each row is a Label + Entry pair; the rows are
        # gridded/removed by visibility, and each keeps its own StringVar so
        # hiding never disturbs the stored text.
        self._param_rows: dict[str, dict] = {}
        row = self._add_param_row(
            controls, row, gui_params.PARAM_PERIOD, "Period", self.params.period_text
        )
        row = self._add_param_row(
            controls, row, gui_params.PARAM_SCALE, "Scale", self.params.scale_text
        )
        row = self._add_param_row(
            controls, row, gui_params.PARAM_ANGLE, "Angle", self.params.angle_text
        )
        row = self._add_param_row(
            controls, row, gui_params.PARAM_ARMS, "Arms", self.params.arms_text
        )
        row = self._add_param_row(
            controls,
            row,
            gui_params.PARAM_LINE_WIDTH,
            "Line width",
            self.params.line_width_text,
        )

        ttk.Separator(controls, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6
        )
        row += 1

        # Resolution readout --------------------------------------------
        self._source_res_var = tk.StringVar(value="Source: \u2014")
        ttk.Label(controls, textvariable=self._source_res_var).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        self._output_res_var = tk.StringVar(value="Output: \u2014")
        ttk.Label(controls, textvariable=self._output_res_var).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        ttk.Separator(controls, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6
        )
        row += 1

        # Actions (both disabled in GUI-001; wired for GUI-002) -----------
        actions = ttk.Frame(controls)
        actions.grid(row=row, column=0, columnspan=2, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self._generate_btn = ttk.Button(
            actions, text="Generate", command=self._on_generate
        )
        self._generate_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._save_btn = ttk.Button(
            actions, text="Save PNG\u2026", command=self._on_save
        )
        self._save_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _add_param_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        name: str,
        label: str,
        initial_text: str,
    ) -> int:
        """Create one ``Label`` + ``Entry`` row and remember its widgets."""
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=initial_text)
        entry = ttk.Entry(parent, textvariable=var, width=14)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        var.trace_add("write", self._make_param_trace(name))
        self._param_rows[name] = {
            "label": label_widget,
            "entry": entry,
            "var": var,
            "row": row,
        }
        return row + 1

    def _make_param_trace(self, name: str):
        def _callback(*_args) -> None:
            self._on_param_text_changed(name)

        return _callback

    # -- right column: previews ----------------------------------------

    def _build_previews(self, parent: ttk.Frame) -> None:
        previews = ttk.Frame(parent)
        previews.grid(row=0, column=1, sticky="nsew")
        previews.columnconfigure(0, weight=1, uniform="preview")
        previews.columnconfigure(1, weight=1, uniform="preview")
        previews.rowconfigure(0, weight=1)

        self._source_preview = self._make_preview_panel(previews, 0, "Source Preview")
        self._result_preview = self._make_preview_panel(previews, 1, "Result Preview")

        # The result preview has no generator in GUI-001: show an explicit
        # neutral placeholder rather than faking any output.
        self._result_image_label.configure(
            text="No result yet",
            image="",
        )
        self._result_image_label.image = None  # type: ignore[attr-defined]

    def _make_preview_panel(
        self, parent: ttk.Frame, column: int, title: str
    ) -> ttk.LabelFrame:
        panel = ttk.LabelFrame(parent, text=title, padding=(6, 6))
        panel.grid(
            row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0)
        )
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        # The image lives in a *fixed-size* holder frame with propagation
        # disabled.  This is what breaks the size feedback loop: if the label
        # itself sized the box, setting an image would change the box, which
        # would trigger another rebuild, and so on.  The holder's size is set
        # once here and then only changes when the window is genuinely resized.
        holder = ttk.Frame(panel, width=_PREVIEW_BOX, height=_PREVIEW_BOX)
        holder.grid(row=0, column=0, sticky="nsew")
        holder.grid_propagate(False)
        holder.pack_propagate(False)

        label = ttk.Label(
            holder,
            text="No image loaded" if column == 0 else "No result yet",
            anchor="center",
            justify="center",
        )
        label.pack(fill="both", expand=True)

        if column == 0:
            self._source_image_label = label
            self._source_preview_holder = holder
            holder.bind("<Configure>", self._make_preview_resize_trace(column))
        else:
            self._result_image_label = label
            self._result_preview_holder = holder
        return panel

    def _make_preview_resize_trace(self, column: int):
        def _callback(_event=None) -> None:
            # Defer the rebuild to idle time and coalesce bursts: a <Configure>
            # storm during window resize would otherwise rebuild the preview on
            # every intermediate size.  Only ever redraws the on-screen copy.
            self._schedule_preview_refresh()

        return _callback

    def _schedule_preview_refresh(self) -> None:
        """Schedule at most one pending source-preview rebuild.

        Setting a label image triggers further geometry events; without
        coalescing, the resize handler and those events would feed each other.
        A single ``after_idle`` token keeps the work bounded to one rebuild per
        idle cycle.
        """
        if self._preview_pending:
            return
        self._preview_pending = True
        self.root.after_idle(self._run_scheduled_preview_refresh)

    def _run_scheduled_preview_refresh(self) -> None:
        self._preview_pending = False
        self._refresh_source_preview()

    # -- bottom: status bar --------------------------------------------

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 4, 8, 8))
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        self._status_var = tk.StringVar(value="")
        ttk.Label(
            bar, textvariable=self._status_var, anchor="w"
        ).grid(row=0, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------

    def _on_open_image(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Open Image",
            filetypes=_FILE_TYPES,
        )
        if not path_str:
            return
        self._load_source(Path(path_str))

    def _on_mode_changed(self, _event=None) -> None:
        self.params.mode = self._label_to_mode(self._mode_var.get())
        self._sync_mode_state()
        self.refresh()

    def _on_width_changed(self, _event=None) -> None:
        self.params.width_mode = self._label_to_width(self._width_var.get())
        self.refresh()

    def _on_render_changed(self, _event=None) -> None:
        self.params.render = _RENDER_BY_LABEL.get(
            self._render_var.get(), self.params.render
        )
        self.refresh()

    def _on_param_text_changed(self, name: str) -> None:
        # Mirror the raw text into the params container and revalidate.  The
        # value is *not* parsed into a number here; the pure validation layer
        # decides what is legal, so an in-progress edit stays representable.
        text = self._param_rows[name]["var"].get()
        setattr(self.params, f"{name}_text", text)
        self.refresh()

    def _on_generate(self) -> None:
        # GUI-001: no pipeline exists.  The button is disabled, so this is a
        # defensive no-op rather than a stub implementation.
        pass

    def _on_save(self) -> None:
        # GUI-001: no Save implementation and no result to save.
        pass

    # ------------------------------------------------------------------
    # source loading
    # ------------------------------------------------------------------

    def _load_source(self, path: Path) -> None:
        """Load ``path`` as the new source, or report a readable error.

        Frozen rule: a *failed* load must not disturb an existing valid
        source / result.  The new image is therefore fully loaded into a local
        first, and only installed once it is known to be good.
        """
        try:
            image = imageutil.load_single_frame(path)
        except imageutil.ImageLoadError as exc:
            # Explicit action error -> a message box is appropriate (unlike
            # ordinary field validation, which stays inline).
            self._show_error("Cannot open image", str(exc))
            return

        # Success: replace the source and clear any old result.
        source = SourceInfo(path=path, width=image.size[0], height=image.size[1])
        self.state.set_source(source)
        self._source_image = image

        self._filename_var.set(source.filename)
        self._source_res_var.set(
            f"Source: {gui_params.format_resolution(source.size)}"
        )
        self._clear_result_preview()
        self.refresh()

    def _clear_result_preview(self) -> None:
        self._result_photo = None
        self._result_image_label.configure(text="No result yet", image="")
        self._result_image_label.image = None  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # view reconciliation
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reconcile the whole view from the pure state model.

        One method so there is exactly one place where state -> widgets
        happens.  Order matters only in that visibility is applied before the
        status text is computed.
        """
        self._apply_parameter_visibility()
        self._apply_output_resolution()
        self._apply_status()
        self._apply_lock_state()
        self._schedule_preview_refresh()

    def _apply_parameter_visibility(self) -> None:
        visible = gui_params.visible_parameters(
            self.params.mode, self.params.width_mode
        )
        for name, widgets in self._param_rows.items():
            if name in visible:
                widgets["label"].grid()
                widgets["entry"].grid()
            else:
                widgets["label"].grid_remove()
                widgets["entry"].grid_remove()

    def _apply_output_resolution(self) -> None:
        size = gui_params.output_resolution(
            self.params.mode, self.state.source_size, self.params.scale_text
        )
        self._output_res_var.set(f"Output: {gui_params.format_resolution(size)}")

    def _apply_status(self) -> None:
        """Compute the single inline status line.

        Priority order (highest first):

        1. a validation error on a visible field;
        2. a neutral "ready" / idle message.

        A rectangular source with ``Mode = Spiral`` is a *legal* configuration
        now (the GUI takes the centered maximum square automatically), so it
        never produces a compatibility error here.  Nothing here ever shows a
        traceback; the messages are plain sentences.
        """
        message = self._compose_status()
        self._status_var.set(message)

    def _compose_status(self) -> str:
        errors = gui_params.validate(self.params)
        if errors:
            # Preserve a stable, readable order rather than dict order.
            order = (
                gui_params.PARAM_PERIOD,
                gui_params.PARAM_SCALE,
                gui_params.PARAM_ANGLE,
                gui_params.PARAM_ARMS,
                gui_params.PARAM_LINE_WIDTH,
            )
            for name in order:
                if name in errors:
                    return errors[name]

        if not self.state.has_source:
            return "Open an image to begin."
        return "Ready."

    def _apply_lock_state(self) -> None:
        """Enable / disable widgets from the state model.

        In GUI-001 ``Generate`` and ``Save PNG...`` are always disabled because
        no pipeline exists.  The *rules* (``can_generate`` / ``can_save``)
        already live in the state model, so GUI-002 only has to relax this
        method.  When a job is running, all user inputs lock -- also expressed
        here ahead of time.
        """
        locked = self.state.inputs_locked
        input_state = "disabled" if locked else "normal"

        for name in self._param_rows:
            self._param_rows[name]["entry"].configure(state=input_state)
        self._mode_box.configure(state="disabled" if locked else "readonly")
        self._width_box.configure(state="disabled" if locked else "readonly")
        self._render_box.configure(state="disabled" if locked else "readonly")

        # GUI-001: no real generation -> both actions stay disabled.
        self._generate_btn.configure(state="disabled")
        self._save_btn.configure(state="disabled")

    def _refresh_source_preview(self) -> None:
        """Rebuild the on-screen source preview for the current box size.

        Guarded by :attr:`_preview_key`: setting a label's image changes its
        requested size and fires another ``<Configure>`` event, so without the
        guard this handler would rebuild forever.  The key encodes the box size
        and the source identity, so the preview is rebuilt exactly when it
        actually needs to change.
        """
        if self._source_image is None:
            self._source_photo = None
            self._preview_key = None
            self._source_image_label.configure(
                text="No image loaded", image=""
            )
            self._source_image_label.image = None  # type: ignore[attr-defined]
            return

        box = self._preview_box()
        key = (box[0], box[1], id(self._source_image))
        if key == self._preview_key and self._source_photo is not None:
            return

        preview = imageutil.scaled_preview(self._source_image, box)
        if preview is None:
            self._source_photo = None
            self._preview_key = key
            self._source_image_label.configure(image="", text="")
            self._source_image_label.image = None  # type: ignore[attr-defined]
            return

        photo = ImageTk.PhotoImage(preview)
        # Assign on the *label* as well as keeping our own reference: Tk keeps
        # only a weak reference to a PhotoImage, so this is what stops the
        # preview from vanishing after garbage collection.
        self._source_photo = photo
        self._preview_key = key
        self._source_image_label.configure(image=photo, text="")
        self._source_image_label.image = photo  # type: ignore[attr-defined]

    def _preview_box(self) -> tuple[int, int]:
        """Return the usable preview box from the source-preview holder.

        The *holder* frame is measured, not the label: the holder has a fixed
        size with propagation disabled, so it reflects the actual available
        space and is never influenced by the image currently displayed.
        Falls back to the fixed default before the first real ``<Configure>``
        event, when Tk still reports a 1x1 window.
        """
        width = self._source_preview_holder.winfo_width()
        height = self._source_preview_holder.winfo_height()
        if width <= 1 or height <= 1:
            return (_PREVIEW_BOX, _PREVIEW_BOX)
        return (width, height)

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_to_mode(label: str) -> str:
        for key, text in _MODE_LABELS.items():
            if text == label:
                return key
        return gui_params.MODE_STRIPE

    @staticmethod
    def _label_to_width(label: str) -> str:
        for key, text in _WIDTH_LABELS.items():
            if text == label:
                return key
        return gui_params.WIDTH_VARIABLE

    def _sync_mode_state(self) -> None:
        """Keep the mode combobox text and the params in step on mode change."""
        self._mode_var.set(_MODE_LABELS[self.params.mode])

    def _show_error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message, parent=self.root)


def run(argv: Optional[list[str]] = None) -> int:
    """Create the Tk root, build the app and enter the main loop.

    ``argv`` is accepted for symmetry with the CLI entry point and future use;
    GUI-001 defines no command-line options of its own.
    """
    root = tk.Tk()
    HalftoneApp(root)
    root.mainloop()
    return 0
