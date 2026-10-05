"""Tkinter / ttk GUI package for halftone-playground.

The package is split so that the *decisions* can be tested without a display
server and the *widgets* stay a thin shell:

* :mod:`halftone_playground.gui.params` -- pure: choices, visibility and
  validation (reuses the core contracts);
* :mod:`halftone_playground.gui.state` -- pure: the Source / Job / Result
  state model;
* :mod:`halftone_playground.gui.imageutil` -- pure (Pillow only): preview
  sizing and single-frame loading;
* :mod:`halftone_playground.gui.app` -- the Tk/ttk shell itself.

Importing this package must stay cheap and must **not** require a display:
``app`` imports ``tkinter``, so it is deliberately *not* imported here.  Import
:mod:`halftone_playground.gui.app` explicitly when a window is actually
wanted.
"""

from __future__ import annotations

from .params import (
    RenderParams,
    center_square_crop_box,
    center_square_side,
    effective_source_size,
    field_error,
    format_resolution,
    output_resolution,
    validate,
    visible_parameters,
)
from .state import GuiState, ResultInfo, SourceInfo

__all__ = [
    "GuiState",
    "RenderParams",
    "ResultInfo",
    "SourceInfo",
    "center_square_crop_box",
    "center_square_side",
    "effective_source_size",
    "field_error",
    "format_resolution",
    "output_resolution",
    "validate",
    "visible_parameters",
]
