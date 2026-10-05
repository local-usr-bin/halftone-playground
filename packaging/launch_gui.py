"""PyInstaller entry script for the halftone-playground GUI.

``python -m halftone_playground.gui`` is implemented with package-relative
imports, so it cannot be used directly as a top-level freeze entry script.
This shim is the equivalent absolute-import entry point and nothing more.
"""

from halftone_playground.gui.app import run

raise SystemExit(run())
