"""``python -m halftone_playground`` entry point.

Kept deliberately thin: it only defers to :func:`halftone_playground.cli.main`
so module invocation behaves exactly like the installed ``halftone-playground``
console script.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
