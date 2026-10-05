"""``python -m halftone_playground.gui`` entry point.

Kept deliberately thin, mirroring :mod:`halftone_playground.__main__`: it only
defers to :func:`halftone_playground.gui.app.run`.  ``app`` is imported
*inside* ``main`` so that merely importing this module never requires a display
server.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from .app import run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
