"""GUI state model: Source / Job / Result (pure logic, no toolkit import).

GUI Round 0 froze three *conceptual* state dimensions and explicitly banned a
single giant combination enum:

======================  ====================================
``Source``              ``none`` | ``loaded``
``Job``                 ``idle`` | ``running``
``Result``              ``none`` | ``current`` | ``stale``
======================  ====================================

This module models them as a few small, explicit fields on one
:class:`GuiState` object plus a handful of pure transition helpers.  There is
deliberately **no** ``READY_WITH_SOURCE_AND_STALE_RESULT...`` enumeration.

GUI-002A fills in the lifecycle that GUI-001 only sketched:

* loading a new source replaces the old source and **clears** any result
  (a superseded result is dropped, never kept as ``stale``);
* a result is ``stale`` while an output-affecting parameter differs from the
  one that produced it, and becomes ``current`` again automatically when the
  parameters return to the **same semantic configuration** -- including the
  ``"1"`` / ``"1.0"`` case, because the comparison is done on a normalized
  :class:`~halftone_playground.gui.pipeline.GenerationKey` rather than on the
  raw text the user typed;
* hiding a parameter never changes state -- visibility is a view concern, and
  the key only ever records parameters that can actually affect the output.

The module knows nothing about widgets or images: a source is represented by
its path and size only, and a result by its size, provenance key and the
identity of the source it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "JOB_IDLE",
    "JOB_RUNNING",
    "RESULT_CURRENT",
    "RESULT_NONE",
    "RESULT_STALE",
    "SOURCE_LOADED",
    "SOURCE_NONE",
    "GuiState",
    "ResultInfo",
    "SourceInfo",
]

SOURCE_NONE = "none"
SOURCE_LOADED = "loaded"

JOB_IDLE = "idle"
JOB_RUNNING = "running"

RESULT_NONE = "none"
RESULT_CURRENT = "current"
RESULT_STALE = "stale"


@dataclass(frozen=True)
class SourceInfo:
    """The loaded source image's identity: its path and pixel size.

    Deliberately tiny: the GUI never needs to keep the full decoded image in
    the state model, and keeping only the metadata makes the transition rules
    trivial to test without touching Pillow.
    """

    path: Path
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def is_square(self) -> bool:
        return self.width == self.height


@dataclass(frozen=True)
class ResultInfo:
    """Metadata for one generated result.

    The pixel data itself deliberately does **not** live here: the state model
    is the small, headless-testable description of *what* exists, while the
    full-resolution array is owned by the shell.  Keeping the two apart is
    what lets every lifecycle rule below be tested without Pillow, Tk or a
    display.

    ``generation_key`` is the normalized semantic configuration that produced
    this result (see
    :class:`halftone_playground.gui.pipeline.GenerationKey`).  Comparing it to
    the live parameters is the whole of the ``current`` / ``stale`` decision.
    It is typed ``object`` so this module stays free of a pipeline import (and
    therefore free of numpy) while still round-tripping the key untouched.

    ``source_path`` records which file the result came from, so a result can
    never be mistaken for one belonging to a different image.  ``mode`` is the
    Pillow mode of the real output (``"RGB"`` or ``"RGBA"``).
    """

    width: int
    height: int
    generation_key: object = None
    source_path: Optional[Path] = None
    mode: str = "RGB"

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def has_alpha(self) -> bool:
        return self.mode == "RGBA"


@dataclass
class GuiState:
    """The GUI's small, explicit state container.

    Three *independent* concepts are represented as independent fields, so the
    state is always inspectable without a combinatorial enum:

    * ``source`` / ``result``: :class:`SourceInfo` / :class:`ResultInfo` or
      ``None``;
    * ``job``: :data:`JOB_IDLE` or :data:`JOB_RUNNING`.

    ``result_status`` is the one derived-ish field: it is ``none`` exactly when
    there is no result, and otherwise ``current`` or ``stale``.
    """

    source: Optional[SourceInfo] = None
    result: Optional[ResultInfo] = None
    result_status: str = RESULT_NONE
    job: str = JOB_IDLE

    #: Inline status message for the bottom area (validation / info).  A plain
    #: string field so the pure model can express what the shell should show
    #: without importing any widget.
    status_message: str = field(default="")

    # -- derived predicates ------------------------------------------------

    @property
    def has_source(self) -> bool:
        return self.source is not None

    @property
    def source_size(self) -> Optional[tuple[int, int]]:
        return None if self.source is None else self.source.size

    @property
    def has_result(self) -> bool:
        return self.result is not None and self.result_status != RESULT_NONE

    @property
    def is_running(self) -> bool:
        return self.job == JOB_RUNNING

    @property
    def can_generate(self) -> bool:
        """Whether ``Generate`` may be enabled.

        GUI-002A rule: a source must be loaded, every visible parameter must
        validate, and no job may already be running.  A rectangular source
        with ``Mode = Spiral`` is a *legal* configuration (the GUI takes the
        centered maximum square itself), so it never blocks generation.

        ``params_valid`` is passed in rather than computed here because
        parameter validation belongs to
        :mod:`halftone_playground.gui.params`; this module stays the decision
        about *state*, not about field rules.
        """
        return self.has_source and not self.is_running

    def generate_allowed(self, *, params_valid: bool) -> bool:
        """``Generate`` eligibility: source + valid params + no running job.

        A thin wrapper over :attr:`can_generate` and ``params_valid`` so the
        shell has one call to make and the full rule is testable here.
        """
        return self.can_generate and params_valid

    @property
    def can_save(self) -> bool:
        """Whether ``Save PNG...`` may be enabled.

        GUI-002B rule: saving is allowed only for a ``current`` (non-stale)
        result while the worker is **idle**.  A ``none`` result is not
        saveable, a ``stale`` result is not saveable, and a running job locks
        saving like every other input -- all three cases collapse into this one
        predicate, so the button and the save callback share a single rule.
        """
        return (
            self.has_result
            and self.result_status == RESULT_CURRENT
            and not self.is_running
        )

    @property
    def inputs_locked(self) -> bool:
        """Whether user inputs should be disabled (a job is running)."""
        return self.is_running

    # -- transitions -------------------------------------------------------

    def set_source(self, source: SourceInfo) -> None:
        """Install a freshly loaded source and clear any existing result.

        Frozen rule: a successful new source **clears** the old result outright
        -- it is never demoted to ``stale``.  Keeping a superseded picture
        around would mislead, and the old result belonged to a different
        image entirely.
        """
        self.source = source
        self.result = None
        self.result_status = RESULT_NONE

    def clear_source(self) -> None:
        """Drop the source and any result (used by a failed load rollback)."""
        self.source = None
        self.result = None
        self.result_status = RESULT_NONE

    def set_result(self, result: ResultInfo, *, stale: bool = False) -> None:
        """Install a result and mark it ``current`` or ``stale``."""
        self.result = result
        self.result_status = RESULT_STALE if stale else RESULT_CURRENT

    def clear_result(self) -> None:
        self.result = None
        self.result_status = RESULT_NONE

    def mark_result_stale(self) -> None:
        """Demote a current result to stale (an output-affecting param moved).

        A stale result may keep being *displayed*, but it is no longer
        saveable; this only flips the status flag.
        """
        if self.result is not None:
            self.result_status = RESULT_STALE

    def mark_result_current(self) -> None:
        """Promote a stale result back to current (params returned to match)."""
        if self.result is not None:
            self.result_status = RESULT_CURRENT

    def sync_result_status(self, current_key: object) -> bool:
        """Re-derive ``current`` / ``stale`` from a live generation key.

        This is the single entry point GUI-002A uses after *any* output-
        affecting change, and it is deliberately a *reconciliation* rather
        than a pair of manual ``mark_*`` calls: it compares
        :attr:`ResultInfo.generation_key` with ``current_key`` and sets the
        status to match.  That gives the "auto-revert to current" rule for
        free -- moving ``Scale`` from ``1`` to ``1.3`` and back to ``1`` ends
        with the result ``current`` again, with no special case anywhere.

        Equality is plain ``==`` on the keys, which is exactly why the key is
        normalized before it is stored: the comparison must not be confused by
        the *text* the user typed (``"1"`` vs ``"1.0"``).

        Returns ``True`` when the status ended up ``current``.  With no result
        installed it is a no-op (and returns ``False``) -- there is nothing to
        be stale about.
        """
        if self.result is None:
            self.result_status = RESULT_NONE
            return False
        if self.result.generation_key == current_key:
            self.result_status = RESULT_CURRENT
            return True
        self.result_status = RESULT_STALE
        return False

    def belongs_to_source(self, path: Path) -> bool:
        """Whether the installed result came from ``path``.

        Guard for the worker hand-off: when a finished job is delivered, the
        result must still belong to the source that is currently loaded.
        """
        return (
            self.result is not None
            and self.result.source_path is not None
            and Path(self.result.source_path) == Path(path)
        )

    def begin_job(self) -> None:
        """Enter the single-job ``running`` state."""
        self.job = JOB_RUNNING

    def end_job(self) -> None:
        """Return to ``idle`` when the worker finishes (success or failure)."""
        self.job = JOB_IDLE
