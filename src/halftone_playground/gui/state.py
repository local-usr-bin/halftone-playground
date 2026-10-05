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

GUI-001 does not run a worker or produce a result yet, but the transitions
here are the ones GUI-002 will need, so the shape is already right:

* loading a new source replaces the old source and **clears** any result
  (a superseded result is dropped, never kept as ``stale``);
* a result can later be marked ``stale`` when an output-affecting parameter
  changes, and ``current`` again when the parameters return to the same
  semantic configuration (that comparison lives in GUI-002);
* hiding a parameter never changes state -- visibility is a view concern.

The module knows nothing about widgets or images: a source is represented by
its path and size only.
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

    Empty in GUI-001 (no generation yet), but defined now so the state model
    is complete and GUI-002 has an obvious place to attach a result handle.
    ``params_fingerprint`` is the semantic configuration that produced the
    result; GUI-002 compares it to the live parameters to decide
    ``current`` vs ``stale``.
    """

    width: int
    height: int
    params_fingerprint: object = None


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

        GUI-001 keeps the actual button disabled (no pipeline yet), but the
        *rule* is expressed here for GUI-002: at least one job at a time, a
        source must be loaded, and no job may already be running.
        """
        return self.has_source and not self.is_running

    @property
    def can_save(self) -> bool:
        """Whether ``Save PNG...`` may be enabled.

        Only a ``current`` (non-stale) result is saveable; GUI-001 therefore
        always reports ``False`` because there is never a result.
        """
        return self.has_result and self.result_status == RESULT_CURRENT

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
        """Install a result and mark it ``current`` or ``stale``.

        Not exercised by GUI-001; present so the shape is ready for GUI-002.
        """
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

    def begin_job(self) -> None:
        """Enter the single-job ``running`` state."""
        self.job = JOB_RUNNING

    def end_job(self) -> None:
        """Return to ``idle`` when the worker finishes (success or failure)."""
        self.job = JOB_IDLE
