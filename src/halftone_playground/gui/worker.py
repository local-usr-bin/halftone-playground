"""Single-job generation worker (no Tk import, no widget access).

GUI v1 freezes "**at most one** generate at a time": no job manager, no work
queue, no thread pool and **no Cancel**.  This module is the whole of that
concurrency story.

Design
------

* One :class:`threading.Thread`, started per job and marked **daemon**, is the
  only thread ever created.  A daemon thread cannot keep the process alive, so
  closing the window while a job runs needs no ``join`` and no cancellation
  protocol: the interpreter simply exits.
* The worker thread **never touches Tk**.  It cannot -- it does not import it.
  It computes a result (or an error) and puts exactly one message on a
  :class:`queue.Queue`.
* The Tk main thread is the only reader of that queue, from an
  ``after``-scheduled poll.  That is the standard, safest Tkinter pattern:
  ``queue.Queue`` for the hand-off, ``root.after`` for the cross-thread wake-up.
* At most one job is in flight.  :meth:`GenerationWorker.start` refuses to
  start a second one, and the shell additionally disables ``Generate`` while
  running -- the guard here is the backstop, not the only defence.
* The worker hands back **either** a result **or** a plain user-facing error
  message.  A raw traceback is never delivered: an unexpected exception is
  converted to a generic sentence, with the technical detail kept on the
  message object for stderr.

The full-resolution pixels travel as ``numpy`` data inside the result object.
NumPy arrays are ordinary heap objects, so handing one from the worker thread
to the main thread is safe; nothing is shared mutably and the worker never
revisits the array after putting it on the queue.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

from .pipeline import GenerationRequest, GenerationResult, PipelineError

__all__ = ["GenerationJob", "GenerationWorker", "WorkerOutcome"]


#: Generic message shown for a bug rather than a user error.  The traceback is
#: kept out of the GUI on purpose (see the round's runtime-error requirement).
_GENERIC_FAILURE = (
    "Generation failed unexpectedly. The details were written to the "
    "console."
)


@dataclass(frozen=True)
class GenerationJob:
    """The immutable work item handed to the worker thread."""

    request: GenerationRequest


@dataclass(frozen=True)
class WorkerOutcome:
    """One message from the worker: either a result or a user-facing error.

    Exactly one of ``result`` / ``error`` is set.  ``job`` is echoed back so
    the main thread can tell which job finished without keeping parallel
    bookkeeping -- important because a stale outcome for a superseded job must
    be ignored rather than displayed.
    """

    job: GenerationJob
    result: Optional[GenerationResult] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.result is not None


def _run_job(job: GenerationJob, out: "queue.Queue[WorkerOutcome]") -> None:
    """Thread body: run the pipeline and post exactly one outcome.

    Never raises: every path ends with exactly one ``put``, so the main
    thread's poll can never wait forever for a job that silently died.
    """
    try:
        from .pipeline import generate

        result = generate(job.request)
        out.put(WorkerOutcome(job=job, result=result))
    except PipelineError as exc:
        # Expected, user-facing: show the sentence as-is.
        out.put(WorkerOutcome(job=job, error=str(exc)))
    except Exception:  # noqa: BLE001 - deliberate catch-all backstop
        # A programming bug.  Keep the GUI clean and the console useful.
        traceback.print_exc()
        out.put(WorkerOutcome(job=job, error=_GENERIC_FAILURE))


class GenerationWorker:
    """Owns the single worker thread and its result queue.

    The worker is *not* a long-lived pool: each job gets a fresh daemon thread
    that exits when its one outcome has been posted.  That keeps the lifecycle
    trivial (there is no idle thread to shut down) while still guaranteeing
    "at most one at a time", because :meth:`start` refuses re-entry.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[WorkerOutcome]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._current: Optional[GenerationJob] = None

    # -- queries ---------------------------------------------------------

    @property
    def busy(self) -> bool:
        """Whether a job is currently running."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def current_job(self) -> Optional[GenerationJob]:
        return self._current

    # -- commands --------------------------------------------------------

    def start(self, job: GenerationJob) -> bool:
        """Start ``job``; return ``False`` if a job is already running.

        A ``False`` return is not an error path the user should normally see
        (the shell disables ``Generate`` while running); it exists so the
        single-job rule is enforced in one place rather than assumed.
        """
        if self.busy:
            return False
        self._current = job
        thread = threading.Thread(
            target=_run_job,
            args=(job, self._queue),
            name="halftone-generate",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return True

    def drain(self) -> list[WorkerOutcome]:
        """Return every outcome currently queued, without blocking.

        Called on the Tk main thread only.  Non-blocking on purpose: the poll
        must never stall the event loop, and a job that is still running
        simply yields an empty list.
        """
        outcomes: list[WorkerOutcome] = []
        while True:
            try:
                outcomes.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return outcomes

    def reset(self) -> None:
        """Forget the finished job so the worker is idle again.

        Deliberately does **not** try to stop a running thread -- there is no
        cancel in GUI v1 and the thread is a daemon, so it is harmless to let
        it finish and drop its outcome (the main thread ignores outcomes for
        jobs it no longer recognises).
        """
        self._current = None


def poll_worker(
    worker: GenerationWorker,
    on_outcome: Callable[[WorkerOutcome], None],
) -> None:
    """Deliver every queued outcome to ``on_outcome``.

    A tiny helper so the shell's ``after`` callback has exactly one job.  Kept
    module-level (rather than a method) so it can be unit-tested against a
    fake worker without touching Tk.
    """
    for outcome in worker.drain():
        on_outcome(outcome)
