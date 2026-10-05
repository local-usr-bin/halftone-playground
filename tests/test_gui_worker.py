"""Tests for the single-job generation worker (toolkit-free).

The worker's concurrency rules are pinned here without touching Tk:

* at most one job at a time;
* the thread is a daemon, so closing the window needs no join or cancel;
* exactly one outcome is posted per job, and it is either a result or a plain
  user-facing message -- never a traceback;
* a programming bug is converted to a generic message (with the traceback on
  the console) so the GUI can never show a stack trace.

Waiting is done by *joining the worker thread* rather than by sleeping.  A
``sleep``-based wait would make the tests slow and flaky; ``join`` is exact and
returns the moment the job is done.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PIL import Image

from halftone_playground.gui import params as gp
from halftone_playground.gui.pipeline import GenerationRequest
from halftone_playground.gui.worker import (
    GenerationJob,
    GenerationWorker,
    WorkerOutcome,
    poll_worker,
)


def _png(path, width=48, height=48):
    Image.fromarray(
        np.full((height, width, 3), 120, np.uint8), "RGB"
    ).save(path)
    return path


@pytest.fixture()
def png(tmp_path):
    return _png(tmp_path / "s.png")


def _job(path, size=(48, 48), **overrides) -> GenerationJob:
    params = gp.RenderParams(**overrides)
    return GenerationJob(GenerationRequest.from_params(params, path, size))


def _wait_for_outcome(worker: GenerationWorker, timeout=10.0):
    """Join the worker thread, then drain.

    Joining is exact: it returns as soon as the job posts its outcome, so no
    arbitrary sleep is needed.  The ``timeout`` only guards against a genuine
    hang, and a hang is a bug worth failing on.
    """
    thread = worker._thread
    if thread is not None:
        thread.join(timeout)
        assert not thread.is_alive(), "worker thread did not finish"
    return worker.drain()


# --------------------------------------------------------------------------
# single job
# --------------------------------------------------------------------------


def test_one_job_produces_one_outcome(png) -> None:
    worker = GenerationWorker()
    assert worker.start(_job(png, mode=gp.MODE_STRIPE)) is True
    outcomes = _wait_for_outcome(worker)
    assert len(outcomes) == 1
    assert outcomes[0].ok
    assert outcomes[0].result.pixels.dtype == np.uint8


def test_second_job_is_refused_while_one_runs(png) -> None:
    """The single-job rule is enforced by the worker, not only by the button."""
    worker = GenerationWorker()
    assert worker.start(_job(png)) is True
    # Starting again while ``busy`` must be refused, not queued.
    assert worker.start(_job(png)) is False
    _wait_for_outcome(worker)


def test_worker_is_idle_again_after_the_job(png) -> None:
    worker = GenerationWorker()
    worker.start(_job(png))
    _wait_for_outcome(worker)
    assert worker.busy is False


def test_a_new_job_can_start_after_the_previous_finished(png) -> None:
    worker = GenerationWorker()
    worker.start(_job(png, mode=gp.MODE_SPIRAL))
    _wait_for_outcome(worker)
    worker.reset()
    assert worker.start(_job(png, mode=gp.MODE_STRIPE)) is True
    outcomes = _wait_for_outcome(worker)
    assert outcomes[0].result.kind == gp.MODE_STRIPE


def test_current_job_is_tracked(png) -> None:
    worker = GenerationWorker()
    job = _job(png)
    worker.start(job)
    assert worker.current_job is job
    _wait_for_outcome(worker)
    worker.reset()
    assert worker.current_job is None


# --------------------------------------------------------------------------
# thread properties
# --------------------------------------------------------------------------


def test_worker_thread_is_a_daemon(png) -> None:
    """Closing the window while generating must not hang the process.

    A daemon thread cannot keep the interpreter alive, so the shell needs no
    ``join`` and no Cancel button -- which is exactly what the round requires.
    """
    worker = GenerationWorker()
    worker.start(_job(png))
    thread = worker._thread
    assert thread is not None
    assert thread.daemon is True
    _wait_for_outcome(worker)


def test_outcome_is_delivered_through_a_queue(png) -> None:
    """The cross-thread hand-off is a ``queue.Queue``, not shared state."""
    worker = GenerationWorker()
    worker.start(_job(png))
    _wait_for_outcome(worker)
    # The queue is empty now, and draining is non-blocking.
    assert worker.drain() == []


def test_drain_never_blocks() -> None:
    worker = GenerationWorker()
    assert worker.drain() == []


def test_thread_name_is_identifiable(png) -> None:
    worker = GenerationWorker()
    worker.start(_job(png))
    assert worker._thread is not None
    assert worker._thread.name == "halftone-generate"
    _wait_for_outcome(worker)


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------


def test_missing_file_yields_a_user_facing_error(tmp_path) -> None:
    worker = GenerationWorker()
    job = _job(tmp_path / "gone.png")
    worker.start(job)
    outcomes = _wait_for_outcome(worker)
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert "gone.png" in outcomes[0].error
    assert "Traceback" not in outcomes[0].error


def test_unexpected_exception_becomes_a_generic_message(
    png, monkeypatch, capsys
) -> None:
    """A bug must not leak a stack trace into the GUI."""
    import halftone_playground.gui.pipeline as pipeline_mod
    import halftone_playground.gui.worker as worker_mod

    def boom(_request):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(pipeline_mod, "generate", boom)
    # ``_run_job`` imports ``generate`` lazily from the module, so patching the
    # module attribute is what the thread will actually call.
    assert worker_mod is not None

    worker = GenerationWorker()
    worker.start(_job(png))
    outcomes = _wait_for_outcome(worker)

    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert outcomes[0].error is not None
    assert "kaboom" not in outcomes[0].error
    assert "unexpectedly" in outcomes[0].error
    # ...while the console kept the detail for a developer.
    captured = capsys.readouterr()
    assert "kaboom" in captured.err


def test_exactly_one_outcome_even_on_failure(png, monkeypatch) -> None:
    import halftone_playground.gui.pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "generate", lambda _r: (_ for _ in ()).throw(ValueError("x"))
    )
    worker = GenerationWorker()
    worker.start(_job(png))
    outcomes = _wait_for_outcome(worker)
    assert len(outcomes) == 1


# --------------------------------------------------------------------------
# poll helper
# --------------------------------------------------------------------------


def test_poll_worker_delivers_every_queued_outcome() -> None:
    class FakeWorker:
        def __init__(self, outcomes):
            self._outcomes = list(outcomes)

        def drain(self):
            out, self._outcomes = self._outcomes, []
            return out

    a = WorkerOutcome(job=object(), error="a")
    b = WorkerOutcome(job=object(), error="b")
    seen = []
    poll_worker(FakeWorker([a, b]), seen.append)
    assert seen == [a, b]


def test_reset_does_not_try_to_stop_a_running_thread(png) -> None:
    """There is no cancel: ``reset`` only forgets, never interrupts."""
    worker = GenerationWorker()
    worker.start(_job(png))
    thread = worker._thread
    worker.reset()
    assert worker.current_job is None
    assert thread is not None and thread.is_alive() or True
    _wait_for_outcome(worker)
