"""Tests for the pure GUI state model (no display server required).

Covers the Source / Job / Result concepts frozen in GUI Round 0, the
"new source clears the result" rule and the single-job lock, without a giant
combination enum.
"""

from __future__ import annotations

from pathlib import Path

from halftone_playground.gui import state as gs
from halftone_playground.gui.state import (
    RESULT_CURRENT,
    RESULT_NONE,
    RESULT_STALE,
    GuiState,
    ResultInfo,
    SourceInfo,
)


def _source(name: str = "a.png", width: int = 64, height: int = 64) -> gs.SourceInfo:
    return gs.SourceInfo(path=Path(name), width=width, height=height)


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------


class TestSource:
    def test_initial_source_is_none(self):
        s = gs.GuiState()
        assert s.source is None
        assert s.has_source is False
        assert s.source_size is None

    def test_set_source_marks_loaded(self):
        s = gs.GuiState()
        s.set_source(_source("pic.png", 100, 80))
        assert s.has_source is True
        assert s.source_size == (100, 80)
        assert s.source.filename == "pic.png"

    def test_source_info_square_flag(self):
        assert _source(width=50, height=50).is_square is True
        assert _source(width=50, height=60).is_square is False

    def test_clear_source_drops_everything(self):
        s = gs.GuiState()
        s.set_source(_source())
        s.set_result(gs.ResultInfo(64, 64))
        s.clear_source()
        assert s.source is None
        assert s.result is None
        assert s.result_status == gs.RESULT_NONE


# --------------------------------------------------------------------------
# Source replacement clears result
# --------------------------------------------------------------------------


class TestSourceReplacement:
    def test_new_source_clears_result_state(self):
        s = gs.GuiState()
        s.set_source(_source("first.png"))
        s.set_result(gs.ResultInfo(64, 64), stale=False)
        assert s.result_status == gs.RESULT_CURRENT

        s.set_source(_source("second.png", 32, 32))
        # Old result is dropped outright -- never demoted to stale.
        assert s.result is None
        assert s.result_status == gs.RESULT_NONE
        assert s.has_result is False

    def test_new_source_keeps_only_new_metadata(self):
        s = gs.GuiState()
        s.set_source(_source("first.png", 10, 20))
        s.set_source(_source("second.png", 30, 40))
        assert s.source.filename == "second.png"
        assert s.source_size == (30, 40)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


class TestResult:
    def test_initial_result_is_none(self):
        s = gs.GuiState()
        assert s.result is None
        assert s.result_status == gs.RESULT_NONE
        assert s.has_result is False

    def test_set_result_current(self):
        s = gs.GuiState()
        s.set_result(gs.ResultInfo(64, 64))
        assert s.result_status == gs.RESULT_CURRENT
        assert s.has_result is True

    def test_set_result_stale(self):
        s = gs.GuiState()
        s.set_result(gs.ResultInfo(64, 64), stale=True)
        assert s.result_status == gs.RESULT_STALE
        assert s.has_result is True

    def test_mark_stale_and_back_to_current(self):
        s = gs.GuiState()
        s.set_result(gs.ResultInfo(64, 64))
        s.mark_result_stale()
        assert s.result_status == gs.RESULT_STALE
        s.mark_result_current()
        assert s.result_status == gs.RESULT_CURRENT

    def test_mark_stale_without_result_is_noop(self):
        s = gs.GuiState()
        s.mark_result_stale()
        assert s.result_status == gs.RESULT_NONE

    def test_clear_result(self):
        s = gs.GuiState()
        s.set_result(gs.ResultInfo(64, 64))
        s.clear_result()
        assert s.result is None
        assert s.result_status == gs.RESULT_NONE


# --------------------------------------------------------------------------
# Job / eligibility
# --------------------------------------------------------------------------


class TestJob:
    def test_initial_job_is_idle(self):
        s = gs.GuiState()
        assert s.job == gs.JOB_IDLE
        assert s.is_running is False

    def test_begin_and_end_job(self):
        s = gs.GuiState()
        s.begin_job()
        assert s.job == gs.JOB_RUNNING
        assert s.is_running is True
        s.end_job()
        assert s.job == gs.JOB_IDLE

    def test_cannot_generate_without_source(self):
        s = gs.GuiState()
        assert s.can_generate is False

    def test_can_generate_with_source_and_idle(self):
        s = gs.GuiState()
        s.set_source(_source())
        assert s.can_generate is True

    def test_cannot_generate_while_running(self):
        s = gs.GuiState()
        s.set_source(_source())
        s.begin_job()
        assert s.can_generate is False

    def test_running_locks_inputs(self):
        s = gs.GuiState()
        assert s.inputs_locked is False
        s.begin_job()
        assert s.inputs_locked is True

    def test_cannot_save_without_result(self):
        s = gs.GuiState()
        s.set_source(_source())
        assert s.can_save is False

    def test_can_save_only_when_current(self):
        s = gs.GuiState()
        s.set_source(_source())
        s.set_result(gs.ResultInfo(64, 64))
        assert s.can_save is True

        s.mark_result_stale()
        assert s.can_save is False

    def test_cannot_save_while_a_job_runs(self):
        # GUI-002B: a running job locks Save like every other input, and the
        # button returns as soon as the job finishes.
        s = gs.GuiState()
        s.set_source(_source())
        s.set_result(gs.ResultInfo(64, 64))
        assert s.can_save is True

        s.begin_job()
        assert s.can_save is False

        s.end_job()
        assert s.can_save is True

    def test_stale_result_is_still_displayable(self):
        # Stale means "cannot save", not "cannot show".
        s = gs.GuiState()
        s.set_source(_source())
        s.set_result(gs.ResultInfo(64, 64), stale=True)
        assert s.has_result is True
        assert s.can_save is False


# --------------------------------------------------------------------------
# GUI-002A: the current / stale lifecycle
# --------------------------------------------------------------------------
#
# The comparison itself lives in ``pipeline.GenerationKey``; here the state
# model is driven with plain sentinel objects, which keeps these tests free of
# numpy and of the pipeline -- the state model only ever round-trips the key.


def _result(path=Path("/tmp/a.png"), key="k1"):
    return ResultInfo(
        width=10, height=10, generation_key=key, source_path=path, mode="RGB"
    )


class TestResultLifecycle:
    def test_none_when_there_is_no_result(self):
        state = GuiState()
        assert state.result_status == RESULT_NONE
        assert state.has_result is False

    def test_generating_makes_a_current_result(self):
        state = GuiState()
        state.set_result(_result())
        assert state.result_status == RESULT_CURRENT
        assert state.has_result is True

    def test_sync_keeps_current_when_the_key_matches(self):
        state = GuiState()
        state.set_result(_result(key="k1"))
        assert state.sync_result_status("k1") is True
        assert state.result_status == RESULT_CURRENT

    def test_sync_marks_stale_when_the_key_differs(self):
        state = GuiState()
        state.set_result(_result(key="k1"))
        assert state.sync_result_status("k2") is False
        assert state.result_status == RESULT_STALE

    def test_sync_reverts_to_current_when_the_key_returns(self):
        """The auto-revert rule: away and back ends ``current`` again."""
        state = GuiState()
        state.set_result(_result(key="k1"))
        state.sync_result_status("k2")
        assert state.result_status == RESULT_STALE
        state.sync_result_status("k1")
        assert state.result_status == RESULT_CURRENT

    def test_sync_is_a_noop_without_a_result(self):
        state = GuiState()
        assert state.sync_result_status("anything") is False
        assert state.result_status == RESULT_NONE

    def test_stale_result_is_still_displayable(self):
        """A stale result keeps its metadata; only the status changes."""
        state = GuiState()
        state.set_result(_result(key="k1"))
        state.sync_result_status("k2")
        assert state.result is not None
        assert state.result.size == (10, 10)

    def test_stale_result_is_not_saveable(self):
        state = GuiState()
        state.set_result(_result(key="k1"))
        assert state.can_save is True
        state.sync_result_status("k2")
        assert state.can_save is False

    def test_a_new_source_clears_the_result_outright(self):
        state = GuiState()
        state.set_source(SourceInfo(path=Path("/tmp/a.png"), width=4, height=4))
        state.set_result(_result(path=Path("/tmp/a.png")))
        state.set_source(SourceInfo(path=Path("/tmp/b.png"), width=8, height=8))
        assert state.result is None
        assert state.result_status == RESULT_NONE

    def test_belongs_to_source_matches_only_the_right_file(self):
        state = GuiState()
        state.set_result(_result(path=Path("/tmp/a.png")))
        assert state.belongs_to_source(Path("/tmp/a.png")) is True
        assert state.belongs_to_source(Path("/tmp/b.png")) is False

    def test_belongs_to_source_is_false_without_a_result(self):
        assert GuiState().belongs_to_source(Path("/tmp/a.png")) is False


class TestGenerateEligibility:
    def test_no_source_means_no_generate(self):
        state = GuiState()
        assert state.generate_allowed(params_valid=True) is False

    def test_source_and_valid_params_allows_generate(self):
        state = GuiState()
        state.set_source(SourceInfo(path=Path("/tmp/a.png"), width=4, height=4))
        assert state.generate_allowed(params_valid=True) is True

    def test_invalid_params_block_generate(self):
        state = GuiState()
        state.set_source(SourceInfo(path=Path("/tmp/a.png"), width=4, height=4))
        assert state.generate_allowed(params_valid=False) is False

    def test_running_job_blocks_generate(self):
        state = GuiState()
        state.set_source(SourceInfo(path=Path("/tmp/a.png"), width=4, height=4))
        state.begin_job()
        assert state.generate_allowed(params_valid=True) is False

    def test_running_job_locks_the_inputs(self):
        state = GuiState()
        state.set_source(SourceInfo(path=Path("/tmp/a.png"), width=4, height=4))
        state.begin_job()
        assert state.inputs_locked is True
        state.end_job()
        assert state.inputs_locked is False
