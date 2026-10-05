"""Tests for the pure GUI state model (no display server required).

Covers the Source / Job / Result concepts frozen in GUI Round 0, the
"new source clears the result" rule and the single-job lock, without a giant
combination enum.
"""

from __future__ import annotations

from pathlib import Path

from halftone_playground.gui import state as gs


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

    def test_stale_result_is_still_displayable(self):
        # Stale means "cannot save", not "cannot show".
        s = gs.GuiState()
        s.set_source(_source())
        s.set_result(gs.ResultInfo(64, 64), stale=True)
        assert s.has_result is True
        assert s.can_save is False
