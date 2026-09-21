"""The nightly job must not report success when a step failed.

WHAT HAPPENED, TWICE IN THREE DAYS, both times invisibly:

  16-19 Sep  fetch_tail only backfilled symbols the store had never seen,
             so the daily store fell four days behind and prev_close
             became a different date per symbol - while the log said
             "daily store now holds 2,521 symbols", which was true.
  20 Sep     fetch_tail died on an expired Kite token and the job carried
             on through eight more steps, finishing with no indication
             that the one step that mattered had aborted.

The job ran nine steps unconditionally and never looked at an exit code.

These tests read the script rather than executing it - running it would
take half an hour and hit Kite - so they check the properties that make
the checking real: every step is followed by a check, the exit code is
captured BEFORE anything that would reset it, and the script exits
non-zero when a step failed. The cmd control flow itself was verified by
probe (two failing steps out of four produced exit code 2, the right two
names, and left the passing steps unflagged).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

JOB = Path(__file__).resolve().parent / "run_nightly.cmd"
TEXT = JOB.read_text(encoding="utf-8", errors="replace")
LINES = [line.rstrip() for line in TEXT.splitlines()]


def _commands() -> list:
    """Every line that actually runs a python step."""
    return [line.strip() for line in LINES
            if '"%PY%"' in line and not line.strip().startswith("REM")]


def test_the_job_exists_and_runs_python_steps() -> None:
    assert _commands(), "no python steps found - did the file move?"


def test_every_step_is_followed_by_a_check() -> None:
    """A step whose exit code nobody reads is the original defect."""
    runs = [i for i, line in enumerate(LINES)
            if '"%PY%"' in line and not line.strip().startswith("REM")]
    assert runs, "no steps found"
    for index in runs:
        following = LINES[index + 1].strip()
        assert following.startswith("call :check"), (
            f"line {index + 1} runs a step but the next line is "
            f"{following!r}, so its exit code is never read")


def test_every_check_names_a_distinct_step() -> None:
    """A repeated name makes the failure summary ambiguous."""
    names = re.findall(r"^\s*call :check (\S+)", TEXT, flags=re.M)
    assert len(names) == len(set(names)), names
    assert len(names) == len(_commands())


def test_the_exit_code_is_captured_before_anything_resets_it() -> None:
    """`set /a` resets ERRORLEVEL, so the capture has to come first.

    Reading %ERRORLEVEL% after `set /a FAILS+=1` would record 0 for every
    failure and the summary would name steps with "exit code 0".
    """
    body = TEXT.split(":check", 1)[1]
    capture = body.index('set "CODE=%ERRORLEVEL%"')
    arithmetic = body.index("set /a FAILS")
    assert capture < arithmetic, (
        "ERRORLEVEL is read after set /a, which has already reset it")


def test_a_failure_is_counted_and_named() -> None:
    body = TEXT.split(":check", 1)[1]
    assert "set /a FAILS+=1" in body
    assert 'set "BROKEN=%BROKEN% %1"' in body


def test_a_successful_step_is_not_flagged() -> None:
    body = TEXT.split(":check", 1)[1]
    assert 'if "%CODE%"=="0" goto :eof' in body


def test_the_job_exits_non_zero_when_a_step_failed() -> None:
    """The exit code is what a scheduler or a caller can act on.

    Without it the only record is prose in a log nobody reads until
    something looks wrong days later.
    """
    assert "exit /b %FAILS%" in TEXT


def test_the_job_exits_zero_only_on_the_all_good_path() -> None:
    assert "exit /b 0" in TEXT
    good = TEXT.index(":allgood")
    assert TEXT.index("exit /b 0", good) > good, \
        "exit /b 0 is reachable before the success branch"


def test_a_failure_is_banner_marked_in_the_log() -> None:
    """So it can be found by eye in a log that is thousands of lines."""
    body = TEXT.split(":check", 1)[1]
    assert "STEP FAILED" in body
    assert "*****" in body


def _failure_branch() -> str:
    """The block between `goto :allgood` and the `:allgood` label.

    Split on the GOTO first: ":allgood" appears in the goto before the
    label, so splitting on the bare string takes the wrong side and the
    assertion passes or fails for the wrong reason.
    """
    after_goto = TEXT.split("goto :allgood", 1)[1]
    return after_goto.split("\n:allgood", 1)[0]


def test_the_summary_lists_which_steps_failed() -> None:
    assert "%BROKEN%" in _failure_branch()
    assert "%FAILS%" in _failure_branch()


def test_the_summary_reaches_the_console_not_only_the_log() -> None:
    """A run started by hand shows nothing without this."""
    console = [line.strip() for line in _failure_branch().splitlines()
               if line.strip().startswith("echo ")
               and not line.strip().startswith("echo.")]
    assert console, "nothing is echoed to the console on failure"


def test_each_run_is_separated_by_a_dated_banner() -> None:
    """Runs were concatenated with nothing between them, so a finished
    run's last line and a running one's look identical - which is how a
    stale line was read as current on 2026-09-20."""
    assert "NIGHTLY RUN STARTED" in TEXT
    assert "%DATE%" in TEXT and "%TIME%" in TEXT


def test_the_claimed_step_count_matches_the_steps() -> None:
    """A summary saying "all 9 steps" while running eight is a lie the
    log would carry every night."""
    claimed = re.findall(r"all (\d+) steps succeeded", TEXT)
    assert claimed, "the success line does not state a count"
    for number in claimed:
        assert int(number) == len(_commands()), (
            f"claims {number} steps, runs {len(_commands())}")


@pytest.mark.parametrize("step", [
    "fetch_tail", "publish_turnover", "prune_cache", "fetch_scan_log",
    "outcomes", "daily_context",
])
def test_the_steps_that_matter_are_still_in_the_job(step) -> None:
    """Guards against a step being dropped while rewiring the checks."""
    assert f"call :check {step}" in TEXT or f"{step}.py" in TEXT
