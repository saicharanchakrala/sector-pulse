"""Calibration counts sessions, not just rows, and drops stale scan times.

WHAT WAS WRONG.

  THE ROW FLOOR ALONE. MIN_SAMPLE counted rows, and rows inside one
  session share that session's market move. Measured 2026-09-24:
  outcomes.csv held 9,800 resolved rows from three sessions, 1,054 of
  them taken, and both cleared the floor of 30 and printed a full
  expectancy block - a three-day track record printed as a thousand
  trades.

  NO INTERVAL. The mean net R was printed bare, so 1,054 correlated rows
  looked as precise as 1,054 independent ones.

  STALE SCAN TIMES. 2,350 of those rows carry a run_time before 09:30:
  the feed scanned on a timer before the open, measured the PREVIOUS
  session's bars, and stamped them with the current date.

WHAT IS PINNED HERE. Both floors, and both counts in the refusal; the
session count beside every n; an interval that resamples whole sessions
through forecast_stats.block_bootstrap, deterministic, and said to be
missing rather than faked when it cannot be estimated; and rows outside
the scan window excluded and counted, replays judged by the same window
on their own run_time.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

import calibrate
import forecast_stats
import scan_publish

STOP_PCT = 1.5
BREAKEVEN = 0.1224

# Weekdays only, so every one of these is a valid scan date.
DAYS = [day for day in (date(2026, 7, 1) + timedelta(days=offset)
                        for offset in range(80))
        if day.weekday() < 5][:calibrate.MIN_SESSIONS + 5]


def _row(run_date: date, run_time: str = "10:30:00", **over) -> dict:
    """One resolved 2:1 setup, scanned at `run_date` `run_time`."""
    row = {
        "row_id": f"{run_date}-{run_time}-{len(over)}-{id(over)}",
        "run_date": run_date.isoformat(), "run_time": run_time,
        "replayed": "", "taken": "True", "score": "0.65",
        "entry": "1000.0", "stop": "990.0", "target": "1020.0",
        "stop_pct": str(STOP_PCT), "breakeven_pct": str(BREAKEVEN),
        "outcome": "STOP", "r_multiple": "-1.0",
    }
    row.update({key: str(value) for key, value in over.items()})
    return row


def _spread(count: int, days: list, **over) -> list:
    """`count` rows dealt round-robin across `days`."""
    return [_row(days[i % len(days)], **over) for i in range(count)]


def _mixed(days: list, per_day: int = 10) -> list:
    """Per session, a mix of winners and losers, so the mean is not flat."""
    rows = []
    for index, day in enumerate(days):
        for slot in range(per_day):
            win = (slot + index) % 3 == 0
            rows.append(_row(day, outcome="TARGET" if win else "STOP",
                             r_multiple="2.0" if win else "-1.0"))
    return rows


# --- the session floor ---------------------------------------------------

def test_the_row_floor_passes_but_the_session_floor_refuses() -> None:
    """THE REGRESSION TEST. 100 rows is over MIN_SAMPLE; 3 sessions is the
    shape of the real file, and it printed a full expectancy block."""
    rows = _spread(100, DAYS[:3])
    assert len(rows) >= calibrate.MIN_SAMPLE
    out = calibrate.report(rows)
    assert "NOT ENOUGH TO CALIBRATE" in out
    assert "PER TRADE" not in out
    assert "100 usable live outcomes from 3 sessions" in out, out
    assert (f"floors of {calibrate.MIN_SAMPLE} outcomes and "
            f"{calibrate.MIN_SESSIONS} sessions") in out


def test_a_session_refusal_does_not_ask_for_more_rows() -> None:
    """Beside 100 usable rows, "a few hundred is where a band starts to
    mean something" would read as the refusal contradicting itself. The
    shortfall is days, and the refusal says so."""
    out = calibrate.report(_spread(100, DAYS[:3]))
    assert "A few hundred" not in out
    assert "more rows per day do not substitute for more days" in out


def test_one_session_short_of_the_floor_still_refuses() -> None:
    rows = _spread(200, DAYS[:calibrate.MIN_SESSIONS - 1])
    assert "NOT ENOUGH TO CALIBRATE" in calibrate.report(rows)


def test_exactly_at_the_session_floor_reports() -> None:
    rows = _spread(200, DAYS[:calibrate.MIN_SESSIONS])
    out = calibrate.report(rows)
    assert "NOT ENOUGH TO CALIBRATE" not in out
    assert "PER TRADE" in out


def test_many_sessions_do_not_rescue_too_few_rows() -> None:
    """The row floor stays: twenty sessions of one setup each is its own
    kind of too small."""
    rows = _spread(calibrate.MIN_SAMPLE - 1, DAYS)
    assert calibrate.sessions(rows) >= calibrate.MIN_SESSIONS
    assert "NOT ENOUGH TO CALIBRATE" in calibrate.report(rows)


def test_the_taken_rows_need_the_session_floor_too() -> None:
    """Enough blocked rows across 25 sessions, and 60 taken rows from 3:
    the headline prints, the taken block must not."""
    rows = _spread(250, DAYS, taken="False") + \
        _spread(60, DAYS[:3], taken="True")
    out = calibrate.report(rows)
    assert "EXPECTANCY over 310 live setups" in out
    assert "TAKEN live setups" not in out
    assert "NO EXPECTANCY FOR THE TAKEN ROWS: 60 of them resolved over " \
           "3 sessions" in out, out


def test_sessions_counts_distinct_run_dates() -> None:
    rows = [*_spread(30, DAYS[:4]), dict(_row(DAYS[0]), run_date="")]
    assert calibrate.sessions(rows) == 4


# --- the session count beside every n ------------------------------------

def test_the_session_count_sits_beside_n() -> None:
    rows = _mixed(DAYS)
    out = calibrate.report(rows)
    assert f"EXPECTANCY over {len(rows)} live setups from {len(DAYS)} " \
           f"sessions" in out, out
    assert f"EXPECTANCY over {len(rows)} TAKEN live setups from " \
           f"{len(DAYS)} sessions" in out
    assert f"usable live rows from {len(DAYS)} sessions" in out


# --- the interval --------------------------------------------------------

def test_enough_sessions_prints_the_interval() -> None:
    out = calibrate.report(_mixed(DAYS))
    assert "95% interval, sessions" in out
    assert "not estimable" not in out


def test_the_interval_is_the_session_block_bootstrap() -> None:
    """Exactly forecast_stats' number, grouped by run_date, fixed seed."""
    rows = _mixed(DAYS)
    exp = calibrate.expectancy(rows)
    values = [calibrate.net_r(row) for row in rows]
    groups = [row["run_date"] for row in rows]
    _, low, high, _ = forecast_stats.block_bootstrap(
        values, groups, seed=forecast_stats.DEFAULT_SEED)
    assert exp["ci_low"] == pytest.approx(low)
    assert exp["ci_high"] == pytest.approx(high)
    assert exp["sessions"] == len(DAYS)


def test_the_interval_resamples_sessions_not_rows() -> None:
    """THE MUTANT THIS CATCHES: grouping by row instead of run_date. Here
    each session is all winners or all losers, the correlation the floor
    exists for, and the honest interval is much wider than a row-level
    one."""
    rows = []
    for index, day in enumerate(DAYS):
        win = index % 2 == 0
        rows.extend(_row(day, outcome="TARGET" if win else "STOP",
                         r_multiple="2.0" if win else "-1.0")
                    for _ in range(10))
    exp = calibrate.expectancy(rows)
    values = [calibrate.net_r(row) for row in rows]
    _, row_low, row_high, _ = forecast_stats.block_bootstrap(
        values, list(range(len(values))), seed=forecast_stats.DEFAULT_SEED)
    assert (exp["ci_high"] - exp["ci_low"]) > 2 * (row_high - row_low)


def test_the_interval_is_deterministic() -> None:
    rows = _mixed(DAYS)
    first = calibrate.expectancy(rows)
    second = calibrate.expectancy(list(rows))
    assert (first["ci_low"], first["ci_high"]) == \
        (second["ci_low"], second["ci_high"])


def test_too_few_sessions_says_so_rather_than_faking_an_interval() -> None:
    """block_bootstrap refuses below five groups. The report must say the
    interval is missing, not print the mean as if it had one."""
    exp = calibrate.expectancy(_mixed(DAYS[:3]))
    assert exp["ci_low"] is None and exp["ci_high"] is None
    text = "\n".join(calibrate._expectancy_lines(exp, "live setups"))
    assert "not estimable" in text
    assert "95% interval, sessions" not in text


def test_an_interval_spanning_zero_is_named() -> None:
    out = calibrate.report(_mixed(DAYS))
    exp = calibrate.expectancy(_mixed(DAYS))
    spans = exp["ci_low"] <= 0.0 <= exp["ci_high"]
    assert ("The interval spans zero" in out) is spans


# --- the scan window -----------------------------------------------------

def test_rows_outside_the_window_are_excluded_and_counted() -> None:
    """THE REGRESSION TEST. Pre-open rows measured the previous session;
    at or after the close there is nothing left to trade. Winners, so a
    leak would show in the mean as well as the count."""
    first, close = scan_publish.window_bounds()
    good = _mixed(DAYS)
    stale = ([_row(DAYS[0], "06:37:00", outcome="TARGET", r_multiple="2.0")
              for _ in range(4)]
             + [_row(DAYS[1], "08:08:00", outcome="TARGET",
                     r_multiple="2.0") for _ in range(2)]
             + [_row(DAYS[2], f"{first:%H:%M}", outcome="TARGET",
                     r_multiple="2.0")]                   # inside: kept
             + [_row(DAYS[3], f"{close:%H:%M}:00", outcome="TARGET",
                     r_multiple="2.0")])
    out = calibrate.report(good + stale)
    assert ", 7 outside the scan window." in out, out
    assert "or with no readable scan time: 7, of which 0 replayed" in out
    assert f"EXPECTANCY over {len(good) + 1} live setups" in out


@pytest.mark.parametrize("run_time, inside", [
    ("06:37:00", False), ("09:29:59", False), ("09:30:00", True),
    ("15:29:59", True), ("15:30:00", False), ("", False),
])
def test_the_predicate_uses_the_scan_window(run_time, inside) -> None:
    """Default config: opening range closes 09:30, session closes 15:30."""
    assert calibrate.in_scan_window(_row(DAYS[0], run_time)) is inside


def test_a_weekend_scan_is_outside_the_window() -> None:
    saturday = date(2026, 9, 26)
    assert saturday.weekday() == 5
    assert not calibrate.in_scan_window(_row(saturday))


def test_a_replay_at_ten_is_kept() -> None:
    """A replay carries a past instant legitimately; 10:00 is inside."""
    assert calibrate.in_scan_window(_row(DAYS[0], "10:00:00",
                                         replayed="True"))


def test_a_replay_at_six_thirty_seven_is_excluded() -> None:
    """Judged by the same window on its own run_time: nothing of that
    day's session existed at 06:37 for a replay to measure either."""
    assert not calibrate.in_scan_window(_row(DAYS[0], "06:37:00",
                                             replayed="True"))


def test_replays_are_split_by_the_window_in_the_report() -> None:
    rows = [*_mixed(DAYS), _row(DAYS[0], "10:00:00", replayed="True"), _row(DAYS[1], "06:37:00", replayed="True")]
    out = calibrate.report(rows)
    assert "1 replayed, 1 outside the scan window." in out, out
    assert "no readable scan time: 1, of which 1 replayed" in out


# --- the score-band verdict ------------------------------------------------

def _band(score: str, hits: int, days: list) -> list:
    """One row per day in `days` at `score`, the first `hits` of them wins."""
    return [_row(day, score=score,
                 outcome="TARGET" if slot < hits else "STOP",
                 r_multiple="2.0" if slot < hits else "-1.0")
            for slot, day in enumerate(days)]


def test_a_hit_rate_that_falls_with_the_score_is_not_a_pass() -> None:
    """THE BUG. The verdict accepted a falling hit rate as "yes", beside a
    sentence saying a score whose hit rate does not rise orders nothing."""
    days = DAYS[:calibrate.MIN_SESSIONS]
    rows = (_band("0.3", 12, days) + _band("0.5", 8, days)
            + _band("0.7", 4, days))
    out = calibrate.report(rows)
    assert "Rises with score: NO, it falls" in out, out


def test_a_hit_rate_that_rises_with_the_score_passes() -> None:
    days = DAYS[:calibrate.MIN_SESSIONS]
    rows = (_band("0.3", 4, days) + _band("0.5", 8, days)
            + _band("0.7", 12, days))
    assert "Rises with score: yes" in calibrate.report(rows)


def test_a_band_short_of_the_session_floor_is_left_out_of_the_verdict(
        ) -> None:
    """Twelve rows from five days is five observations of that band. With
    it left out only two bands remain, too few to call an ordering."""
    days = DAYS[:calibrate.MIN_SESSIONS]
    thin = [_row(DAYS[slot % 5], score="0.7", outcome="TARGET",
                 r_multiple="2.0") for slot in range(12)]
    rows = _band("0.3", 4, days) + _band("0.5", 8, days) + thin
    out = calibrate.report(rows)
    assert "EXPECTANCY" in out, "the population itself must clear the floors"
    assert "Rises with score" not in out, out
