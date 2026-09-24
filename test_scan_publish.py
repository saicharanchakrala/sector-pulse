"""Tests for the scan that runs where the bars are.

Two defects are guarded first, because both were found by running it
rather than reading it, and both would have shipped a table that looked
right:

  * flattening through scan_intraday._row silently DROPPED every
    non-actionable setup - it reads trade.entry, which is None when no
    levels were built. A pass over 216 F&O underlyings published 182 rows
    and discarded 34, taking with them the blocked names and the reasons
    they were blocked. That audit trail is the scanner's whole subtractive
    value.

  * numeric blanks written as "" made those columns object dtype, and
    parquet refused the table outright: "Could not convert '' with type
    str: tried to convert to double".

The third property pinned here is that assemble() splices with the same
precedence live_bars.combined uses, because a table built from a different
splice would disagree with the UI about the same symbol.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import scan_publish

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 16, 13, 0, tzinfo=IST)


def _bars(day, start_hour=9, start_min=15, bars=5, close=100.0):
    stamps = pd.date_range(f"{day} {start_hour:02d}:{start_min:02d}",
                           periods=bars, freq="3min", tz=IST)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": 1000}, index=stamps)


class _Levels:
    entry = 100.0
    stop = 98.0
    target = 104.0
    quantity = 10
    stop_pct = 2.0
    target_pct = 4.0
    breakeven_pct = 0.12
    cost_rupees = 12.0
    required_win_rate = 0.34


class _Readings:
    def __init__(self, symbol):
        self.symbol = symbol
        self.rvol = 1.5
        self.relative_strength = 0.8
        self.oi_change_pct = None
        self.futures_share = None
        self.vwap = 100.2
        self.turnover_20d = 8e8


class _Setup:
    def __init__(self, symbol, direction="LONG", levels=None, passed=False,
                 reasons=None):
        self.readings = _Readings(symbol)
        self.direction = direction
        self.levels = levels
        self.passed = passed
        self.rank_score = 0.5
        self.reasons = reasons or []

    @property
    def actionable(self):
        return self.passed and self.levels is not None


# --- the dropped rows ----------------------------------------------------

def test_a_setup_without_levels_still_produces_a_row() -> None:
    # scan_intraday._row raises here - it reads trade.entry directly - so
    # a published table built on it would be missing this symbol entirely.
    blocked = _Setup("BLOCKED", direction="LONG", levels=None,
                     reasons=["turnover below floor [FAIL]"])
    row = scan_publish.row_for(blocked, NOW)
    assert row["symbol"] == "BLOCKED"
    assert row["direction"] == "LONG"
    assert row["actionable"] is False
    assert row["entry"] is None, "no levels means no entry, not a fabricated one"
    assert "turnover below floor" in row["reasons"]


def test_the_reasons_survive_because_they_are_the_point() -> None:
    setup = _Setup("X", reasons=["gate a [PASS]", "gate b [FAIL]"])
    row = scan_publish.row_for(setup, NOW)
    assert row["reasons"] == "gate a [PASS] | gate b [FAIL]"


def test_an_actionable_setup_carries_its_levels() -> None:
    setup = _Setup("Y", levels=_Levels(), passed=True)
    row = scan_publish.row_for(setup, NOW)
    assert row["actionable"] is True
    assert row["entry"] == pytest.approx(100.0)
    assert row["required_win_rate"] == pytest.approx(0.34)


# --- the dtype that broke parquet ---------------------------------------

def test_missing_numbers_are_none_so_the_table_serialises() -> None:
    # "" in a numeric column makes it object dtype and parquet refuses the
    # whole table. None becomes NaN, which is what missing means.
    row = scan_publish.row_for(_Setup("X", levels=None), NOW)
    for field in ("entry", "stop", "target", "stop_pct", "breakeven_pct"):
        assert row[field] is None, f"{field} must be None, not a blank string"


def test_a_mixed_table_round_trips_through_parquet() -> None:
    import io

    rows = [scan_publish.row_for(_Setup("A", levels=_Levels(), passed=True), NOW),
            scan_publish.row_for(_Setup("B", levels=None), NOW)]
    table = pd.DataFrame(rows)
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False)          # must not raise
    back = pd.read_parquet(io.BytesIO(buffer.getvalue()))
    assert len(back) == 2
    assert back["entry"].isna().sum() == 1


# --- the splice ----------------------------------------------------------

def test_assemble_lets_the_freshest_source_win_on_a_duplicate_stamp() -> None:
    # Same precedence as live_bars.combined: history, then seed, then the
    # stream. A different rule here would make the published table
    # disagree with the UI about the same bar.
    stamp = pd.date_range("2026-09-16 09:15", periods=1, freq="3min", tz=IST)
    history = {"X": pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                                  "Close": 1.0, "Volume": 1}, index=stamp)}
    today = {"X": pd.DataFrame({"Open": 9.0, "High": 9.0, "Low": 9.0,
                                "Close": 9.0, "Volume": 9}, index=stamp)}
    out = scan_publish.assemble(["X"], history, today)
    assert len(out["X"]) == 1
    assert out["X"]["Close"].iloc[0] == pytest.approx(9.0), "the stream wins"


def test_assemble_joins_prior_sessions_to_today() -> None:
    history = {"X": _bars(date(2026, 9, 15))}
    today = {"X": _bars(date(2026, 9, 16))}
    out = scan_publish.assemble(["X"], history, today)
    assert len(out["X"]) == 10
    assert out["X"].index.is_monotonic_increasing


def test_a_symbol_with_no_bars_anywhere_is_omitted() -> None:
    out = scan_publish.assemble(["X", "GHOST"], {"X": _bars(date(2026, 9, 16))},
                                {})
    assert set(out) == {"X"}


def test_an_empty_frame_is_not_treated_as_data() -> None:
    out = scan_publish.assemble(["X"], {"X": pd.DataFrame()},
                                {"X": _bars(date(2026, 9, 16))})
    assert len(out["X"]) == 5


# --- publication ---------------------------------------------------------

def test_the_object_is_named_by_session_so_yesterday_cannot_be_read_as_today(
) -> None:
    a = scan_publish.object_name(datetime(2026, 9, 16, 10, 0, tzinfo=IST))
    b = scan_publish.object_name(datetime(2026, 9, 17, 10, 0, tzinfo=IST))
    assert a != b
    assert "20260916" in a


def test_publishing_nothing_writes_nothing(monkeypatch) -> None:
    import object_store

    calls = []
    monkeypatch.setattr(object_store, "put",
                        lambda name, body: calls.append(name))
    assert scan_publish.publish(pd.DataFrame()) == 0
    assert scan_publish.publish(None) == 0
    assert calls == []


def test_the_scan_interval_leaves_headroom_over_a_measured_pass() -> None:
    # A pass over the F&O set measured 6.1s. An interval below it means
    # the loop never sleeps and the tick stream competes with it.
    assert scan_publish.DEFAULT_EVERY >= 15


# --- the loop must not take the feed down -------------------------------

def test_a_failing_scan_does_not_escape_the_loop(monkeypatch) -> None:
    # This thread runs beside the tick stream. Bars still being written is
    # worth more than a table, so a scan that throws must be swallowed.
    import threading

    scans = {"n": 0}

    def boom(*a, **k):
        scans["n"] += 1
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(scan_publish, "run_once", boom)
    # The loop now idles outside the scan window and without bars stamped
    # today, so without these two it never reached run_once at all and
    # this test passed without exercising the failure it is named for.
    # The window and the bar check have their own tests in
    # test_scan_window.py.
    monkeypatch.setattr(scan_publish, "scan_window",
                        lambda now: (True, "open"))
    monkeypatch.setattr(scan_publish, "session_symbols",
                        lambda symbols, today, seed, now: list(symbols))

    class _Builder:
        def snapshot(self):
            return pd.DataFrame()

    stop = threading.Event()
    calls = {"n": 0}
    real_wait = stop.wait

    def wait(timeout=None):
        calls["n"] += 1
        return calls["n"] > 2       # two iterations, then stop
    stop.wait = wait

    scan_publish.loop(_Builder(), ["X"], {}, {}, {}, stop, every=0)
    stop.wait = real_wait
    assert calls["n"] > 2, "the loop must have kept going after the failure"
    assert scans["n"] == 2, "both iterations must have reached the scan"
