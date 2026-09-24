"""The feed must not scan, publish or log outside the scan window.

WHAT WAS WRONG. scan_publish.loop scanned on a timer with no clock check,
and setups.measure falls back to the last session it holds - so a scan
before the open measured YESTERDAY's bars and stamped them with TODAY's
run_date. Measured 2026-09-24: 2,350 of 9,800 resolved outcomes carry a
run_time before 09:30 (06:37, 08:08, 09:21), and all 2,485 rows of
scan_log_20260924.csv are stamped 00:00-06:48. remember() kept each as the
first sighting, adoption carried it across restarts, and outcomes.py then
scored yesterday's levels against today's bars.

WHAT IS PINNED HERE.

  THE WINDOW. Open on a weekday from the opening range's close up to but
  not including the session close. Boundaries are built from the same
  config constants the code reads, so a change to the opening range moves
  the test with it rather than breaking it.

  THE LOOP. Outside the window an iteration touches nothing - not the
  snapshot, not the scan, not the publish, not the log - and inside it
  only symbols with a bar stamped today are scanned, which is what covers
  a holiday the clock cannot know about and a feed restarted before its
  seed has arrived. The idle state is logged once per change, not once
  per tick.

  THE SECOND WALL. remember() and log_row() refuse a row whose own stamp
  is outside the window, whoever calls them, and a published log adopted
  on a restart is held to the same rule.

Time is frozen by passing `now` explicitly, which is why scan_window and
_iteration take it as an argument.
"""
from __future__ import annotations

import csv
import io
import logging
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import config
import object_store
import scan_publish

IST = ZoneInfo("Asia/Kolkata")
MONDAY = datetime(2026, 9, 21, tzinfo=IST)          # a trading weekday
FRIDAY = MONDAY - timedelta(days=3)                 # the session before it
SATURDAY = MONDAY + timedelta(days=5)


def _at(day: datetime, hour: int, minute: int, second: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute, second=second)


def _first() -> datetime:
    """The first scannable instant on MONDAY, from the constants used."""
    open_hour, open_minute = config.SCAN_SESSION_OPEN
    return _at(MONDAY, open_hour, open_minute) + timedelta(
        minutes=config.SCAN_OPENING_RANGE_MINUTES)


def _close() -> datetime:
    close_hour, close_minute = config.SCAN_SESSION_CLOSE
    return _at(MONDAY, close_hour, close_minute)


def _published(symbol="AAA", run_date="2026-09-21", run_time="10:30:00",
               actionable=False, entry=100.0):
    return {
        "run_date": run_date, "run_time": run_time, "symbol": symbol,
        "direction": "LONG", "passed": actionable,
        "actionable": actionable, "score": 0.5, "rvol": 1.5,
        "relative_strength": 1.0, "oi_change_pct": None,
        "futures_share": None, "vwap": 99.5, "turnover_20d": 5e8,
        "reasons": "", "entry": entry, "stop": 98.0, "target": 104.0,
        "quantity": 10, "stop_pct": 2.0, "target_pct": 4.0,
        "breakeven_pct": 0.12, "cost_rupees": 12.0,
        "required_win_rate": 0.34,
    }


def _snapshot(day: datetime, token: int = 1, bars: int = 3) -> pd.DataFrame:
    """A BarBuilder-shaped snapshot: completed 3-minute bars from 09:15."""
    stamps = pd.date_range(f"{day:%Y-%m-%d} 09:15", periods=bars,
                           freq="3min", tz=IST)
    return pd.DataFrame({
        "instrument_token": token, "Date": stamps, "Open": 100.0,
        "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000.0,
        "ticks": 5, "partial": False,
    })


class _Builder:
    """Hands out a fixed snapshot and counts how often it was asked."""

    def __init__(self, frame: "pd.DataFrame | None" = None) -> None:
        self.frame = pd.DataFrame() if frame is None else frame
        self.calls = 0

    def snapshot(self) -> pd.DataFrame:
        self.calls += 1
        return self.frame


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """An empty session log and an object store that holds nothing.

    The store is stubbed because remember() reads the published log back
    on its first call of a day, and a developer machine with the bucket
    variable set would otherwise send that read to S3.
    """
    monkeypatch.setattr(scan_publish, "_SESSION_LOG", {})
    monkeypatch.setattr(scan_publish, "_SESSION_LOG_DAY", None)
    monkeypatch.setattr(object_store, "enabled", lambda: False)
    monkeypatch.setattr(object_store, "get", lambda name: None)


@pytest.fixture
def _recorded(monkeypatch):
    """Every side effect of a scan, recorded instead of performed."""
    seen = {"run_once": [], "scanned": [], "publish": 0, "remember": 0,
            "publish_log": 0}

    def run_once(symbols, history, today, daily, seed=None, now=None):
        seen["run_once"].append(now)
        seen["scanned"].append(list(symbols))
        return pd.DataFrame(), len(symbols)

    def publish(table, when=None):
        seen["publish"] += 1
        return 0

    def remember(table, when=None):
        seen["remember"] += 1
        return 0

    def publish_log(when=None):
        seen["publish_log"] += 1
        return 0

    monkeypatch.setattr(scan_publish, "run_once", run_once)
    monkeypatch.setattr(scan_publish, "publish", publish)
    monkeypatch.setattr(scan_publish, "remember", remember)
    monkeypatch.setattr(scan_publish, "publish_log", publish_log)
    return seen


# --- the window ----------------------------------------------------------

def test_the_window_is_the_opening_range_close_to_the_session_close() -> None:
    """The README's 09:30, derived rather than restated."""
    first, close = scan_publish.window_bounds()
    assert (first.hour, first.minute) == (_first().hour, _first().minute)
    assert (close.hour, close.minute) == (_close().hour, _close().minute)


@pytest.mark.parametrize("when", [
    _at(MONDAY, 6, 37),                       # the observed pre-open scans
    _at(MONDAY, 8, 8),
    _at(MONDAY, 9, 21),                       # inside the opening range
    _first() - timedelta(minutes=1),          # 09:29
    _first() - timedelta(seconds=1),          # 09:29:59
    _close(),                                 # 15:30 exactly
    _close() + timedelta(minutes=30),
    _at(SATURDAY, 10, 0),
    _at(SATURDAY + timedelta(days=1), 11, 0),
])
def test_closed(when) -> None:
    is_open, reason = scan_publish.scan_window(when)
    assert is_open is False, when
    assert reason


@pytest.mark.parametrize("when", [
    _first(),                                 # 09:30 exactly
    _at(MONDAY, 12, 0),
    _close() - timedelta(minutes=1),          # 15:29
    _close() - timedelta(seconds=1),          # 15:29:59
])
def test_open(when) -> None:
    assert scan_publish.scan_window(when)[0] is True, when


def test_each_closed_state_names_itself() -> None:
    """loop() logs a change of state, so the reasons must differ by state
    and must NOT differ by the minute - or it would log every tick."""
    early = scan_publish.scan_window(_at(MONDAY, 6, 37))[1]
    also_early = scan_publish.scan_window(_at(MONDAY, 8, 8))[1]
    late = scan_publish.scan_window(_close())[1]
    weekend = scan_publish.scan_window(_at(SATURDAY, 10, 0))[1]
    assert early == also_early
    assert len({early, late, weekend}) == 3


def test_an_instant_in_another_zone_is_converted_not_misread() -> None:
    """04:00 UTC is 09:30 IST. Reading the wall-clock digits would call
    it pre-open."""
    utc = _first().astimezone(timezone.utc)
    assert utc.hour == 4
    assert scan_publish.scan_window(utc)[0] is True


def test_a_naive_clock_is_closed_not_assumed() -> None:
    """Read as IST, a naive UTC clock would open the window from 15:00 to
    21:00 IST and scan a finished session for six hours."""
    assert scan_publish.scan_window(datetime(2026, 9, 21, 10, 0))[0] is False


def test_the_window_follows_the_configured_opening_range(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_OPENING_RANGE_MINUTES", 30)
    assert scan_publish.scan_window(_at(MONDAY, 9, 44))[0] is False
    assert scan_publish.scan_window(_at(MONDAY, 9, 45))[0] is True


@pytest.mark.parametrize("row, inside", [
    ({"run_date": "2026-09-21", "run_time": "10:00:00"}, True),
    ({"run_date": "2026-09-21", "run_time": "10:00"}, True),
    ({"run_date": "2026-09-21", "run_time": "06:37:00"}, False),
    ({"run_date": "2026-09-26", "run_time": "10:00:00"}, False),   # Saturday
    ({"run_date": "2026-09-21", "run_time": ""}, False),
    ({"run_date": "", "run_time": "10:00:00"}, False),
    ({"run_date": float("nan"), "run_time": "10:00:00"}, False),
    ({}, False),
])
def test_a_row_is_judged_on_its_own_stamp_and_fails_closed(row,
                                                           inside) -> None:
    assert scan_publish.in_scan_window(row) is inside


# --- which symbols have today's session ----------------------------------

def _frames(*pairs) -> dict:
    """Per-symbol frames through live_bars.frames_from, as the loop gets
    them: pairs of (symbol, day)."""
    import live_bars

    tokens = {symbol: token for token, (symbol, _) in enumerate(pairs, 1)}
    snapshot = pd.concat([_snapshot(day, token=tokens[symbol])
                          for symbol, day in pairs], ignore_index=True)
    return live_bars.frames_from(snapshot, tokens)


def test_only_yesterdays_bars_are_not_todays_session() -> None:
    """A holiday, or a feed process that outlived its session: the clock
    says open and the bars are the last session's."""
    today = _frames(("AAA", FRIDAY))
    assert today, "fixture must produce a frame"
    assert scan_publish.session_symbols(["AAA"], today, None,
                                        _at(MONDAY, 10, 0)) == []


def test_bars_stamped_today_are_todays_session() -> None:
    today = _frames(("AAA", MONDAY))
    assert scan_publish.session_symbols(["AAA"], today, None,
                                        _at(MONDAY, 10, 0)) == ["AAA"]


def test_the_check_is_per_symbol_not_per_session() -> None:
    """THE REGRESSION TEST for the first draft of this guard, which asked
    whether ANY frame had a bar today and then scanned every symbol - so a
    name with only Friday's bars was measured on Friday and stamped
    Monday, because another name had traded."""
    today = _frames(("AAA", MONDAY), ("BBB", FRIDAY))
    assert scan_publish.session_symbols(["AAA", "BBB"], today, None,
                                        _at(MONDAY, 10, 0)) == ["AAA"]


def test_the_seed_counts_for_a_feed_that_has_not_closed_a_bar() -> None:
    """Restarted at 11:00, the stream closes its first bar at 11:03 while
    the seed already holds 09:15 onwards."""
    seed = _frames(("AAA", MONDAY))
    assert scan_publish.session_symbols(["AAA"], {}, seed,
                                        _at(MONDAY, 11, 0)) == ["AAA"]


def test_a_seed_holding_the_prior_session_does_not_count() -> None:
    seed = _frames(("AAA", FRIDAY))
    assert scan_publish.session_symbols(["AAA"], {}, seed,
                                        _at(MONDAY, 11, 0)) == []


def test_nothing_at_all_is_no_session() -> None:
    assert scan_publish.session_symbols(["AAA"], {}, None,
                                        _at(MONDAY, 10, 0)) == []
    assert scan_publish.session_symbols(["AAA"], {}, {},
                                        _at(MONDAY, 10, 0)) == []


# --- one iteration of the loop -------------------------------------------

def test_a_closed_window_touches_nothing(_recorded) -> None:
    """THE REGRESSION TEST. At 06:37 the old loop scanned, published the
    table and logged every setup - on the previous session's bars."""
    builder = _Builder(_snapshot(FRIDAY))
    reason = scan_publish._iteration(builder, ["AAA"], {}, {"AAA": 1}, {},
                                     None, _at(MONDAY, 6, 37))
    assert reason
    assert builder.calls == 0, "not even the snapshot should be taken"
    assert _recorded == {"run_once": [], "scanned": [], "publish": 0,
                         "remember": 0, "publish_log": 0}


def test_an_open_window_without_todays_bars_touches_nothing(_recorded) -> None:
    builder = _Builder(_snapshot(FRIDAY))
    reason = scan_publish._iteration(builder, ["AAA"], {}, {"AAA": 1}, {},
                                     None, _at(MONDAY, 10, 0))
    assert reason == scan_publish.NO_SESSION_BARS
    assert _recorded["run_once"] == []
    assert _recorded["publish"] == 0
    assert _recorded["remember"] == 0
    assert _recorded["publish_log"] == 0


def test_an_open_window_with_todays_bars_scans_at_that_instant(
        _recorded) -> None:
    now = _at(MONDAY, 10, 0)
    reason = scan_publish._iteration(_Builder(_snapshot(MONDAY)), ["AAA"],
                                     {}, {"AAA": 1}, {}, None, now)
    assert reason is None
    assert _recorded["run_once"] == [now]
    assert _recorded["publish"] == 1
    assert _recorded["remember"] == 1
    assert _recorded["publish_log"] == 1


def test_the_seed_alone_is_enough_to_scan(_recorded) -> None:
    import live_bars

    seed = live_bars.frames_from(_snapshot(MONDAY), {"AAA": 1})
    reason = scan_publish._iteration(_Builder(), ["AAA"], {}, {"AAA": 1},
                                     {}, seed, _at(MONDAY, 11, 0))
    assert reason is None
    assert len(_recorded["run_once"]) == 1


def test_only_symbols_with_a_bar_today_are_scanned(_recorded) -> None:
    """BBB holds only Friday's bars, so it must not reach run_once to be
    measured on Friday and stamped Monday."""
    snapshot = pd.concat([_snapshot(MONDAY, token=1),
                          _snapshot(FRIDAY, token=2)], ignore_index=True)
    scan_publish._iteration(_Builder(snapshot), ["AAA", "BBB"], {},
                            {"AAA": 1, "BBB": 2}, {}, None,
                            _at(MONDAY, 10, 0))
    assert _recorded["scanned"] == [["AAA"]]


def test_the_benchmark_alone_is_not_a_session(monkeypatch,
                                              _recorded) -> None:
    """An index ticks when nothing else has yet; there is nothing to scan
    until a tradable symbol has a bar."""
    monkeypatch.setattr(config, "SCAN_BENCHMARK", "BENCH")
    reason = scan_publish._iteration(
        _Builder(_snapshot(MONDAY, token=9)), ["BENCH", "AAA"], {},
        {"BENCH": 9, "AAA": 1}, {}, None, _at(MONDAY, 10, 0))
    assert reason == scan_publish.NO_SESSION_BARS
    assert _recorded["run_once"] == []


# --- the loop ------------------------------------------------------------

def _frozen(moment: datetime):
    """A datetime stand-in whose now() is `moment`, for loop()'s clock.

    The module-level subclass pattern test_scan_data_store_first uses for
    date.today(), applied to the one clock read loop() makes.
    """
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment if tz is None else moment.astimezone(tz)

    return _Clock


def _stopping_after(ticks: int) -> threading.Event:
    """A stop event whose wait() returns True after `ticks` iterations."""
    stop = threading.Event()
    count = {"n": 0}

    def wait(timeout=None):
        count["n"] += 1
        return count["n"] > ticks
    stop.wait = wait
    return stop


def _idle_lines(caplog) -> list:
    return [r.getMessage() for r in caplog.records
            if "scan loop idle" in r.getMessage()]


def test_the_loop_does_not_scan_while_closed(monkeypatch, caplog,
                                             _recorded) -> None:
    monkeypatch.setattr(scan_publish, "datetime",
                        _frozen(_at(MONDAY, 6, 37)))
    builder = _Builder(_snapshot(FRIDAY))
    with caplog.at_level(logging.INFO, logger="scan_publish"):
        scan_publish.loop(builder, ["AAA"], {}, {"AAA": 1}, {},
                          _stopping_after(5), every=0)
    assert builder.calls == 0
    assert _recorded["run_once"] == []
    assert _recorded["publish"] == 0 and _recorded["publish_log"] == 0
    assert len(_idle_lines(caplog)) == 1, "idle is logged once, not per tick"


def test_the_loop_logs_each_change_of_state_once(monkeypatch, caplog,
                                                 _recorded) -> None:
    """Closed, closed, open, open, closed: two idle lines and one resume,
    and the scan runs exactly on the two open ticks."""
    states = iter([(False, "early"), (False, "early"), (True, "open"),
                   (True, "open"), (False, "late")])
    monkeypatch.setattr(scan_publish, "scan_window",
                        lambda now: next(states))
    # Frozen as well, so the bars below are "today" for session_symbols
    # whatever day the suite runs on.
    monkeypatch.setattr(scan_publish, "datetime",
                        _frozen(_at(MONDAY, 10, 0)))
    with caplog.at_level(logging.INFO, logger="scan_publish"):
        scan_publish.loop(_Builder(_snapshot(MONDAY)), ["AAA"], {},
                          {"AAA": 1}, {}, _stopping_after(5), every=0)
    messages = [r.getMessage() for r in caplog.records]
    assert len(_idle_lines(caplog)) == 2
    assert sum("resuming" in m for m in messages) == 1
    assert len(_recorded["run_once"]) == 2


# --- the second wall: remember() and log_row() ---------------------------

def test_remember_refuses_a_pre_open_row() -> None:
    held = scan_publish.remember(
        pd.DataFrame([_published(run_time="06:37:00")]),
        when=_at(MONDAY, 6, 37))
    assert held == 0
    assert scan_publish._SESSION_LOG == {}


def test_a_pre_open_sighting_cannot_block_the_valid_one() -> None:
    """First sighting wins, which is exactly what made the stale rows
    stick: a 06:37 row held the key all day."""
    scan_publish.remember(pd.DataFrame([_published(run_time="06:37:00",
                                                   entry=90.0)]),
                          when=_at(MONDAY, 6, 37))
    scan_publish.remember(pd.DataFrame([_published(run_time="10:05:00",
                                                   entry=100.0)]),
                          when=_at(MONDAY, 10, 5))
    row = next(iter(scan_publish._SESSION_LOG.values()))
    assert row["entry"] == 100.0
    assert row["run_time"] == "10:05:00"


def test_remember_keeps_the_in_window_rows_of_a_mixed_table() -> None:
    held = scan_publish.remember(pd.DataFrame([
        _published(symbol="STALE", run_time="08:08:00"),
        _published(symbol="GOOD", run_time="10:30:00"),
    ]), when=_at(MONDAY, 10, 30))
    assert held == 1
    assert next(iter(scan_publish._SESSION_LOG))[0] == "GOOD"


def test_remember_judges_the_row_stamp_not_the_logging_clock() -> None:
    """Scanned at 15:20 and logged at 15:31 is a valid row."""
    held = scan_publish.remember(
        pd.DataFrame([_published(run_time="15:20:00")]),
        when=_close() + timedelta(minutes=1))
    assert held == 1


def test_log_row_refuses_a_row_outside_the_window() -> None:
    with pytest.raises(ValueError, match="outside the scan window"):
        scan_publish.log_row(_published(run_time="06:37:00"))
    with pytest.raises(ValueError):
        scan_publish.log_row(_published(run_date="", run_time=""))


def test_log_row_still_builds_an_in_window_row() -> None:
    assert scan_publish.log_row(_published())["symbol"] == "AAA"


def test_an_adopted_stale_row_is_not_republished(monkeypatch) -> None:
    """A log published before this guard existed - scan_log_20260924.csv
    is entirely pre-open - must not be adopted on a restart and written
    straight back out for outcomes.py to resolve again."""
    import scan_intraday

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=scan_intraday._CSV_FIELDS,
                            extrasaction="ignore")
    writer.writeheader()
    for symbol, stamp in (("STALE", "06:37:00"), ("MORNING", "10:05:00")):
        row = dict(_published(symbol=symbol, run_time=stamp),
                   replayed=False, taken=False, blocked_by="",
                   risk_rupees=20.0)
        writer.writerow(row)
    held = {scan_publish.log_object_name(MONDAY):
            buffer.getvalue().encode("utf-8")}
    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(object_store, "get", lambda name: held.get(name))
    monkeypatch.setattr(object_store, "put",
                        lambda name, payload: held.__setitem__(name, payload))

    scan_publish.remember(pd.DataFrame([_published(symbol="AFTER",
                                                   run_time="11:00:00")]),
                          when=_at(MONDAY, 11, 0))
    scan_publish.publish_log(_at(MONDAY, 11, 0))

    rows = list(csv.DictReader(io.StringIO(
        held[scan_publish.log_object_name(MONDAY)].decode("utf-8"))))
    assert sorted(r["symbol"] for r in rows) == ["AFTER", "MORNING"]


# --- the CLI's log guard ---------------------------------------------------

def _setup(symbol: str):
    """The one attribute loggable_now reads off a setup."""
    from types import SimpleNamespace

    return SimpleNamespace(readings=SimpleNamespace(symbol=symbol))


def test_the_cli_logs_nothing_before_the_window_opens() -> None:
    """06:37: the frames end on the previous session, which is exactly the
    bug. Nothing is logged, and the reason says the table is stale."""
    import scan_intraday

    fresh, why = scan_intraday.loggable_now(
        [_setup("AAA")], _frames(("AAA", FRIDAY)), _at(MONDAY, 6, 37))
    assert fresh == []
    assert "last session held" in why


def test_the_cli_logs_nothing_measured_on_an_earlier_session() -> None:
    """A holiday, or a replay of one: the clock window is open, but one
    name's frame ends on the previous session. Only the current one is
    logged, and the count held back is stated."""
    import scan_intraday

    current, stale = _setup("AAA"), _setup("BBB")
    fresh, why = scan_intraday.loggable_now(
        [current, stale], _frames(("AAA", MONDAY), ("BBB", FRIDAY)),
        _at(MONDAY, 10, 0))
    assert fresh == [current]
    assert "1 of 2" in why


def test_the_cli_logs_everything_current_and_says_nothing() -> None:
    import scan_intraday

    setups = [_setup("AAA"), _setup("BBB")]
    fresh, why = scan_intraday.loggable_now(
        setups, _frames(("AAA", MONDAY), ("BBB", MONDAY)),
        _at(MONDAY, 10, 0))
    assert fresh == setups
    assert why is None
