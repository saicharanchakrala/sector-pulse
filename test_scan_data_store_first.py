"""The UI serves bars from the store, proves coverage, and never downloads.

WHAT HAPPENED. scan_data.fetch_bars answered a cache miss by fetching per
symbol from Kite, inside the Streamlit process. On 2026-09-17 the widest
scope did that for about 2,570 symbols during market hours: bar_cache went
from roughly 12,000 files to 31,537, a directory listing timed out at two
minutes, the tab stopped answering, and the machine ran out of CPU and
memory. Those requests also competed with the live feed for Kite's
three-a-second budget.

PROOF IS THE HARD PART, and a first version of this change got it wrong.
It promised in its docstring that a frame is returned only when it spans
the window asked for, implemented "not None and not empty", and shipped a
fixture whose frames ended two days before the window did - so the tests
certified the bug. setups.measure takes the LAST session present in a
frame and never compares it to the clock, so a frame ending yesterday
becomes today's opening range, rvol, ATR, high and low, combined with a
live minutes_left, ranked into the table, with nothing reporting it.

_from_store now delegates to live_bars.from_store, which refuses the store
when its freshest session is too far behind and drops any symbol not
reaching that session. The tests below pin the refusals, because those are
what the promise is made of.
"""
from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import config
import scan_data

IST = ZoneInfo("Asia/Kolkata")

# The window fetch_bars asks of the store for a live scan.
END = date(2026, 9, 16)


def _frame(through=END, days=5):
    """Sessions ending at `through`, so the store can prove coverage."""
    index = pd.date_range(end=pd.Timestamp(through, tz=IST), periods=days,
                          freq="D")
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                         "Close": 100.0, "Volume": 1000.0}, index=index)


@pytest.fixture
def _no_network(monkeypatch):
    """Any Kite call is a test failure.

    Records BEFORE raising: _fetch wraps the call in `except Exception`,
    so the exception alone would be swallowed and prove nothing.
    """
    calls = []

    def explode(symbols, start, end, interval="5minute", **kwargs):
        calls.append((tuple(symbols), interval))
        raise AssertionError(f"downloaded {len(symbols)} symbol(s)")

    import market_source
    monkeypatch.setattr(market_source, "bars", explode)
    return calls


@pytest.fixture
def _store(monkeypatch):
    """A stub consolidated store, recording the interval it was asked for."""
    asked = []

    def install(frames):
        import bar_store

        def load(interval="day", symbols=None, start=None, end=None):
            asked.append(interval)
            table = frames.get(interval, {})
            if symbols is not None:
                table = {s: f for s, f in table.items() if s in set(symbols)}
            return dict(table)

        monkeypatch.setattr(bar_store, "load", load)
        return asked
    return install


@pytest.fixture(autouse=True)
def _no_live_feed(monkeypatch):
    """The live feed wins when present; these tests are about the store."""
    monkeypatch.setattr(scan_data, "_live_intraday",
                        lambda symbols: ({}, 0, float("nan")))


# --- the download gate ---------------------------------------------------

def test_a_caller_that_may_not_download_does_not(_no_network) -> None:
    got = scan_data._fetch(["AAA", "BBB"], date(2026, 9, 1), END, "day",
                           may_download=False)
    assert got == {}
    assert _no_network == []


def test_fetch_defaults_to_allowing_downloads(monkeypatch) -> None:
    """The refusal is the UI's policy, not this function's.

    scan_intraday and instrument_report are command-line tools whose job
    is to answer for the symbols asked for.
    """
    seen = {}

    def fake(symbols, start, end, interval="5minute", **kwargs):
        seen["symbols"] = list(symbols)
        return {s: _frame() for s in symbols}

    import market_source
    monkeypatch.setattr(market_source, "bars", fake)
    got = scan_data._fetch(["AAA"], date(2026, 9, 1), END, "day")
    assert list(got) == ["AAA"]
    assert seen["symbols"] == ["AAA"]


def test_fetch_bars_takes_its_default_from_config(monkeypatch, _store,
                                                  _no_network) -> None:
    """The UI passes nothing, so the config default is what guards it."""
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    _store({})
    got = scan_data.fetch_bars(["AAA"])
    assert got.failed == ["AAA"]
    assert _no_network == []


def test_fetch_bars_honours_an_explicit_override(monkeypatch,
                                                 _store) -> None:
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    _store({})
    fetched = []

    def fake(symbols, start, end, interval="5minute", **kwargs):
        fetched.append(interval)
        return {s: _frame() for s in symbols}

    import market_source
    monkeypatch.setattr(market_source, "bars", fake)
    got = scan_data.fetch_bars(["AAA"], may_download=True)
    assert "AAA" in got.intraday
    assert fetched, "the override did not reach the fetch"


def test_a_replay_may_always_download(monkeypatch, _store) -> None:
    """There is no other source for a past instant.

    The feed holds today and the store ends at the last nightly fold, so
    refusing here would turn every replay into "no bars came back".
    """
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    _store({})
    fetched = []

    def fake(symbols, start, end, interval="5minute", **kwargs):
        fetched.append(interval)
        return {s: _frame(through=date(2026, 9, 10)) for s in symbols}

    import market_source
    monkeypatch.setattr(market_source, "bars", fake)
    got = scan_data.fetch_bars(["AAA"], target=date(2026, 9, 10))
    assert fetched, "a replay was refused its download"
    assert "AAA" in got.intraday


def test_refusing_says_how_many_rather_than_failing_silently(caplog) -> None:
    with caplog.at_level("WARNING"):
        scan_data._fetch(["AAA", "BBB", "CCC"], date(2026, 9, 1), END,
                         "day", may_download=False)
    assert any("3 symbol" in r.getMessage() for r in caplog.records), \
        caplog.text


def test_an_empty_request_does_not_warn(caplog) -> None:
    """Every symbol covered by the store is the GOOD case, not a warning."""
    with caplog.at_level("WARNING"):
        assert scan_data._fetch([], date(2026, 9, 1), END, "day",
                                may_download=False) == {}
    assert not [r for r in caplog.records if "uncovered" in r.getMessage()]


# --- the proof, which is the point --------------------------------------

def test_a_store_frame_that_stops_short_of_the_window_is_refused(
        _store) -> None:
    """THE REGRESSION TEST. A frame ending days before the window's end
    would otherwise become today's session, silently."""
    _store({"3minute": {"AAA": _frame(through=END - timedelta(days=4))}})
    got = scan_data._from_store(["AAA"], END - timedelta(days=10), END,
                                config.SCAN_BAR_INTERVAL)
    assert got == {}, "served a stale session as the current one"


def test_a_symbol_lagging_the_rest_of_the_store_is_dropped(
        _store) -> None:
    """A name that stopped trading is not covered merely by being present."""
    _store({"3minute": {"FRESH": _frame(through=END),
                        "STALLED": _frame(through=END - timedelta(days=3))}})
    got = scan_data._from_store(["FRESH", "STALLED"],
                                END - timedelta(days=10), END,
                                config.SCAN_BAR_INTERVAL)
    assert "FRESH" in got
    assert "STALLED" not in got


def test_a_current_store_frame_is_served(_store) -> None:
    _store({"3minute": {"AAA": _frame(through=END)}})
    got = scan_data._from_store(["AAA"], END - timedelta(days=10), END,
                                config.SCAN_BAR_INTERVAL)
    assert "AAA" in got


# --- the interval-spelling trap -----------------------------------------

def test_the_store_is_asked_in_kites_spelling_not_the_configs(
        _store) -> None:
    """The whole change is inert if this is wrong, and inert in silence."""
    asked = _store({"3minute": {"AAA": _frame(through=END)}})
    got = scan_data._from_store(["AAA"], END - timedelta(days=10), END,
                                config.SCAN_BAR_INTERVAL)
    assert asked == ["3minute"], asked
    assert "AAA" in got


def test_the_config_spelling_alone_would_have_found_nothing() -> None:
    """Pins WHY the resolution is needed, so it is not tidied away."""
    import market_source
    assert config.SCAN_BAR_INTERVAL == "3m"
    assert market_source.kite_interval(config.SCAN_BAR_INTERVAL) == "3minute"


def test_an_unreadable_store_is_empty_rather_than_an_exception(
        monkeypatch) -> None:
    import bar_store

    def boom(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(bar_store, "load", boom)
    assert scan_data._from_store(["AAA"], date(2026, 9, 10), END,
                                 "day") == {}


# --- the two together ----------------------------------------------------

def test_a_fully_covered_scan_touches_no_network(monkeypatch, _store,
                                                 _no_network) -> None:
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    _store({"3minute": {"AAA": _frame(), "BBB": _frame()},
            "day": {"AAA": _frame(through=END - timedelta(days=1)),
                    "BBB": _frame(through=END - timedelta(days=1))}})
    got = scan_data.fetch_bars(["AAA", "BBB"])
    assert sorted(got.intraday) == ["AAA", "BBB"]
    assert sorted(got.daily) == ["AAA", "BBB"]
    assert got.failed == []
    assert got.daily_failed == []
    assert _no_network == []


def test_a_covered_scan_is_not_labelled_a_download(monkeypatch, _store,
                                                   _no_network) -> None:
    """The UI warns that downloaded bars can be minutes old.

    Saying that when nothing was downloaded is a lie about the data's
    provenance, and it fired whenever a single symbol was uncovered.
    """
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    _store({"3minute": {"AAA": _frame()},
            "day": {"AAA": _frame(through=END - timedelta(days=1))}})
    got = scan_data.fetch_bars(["AAA", "UNCOVERED"])
    assert got.source == "store", got.source


def test_what_the_store_cannot_cover_is_reported_not_fetched(
        monkeypatch, _store, _no_network) -> None:
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    _store({"3minute": {"AAA": _frame()},
            "day": {"AAA": _frame(through=END - timedelta(days=1))}})
    got = scan_data.fetch_bars(["AAA", "ILLIQUID"])
    assert "AAA" in got.intraday
    assert got.failed == ["ILLIQUID"]
    assert got.requested == 2
    assert _no_network == []


def test_a_missing_daily_context_is_reported_separately(
        monkeypatch, _store, _no_network) -> None:
    """It is NOT in `failed` - the symbol scans, with every gate needing a
    previous close failing closed and nothing saying why."""
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    _store({"3minute": {"AAA": _frame()}, "day": {}})
    got = scan_data.fetch_bars(["AAA"])
    assert got.failed == []
    assert got.daily_failed == ["AAA"]


def test_the_live_feed_still_wins_over_the_store(monkeypatch, _store,
                                                 _no_network) -> None:
    """Ticks are seconds old; the store ends at the last nightly fold."""
    monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", False)
    live = {"AAA": _frame(days=3)}
    monkeypatch.setattr(scan_data, "_live_intraday",
                        lambda symbols: (live, 1, 12.0))
    asked = _store({"3minute": {"AAA": _frame()},
                    "day": {"AAA": _frame(through=END - timedelta(days=1))}})
    got = scan_data.fetch_bars(["AAA"])
    assert got.source == "live feed"
    assert got.intraday["AAA"] is live["AAA"]
    assert "3minute" not in asked, "read the store when ticks were available"
