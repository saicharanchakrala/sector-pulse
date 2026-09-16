"""Tests for reading prior sessions from the consolidated store.

THE WHOLE POINT of these is that the fast path must never quietly hand
back a SHORTER history than the slow one. The relative-volume gate medians
across the sessions in the prewarm window, and this project has already
measured the consequence of getting that window wrong: the gate's verdict
differed on 7 of 40 symbols depending on which path supplied the bars.

So a stale store must be refused outright, and a symbol whose own coverage
stops early must be refused individually - because a scan in which some
names carry seventeen days of baseline and others carry ten, with nothing
saying which, is worse than a slow scan.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import live_bars

IST = "Asia/Kolkata"


def _frame(last_day: date, sessions: int = 12):
    """A 3-minute frame whose newest bar is on `last_day`."""
    blocks = []
    for back in range(sessions):
        day = last_day - timedelta(days=back)
        stamps = pd.date_range(f"{day} 09:15", periods=5, freq="3min", tz=IST)
        blocks.append(pd.DataFrame(
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5,
             "Volume": 1000},
            index=stamps))
    return pd.concat(blocks).sort_index()


@pytest.fixture()
def store(monkeypatch):
    """Stand in for bar_store.load with a dict of frames."""
    def build(frames, raises=None):
        import bar_store

        def fake(interval, symbols=None, start=None, end=None):
            if raises:
                raise raises
            if symbols is None:
                return dict(frames)
            return {s: f for s, f in frames.items() if s in symbols}

        monkeypatch.setattr(bar_store, "load", fake)
        return frames
    return build


# --- the staleness gate --------------------------------------------------

def test_a_store_behind_the_window_is_refused_outright(store) -> None:
    # THE DEFECT THIS GUARDS. A store reaching 11 September against a
    # window ending the 15th is not a smaller window, it is a different
    # measurement - four sessions missing from every baseline.
    end = date(2026, 9, 15)
    store({"AAA": _frame(date(2026, 9, 11))})
    got = live_bars.from_store(["AAA"], end - timedelta(days=17), end,
                               "3minute")
    assert got == {}, "a four-day-stale store must not be used"


def test_a_store_one_session_behind_is_accepted(store) -> None:
    # One session of slack, because a window ending yesterday is exactly
    # what history_window asks for and the fold runs after the close.
    end = date(2026, 9, 15)
    store({"AAA": _frame(date(2026, 9, 14))})
    got = live_bars.from_store(["AAA"], end - timedelta(days=17), end,
                               "3minute")
    assert set(got) == {"AAA"}


def test_a_current_store_is_accepted(store) -> None:
    end = date(2026, 9, 15)
    store({"AAA": _frame(end), "BBB": _frame(end)})
    got = live_bars.from_store(["AAA", "BBB"], end - timedelta(days=17), end,
                               "3minute")
    assert set(got) == {"AAA", "BBB"}


def test_the_slack_is_exactly_one_day() -> None:
    # Pinned, because widening it is the easy way to make a scan faster
    # and a baseline wrong.
    assert live_bars.STORE_MAX_STALE_DAYS == 1


# --- per-symbol coverage -------------------------------------------------

def test_a_symbol_that_stopped_early_is_refused_individually(store) -> None:
    # The store is current overall, but this name's own bars stop a week
    # back. Returning it would give one symbol a ten-day baseline while
    # its neighbours carry seventeen, in the same scan, silently.
    end = date(2026, 9, 15)
    store({"FRESH": _frame(end), "STALE": _frame(end - timedelta(days=7))})
    got = live_bars.from_store(["FRESH", "STALE"], end - timedelta(days=17),
                               end, "3minute")
    assert set(got) == {"FRESH"}, "a name whose coverage stops early is not covered"


def test_an_empty_frame_is_not_coverage(store) -> None:
    end = date(2026, 9, 15)
    store({"AAA": _frame(end), "EMPTY": pd.DataFrame()})
    got = live_bars.from_store(["AAA", "EMPTY"], end - timedelta(days=17),
                               end, "3minute")
    assert set(got) == {"AAA"}


def test_an_absent_store_falls_back_quietly(store) -> None:
    store({}, raises=FileNotFoundError("no store"))
    assert live_bars.from_store(["AAA"], date(2026, 8, 29), date(2026, 9, 15),
                                "3minute") == {}


def test_an_empty_store_is_not_an_error(store) -> None:
    store({})
    assert live_bars.from_store(["AAA"], date(2026, 8, 29), date(2026, 9, 15),
                                "3minute") == {}


# --- what prewarm does with it -------------------------------------------

def test_prewarm_fetches_only_what_the_store_does_not_cover(
        store, monkeypatch) -> None:
    import market_source

    end = date(2026, 9, 15)
    monkeypatch.setattr(live_bars, "history_window",
                        lambda days=None: (end - timedelta(days=17), end))
    store({"COVERED": _frame(end)})

    asked = []

    def fake_bars(symbols, start, stop, interval="5minute", **kw):
        asked.append(list(symbols))
        return {s: _frame(end) for s in symbols}

    monkeypatch.setattr(market_source, "bars", fake_bars)
    got = live_bars.prewarm(["COVERED", "MISSING"], interval="3minute")

    assert set(got) == {"COVERED", "MISSING"}
    assert asked == [["MISSING"]], (
        "the covered symbol must not be refetched, and the missing one must be")


def test_prewarm_skips_the_fetch_entirely_when_the_store_covers_all(
        store, monkeypatch) -> None:
    import market_source

    end = date(2026, 9, 15)
    monkeypatch.setattr(live_bars, "history_window",
                        lambda days=None: (end - timedelta(days=17), end))
    store({"AAA": _frame(end), "BBB": _frame(end)})

    def boom(*a, **k):
        raise AssertionError("market_source.bars must not be called")

    monkeypatch.setattr(market_source, "bars", boom)
    got = live_bars.prewarm(["AAA", "BBB"], interval="3minute")
    assert set(got) == {"AAA", "BBB"}


def test_a_stale_store_means_prewarm_fetches_everything(
        store, monkeypatch) -> None:
    # The fast path degrades to the old behaviour rather than to a wrong
    # one. Slow and correct is the required failure mode.
    import market_source

    end = date(2026, 9, 15)
    monkeypatch.setattr(live_bars, "history_window",
                        lambda days=None: (end - timedelta(days=17), end))
    store({"AAA": _frame(date(2026, 9, 1)), "BBB": _frame(date(2026, 9, 1))})

    asked = []

    def fake_bars(symbols, start, stop, interval="5minute", **kw):
        asked.append(sorted(symbols))
        return {s: _frame(end) for s in symbols}

    monkeypatch.setattr(market_source, "bars", fake_bars)
    got = live_bars.prewarm(["AAA", "BBB"], interval="3minute")
    assert asked == [["AAA", "BBB"]]
    assert set(got) == {"AAA", "BBB"}


def test_a_no_session_error_still_returns_what_the_store_had(
        store, monkeypatch) -> None:
    # Losing the Kite session must not throw away the bars already in
    # hand: a partial scan beats no scan.
    import market_source

    end = date(2026, 9, 15)
    monkeypatch.setattr(live_bars, "history_window",
                        lambda days=None: (end - timedelta(days=17), end))
    store({"COVERED": _frame(end)})

    def no_session(*a, **k):
        raise market_source.NoSession("no token")

    monkeypatch.setattr(market_source, "bars", no_session)
    got = live_bars.prewarm(["COVERED", "MISSING"], interval="3minute")
    assert set(got) == {"COVERED"}
