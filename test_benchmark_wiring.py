"""The benchmark has to reach the containerised scan, as a number.

WHAT BROKE. The relative-strength gate compares a symbol's day change
against the index's. In the container it had neither half, and neither
failure was loud:

  1. A NAME, not a subscription. The index was always being streamed - the
     F&O master calls it "NIFTY", it is in fo_underlyings, and
     token_for("NIFTY") is 256265, the same token as "NIFTY 50".
     live_bars.frames_from inverts {symbol: token} into {token: symbol},
     so the bars arrived keyed "NIFTY" while the scan and daily_context
     both key the index "NIFTY 50". The lookup missed.
  2. scan_publish.run_once() then passed `combined.get("NIFTY 50")` - a
     DATA FRAME - into setups.measure, which takes a FLOAT. Even with the
     right key that argument could not have produced a number.

Either one alone makes relative_strength None for every symbol, which
fails the gate for every symbol, which means score_setup never runs and
nothing is ever actionable. Measured on the live feed 2026-09-17: 2,498 of
2,498 published rows had a null relative_strength, a score of exactly 0.0
and actionable False - and the UI rendered that as "Cleared every gate:
0", which is what a quiet market looks like too.

WHY THE NAMING TESTS MATTER MOST. The first fix attempt simply added
"NIFTY 50" alongside "NIFTY". That put two names on one token, and
frames_from's inverted map keeps whichever was inserted last - so it
worked only because `sorted()` puts "NIFTY" before "NIFTY 50". Spell the
benchmark "^NSEI" and it silently breaks again. The tests below pin the
token map and the canonical name, not just membership in a list.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import config
import live_feed
import market_source
import scan_publish

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 16, 13, 0, tzinfo=IST)
BENCH = config.SCAN_BENCHMARK


def _intraday(day, bars=60, open_=100.0, close=None):
    """A session's 3-minute bars, walking from open_ to close."""
    close = open_ if close is None else close
    stamps = pd.date_range(f"{day} 09:15", periods=bars, freq="3min", tz=IST)
    # numpy, not a Series: a Series carries its own RangeIndex and pandas
    # aligns it against `stamps`, which silently produces an all-NaN frame.
    steps = np.arange(bars, dtype=float) / max(1, bars - 1)
    prices = open_ + (close - open_) * steps
    return pd.DataFrame({"Open": prices, "High": prices + 0.5,
                         "Low": prices - 0.5, "Close": prices,
                         "Volume": 5_000.0}, index=stamps)


def _daily(last_close=100.0, sessions=25, through="2026-09-15"):
    stamps = pd.date_range(end=through, periods=sessions, freq="D", tz=IST)
    return pd.DataFrame({"Open": last_close, "High": last_close + 2,
                         "Low": last_close - 2, "Close": last_close,
                         "Volume": 1_000_000.0}, index=stamps)


# --- the streamed universe ----------------------------------------------

class _Snapshot:
    def __init__(self, symbols):
        self.fo_underlyings = [type("I", (), {"symbol": s})() for s in symbols]


@pytest.fixture
def _ranked(monkeypatch):
    """universe() with a stub snapshot and a stub turnover ranking."""
    def arrange(fo, liquid, cap=3000):
        monkeypatch.setattr(live_feed.instruments, "load_latest",
                            lambda: _Snapshot(fo))
        monkeypatch.setattr(live_feed, "by_turnover", lambda n: list(liquid))
        monkeypatch.setattr(live_feed.kt, "MAX_INSTRUMENTS_PER_CONNECTION",
                            cap)
    return arrange


def test_the_scan_asks_for_the_canonical_name(monkeypatch) -> None:
    """The invariant the whole chain rests on.

    universe() streams canonical(SCAN_BENCHMARK); run_once looks up
    SCAN_BENCHMARK; daily_context is keyed canonically. If SCAN_BENCHMARK
    is ever set to an alias - "NIFTY", "^NSEI" - those are different
    strings and every lookup misses, with no error anywhere.
    """
    assert market_source.canonical(BENCH) == BENCH, (
        f"SCAN_BENCHMARK is {BENCH!r} but canonicalises to "
        f"{market_source.canonical(BENCH)!r}, so the bars and the daily "
        f"context will be keyed under a name the scan never asks for")


def test_the_benchmark_is_streamed_under_the_name_the_scan_uses(
        _ranked) -> None:
    _ranked(fo=["RELIANCE", "TCS"], liquid=["RELIANCE", "INFY", "WIPRO"])
    got = live_feed.universe("")
    assert BENCH in got, "without its bars every relative-strength gate fails"


def test_an_alias_in_the_fo_set_is_collapsed_not_added_alongside(
        _ranked) -> None:
    """The regression the first fix attempt left behind.

    "NIFTY" and "NIFTY 50" are one instrument on one token. Streaming both
    means frames_from keeps whichever was inserted last, which made the
    fix depend on string ordering.
    """
    _ranked(fo=["NIFTY", "RELIANCE"], liquid=["RELIANCE", "INFY"])
    got = live_feed.universe("")
    assert BENCH in got
    assert "NIFTY" not in got, "two names on token 256265"


def test_an_explicit_symbol_list_still_gets_the_benchmark() -> None:
    got = live_feed.universe("RELIANCE,TCS")
    assert got[:2] == ["RELIANCE", "TCS"]
    assert BENCH in got


@pytest.mark.parametrize("spelling", ["NIFTY 50", "NIFTY"])
def test_naming_the_benchmark_explicitly_does_not_double_it(spelling) -> None:
    got = live_feed.universe(f"RELIANCE,{spelling}")
    canonical = [market_source.canonical(s) for s in got]
    assert canonical.count(BENCH) == 1, got


def test_no_two_streamed_names_share_one_instrument_token(_ranked) -> None:
    """The failure this whole class of bug comes from, asserted directly.

    frames_from builds {token: symbol}, so two names on one token silently
    lose one of them. Nothing downstream reports it.
    """
    _ranked(fo=["NIFTY", "RELIANCE", "TCS"], liquid=["RELIANCE", "INFY"])
    got = live_feed.universe("")
    tokens = {s: market_source.token_for(s) for s in got}
    tokens = {s: t for s, t in tokens.items() if t}
    assert len(set(tokens.values())) == len(tokens), (
        "two symbols map to one token: "
        f"{sorted(tokens.items(), key=lambda kv: kv[1])}")


def test_the_streamed_bars_come_back_under_the_key_the_scan_asks_for(
        _ranked) -> None:
    """The seam itself: universe -> tokens -> frames_from -> run_once.

    This is where the bug actually lived, and it is the only test here
    that would have caught it. Membership in the symbol list was never
    the problem.
    """
    import live_bars

    _ranked(fo=["NIFTY", "RELIANCE"], liquid=["RELIANCE"])
    symbols = live_feed.universe("")
    tokens = {s: market_source.token_for(s) for s in symbols}
    tokens = {s: t for s, t in tokens.items() if t}

    stamps = pd.date_range("2026-09-16 09:15", periods=3, freq="3min", tz=IST)
    rows = []
    for token in tokens.values():
        for stamp in stamps:
            rows.append({"instrument_token": token, "Date": stamp,
                         "Open": 100.0, "High": 101.0, "Low": 99.0,
                         "Close": 100.0, "Volume": 1000.0, "partial": False})
    frames = live_bars.frames_from(pd.DataFrame(rows), tokens)
    assert BENCH in frames, (
        f"bars arrived under {sorted(frames)} but run_once asks for {BENCH!r}")


def test_the_cap_trims_the_ranking_tail_and_never_the_benchmark(
        _ranked) -> None:
    # A cap smaller than the ranking, so the trim branch runs.
    _ranked(fo=["RELIANCE"], liquid=[f"SYM{i}" for i in range(50)], cap=10)
    got = live_feed.universe("")
    assert len(got) <= 10
    assert BENCH in got, "the benchmark is core, not ranking tail"
    assert "RELIANCE" in got


def test_a_missing_turnover_ranking_still_leaves_the_benchmark(
        _ranked) -> None:
    _ranked(fo=["RELIANCE", "TCS"], liquid=[])
    got = live_feed.universe("")
    assert BENCH in got


# --- the benchmark's day change -----------------------------------------

def test_the_day_change_is_measured_against_the_previous_close() -> None:
    intraday = _intraday("2026-09-16", open_=100.0, close=102.0)
    daily = _daily(last_close=100.0, through="2026-09-15")
    got = scan_publish.benchmark_change_pct(intraday, daily, NOW)
    assert got == pytest.approx(2.0, rel=1e-6)


def test_todays_own_daily_bar_cannot_become_the_previous_close() -> None:
    """The lookahead guard, at the benchmark rather than at the symbol.

    A daily frame carrying today's partial bar would make the change read
    against a close the session has not produced yet. Anchored on the
    clock, so a frame that runs through today is cut at today.
    """
    intraday = _intraday("2026-09-16", open_=100.0, close=102.0)
    through_today = _daily(last_close=100.0, through="2026-09-16")
    # Today's bar carries a different close, so using it would show.
    through_today.iloc[-1, through_today.columns.get_loc("Close")] = 102.0
    got = scan_publish.benchmark_change_pct(through_today, through_today, NOW)
    assert got != pytest.approx(0.0), "read against today's own close"

    got = scan_publish.benchmark_change_pct(intraday, through_today, NOW)
    assert got == pytest.approx(2.0, rel=1e-6)


def test_a_benchmark_that_has_not_ticked_today_is_none_not_zero() -> None:
    """The dangerous reading, because 0.0 survives every None check.

    session_bars() defaults to the frame's LAST session while the daily
    cutoff is anchored on the clock. A benchmark holding only prior
    sessions therefore compared its last intraday close against that same
    day's daily close - the same number - and returned 0.0. Zero is not
    missing: it silences run_once's warning and makes every symbol's
    relative strength equal its own day change, so in a rising market
    every rising name reads as leading the index.

    Reachable on every feed start before the index's first 3-minute bar
    closes, and all session if the index token stops ticking.
    """
    stale = _intraday("2026-09-15", open_=20_000.0, close=20_100.0)
    daily = _daily(last_close=20_000.0, through="2026-09-15")
    got = scan_publish.benchmark_change_pct(stale, daily, NOW)
    assert got is None, f"stale benchmark read as {got}, not missing"


def test_a_stale_benchmark_does_not_become_every_symbols_own_day_change(
        caplog) -> None:
    """The consequence of the above, at the published artifact."""
    today = {
        "AAA": _intraday("2026-09-16", open_=100.0, close=103.0),
        BENCH: _intraday("2026-09-15", open_=20_000.0, close=20_100.0),
    }
    daily = {"AAA": _daily(last_close=100.0),
             BENCH: _daily(last_close=20_000.0)}
    with caplog.at_level("WARNING"):
        table, _ = scan_publish.run_once(["AAA", BENCH], {}, today, daily,
                                         now=NOW)
    assert any("relative-strength" in r.getMessage() for r in caplog.records)
    assert not table.empty
    strength = table.set_index("symbol")["relative_strength"]
    assert strength.isna().all(), (
        "a stale index must block the gate, not hand each symbol its own "
        f"day change: {strength.to_dict()}")


@pytest.mark.parametrize("intraday,daily", [
    (None, "daily"), ("intraday", None), (None, None),
])
def test_a_missing_side_fails_closed_rather_than_comparing_to_zero(
        intraday, daily) -> None:
    frames = {"intraday": _intraday("2026-09-16"), "daily": _daily()}
    assert scan_publish.benchmark_change_pct(
        frames.get(intraday), frames.get(daily), NOW) is None


def test_a_malformed_benchmark_frame_does_not_kill_the_whole_scan() -> None:
    """It is called OUTSIDE run_once's per-symbol handler.

    A frame without a Close column raised straight out of run_once into
    loop's broad handler, which would have logged one bare "scan failed"
    every 30 seconds for the rest of the session.
    """
    no_close = _intraday("2026-09-16").drop(columns=["Close"])
    assert scan_publish.benchmark_change_pct(no_close, _daily(), NOW) is None
    assert scan_publish.benchmark_change_pct(
        _intraday("2026-09-16"), _daily().drop(columns=["Close"]),
        NOW) is None
    assert scan_publish.benchmark_change_pct("not a frame", _daily(),
                                             NOW) is None

    # And the scan still runs, rather than the exception escaping.
    today = {"AAA": _intraday("2026-09-16", open_=100.0, close=103.0),
             BENCH: no_close}
    daily = {"AAA": _daily(last_close=100.0), BENCH: _daily()}
    table, _ = scan_publish.run_once(["AAA", BENCH], {}, today, daily, now=NOW)
    assert not table.empty


def test_a_naive_clock_is_read_as_exchange_local() -> None:
    """astimezone() on a naive datetime reads it as SYSTEM local time.

    On any machine not set to IST that shifts the cutoff day, which is the
    difference between a previous close and today's partial one.
    """
    intraday = _intraday("2026-09-16", open_=100.0, close=102.0)
    daily = _daily(last_close=100.0, through="2026-09-15")
    naive = NOW.replace(tzinfo=None)
    assert scan_publish.benchmark_change_pct(intraday, daily, naive) == \
        pytest.approx(scan_publish.benchmark_change_pct(intraday, daily, NOW))


def test_an_untimestamped_daily_index_is_refused() -> None:
    daily = _daily().reset_index(drop=True)
    assert scan_publish.benchmark_change_pct(
        _intraday("2026-09-16"), daily, NOW) is None


def test_a_daily_frame_entirely_from_today_leaves_no_previous_close() -> None:
    only_today = _daily(sessions=1, through="2026-09-16")
    assert scan_publish.benchmark_change_pct(
        _intraday("2026-09-16"), only_today, NOW) is None


# --- the wire, end to end ------------------------------------------------

def test_the_published_scan_carries_a_relative_strength() -> None:
    """A run_once unit test, and the one that pins the float-vs-frame half.

    It does NOT exercise the naming half - `symbols` and the benchmark
    frame are both handed in here, so the keys line up by construction.
    test_the_streamed_bars_come_back_under_the_key_the_scan_asks_for is
    the one that covers that.

    With the frame passed through instead of a float, measure's arithmetic
    dies in indicators._finite and every symbol is dropped, so the table
    comes back empty.
    """
    symbols = ["AAA", "BBB", BENCH]
    today = {
        "AAA": _intraday("2026-09-16", open_=100.0, close=104.0),
        "BBB": _intraday("2026-09-16", open_=100.0, close=98.0),
        BENCH: _intraday("2026-09-16", open_=20_000.0, close=20_200.0),
    }
    daily = {
        "AAA": _daily(last_close=100.0),
        "BBB": _daily(last_close=100.0),
        BENCH: _daily(last_close=20_000.0),
    }
    table, assembled = scan_publish.run_once(symbols, {}, today, daily,
                                             now=NOW)
    assert assembled == 3, "the benchmark is assembled, then skipped"
    assert not table.empty
    assert set(table["symbol"]) == {"AAA", "BBB"}, "the index is not scanned"

    strengths = table.set_index("symbol")["relative_strength"]
    assert strengths.notna().all(), "the gate cannot judge a None"
    # AAA rose 4% against the index's 1%, BBB fell 2% against it.
    assert strengths["AAA"] == pytest.approx(3.0, abs=0.05)
    assert strengths["BBB"] == pytest.approx(-3.0, abs=0.05)


def test_a_missing_benchmark_says_so_rather_than_publishing_silent_nulls(
        caplog) -> None:
    """Failing closed is right; failing closed quietly is what cost a session."""
    today = {"AAA": _intraday("2026-09-16", open_=100.0, close=104.0)}
    daily = {"AAA": _daily(last_close=100.0)}
    with caplog.at_level("WARNING"):
        table, _ = scan_publish.run_once(["AAA"], {}, today, daily, now=NOW)
    assert any("relative-strength" in r.getMessage()
               for r in caplog.records), caplog.text
    # Asserted unconditionally: guarding this on `not table.empty` lets it
    # pass vacuously in exactly the case worth checking.
    assert not table.empty
    assert table["relative_strength"].isna().all()
