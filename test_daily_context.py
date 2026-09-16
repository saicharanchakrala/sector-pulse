"""Tests for the daily slice the containerised scan reads.

THE PROPERTY THIS FILE EXISTS FOR is that the slice is a drop-in for the
559 MB store. measure() reads four things from a daily frame - previous
close, previous high and low for the pivot range, and a twenty-session
mean volume - and if the slice disagrees with the store about any of them,
a scan running in the container silently differs from one running on the
laptop, on prev_close, day change, CPR, turnover and every gate downstream
of those.

Verified live on 2026-09-16 across 216 F&O underlyings: 216 identical, 0
differing. These tests keep it that way.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import daily_context
import setups

IST = ZoneInfo("Asia/Kolkata")


def _daily(sessions: int = 40, last_day: date = date(2026, 9, 15)):
    """A daily frame with a distinct value per session."""
    stamps = pd.bdate_range(end=pd.Timestamp(last_day), periods=sessions,
                            tz=IST)
    base = [100.0 + i for i in range(sessions)]
    return pd.DataFrame({
        "Open": base,
        "High": [v + 2 for v in base],
        "Low": [v - 2 for v in base],
        "Close": [v + 1 for v in base],
        "Volume": [1_000_000 + i * 1000 for i in range(sessions)],
    }, index=stamps)


# --- the slice keeps enough --------------------------------------------

def test_the_tail_covers_the_twenty_sessions_measure_needs() -> None:
    # measure() takes tail(20) of bars strictly BEFORE the scan's date, so
    # twenty-one is the arithmetic minimum. Anything less and the mean
    # volume - and therefore the turnover gate - is computed over a
    # shorter window than the store would have used.
    assert daily_context.TAIL_SESSIONS >= 21


def test_build_keeps_the_most_recent_sessions_not_the_oldest() -> None:
    table = daily_context.build({"AAA": _daily(sessions=60)})
    kept = table.sort_values("stamp")
    assert len(kept) == daily_context.TAIL_SESSIONS
    # The newest session must survive; taking head() instead of tail()
    # would publish a month-old context that still looked well formed.
    assert kept["Close"].iloc[-1] == pytest.approx(100.0 + 59 + 1)


def test_a_symbol_missing_columns_is_skipped_not_half_published() -> None:
    frames = {"GOOD": _daily(), "BAD": _daily()[["Close"]]}
    table = daily_context.build(frames)
    assert set(table["symbol"]) == {"GOOD"}


def test_an_empty_store_publishes_nothing_rather_than_an_empty_file(
        monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_context, "LOCAL", tmp_path / "ctx.parquet")
    assert daily_context.publish({}) == 0
    assert not (tmp_path / "ctx.parquet").exists()


# --- the round trip ------------------------------------------------------

@pytest.fixture()
def published(monkeypatch, tmp_path):
    def build(frames):
        monkeypatch.setattr(daily_context, "LOCAL", tmp_path / "ctx.parquet")
        daily_context.publish(frames)
        return tmp_path / "ctx.parquet"
    return build


def test_the_round_trip_preserves_the_index_and_columns(published) -> None:
    published({"AAA": _daily()})
    back = daily_context.load()
    assert set(back) == {"AAA"}
    frame = back["AAA"]
    assert list(frame.columns) == daily_context.COLUMNS
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is not None, "a naive index would shift the cutoff"


def test_symbols_filter_before_the_frames_are_built(published) -> None:
    # The whole point: a caller scanning 1 name must not pay to build 3.
    published({"AAA": _daily(), "BBB": _daily(), "CCC": _daily()})
    assert set(daily_context.load(["BBB"])) == {"BBB"}
    assert set(daily_context.load()) == {"AAA", "BBB", "CCC"}


def test_an_unknown_symbol_yields_nothing_rather_than_everything(
        published) -> None:
    published({"AAA": _daily()})
    assert daily_context.load(["NOPE"]) == {}


def test_an_absent_file_is_empty_not_an_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_context, "LOCAL", tmp_path / "missing.parquet")
    assert daily_context.load() == {}
    assert daily_context.age() is None


# --- THE ONE THAT MATTERS ------------------------------------------------

def _intraday(day: date = date(2026, 9, 16), bars: int = 40):
    stamps = pd.date_range(f"{day} 09:15", periods=bars, freq="3min", tz=IST)
    base = [200.0 + (i % 7) for i in range(bars)]
    return pd.DataFrame({
        "Open": base, "High": [v + 1 for v in base],
        "Low": [v - 1 for v in base], "Close": base,
        "Volume": [10_000] * bars,
    }, index=stamps)


def test_measure_reads_the_same_numbers_from_the_slice_and_the_store(
        published) -> None:
    """The drop-in property, asserted rather than assumed.

    A full store with sixty sessions against the published tail of
    twenty-five: measure() must not notice the difference, because it
    only ever looks at the last twenty before the scan's own date.
    """
    full = _daily(sessions=60)
    published({"AAA": full})
    slice_frame = daily_context.load(["AAA"])["AAA"]
    assert len(slice_frame) < len(full), "the fixture must actually be a slice"

    intraday = _intraday()
    now = datetime(2026, 9, 16, 13, 0, tzinfo=IST)

    from_store = setups.measure("AAA", "AAA", intraday, full, 0.5, now)
    from_slice = setups.measure("AAA", "AAA", intraday, slice_frame, 0.5, now)
    assert from_store is not None and from_slice is not None

    for field in ("prev_close", "day_change_pct", "turnover_20d", "last",
                  "rvol", "relative_strength", "vwap"):
        a, b = getattr(from_store, field), getattr(from_slice, field)
        if a is None and b is None:
            continue
        assert a == pytest.approx(b), f"{field}: store {a} vs slice {b}"

    assert (from_store.cpr is None) == (from_slice.cpr is None)
    if from_store.cpr is not None:
        for part in ("pivot", "top", "bottom"):
            assert getattr(from_store.cpr, part) == pytest.approx(
                getattr(from_slice.cpr, part)), f"cpr.{part} differs"


def test_a_slice_too_short_would_change_the_turnover_and_is_detectable(
        published, monkeypatch) -> None:
    # The failure mode TAIL_SESSIONS exists to prevent: publish only ten
    # sessions and the twenty-session mean volume silently becomes a
    # ten-session one. This asserts the difference is real, which is why
    # the constant is guarded by its own test above.
    monkeypatch.setattr(daily_context, "TAIL_SESSIONS", 10)
    full = _daily(sessions=60)
    published({"AAA": full})
    short = daily_context.load(["AAA"])["AAA"]

    intraday = _intraday()
    now = datetime(2026, 9, 16, 13, 0, tzinfo=IST)
    from_store = setups.measure("AAA", "AAA", intraday, full, 0.5, now)
    from_short = setups.measure("AAA", "AAA", intraday, short, 0.5, now)
    assert from_store.turnover_20d != pytest.approx(from_short.turnover_20d), (
        "a ten-session slice must not silently match a twenty-session mean")


def test_age_reports_how_far_behind_the_context_is(published) -> None:
    # Silence only means "current" if something says so. A container
    # reading a week-old context should be able to tell.
    published({"AAA": _daily(last_day=date.today() - timedelta(days=5))})
    newest, behind = daily_context.age()
    assert behind >= 4
    assert newest == (date.today() - timedelta(days=5)) or newest < date.today()
