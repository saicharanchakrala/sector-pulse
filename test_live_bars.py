"""Tests for building 5-minute bars out of live ticks.

The volume tests are the point of this file. Kite's tick carries CUMULATIVE
day volume, so the obvious implementation - summing it across a bar - would
multiply the true figure by the tick count. That error is invisible in the
output (bars look normal, just busier) and would push relative volume, one
of the scanner's nine gates, permanently through its floor.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import live_bars

IST = ZoneInfo("Asia/Kolkata")


def tick(token: int, price: float, cumulative: int, when: datetime):
    """A stand-in for kite_ticker.Tick carrying only what the builder reads."""
    return SimpleNamespace(instrument_token=token, last_price=price,
                           volume=cumulative, exchange_timestamp=when,
                           received_at=when)


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 9, 9, hour, minute, second, tzinfo=IST)


# --- bucketing ------------------------------------------------------------

def test_bars_are_stamped_at_the_start_of_their_bucket() -> None:
    # Kite's historical candles are stamped at the START, and every
    # point-in-time guard in this project relies on that. A bar labelled
    # 10:00 covers 10:00 to 10:05.
    assert live_bars.bucket_start(at(10, 0, 1)) == at(10, 0)
    assert live_bars.bucket_start(at(10, 4, 59)) == at(10, 0)
    assert live_bars.bucket_start(at(10, 5, 0)) == at(10, 5)
    assert live_bars.bucket_start(at(9, 17, 30)) == at(9, 15)


def test_bucketing_is_done_in_ist_whatever_the_tick_carries() -> None:
    utc = datetime(2026, 9, 9, 4, 32, 0, tzinfo=ZoneInfo("UTC"))  # 10:02 IST
    assert live_bars.bucket_start(utc) == at(10, 0)


# --- volume is a delta ----------------------------------------------------

def test_bar_volume_is_the_delta_not_the_sum_of_cumulative_readings() -> None:
    builder = live_bars.BarBuilder()
    # First bucket establishes the baseline; second is measurable.
    builder.add([tick(1, 100.0, 1_000, at(10, 0, 5)),
                 tick(1, 101.0, 1_400, at(10, 1)),
                 tick(1, 99.0, 1_900, at(10, 4, 55))])
    builder.add([tick(1, 100.5, 2_500, at(10, 5, 5)),
                 tick(1, 102.0, 3_100, at(10, 7)),
                 tick(1, 101.5, 3_400, at(10, 9, 30))])
    builder.add([tick(1, 101.0, 3_500, at(10, 10, 1))])
    frame = builder.snapshot().sort_values("Date")
    second = frame[frame["Date"] == at(10, 5)].iloc[0]
    # Cumulative went 1,900 -> 3,400 across the bar, so 1,500 traded.
    # Summing the readings would have given 2,500+3,100+3,400 = 9,000.
    assert second["Volume"] == pytest.approx(1_500.0)
    assert second["Volume"] != pytest.approx(9_000.0)


def test_the_first_bar_has_unknown_volume_rather_than_the_whole_day() -> None:
    # Joining mid-session, the cumulative figure includes everything traded
    # before we were listening. Reporting that as one bar's volume would
    # read as a colossal spike.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 5_000_000, at(11, 7, 30))])
    builder.add([tick(1, 100.0, 5_010_000, at(11, 10, 1))])
    frame = builder.snapshot().sort_values("Date")
    first = frame.iloc[0]
    assert pd.isna(first["Volume"])
    assert bool(first["partial"]) is True


def test_a_later_bar_is_complete_and_not_flagged_partial() -> None:
    builder = live_bars.BarBuilder()
    for minute, cumulative in ((7, 100), (10, 250), (15, 400), (20, 600)):
        builder.add([tick(1, 100.0, cumulative, at(11, minute))])
    frame = builder.snapshot().sort_values("Date")
    later = frame[frame["Date"] == at(11, 10)].iloc[0]
    assert bool(later["partial"]) is False
    assert later["Volume"] == pytest.approx(150.0)


def test_a_cumulative_figure_that_goes_backwards_cannot_make_volume_negative() -> None:
    # Should not happen, but a reconnect replaying an older tick would do
    # it, and a negative volume would corrupt every downstream average.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 5_000, at(10, 0)),
                 tick(1, 100.0, 5_000, at(10, 4))])
    builder.add([tick(1, 100.0, 4_000, at(10, 5)),
                 tick(1, 100.0, 4_100, at(10, 6))])
    builder.add([tick(1, 100.0, 6_000, at(10, 10))])
    volumes = builder.snapshot()["Volume"].dropna()
    assert (volumes >= 0).all()


# --- OHLC -----------------------------------------------------------------

def test_ohlc_comes_from_the_prices_seen_in_that_bucket() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 10, at(10, 0)),
                 tick(1, 105.0, 20, at(10, 1)),
                 tick(1, 95.0, 30, at(10, 2)),
                 tick(1, 102.0, 40, at(10, 4))])
    builder.add([tick(1, 103.0, 50, at(10, 5))])
    bar = builder.snapshot().iloc[0]
    assert bar["Open"] == pytest.approx(100.0)
    assert bar["High"] == pytest.approx(105.0)
    assert bar["Low"] == pytest.approx(95.0)
    assert bar["Close"] == pytest.approx(102.0)
    assert bar["ticks"] == 4


def test_a_zero_or_negative_price_is_ignored() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 0.0, 10, at(10, 0)), tick(1, 100.0, 20, at(10, 1))])
    builder.add([tick(1, 101.0, 30, at(10, 5))])
    bar = builder.snapshot().iloc[0]
    assert bar["Open"] == pytest.approx(100.0)
    assert bar["ticks"] == 1


def test_instruments_are_kept_apart() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 10, at(10, 0)), tick(2, 500.0, 70, at(10, 0))])
    builder.add([tick(1, 110.0, 20, at(10, 5)), tick(2, 550.0, 90, at(10, 5))])
    builder.add([tick(1, 111.0, 30, at(10, 10)), tick(2, 551.0, 95, at(10, 10))])
    frame = builder.snapshot()
    assert set(frame["instrument_token"]) == {1, 2}
    one = frame[(frame["instrument_token"] == 1) & (frame["Date"] == at(10, 5))]
    two = frame[(frame["instrument_token"] == 2) & (frame["Date"] == at(10, 5))]
    assert one.iloc[0]["Volume"] == pytest.approx(10.0)
    assert two.iloc[0]["Volume"] == pytest.approx(20.0)


# --- the forming bar ------------------------------------------------------

def test_the_forming_bar_is_excluded_by_default() -> None:
    # Every indicator here treats a bar as a finished period. Half a bar
    # served as a whole one understates range and volume.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 10, at(10, 0)), tick(1, 101.0, 20, at(10, 2))])
    assert builder.snapshot().empty
    with_forming = builder.snapshot(include_forming=True)
    assert len(with_forming) == 1
    assert bool(with_forming.iloc[0]["partial"]) is True


def test_an_empty_builder_returns_a_frame_with_the_right_columns() -> None:
    frame = live_bars.BarBuilder().snapshot()
    assert frame.empty
    for column in ("instrument_token", "Date", "Open", "High", "Low",
                   "Close", "Volume"):
        assert column in frame.columns


# --- persistence and reading back ----------------------------------------

def test_flush_then_load_round_trips_and_maps_tokens_to_symbols(monkeypatch,
                                                                tmp_path) -> None:
    monkeypatch.setattr(live_bars, "STORE", tmp_path)
    builder = live_bars.BarBuilder()
    for minute, cumulative in ((0, 100), (5, 300), (10, 450), (15, 600)):
        builder.add([tick(738561, 100.0 + minute, cumulative, at(10, minute))])
    written = builder.flush(when=at(10, 0).date())
    assert written >= 3
    loaded = live_bars.load_today({"RELIANCE": 738561}, when=at(10, 0).date())
    assert list(loaded) == ["RELIANCE"]
    frame = loaded["RELIANCE"]
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(frame.index.tz) == "Asia/Kolkata"
    assert frame.index.is_monotonic_increasing


def test_load_drops_partial_bars_unless_asked_to_keep_them(monkeypatch,
                                                           tmp_path) -> None:
    monkeypatch.setattr(live_bars, "STORE", tmp_path)
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 9_999, at(11, 7))])     # partial, no baseline
    builder.add([tick(1, 101.0, 10_100, at(11, 10))])
    builder.add([tick(1, 102.0, 10_200, at(11, 15))])
    builder.flush(when=at(11, 0).date())
    kept = live_bars.load_today({"X": 1}, when=at(11, 0).date())
    assert len(kept["X"]) == 1                          # only the clean bar
    everything = live_bars.load_today({"X": 1}, when=at(11, 0).date(),
                                      drop_partial=False)
    assert len(everything["X"]) == 2


def test_load_returns_empty_when_the_feed_has_written_nothing(monkeypatch,
                                                              tmp_path) -> None:
    monkeypatch.setattr(live_bars, "STORE", tmp_path)
    assert live_bars.load_today({"X": 1}) == {}
    assert live_bars.status()["present"] is False


# --- the history window ---------------------------------------------------

def test_the_history_window_ends_yesterday_so_it_caches_all_day() -> None:
    # A window ending TODAY changes every session and misses the parquet
    # cache every morning, which is the delay this module exists to remove.
    start, end = live_bars.history_window(12)
    assert end < datetime.now(IST).date()
    assert (end - start) == timedelta(days=12)
