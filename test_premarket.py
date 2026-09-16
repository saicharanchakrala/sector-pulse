"""Tests for the pre-open watchlist.

Two defects are guarded first, because both were found by running the
thing rather than by reading it, and both made every number on the list
quietly wrong:

  * the daily store is folded DURING the session, so its last row is a
    partial day. Measured 2026-09-15: RELIANCE carried 3.2M of volume in
    it against a 13M ten-session mean. A clock check alone did not catch
    it - at 20:35 the session was long over, but the store had been
    written at 14:42, so the row for "today" was a mid-session snapshot.

  * the liquidity gate took a MEAN. WEL traded 129.5M shares on one day
    in August and 402K on 11 September; its mean turnover is 110 crore
    against a median of 9. The gate asks whether you could get out
    tomorrow, and a mean answers a different question.

The third thing this file pins is what the list refuses to do. It ranks on
expected MOVEMENT and never on direction, because that is what this
project's measurements support - so a test asserts the ordering ignores
the sign of the last move.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import premarket

IST = ZoneInfo("Asia/Kolkata")


def _frame(sessions=300, close=100.0, spread=2.0, volume=1_000_000,
           end=date(2026, 9, 11)):
    """A daily frame ending on `end`, with a constant true range."""
    stamps = pd.bdate_range(end=pd.Timestamp(end), periods=sessions, tz=IST)
    closes = np.full(sessions, close, dtype=float)
    return pd.DataFrame({
        "Open": closes,
        "High": closes + spread / 2,
        "Low": closes - spread / 2,
        "Close": closes,
        "Volume": np.full(sessions, volume, dtype=float),
    }, index=stamps)


# --- what counts as a settled session ------------------------------------

def test_before_the_close_today_is_not_settled() -> None:
    now = datetime(2026, 9, 15, 9, 0, tzinfo=IST)
    assert premarket.settled_through(now) == date(2026, 9, 14)


def test_after_the_close_with_a_freshly_folded_store_today_counts() -> None:
    now = datetime(2026, 9, 15, 20, 0, tzinfo=IST)
    folded = datetime(2026, 9, 15, 19, 0, tzinfo=IST)
    assert premarket.settled_through(now, written_at=folded) == date(2026, 9, 15)


def test_a_store_folded_before_the_close_does_not_make_today_settled() -> None:
    # THE ONE A CLOCK CHECK MISSES. 20:35 is hours past the close, so the
    # session is over - but the store was written at 14:42, so the row it
    # holds for today is a mid-session snapshot with a partial volume and
    # a truncated range.
    now = datetime(2026, 9, 15, 20, 35, tzinfo=IST)
    folded = datetime(2026, 9, 15, 14, 42, tzinfo=IST)
    assert premarket.settled_through(now, written_at=folded) == date(2026, 9, 14)


def test_an_unreadable_store_time_is_treated_as_stale() -> None:
    now = datetime(2026, 9, 15, 20, 35, tzinfo=IST)
    assert premarket.settled_through(now, written_at=None) is not None


def test_a_partial_last_session_is_excluded_from_every_figure() -> None:
    # The partial bar carries a fifth of the usual volume. If it reached
    # rvol_10d the ratio would collapse, which is exactly what was seen.
    frame = _frame(sessions=300, volume=1_000_000, end=date(2026, 9, 14))
    partial = frame.copy()
    partial.loc[pd.Timestamp("2026-09-15", tz=IST)] = {
        "Open": 100.0, "High": 100.2, "Low": 99.8, "Close": 100.0,
        "Volume": 200_000.0}
    got = premarket.describe("X", partial, through=date(2026, 9, 14))
    assert got is not None
    assert got["rvol_10d"] == pytest.approx(1.0), (
        "the partial session leaked into the volume baseline")


# --- the liquidity gate ---------------------------------------------------

def test_the_turnover_gate_is_a_median_so_one_huge_day_cannot_carry_it() -> None:
    frame = _frame(sessions=300, close=100.0, volume=1_000)
    # One enormous day, as WEL had: 20-session mean turnover soars, median
    # is untouched, and only the median describes tomorrow's exit.
    frame.iloc[-5, frame.columns.get_loc("Volume")] = 100_000_000
    got = premarket.describe("X", frame, through=date(2026, 9, 11))
    assert got["turnover_20d"] == pytest.approx(100_000.0), "median, not mean"
    assert got["spike_ratio"] > 1.5, "the distortion must be visible"


def test_a_steady_name_has_a_spike_ratio_of_about_one() -> None:
    got = premarket.describe("X", _frame(), through=date(2026, 9, 11))
    assert got["spike_ratio"] == pytest.approx(1.0, abs=0.05)


# --- the arithmetic -------------------------------------------------------

def test_the_atr_is_a_mean_true_range_in_rupees() -> None:
    # Constant 2.0 range, flat closes, so every true range is exactly 2.0.
    got = premarket.describe("X", _frame(spread=2.0, close=100.0),
                             through=date(2026, 9, 11))
    assert got["atr"] == pytest.approx(2.0)
    assert got["atr_pct"] == pytest.approx(2.0)


def test_the_expected_band_is_one_atr_either_side_of_the_close() -> None:
    got = premarket.describe("X", _frame(spread=2.0, close=100.0),
                             through=date(2026, 9, 11))
    assert got["expected_low"] == pytest.approx(98.0)
    assert got["expected_high"] == pytest.approx(102.0)


def test_cost_in_atr_rises_as_the_range_narrows() -> None:
    # The column exists to show when costs eat the day. A name that moves
    # 0.4% must look far worse than one that moves 4%.
    wide = premarket.describe("W", _frame(spread=8.0, close=100.0),
                              through=date(2026, 9, 11))
    narrow = premarket.describe("N", _frame(spread=0.4, close=100.0),
                                through=date(2026, 9, 11))
    assert narrow["cost_in_atr"] > wide["cost_in_atr"] * 10


def test_too_little_history_is_refused_rather_than_extrapolated() -> None:
    # pos_52w over 40 sessions would describe a window its name denies.
    assert premarket.describe("X", _frame(sessions=40),
                              through=date(2026, 9, 11)) is None


# --- what it ranks on, and what it refuses --------------------------------

def test_the_ranking_is_by_expected_movement() -> None:
    frames = {
        "QUIET": _frame(spread=0.5, close=100.0, volume=10_000_000),
        "WILD": _frame(spread=6.0, close=100.0, volume=10_000_000),
        "MIDDLING": _frame(spread=2.0, close=100.0, volume=10_000_000),
    }
    table = premarket.build(frames=frames, now=datetime(2026, 9, 15, 9,
                                                        tzinfo=IST))
    assert table["symbol"].tolist() == ["WILD", "MIDDLING", "QUIET"]


def test_the_ranking_ignores_which_way_the_last_move_went() -> None:
    # THE PROPERTY THE MODULE EXISTS TO HOLD. Direction scored 0.5121 AUC
    # against a 0.4990 null here; ordering by it would assert something
    # the measurements rejected. Two names with identical ranges, one that
    # rose hard and one that fell hard, must not be separated by it.
    # Holding ATR equal while moving the close is impossible - the jump
    # enters that bar's own true range. So the test is built the other
    # way: the name with much the larger RISE has much the smaller range,
    # and must rank below the quiet wide one. A list ordered by return
    # would invert this.
    big_rise = _frame(spread=0.5, close=100.0, volume=10_000_000)
    big_rise.iloc[-1, big_rise.columns.get_loc("Close")] = 108.0
    quiet_wide = _frame(spread=6.0, close=100.0, volume=10_000_000)

    table = premarket.build(frames={"BIGRISE": big_rise,
                                    "QUIETWIDE": quiet_wide},
                            now=datetime(2026, 9, 15, 9, tzinfo=IST))
    changes = dict(zip(table["symbol"], table["change_pct"], strict=True))
    assert changes["BIGRISE"] > 7.0, "fixture is wrong: it should have jumped"
    assert abs(changes["QUIETWIDE"]) < 0.1, "fixture is wrong: it should be flat"
    assert table["symbol"].tolist() == ["QUIETWIDE", "BIGRISE"], (
        "the list ranked the big riser first, so it is ordering on return")


def test_names_below_the_price_or_turnover_floor_are_excluded() -> None:
    frames = {
        "PENNY": _frame(close=5.0, volume=10_000_000),
        "THIN": _frame(close=500.0, volume=10),
        "FINE": _frame(close=500.0, volume=10_000_000),
    }
    table = premarket.build(frames=frames,
                            now=datetime(2026, 9, 15, 9, tzinfo=IST))
    assert table["symbol"].tolist() == ["FINE"]


def test_an_empty_store_gives_an_empty_table_with_the_right_columns() -> None:
    table = premarket.build(frames={},
                            now=datetime(2026, 9, 15, 9, tzinfo=IST))
    assert table.empty
    assert list(table.columns) == premarket.COLUMNS


def test_the_limit_is_honoured() -> None:
    frames = {f"S{i:02d}": _frame(spread=1.0 + i / 10, close=100.0,
                                  volume=10_000_000) for i in range(20)}
    table = premarket.build(frames=frames, limit=5,
                            now=datetime(2026, 9, 15, 9, tzinfo=IST))
    assert len(table) == 5


def test_filings_attach_as_a_column_without_reordering(monkeypatch) -> None:
    import filings

    table = premarket.build(frames={"A": _frame(spread=4.0, volume=10_000_000),
                                    "B": _frame(spread=2.0, volume=10_000_000)},
                            now=datetime(2026, 9, 15, 9, tzinfo=IST))
    order = table["symbol"].tolist()
    monkeypatch.setattr(filings, "recent_for",
                        lambda *a, **k: [{"desc": "Credit Rating"}])
    out = premarket.with_filings(table)
    assert out["symbol"].tolist() == order, "context must not reorder"
    assert out["filing_18h"].iloc[0] == "Credit Rating"


def test_a_filings_failure_leaves_the_table_usable(monkeypatch) -> None:
    import filings

    def boom(*a, **k):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(filings, "recent_for", boom)
    table = premarket.build(frames={"A": _frame(volume=10_000_000)},
                            now=datetime(2026, 9, 15, 9, tzinfo=IST))
    out = premarket.with_filings(table)
    assert list(out["filing_18h"]) == [""]


def test_settled_truncates_on_the_date_not_the_timestamp() -> None:
    frame = _frame(sessions=5, end=date(2026, 9, 11))
    kept = premarket.settled(frame, date(2026, 9, 9))
    assert len(kept) == 3
    assert max(pd.DatetimeIndex(kept.index).date) == date(2026, 9, 9)


def test_a_missing_cutoff_leaves_the_frame_alone() -> None:
    frame = _frame(sessions=5)
    assert len(premarket.settled(frame, None)) == 5


def test_the_atr_window_is_the_configured_one() -> None:
    import config

    assert premarket.ATR_SESSIONS == config.SCAN_ATR_BARS


def test_a_frame_missing_columns_is_refused() -> None:
    frame = _frame()[["Close", "Volume"]]
    assert premarket.describe("X", frame, through=date(2026, 9, 11)) is None


def test_yesterday_is_used_when_the_store_time_is_unavailable(
        monkeypatch) -> None:
    monkeypatch.setattr(premarket, "store_written_at", lambda: None)
    now = datetime(2026, 9, 15, 20, 0, tzinfo=IST)
    assert premarket.settled_through(now) == now.date() - timedelta(days=1)
