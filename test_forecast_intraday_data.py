"""Tests for the intraday dataset builder.

This module had NO tests while producing every label the forecast study is
scored against, and an independent review found real defects in it: an ATR
averaging one session while claiming fourteen, a benchmark subtracted from
a different baseline than the stock, and features reading the realised
session length at a bar where the close was still in the future.

So the tests below are mostly regression tests for those, and they are
ordered with the LOOK-AHEAD cases first. A builder that drops rows makes a
result look weak and someone investigates; a builder that leaks the future
makes it look strong and nobody does.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import forecast_intraday_data as fid
import trade_costs

IST = "Asia/Kolkata"


def _session(n: int, day: str = "2026-09-10", start: str = "09:15",
             step: str = "5min", base: float = 100.0) -> pd.DataFrame:
    """One session of bars, gently trending so highs and lows differ."""
    stamps = pd.date_range(f"{day} {start}", periods=n, freq=step, tz=IST)
    close = base + np.arange(n) * 0.1
    return pd.DataFrame({"Open": close - 0.05, "High": close + 0.2,
                         "Low": close - 0.2, "Close": close,
                         "Volume": np.full(n, 1_000.0)}, index=stamps)


# --- look-ahead ----------------------------------------------------------

def test_sigma_and_time_features_use_the_expected_session_length() -> None:
    # bars_left, frac_session_done and sigma all read n = len(close), the
    # REALISED length, at a bar where the closing bell had not happened.
    # They now come from the median of prior sessions.
    bars = {k: v.to_numpy(dtype=float) for k, v in
            _session(80).rename(columns=str.lower).items()}
    bars["stamp"] = _session(80).index.to_numpy()
    curve = np.cumsum(np.full(200, 1_000.0))
    rows = fid.session_rows("X", date(2026, 9, 10), bars, atr=1.0,
                            curve=curve, prev_close=99.0, prev_high=101.0,
                            prev_low=98.0, bench_at=np.zeros(80),
                            expected_n=75)
    assert rows, "expected some sampled bars"
    for row in rows:
        assert row["frac_session_done"] <= 1.0
        # expected_left counts down to the EXPECTED close, not the real one
        assert row["bars_left"] == max(1, 75 - (row["bar"] + 1))


def test_a_long_session_does_not_emit_more_rows_than_a_normal_one() -> None:
    # The sampling loop bounded on the realised length, so a session that
    # happened to run long contributed more observations purely for that
    # reason - weighting the dataset by an unknowable quantity.
    curve = np.cumsum(np.full(200, 1_000.0))

    def count(realised: int) -> int:
        frame = _session(realised)
        bars = {k.lower(): v.to_numpy(dtype=float) for k, v in frame.items()}
        bars["stamp"] = frame.index.to_numpy()
        return len(fid.session_rows("X", date(2026, 9, 10), bars, 1.0, curve,
                                    99.0, 101.0, 98.0,
                                    np.zeros(realised), expected_n=75))

    assert count(92) == count(75), "a long session must not earn extra rows"
    assert count(60) < count(75), "a short session genuinely has fewer bars"


def test_the_benchmark_is_read_at_the_same_bar_index_it_was_aligned_to() -> None:
    # bench_at is reindexed onto the symbol's own stamps before it arrives,
    # so position i is bar i's clock time. A stale min(i, size-1) clamp
    # would serve the benchmark's LAST value to every bar past its end.
    n = 40
    frame = _session(n)
    bars = {k.lower(): v.to_numpy(dtype=float) for k, v in frame.items()}
    bars["stamp"] = frame.index.to_numpy()
    bench = np.arange(n, dtype=float)          # bar i -> value i
    rows = fid.session_rows("X", date(2026, 9, 10), bars, 1.0,
                            np.cumsum(np.full(200, 1_000.0)), 99.0, 101.0,
                            98.0, bench, expected_n=n)
    for row in rows:
        assert row["bench_change"] == float(row["bar"])

    # And a benchmark that is NOT aligned must fail loudly. The old code
    # clamped with min(i, size - 1), which served the index's last known
    # value to every later bar - stale data wearing a fresh timestamp.
    with pytest.raises(IndexError):
        fid.session_rows("X", date(2026, 9, 10), bars, 1.0,
                         np.cumsum(np.full(200, 1_000.0)), 99.0, 101.0,
                         98.0, bench[:5], expected_n=n)


# --- the ATR that sets every stop, target and label ----------------------

def test_the_true_range_window_is_the_period_not_one_session() -> None:
    # It pooled 15 sessions then averaged joined[-ATR_PERIOD*5:] - the last
    # 70 bar-ranges, which on a ~74-bar session is YESTERDAY. The slice
    # looked like a period but indexed an array already pooled across
    # sessions, so two units were mixed.
    order, sessions = [], {}
    for index in range(fid.ATR_PERIOD + 3):
        day = date(2026, 1, 1) + pd.Timedelta(days=index).to_pytimedelta()
        # Early sessions calm, the most recent one wild. A one-session
        # window would follow the wild day; a 14-session mean should not.
        width = 10.0 if index == fid.ATR_PERIOD + 2 else 1.0
        close = np.arange(80, dtype=float)
        sessions[day] = {"high": close + width, "low": close - width,
                         "close": close, "open": close,
                         "volume": np.full(80, 1.0)}
        order.append(day)
    # upto=len(order) so the WILD day is inside the window. The 14-session
    # mean should barely move; the old 70-bar slice would sit on top of it.
    atr = fid.mean_bar_tr(order, sessions, upto=len(order))
    assert 1.5 < atr < 5.0, ("a 14-session mean must not be dominated by "
                             f"the single most recent day, got {atr}")


def test_a_short_window_refuses_rather_than_averaging_what_it_has() -> None:
    order, sessions = [], {}
    for index in range(3):
        day = date(2026, 1, 1) + pd.Timedelta(days=index).to_pytimedelta()
        close = np.arange(80, dtype=float)
        sessions[day] = {"high": close + 1, "low": close - 1, "close": close,
                         "open": close, "volume": np.full(80, 1.0)}
        order.append(day)
    assert np.isnan(fid.mean_bar_tr(order, sessions, upto=3))


def test_overnight_gaps_are_excluded_from_the_true_range() -> None:
    # Each session's ranges start at its SECOND bar, so the gap from the
    # prior close never enters an intraday ATR.
    order, sessions = [], {}
    for index in range(fid.ATR_PERIOD + 1):
        day = date(2026, 1, 1) + pd.Timedelta(days=index).to_pytimedelta()
        base = 100.0 + index * 50.0        # a huge gap every session
        close = base + np.arange(20, dtype=float) * 0.01
        sessions[day] = {"high": close + 0.05, "low": close - 0.05,
                         "close": close, "open": close,
                         "volume": np.full(20, 1.0)}
        order.append(day)
    atr = fid.mean_bar_tr(order, sessions, upto=len(order))
    assert atr < 1.0, f"a 50-point gap leaked into the ATR: {atr}"


# --- the label resolver --------------------------------------------------

def _walk(highs, lows, close_price, long, entry, stop_distance,
          target_distance):
    cmax = np.maximum.accumulate(np.asarray(highs, dtype=float))
    neg_cmin = np.maximum.accumulate(-np.asarray(lows, dtype=float))
    return fid.resolve(cmax, neg_cmin, close_price, long, entry,
                       stop_distance, target_distance)


def test_a_bar_touching_both_levels_counts_as_the_stop() -> None:
    # Intrabar order is unknowable, so the tie must resolve AGAINST the
    # trade. Resolving it as a win would inflate every hit rate measured.
    label, r = _walk([102.0], [98.0], 100.0, True, 100.0, 1.0, 2.0)
    assert label == 0 and r == -1.0


def test_the_target_is_taken_when_it_comes_first() -> None:
    label, r = _walk([100.5, 102.5], [99.5, 99.5], 101.0, True, 100.0,
                     1.0, 2.0)
    assert label == 1 and r == 2.0


def test_an_unresolved_trade_marks_out_at_the_close_not_as_a_loss() -> None:
    # Label 0 lumps stops together with positions that reached neither
    # level. The realised R must carry the mark-out, because EV downstream
    # reads it and assuming -1R there was a measured defect.
    label, r = _walk([100.5], [99.5], 100.4, True, 100.0, 1.0, 2.0)
    assert label == 0
    assert r == pytest.approx(0.4), "must mark out, not book a full loss"


def test_short_side_mirrors_the_long_side() -> None:
    up = _walk([102.5, 102.5], [99.5, 99.5], 102.0, True, 100.0, 1.0, 2.0)
    down = _walk([100.5, 100.5], [97.5, 97.5], 98.0, False, 100.0, 1.0, 2.0)
    assert up[0] == down[0] == 1
    assert up[1] == down[1] == 2.0


def test_a_degenerate_geometry_resolves_to_nothing_rather_than_raising() -> None:
    assert fid.resolve(np.array([101.0]), np.array([-99.0]), 100.0, True,
                       100.0, 0.0, 2.0) == (0, 0.0)


# --- the cost model ------------------------------------------------------

def test_the_cost_model_agrees_with_the_live_cost_stack() -> None:
    # cost_fraction paraphrases trade_costs so the builder stays a
    # standalone script. The paraphrase is only safe while it is checked.
    fid._check_cost_model()
    for price, risk in ((1000.0, 5.0), (100.0, 15.0), (250.0, 0.8)):
        quantity = int((fid.CAPITAL * fid.RISK_PCT / 100.0) / risk)
        want = trade_costs.equity_intraday_cost(
            price, price, quantity).breakeven_pct / 100.0
        assert fid.cost_fraction(price, risk) == pytest.approx(want)


def test_cost_falls_as_the_ticket_grows_because_brokerage_is_capped() -> None:
    # The flat 0.000824 constant assumed cost per rupee was flat. It is
    # hyperbolic: a tighter stop buys more shares and pays a lower
    # fraction. Slippage then more than offsets that, which is why the new
    # figure is HIGHER overall - both directions matter and are asserted.
    tight = fid.cost_fraction(1000.0, 1.0)
    wide = fid.cost_fraction(1000.0, 20.0)
    assert tight < wide, "the per-order cap must make big tickets cheaper"
    assert wide > 0.000824, "slippage must push the realistic case above the old constant"


def test_a_position_too_small_to_buy_one_share_is_nan_not_zero_cost() -> None:
    # At a 1% budget on 1,00,000 this bites whenever the stop exceeds
    # 1,000 rupees. Returning 0.0 would silently price those as free.
    assert np.isnan(fid.cost_fraction(1000.0, 2_000.0))
    assert np.isnan(fid.cost_fraction(0.0, 5.0))
    assert np.isnan(fid.cost_fraction(1000.0, 0.0))


# --- session splitting ---------------------------------------------------

def test_sessions_carry_their_timestamps_for_clock_alignment() -> None:
    order, sessions = fid.sessions_from(_session(40))
    assert order and "stamp" in sessions[order[0]]
    assert len(sessions[order[0]]["stamp"]) == 40


def test_a_session_too_short_to_sample_is_dropped() -> None:
    order, _ = fid.sessions_from(_session(3))
    assert order == []


def test_a_frame_without_the_price_columns_returns_nothing() -> None:
    assert fid.sessions_from(pd.DataFrame({"Close": [1.0]})) == ([], {})
    assert fid.sessions_from(None) == ([], {})
