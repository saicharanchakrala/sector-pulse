"""Tests for the intraday scanner: indicators, levels, costs, gates, chain.

Run with: .venv\\Scripts\\python -m pytest test_intraday_scan.py -q

Every test here is offline and deterministic. Where a value can be computed
by hand it is asserted against the hand-computed number rather than against
whatever the code currently returns, so a regression fails instead of
quietly redefining the expected answer.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import config
import indicators
import levels as levels_mod
import options_chain
import scan_data
import setups
import instruments
import trade_costs
from levels import LONG, SHORT

IST = ZoneInfo("Asia/Kolkata")


def _bars(rows: list[tuple], day: str = "2026-09-08") -> pd.DataFrame:
    """Build a bar frame from (hh, mm, open, high, low, close, volume) rows."""
    index, data = [], []
    for hour, minute, o, h, low, c, v in rows:
        index.append(pd.Timestamp(f"{day} {hour:02d}:{minute:02d}", tz=IST))
        data.append({"Open": o, "High": h, "Low": low, "Close": c, "Volume": v})
    return pd.DataFrame(data, index=pd.DatetimeIndex(index))


def _session(day: str, start_price: float, bars: int = 20,
             volume: float = 1000.0, step: float = 1.0) -> pd.DataFrame:
    """A synthetic ascending session of 5-minute bars."""
    rows = []
    minute = 15
    hour = 9
    for i in range(bars):
        price = start_price + i * step
        rows.append((hour, minute, price, price + 0.5, price - 0.5, price, volume))
        minute += 5
        if minute >= 60:
            minute -= 60
            hour += 1
    return _bars(rows, day)


# --- VWAP ----------------------------------------------------------------

def test_vwap_is_volume_weighted_not_a_simple_mean() -> None:
    frame = _bars([
        (9, 15, 100.0, 102.0, 98.0, 100.0, 100.0),   # typical 100, weight 100
        (9, 20, 100.0, 212.0, 208.0, 210.0, 900.0),  # typical 210, weight 900
    ])
    # (100*100 + 210*900) / 1000 = 199.0; the unweighted mean would be 155.0.
    assert indicators.vwap(frame) == pytest.approx(199.0)


def test_vwap_is_none_when_volume_is_absent_rather_than_a_plain_mean() -> None:
    frame = _bars([(9, 15, 100.0, 101.0, 99.0, 100.0, 0.0),
                   (9, 20, 100.0, 101.0, 99.0, 100.0, 0.0)])
    assert indicators.vwap(frame) is None
    no_column = frame.drop(columns=["Volume"])
    assert indicators.vwap(no_column) is None


# --- opening range -------------------------------------------------------

def test_opening_range_selects_by_clock_not_by_bar_count() -> None:
    frame = _bars([
        (9, 15, 100.0, 105.0, 99.0, 104.0, 10.0),
        (9, 20, 104.0, 106.0, 103.0, 105.0, 10.0),
        (9, 25, 105.0, 108.0, 104.0, 107.0, 10.0),   # at the 15-minute cutoff
        (9, 30, 107.0, 120.0, 90.0, 110.0, 10.0),    # must not be included
    ])
    orb = indicators.opening_range(frame, minutes=15)
    assert orb is not None
    # Bars are stamped at their start, so 09:15, 09:20 and 09:25 all begin
    # inside the first 15 minutes; 09:30 is the exclusive cutoff.
    assert orb.bars == 3
    assert orb.high == pytest.approx(108.0)
    assert orb.low == pytest.approx(99.0)
    assert 120.0 not in (orb.high, orb.low), "09:30 must stay out"


def test_opening_range_shortens_when_the_session_starts_late() -> None:
    # A missing 09:15 print must not slide the window later in the day.
    frame = _bars([(10, 0, 100.0, 103.0, 99.0, 102.0, 10.0),
                   (11, 0, 102.0, 130.0, 101.0, 129.0, 10.0)])
    orb = indicators.opening_range(frame, minutes=15)
    assert orb is not None and orb.bars == 1
    assert orb.high == pytest.approx(103.0)


# --- CPR -----------------------------------------------------------------

def test_central_pivot_range_matches_the_standard_formula() -> None:
    cpr = indicators.central_pivot_range(110.0, 90.0, 105.0)
    assert cpr is not None
    # pivot = (110+90+105)/3 = 101.666..., bc = 100, tc = 2*pivot - bc
    assert cpr.pivot == pytest.approx(101.6666667)
    assert cpr.bottom == pytest.approx(100.0)
    assert cpr.top == pytest.approx(103.3333333)
    assert cpr.width == pytest.approx(3.3333333)


def test_central_pivot_range_orders_its_bands_either_way() -> None:
    # When tc computes below bc the dataclass must still hold top >= bottom.
    cpr = indicators.central_pivot_range(110.0, 90.0, 95.0)
    assert cpr is not None and cpr.top >= cpr.bottom


def test_central_pivot_range_refuses_nonsense_input() -> None:
    assert indicators.central_pivot_range(90.0, 110.0, 100.0) is None
    assert indicators.central_pivot_range(0.0, 0.0, 0.0) is None


# --- ATR -----------------------------------------------------------------

def test_atr_uses_wilder_smoothing_not_a_simple_mean() -> None:
    # Varying true ranges, so the two schemes give different answers. Closes
    # are flat at 100 and each bar is symmetric, making TR = high - low.
    frame = _bars([
        (9, 15, 100.0, 101.0, 99.0, 100.0, 10.0),   # TR dropped (no prior close)
        (9, 20, 100.0, 101.0, 99.0, 100.0, 10.0),   # TR 2
        (9, 25, 100.0, 102.0, 98.0, 100.0, 10.0),   # TR 4
        (9, 30, 100.0, 103.0, 97.0, 100.0, 10.0),   # TR 6
        (9, 35, 100.0, 106.0, 94.0, 100.0, 10.0),   # TR 12
    ])
    ranges = [float(x) for x in indicators.true_range(frame)]
    # The opening bar keeps its high-low span: inside one session there is no
    # earlier close to measure against, and reaching for yesterday's would be
    # the very gap contamination this design excludes.
    assert ranges == pytest.approx([2.0, 2.0, 4.0, 6.0, 12.0])
    # Wilder seeds on the first 3 (mean 8/3) then smooths in 6.0 and 12.0:
    # (8/3*2 + 6)/3 = 3.777778, then (3.777778*2 + 12)/3 = 6.518519.
    assert indicators.atr(frame, period=3) == pytest.approx(6.5185185)
    # A simple mean of all five is 5.2, so this now distinguishes the two.
    assert indicators.atr(frame, period=3) != pytest.approx(5.2)


def test_atr_ignores_overnight_gaps_between_sessions() -> None:
    # A gap is not a move that could have been traded intraday. Before this
    # was fixed, a 20-rupee gap read ATR 2.21 three bars into the session
    # against a true 1.0, doubling every stop and target in the first hour.
    prior = _session("2026-09-04", 300.0, bars=20, step=0.0)
    today = _session("2026-09-08", 320.0, bars=20, step=0.0)
    combined = pd.concat([prior, today]).sort_index()
    # _session builds each bar with a 1.0 high-low span and a flat close.
    assert indicators.atr(combined, period=14) == pytest.approx(1.0)
    assert indicators.atr(today, period=14) == pytest.approx(1.0)


def test_true_range_by_session_drops_each_session_opening_bar() -> None:
    prior = _session("2026-09-04", 300.0, bars=5, step=0.0)
    today = _session("2026-09-08", 320.0, bars=5, step=0.0)
    combined = pd.concat([prior, today]).sort_index()
    ranges = indicators.true_range_by_session(combined)
    assert ranges is not None
    # 5 ranges per session, and critically none of them is the 20-rupee gap:
    # every value is the 1.0 intraday span. Computing across the whole frame
    # instead would put a 20.5 range at today's first bar.
    assert len(ranges) == 10
    assert max(float(x) for x in ranges) == pytest.approx(1.0)
    whole_frame = indicators.true_range(combined)
    assert max(float(x) for x in whole_frame) > 20.0, (
        "the unsegmented series must still show the gap this guards against")


def test_atr_true_range_accounts_for_the_gap_not_just_the_bar() -> None:
    frame = _bars([(9, 15, 100.0, 101.0, 99.0, 100.0, 10.0),
                   (9, 20, 120.0, 121.0, 119.0, 120.0, 10.0)])
    ranges = indicators.true_range(frame)
    assert ranges is not None
    # |121 - 100| = 21 beats the 2.0 high-low span of the gapped bar.
    assert float(ranges.iloc[-1]) == pytest.approx(21.0)


def test_atr_is_none_when_there_are_fewer_bars_than_the_period() -> None:
    frame = _bars([(9, 15, 100.0, 101.0, 99.0, 100.0, 10.0),
                   (9, 20, 100.0, 101.0, 99.0, 100.0, 10.0)])
    assert indicators.atr(frame, period=14) is None


# --- relative volume -----------------------------------------------------

def test_relative_volume_compares_the_same_time_of_day() -> None:
    slow = _session("2026-09-04", 100.0, bars=4, volume=100.0)
    also_slow = _session("2026-09-05", 100.0, bars=4, volume=100.0)
    busy = _session("2026-09-08", 100.0, bars=4, volume=300.0)
    frame = pd.concat([slow, also_slow, busy]).sort_index()
    # Today's cumulative is 3x the prior sessions' at every point.
    assert indicators.relative_volume(frame) == pytest.approx(3.0)


def test_relative_volume_uses_a_median_so_one_frenzy_sets_no_baseline() -> None:
    quiet_a = _session("2026-09-01", 100.0, bars=3, volume=100.0)
    quiet_b = _session("2026-09-02", 100.0, bars=3, volume=100.0)
    frenzy = _session("2026-09-03", 100.0, bars=3, volume=10_000.0)
    today = _session("2026-09-08", 100.0, bars=3, volume=200.0)
    frame = pd.concat([quiet_a, quiet_b, frenzy, today]).sort_index()
    # Median of (100, 100, 10000) cumulative curves is the 100 curve, so 2.0.
    # A mean baseline would report roughly 0.06 and hide a genuine doubling.
    assert indicators.relative_volume(frame) == pytest.approx(2.0)


def test_relative_volume_skips_a_prior_session_that_ended_early() -> None:
    # asof() would return a short session's final total, reading as "quiet at
    # this hour" and inflating today's ratio through the floor on nothing but
    # missing data. Measured before the fix: 2.0 became 3.16.
    full = _session("2026-09-01", 100.0, bars=12, volume=100.0)
    short = _session("2026-09-02", 100.0, bars=3, volume=100.0)
    today = _session("2026-09-08", 100.0, bars=12, volume=200.0)
    frame = pd.concat([full, short, today]).sort_index()
    assert indicators.relative_volume(frame) == pytest.approx(2.0)


def test_relative_volume_is_none_without_a_prior_session() -> None:
    assert indicators.relative_volume(_session("2026-09-08", 100.0)) is None


# --- session helpers -----------------------------------------------------

def test_session_bars_isolates_one_day() -> None:
    frame = pd.concat([_session("2026-09-04", 100.0, bars=3),
                       _session("2026-09-08", 200.0, bars=5)]).sort_index()
    today = indicators.session_bars(frame)
    assert today is not None and len(today) == 5
    assert all(ts.date().isoformat() == "2026-09-08" for ts in today.index)


def test_minutes_left_clamps_at_zero_after_the_close() -> None:
    assert indicators.minutes_left_in_session(datetime(2026, 9, 8, 13, 0)) == 150
    assert indicators.minutes_left_in_session(datetime(2026, 9, 8, 16, 0)) == 0


def test_minutes_left_uses_ist_for_an_aware_datetime() -> None:
    # The scanner always passes IST, which must behave like the naive case.
    ist = datetime(2026, 9, 8, 13, 0, tzinfo=IST)
    assert indicators.minutes_left_in_session(ist) == 150
    # A Sydney-aware 17:30 is 13:00 IST, so it must give the same answer
    # rather than reading 17:30 as though it were Mumbai wall-clock.
    sydney = ist.astimezone(ZoneInfo("Australia/Sydney"))
    assert sydney.hour != 13
    assert indicators.minutes_left_in_session(sydney) == 150


# --- costs ---------------------------------------------------------------

def test_equity_brokerage_is_capped_per_order() -> None:
    # 0.03% of 10 lakh is 300, so both legs must charge the 20-rupee cap.
    cost = trade_costs.equity_intraday_cost(1000.0, 1000.0, 1000)
    assert cost.brokerage == pytest.approx(40.0)


def test_equity_brokerage_is_percentage_based_on_small_orders() -> None:
    # 0.03% of 10,000 is 3.0, below the cap, so it applies per leg.
    cost = trade_costs.equity_intraday_cost(100.0, 100.0, 100)
    assert cost.brokerage == pytest.approx(6.0)


def test_gst_excludes_stt_and_stamp_duty() -> None:
    cost = trade_costs.equity_intraday_cost(1000.0, 1000.0, 100)
    expected = (cost.brokerage + cost.transaction + cost.sebi) * config.COST_GST_PCT
    assert cost.gst == pytest.approx(expected)
    assert cost.stt > 0.0 and cost.stamp_duty > 0.0, "both must be non-zero"


def test_equity_stt_and_stamp_duty_hit_only_their_own_leg() -> None:
    cost = trade_costs.equity_intraday_cost(100.0, 200.0, 10)
    assert cost.stt == pytest.approx(200.0 * 10 * config.COST_EQ_STT_SELL_PCT)
    assert cost.stamp_duty == pytest.approx(100.0 * 10 * config.COST_EQ_STAMP_BUY_PCT)


def test_options_stt_and_stamp_duty_hit_only_their_own_leg() -> None:
    cost = trade_costs.options_cost(10.0, 20.0, 1, 500)
    sell_turnover = 20.0 * 500
    buy_turnover = 10.0 * 500
    assert cost.stt == pytest.approx(sell_turnover * config.COST_OPT_STT_SELL_PCT)
    assert cost.stamp_duty == pytest.approx(
        buy_turnover * config.COST_OPT_STAMP_BUY_PCT)
    # Charging STT on both legs would inflate it by the buy leg's share.
    assert cost.stt < (sell_turnover + buy_turnover) * config.COST_OPT_STT_SELL_PCT
    assert cost.brokerage == pytest.approx(config.COST_OPT_BROKERAGE_FLAT * 2)


def test_cheap_options_cost_far_more_in_percentage_terms() -> None:
    dear = trade_costs.options_breakeven_pct(100.0, 1, 500)
    cheap = trade_costs.options_breakeven_pct(5.0, 1, 500)
    assert cheap > dear * 5, (cheap, dear)


def test_zero_or_negative_size_costs_nothing_rather_than_raising() -> None:
    assert trade_costs.equity_intraday_cost(100.0, 100.0, 0).total == 0.0
    assert trade_costs.equity_intraday_cost(100.0, 100.0, -5).total == 0.0
    assert trade_costs.equity_breakeven_pct(100.0, 0) == 0.0


# --- levels --------------------------------------------------------------

def _long_levels(**kwargs):
    """A LONG position with no structural level, so volatility sets the stop."""
    defaults = dict(direction=LONG, entry=100.0, atr_per_bar=1.0, bars_left=25,
                    capital=1_000_000.0, risk_pct=1.0)
    defaults.update(kwargs)
    return levels_mod.build_levels(**defaults)


def test_stop_is_half_the_remaining_sigma_by_default() -> None:
    trade = _long_levels()
    assert trade is not None
    # sigma = 1.0 * sqrt(25) = 5.0; stop distance = 0.5 * 5.0 = 2.5
    assert trade.stop == pytest.approx(97.5)
    assert trade.risk_per_share == pytest.approx(2.5)


def test_target_lands_on_sigma_so_it_stays_reachable() -> None:
    trade = _long_levels()
    assert trade is not None
    assert trade.target == pytest.approx(105.0)     # 2 * 2.5 above entry
    assert trade.expected_range == pytest.approx(5.0)
    assert trade.reachable, "a 2:1 target on a half-sigma stop must be reachable"


def test_short_levels_mirror_long_levels() -> None:
    trade = levels_mod.build_levels(direction=SHORT, entry=100.0, atr_per_bar=1.0,
                                    bars_left=25, capital=1_000_000.0, risk_pct=1.0)
    assert trade is not None
    assert trade.stop == pytest.approx(102.5)
    assert trade.target == pytest.approx(95.0)
    assert trade.reachable


def test_a_structural_level_inside_the_band_sets_the_stop() -> None:
    # Base stop distance is 2.5, so the accepted window is [1.5, 2.5]. An
    # opening-range low 2.0 away is inside it, so the real level wins over
    # the arithmetic one and the target still lands inside sigma.
    trade = _long_levels(opening_low=98.0)
    assert trade is not None
    assert trade.stop == pytest.approx(98.0)
    assert trade.stop_source == "opening-range low"
    assert trade.reachable, "structure inside the window must stay reachable"


def test_a_structural_level_wider_than_the_volatility_stop_is_ignored() -> None:
    # Base distance is 2.5. A level 3.0 away used to be accepted (inside the
    # old +/-40% band) and then scaled the target to 6.0 against a 5.0 sigma,
    # so reachability rejected it every time. The window now stops at base.
    trade = _long_levels(opening_low=99.0)
    assert trade is not None
    assert trade.stop == pytest.approx(97.5)
    assert trade.stop_source == "volatility"
    assert trade.reachable, "must stay actionable rather than self-reject"


def test_the_session_low_is_tried_when_the_opening_range_is_unusable() -> None:
    # The opening-range low sits above the entry, so it is skipped and the
    # session low is considered instead. Previously only the first candidate
    # was ever returned, so the fallback could not fire.
    trade = _long_levels(opening_low=101.0, day_low=98.0)
    assert trade is not None
    assert trade.stop == pytest.approx(98.0)
    assert trade.stop_source == "session low"


def test_a_structural_level_outside_the_band_is_ignored() -> None:
    # 10.0 away against a 2.5 base is far outside the window. Honouring it
    # would push the 2:1 target to 20.0, past the 5.0 remaining move.
    trade = _long_levels(opening_low=90.0)
    assert trade is not None
    assert trade.stop == pytest.approx(97.5)
    assert trade.stop_source == "volatility"
    assert trade.reachable


def test_a_structural_level_on_the_wrong_side_is_ignored() -> None:
    trade = _long_levels(opening_low=101.0)
    assert trade is not None
    assert trade.stop == pytest.approx(97.5)
    assert trade.stop_source == "volatility"


def test_size_comes_from_the_risk_budget_when_capital_allows() -> None:
    trade = _long_levels(capital=1_000_000.0, risk_pct=1.0)
    assert trade is not None
    # 1% of 10 lakh is 10,000 at risk over a 2.5 stop = 4,000 shares.
    assert trade.quantity == 4000
    assert not trade.capital_capped


def test_size_is_capped_by_funding_not_just_by_risk() -> None:
    # 1% of 1 lakh is 1,000 at risk over a 2.5 stop, which asks for 400
    # shares of a 100-rupee stock: 40,000 notional against 1 lakh at 5x is
    # affordable. Drop leverage to 1x and the same 40,000 is still fine, so
    # squeeze capital instead.
    trade = _long_levels(capital=10_000.0, risk_pct=10.0, leverage=1.0)
    assert trade is not None
    assert trade.capital_capped, "10,000 cannot fund 400 shares at 100"
    assert trade.quantity == 100          # 10,000 / 100
    assert trade.entry * trade.quantity <= 10_000.0 * 1.0


def test_required_win_rate_solves_the_breakeven_equation() -> None:
    trade = _long_levels()
    assert trade is not None
    reward = trade.lot_risk * trade.reward_risk
    expected = (trade.lot_risk + trade.cost_rupees) / (reward + trade.lot_risk)
    assert trade.required_win_rate == pytest.approx(expected)
    # 2:1 before costs needs 33.3%; costs can only push it up, never down.
    assert trade.required_win_rate > 1.0 / 3.0


def test_required_win_rate_is_capped_at_certainty() -> None:
    # Costs above the whole reward make the trade unwinnable at any hit rate,
    # so the figure must saturate at 1.0 rather than exceed it. Built by hand
    # because no realistic sizing gets there.
    hopeless = levels_mod.TradeLevels(
        direction=LONG, entry=100.0, stop=99.0, target=102.0, quantity=1,
        lot_risk=1.0, reward_risk=2.0, breakeven_pct=50.0, cost_rupees=500.0,
        expected_range=5.0, stop_source="volatility", capital_capped=False)
    assert hopeless.required_win_rate == pytest.approx(1.0)
    # Without the clamp this would be (1 + 500) / 3 = 167.0.
    sane = _long_levels()
    assert sane is not None and 0.0 < sane.required_win_rate < 1.0


def test_levels_refuse_to_build_without_volatility_or_time() -> None:
    assert _long_levels(atr_per_bar=0.0) is None
    assert _long_levels(bars_left=0) is None
    assert _long_levels(entry=0.0) is None
    assert _long_levels(capital=0.0) is None


def test_levels_reject_an_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        levels_mod.build_levels(direction="HOLD", entry=100.0, atr_per_bar=1.0,
                                bars_left=25)


def test_expected_range_grows_with_the_square_root_of_time() -> None:
    assert levels_mod.expected_remaining_range(1.0, 100) == pytest.approx(10.0)
    assert levels_mod.expected_remaining_range(1.0, 25) == pytest.approx(5.0)
    assert levels_mod.expected_remaining_range(1.0, 0) == 0.0
    assert levels_mod.expected_remaining_range(0.0, 25) == 0.0


# --- direction agreement -------------------------------------------------

def _reading(**kwargs) -> setups.Readings:
    """A Readings record that passes every gate unless overridden."""
    # The opening-range low is 3.0 below the entry against a 2.5 base stop,
    # so it falls outside the accepted window and the volatility stop is
    # used. That keeps the fixture genuinely actionable, which it was not
    # while the band accepted levels the reachability gate then rejected.
    orb = indicators.OpeningRange(high=101.0, low=99.0, bars=3, minutes=15)
    defaults = dict(
        symbol="TEST", ticker="TEST.NS", last=102.0, prev_close=100.0,
        day_change_pct=2.0, vwap=100.5, vwap_distance_pct=1.49,
        opening_range=orb, cpr=None, atr_bar=1.0, rvol=2.0,
        relative_strength=1.5, turnover_20d=1e9, oi_change_pct=5.0,
        futures_share=0.85, derivatives_turnover=1e9, day_high=102.5,
        day_low=98.8, minutes_left=150, bars_left=25)
    defaults.update(kwargs)
    return setups.Readings(**defaults)


def test_direction_needs_vwap_and_the_opening_range_to_agree() -> None:
    assert setups.choose_direction(_reading())[0] == LONG
    below = _reading(last=98.0, vwap=100.5, day_change_pct=-2.0)
    assert setups.choose_direction(below)[0] == SHORT


def test_price_inside_the_opening_range_is_never_a_setup() -> None:
    inside = _reading(last=100.0, vwap=99.5)
    direction, line = setups.choose_direction(inside)
    assert direction == setups.NO_SETUP
    assert "inside the opening range" in line


def test_a_break_against_the_vwap_side_is_never_a_setup() -> None:
    # Broke the range high but sits below VWAP: the two facts disagree.
    conflicted = _reading(last=101.5, vwap=103.0)
    direction, line = setups.choose_direction(conflicted)
    assert direction == setups.NO_SETUP
    assert "against its VWAP side" in line


def test_opening_range_is_not_closed_until_it_has_elapsed() -> None:
    # 09:30 is the first bar outside the first-15-minutes window, so it is
    # the first instant at which a breakout is representable at all.
    for bars, closed in ((1, False), (2, False), (3, False), (4, True), (5, True)):
        frame = _session("2026-09-08", 100.0, bars=bars, step=0.0)
        assert indicators.opening_range_closed(frame) is closed, bars


def test_no_direction_is_claimed_before_the_opening_range_closes() -> None:
    # While the latest bar is inside the range window, orb.low <= last <=
    # orb.high holds identically, so neither break can ever be True. That
    # must be reported as not computable, never as "no agreement": the same
    # empty output would otherwise appear on a violent gap-up morning.
    orb = indicators.OpeningRange(high=105.0, low=99.0, bars=3, minutes=15)
    premature = _reading(last=104.0, vwap=101.0, opening_range=orb,
                         range_closed=False, session_bar_count=3)
    direction, line = setups.choose_direction(premature)
    assert direction == setups.NO_SETUP
    assert "has not closed yet" in line
    assert "[N/A]" in line, "must not read as a market observation"
    assert "no agreement" not in line


def test_the_same_reading_is_directional_once_the_range_has_closed() -> None:
    orb = indicators.OpeningRange(high=105.0, low=99.0, bars=3, minutes=15)
    ready = _reading(last=106.0, vwap=101.0, opening_range=orb,
                     range_closed=True, session_bar_count=8)
    assert setups.choose_direction(ready)[0] == LONG


def test_measure_reports_session_readiness_from_the_bars() -> None:
    early = _session("2026-09-08", 100.0, bars=3, step=0.0)
    late = _session("2026-09-08", 100.0, bars=20, step=0.0)
    when = datetime(2026, 9, 8, 10, 0, tzinfo=IST)
    first = setups.measure("T", "T.NS", early, None, None, when)
    second = setups.measure("T", "T.NS", late, None, None, when)
    assert first is not None and second is not None
    assert first.session_bar_count == 3 and not first.range_closed
    assert second.session_bar_count == 20 and second.range_closed


def test_reachability_is_reported_as_an_invariant_not_a_pass() -> None:
    # target = 2 * stop and stop is capped at 0.5 * sigma, so the target
    # lands on sigma by construction. Presenting that as a PASS implied a
    # check that cannot bind.
    setup = setups.evaluate(_reading(), capital=1_000_000.0)
    assert setup.levels is not None
    line = next(r for r in setup.reasons if "plausible remaining move" in r)
    assert "[OK]" in line
    assert "[PASS]" not in line


def test_direction_fails_closed_without_vwap_or_an_opening_range() -> None:
    assert setups.choose_direction(_reading(vwap=None))[0] == setups.NO_SETUP
    assert setups.choose_direction(_reading(opening_range=None))[0] == setups.NO_SETUP


# --- gates ---------------------------------------------------------------

def test_a_clean_reading_is_actionable() -> None:
    setup = setups.evaluate(_reading(), capital=1_000_000.0)
    assert setup.actionable, setup.reasons
    assert setup.direction == LONG
    assert setup.rank_score > 0.0


def test_missing_relative_volume_fails_closed() -> None:
    setup = setups.evaluate(_reading(rvol=None), capital=1_000_000.0)
    assert not setup.actionable
    assert any("no relative-volume baseline" in r for r in setup.reasons)


def test_missing_relative_strength_fails_closed() -> None:
    setup = setups.evaluate(_reading(relative_strength=None),
                            capital=1_000_000.0)
    assert not setup.actionable
    assert any("no benchmark comparison" in r for r in setup.reasons)


def test_relative_strength_must_agree_with_the_direction() -> None:
    # A long that is underperforming the index must not pass.
    setup = setups.evaluate(_reading(relative_strength=-1.0),
                            capital=1_000_000.0)
    assert not setup.actionable
    assert any("needs to outperform" in r and "[FAIL]" in r
               for r in setup.reasons)


def test_missing_turnover_history_fails_closed() -> None:
    setup = setups.evaluate(_reading(turnover_20d=None), capital=1_000_000.0)
    assert not setup.actionable
    assert any("depth is unknown" in r for r in setup.reasons)


def test_no_setup_when_the_session_is_nearly_over() -> None:
    setup = setups.evaluate(_reading(minutes_left=10), capital=1_000_000.0)
    assert not setup.actionable
    assert any("min left" in r and "[FAIL]" in r for r in setup.reasons)


def test_a_penny_stock_is_rejected_on_price() -> None:
    setup = setups.evaluate(_reading(last=5.0, vwap=4.0, prev_close=4.0,
                                     opening_range=indicators.OpeningRange(
                                         4.5, 3.5, 3, 15)),
                            capital=1_000_000.0)
    assert not setup.actionable
    assert any("price 5.00" in r and "[FAIL]" in r for r in setup.reasons)


def test_costs_that_demand_an_implausible_win_rate_block_the_setup() -> None:
    # A very quiet stock gets a very tight stop, so round-trip charges become
    # large relative to the money at risk and the breakeven win rate climbs
    # past the ceiling. This is the gate that catches trades which are only
    # profitable for the broker.
    quiet = _reading(atr_bar=0.02)
    setup = setups.evaluate(quiet, capital=1_000_000.0, risk_pct=1.0)
    assert setup.levels is not None, "levels must exist for the gate to bind"
    assert setup.levels.required_win_rate > config.SCAN_MAX_WIN_RATE, (
        setup.levels.required_win_rate)
    assert not setup.actionable
    assert any("win rate" in r and "[FAIL]" in r for r in setup.reasons),         setup.reasons


def test_rank_puts_actionable_setups_first() -> None:
    good = setups.evaluate(_reading(), capital=1_000_000.0)
    bad = setups.evaluate(_reading(rvol=None), capital=1_000_000.0)
    ranked = setups.rank([bad, good])
    assert ranked[0] is good
    assert setups.top_setup([bad, good]) is good
    assert setups.top_setup([bad]) is None


def test_score_is_the_documented_weighted_sum() -> None:
    reading = _reading(rvol=3.0, relative_strength=2.0, vwap_distance_pct=1.5,
                       oi_change_pct=5.0)
    setup = setups.evaluate(reading, capital=1_000_000.0)
    assert setup.levels is not None
    trade = setup.levels
    # rvol, strength and stretch saturate at 1.0 for these inputs; open
    # interest at +5% is half of the 10% that saturates; cost and headroom
    # come from the levels.
    cost = min(1.0, max(0.0, (trade.cost_multiple - 1.0) / 5.0))
    headroom = min(1.0, max(0.0, trade.expected_range / trade.target_distance - 1.0))
    expected = round(0.25 * 1.0 + 0.20 * 1.0 + 0.15 * 1.0
                     + 0.15 * cost + 0.10 * headroom + 0.15 * 0.5, 4)
    assert setups.score_setup(reading, trade) == pytest.approx(expected)
    # Pin the weights: scrambling them changes this number.
    assert 0.70 <= setups.score_setup(reading, trade) <= 0.90


def test_a_name_without_derivatives_scores_zero_on_positioning() -> None:
    # Absent open interest must not flatter a symbol against one that has
    # the data and shows an unwind. A neutral 0.5 default would do exactly
    # that, so missing data scores 0.0 on this term.
    trade = _long_levels()
    assert trade is not None
    none_at_all = setups.score_setup(_reading(oi_change_pct=None), trade)
    unwinding = setups.score_setup(_reading(oi_change_pct=-8.0), trade)
    building = setups.score_setup(_reading(oi_change_pct=10.0), trade)
    assert none_at_all == pytest.approx(unwinding), (
        "no derivatives and an unwind both score 0.0 on positioning")
    assert building > none_at_all
    assert building - none_at_all == pytest.approx(0.15, abs=1e-4)


def test_open_interest_is_reported_but_does_not_veto_by_default() -> None:
    # The convention has no backtest here, so it informs the rank and the
    # reasons without silently halving the output.
    assert not config.SCAN_REQUIRE_OI_CONFIRMATION
    unwinding = setups.evaluate(_reading(oi_change_pct=-12.0),
                                capital=1_000_000.0)
    assert unwinding.actionable, unwinding.reasons
    assert any("open interest -12.00% (unwinding)" in r and "[INFO]" in r
               for r in unwinding.reasons), unwinding.reasons


def test_a_name_without_derivatives_still_reports_the_absence() -> None:
    setup = setups.evaluate(_reading(oi_change_pct=None, futures_share=None),
                            capital=1_000_000.0)
    assert setup.actionable
    assert any("no listed derivatives" in r for r in setup.reasons)


def test_open_interest_can_be_made_binding(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_REQUIRE_OI_CONFIRMATION", True)
    unwinding = setups.evaluate(_reading(oi_change_pct=-12.0),
                                capital=1_000_000.0)
    assert not unwinding.actionable
    assert any("unwinding" in r and "[FAIL]" in r for r in unwinding.reasons)
    building = setups.evaluate(_reading(oi_change_pct=6.0),
                               capital=1_000_000.0)
    assert building.actionable, building.reasons
    # With the gate binding, a name carrying no derivatives at all cannot
    # pass either: absence is not confirmation.
    absent = setups.evaluate(_reading(oi_change_pct=None), capital=1_000_000.0)
    assert not absent.actionable


def test_score_is_independent_of_the_rest_of_the_scan() -> None:
    alone = setups.evaluate(_reading(), capital=1_000_000.0)
    crowd = [setups.evaluate(_reading(symbol=f"S{i}"), capital=1_000_000.0)
             for i in range(5)]
    assert alone.rank_score == pytest.approx(crowd[0].rank_score)


def test_explanation_states_the_required_win_rate() -> None:
    text = setups.explain(setups.evaluate(_reading(), capital=1_000_000.0))
    assert "win rate" in text or "break" in text
    assert "not a" in text.lower(), "must keep the no-forecast caveat"


# --- option contracts ----------------------------------------------------

def _contract(**kwargs) -> options_chain.Contract:
    defaults = dict(symbol="X", expiry="08-Sep-2026", strike=100.0,
                    side=options_chain.CALL, last_price=10.0, bid=9.9,
                    ask=10.1, open_interest=5000, oi_change=100, volume=2000,
                    implied_volatility=18.0)
    defaults.update(kwargs)
    return options_chain.Contract(**defaults)


def test_spread_pct_is_none_on_a_one_sided_book_not_zero() -> None:
    assert _contract(bid=0.0).spread_pct is None
    assert _contract(ask=0.0).spread_pct is None
    assert _contract(bid=9.0, ask=11.0).spread_pct == pytest.approx(20.0)


def test_an_unquoted_contract_fails_the_quality_gate() -> None:
    ok, reasons = options_chain.quality_reasons(_contract(bid=0.0, ask=0.0), 1, 500)
    assert not ok
    assert any("cannot be measured" in r for r in reasons)


def test_a_wide_spread_fails_the_quality_gate() -> None:
    ok, _ = options_chain.quality_reasons(_contract(bid=9.0, ask=11.0), 1, 500)
    assert not ok


def test_thin_open_interest_or_volume_fails_the_quality_gate() -> None:
    assert not options_chain.quality_reasons(_contract(open_interest=1), 1, 500)[0]
    assert not options_chain.quality_reasons(_contract(volume=1), 1, 500)[0]


def test_max_pain_is_the_least_writer_payout_strike() -> None:
    chain = [
        _contract(strike=100.0, side=options_chain.CALL, open_interest=500),
        _contract(strike=100.0, side=options_chain.PUT, open_interest=500),
        _contract(strike=110.0, side=options_chain.CALL, open_interest=100),
        _contract(strike=110.0, side=options_chain.PUT, open_interest=2000),
    ]
    # Settling at 100 costs writers the 110 puts: 2000 * 10 = 20,000.
    # Settling at 110 costs them the 100 calls:    500 * 10 =  5,000.
    # 110 is the cheaper settlement, so max pain sits there, not at spot.
    assert options_chain.max_pain(chain) == pytest.approx(110.0)
    assert options_chain.max_pain([]) is None


def test_pick_contract_buys_calls_for_long_and_puts_for_short() -> None:
    chain = [_contract(strike=100.0, side=options_chain.CALL),
             _contract(strike=100.0, side=options_chain.PUT)]
    call, _ = options_chain.pick_contract(chain, 100.0, LONG, 500)
    put, _ = options_chain.pick_contract(chain, 100.0, SHORT, 500)
    assert call is not None and call.side == options_chain.CALL
    assert put is not None and put.side == options_chain.PUT


def test_pick_contract_returns_nothing_when_every_strike_is_illiquid() -> None:
    chain = [_contract(strike=100.0, open_interest=1, volume=1)]
    picked, reasons = options_chain.pick_contract(chain, 100.0, LONG, 500)
    assert picked is None
    assert "no contract cleared the quality gates" in reasons[0]


def test_atm_strike_is_the_nearest_strike_to_spot() -> None:
    chain = [_contract(strike=95.0), _contract(strike=100.0), _contract(strike=110.0)]
    assert options_chain.atm_strike(chain, 101.0) == pytest.approx(100.0)
    assert options_chain.atm_strike([], 101.0) is None


# --- instrument discovery ---------------------------------------------

_MKTLOTS = (
    "UNDERLYING,SYMBOL,SEP-26,OCT-26\n"
    "NIFTY 50,NIFTY,65,65\n"
    "NIFTY FPI 150,NIFTYFPI,1100,1100\n"
    "Derivatives on Individual Securities,Symbol,SEP-26,OCT-26\n"
    "RELIANCE INDUSTRIES LTD,RELIANCE,500,500\n"
    "ICICI BANK LTD,ICICIBANK,700,700\n"
)

_EQUITY_MASTER = (
    "SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,PAID UP VALUE,"
    "MARKET LOT,ISIN NUMBER,FACE VALUE\n"
    "20MICRONS,20 Microns Limited,EQ,06-OCT-2008,5,1,INE144J01027,5\n"
    "RELIANCE,Reliance Industries Limited,EQ,29-NOV-1995,10,1,INE002A01018,10\n"
)


def test_fo_master_splits_indices_from_stocks_by_the_section_row() -> None:
    # The boundary comes from the file's own "Derivatives on Individual
    # Securities" row, not a maintained list. A hardcoded index list is
    # exactly what left NIFTYFPI treated as an equity on every scan.
    indices, stocks = instruments.parse_fo_master(_MKTLOTS)
    assert [i.symbol for i in indices] == ["NIFTY", "NIFTYFPI"]
    assert [i.symbol for i in stocks] == ["RELIANCE", "ICICIBANK"]
    assert all(i.kind == instruments.KIND_FO_INDEX for i in indices)
    assert all(i.is_index for i in indices)
    assert not any(s.is_index for s in stocks)


def test_fo_master_reads_front_month_lot_sizes() -> None:
    _, stocks = instruments.parse_fo_master(_MKTLOTS)
    lots = {s.symbol: s.lot_size for s in stocks}
    assert lots == {"RELIANCE": 500, "ICICIBANK": 700}


def test_fo_master_without_a_section_row_treats_all_as_stocks() -> None:
    # Fail towards the safer reading: calling an index a stock only wastes a
    # download, while calling a stock an index would silently drop it.
    text = "UNDERLYING,SYMBOL,SEP-26\nRELIANCE INDUSTRIES LTD,RELIANCE,500\n"
    indices, stocks = instruments.parse_fo_master(text)
    assert indices == []
    assert [s.symbol for s in stocks] == ["RELIANCE"]


def test_equity_master_parses_by_header_name() -> None:
    equities = instruments.parse_equity_master(_EQUITY_MASTER)
    assert [e.symbol for e in equities] == ["20MICRONS", "RELIANCE"]
    assert equities[1].name == "Reliance Industries Limited"
    assert equities[1].series == "EQ"
    assert equities[1].isin == "INE002A01018"
    assert all(e.kind == instruments.KIND_EQUITY for e in equities)


def test_equity_master_refuses_a_file_with_no_symbol_column() -> None:
    assert instruments.parse_equity_master("A,B\n1,2\n") == []
    assert instruments.parse_equity_master("") == []


def test_ticker_mapping_returns_kite_symbols_and_strips_yahoo_leftovers() -> None:
    # Kite takes bare tradingsymbols, so this is near-identity for a live
    # symbol and a cleanup for anything written before the migration.
    assert instruments.to_ticker("RELIANCE") == "RELIANCE"
    assert instruments.to_ticker("M&M") == "M&M"
    assert instruments.to_ticker("RELIANCE.NS") == "RELIANCE"
    # The Yahoo index spellings map to Kite index names, not to a bare strip:
    # "NSEI" is not an instrument, "NIFTY 50" is.
    assert instruments.to_ticker("^NSEI") == "NIFTY 50"
    assert instruments.to_ticker("^CNXIT") == "NIFTY IT"


def test_fo_state_keeps_lakh_and_rupee_fields_apart() -> None:
    # Verified against the feed's own arithmetic: total == futures + options
    # premium, both in lakhs, while optValue is options notional in rupees.
    # Summing futValue with optValue made every futures share read 0.0%.
    state = instruments.parse_fo_state([{
        "symbol": "RELIANCE", "underlyingValue": 1294, "latestOI": 538981,
        "prevOI": 519362, "futValue": 151760.8249, "premValue": 17711.30375,
        "optValue": 158116645375, "total": 169472.12865, "volume": 260827,
    }])["RELIANCE"]
    assert state.totals_reconcile
    assert state.oi_change == pytest.approx(19619.0)
    # NSE's own avgInOI for this row was 3.78.
    assert state.oi_change_pct == pytest.approx(3.7776, abs=1e-3)
    assert state.futures_share == pytest.approx(151760.8249 / 169472.12865)
    assert state.futures_share == pytest.approx(0.8954, abs=1e-3)
    assert state.derivatives_turnover_rupees == pytest.approx(169472.12865 * 1e5)


def test_fo_state_flags_a_row_whose_totals_stop_reconciling() -> None:
    # If NSE redefines a column the share must become untrustworthy rather
    # than quietly wrong.
    broken = instruments.parse_fo_state([{
        "symbol": "X", "latestOI": 100, "prevOI": 100, "futValue": 10.0,
        "premValue": 5.0, "total": 999.0,
    }])["X"]
    assert not broken.totals_reconcile


def test_fo_state_has_no_oi_change_without_a_prior_reading() -> None:
    state = instruments.parse_fo_state([
        {"symbol": "X", "latestOI": 100, "prevOI": 0}])["X"]
    assert state.oi_change_pct is None
    assert state.futures_share is None


def test_universe_round_trips_through_its_snapshot(tmp_path) -> None:
    universe = instruments.Universe(
        captured_at=datetime(2026, 9, 8, 10, 0, tzinfo=IST),
        equities=[instruments.Instrument("RELIANCE", instruments.KIND_EQUITY)],
        fo_indices=[instruments.Instrument("NIFTY", instruments.KIND_FO_INDEX,
                                           lot_size=65)],
        fo_stocks=[instruments.Instrument("RELIANCE", instruments.KIND_FO_STOCK,
                                          lot_size=500)],
        fo_state=instruments.parse_fo_state(
            [{"symbol": "RELIANCE", "latestOI": 110, "prevOI": 100}]),
        gaps=["no per-contract futures feed answered"],
    )
    assert instruments.save(universe, tmp_path) is not None
    back = instruments.load_latest(tmp_path)
    assert back is not None
    assert back.counts() == universe.counts()
    assert back.captured_at == universe.captured_at
    assert back.fo_stocks[0].lot_size == 500
    assert back.fo_state["RELIANCE"].oi_change_pct == pytest.approx(10.0)
    assert back.gaps == universe.gaps


def _tiny_universe(when=None) -> instruments.Universe:
    """A minimal universe for store tests."""
    return instruments.Universe(
        captured_at=when or datetime(2026, 9, 8, 10, 0, tzinfo=IST),
        fo_stocks=[instruments.Instrument("RELIANCE",
                                          instruments.KIND_FO_STOCK,
                                          lot_size=500)])


def test_saving_keeps_one_current_file_not_a_history(tmp_path) -> None:
    # The universe describes what is listed right now, so an older copy has
    # no use and the store deliberately keeps only the latest.
    first = instruments.save(_tiny_universe(datetime(2026, 9, 8, 10, 0, tzinfo=IST)),
                             tmp_path)
    second = instruments.save(_tiny_universe(datetime(2026, 9, 8, 15, 0, tzinfo=IST)),
                              tmp_path)
    assert first == second, "both syncs must write the same path"
    assert [p.name for p in sorted(tmp_path.glob("*.json"))] == ["universe.json"]
    back = instruments.load_latest(tmp_path)
    assert back is not None
    assert back.captured_at.hour == 15, "the later sync must win"


def test_saving_prunes_superseded_timestamped_files(tmp_path) -> None:
    # Installs that predate the single-file store must be cleaned up rather
    # than left to accumulate beside the current file.
    for stamp in ("20260901_100000", "20260902_100000"):
        (tmp_path / f"universe_{stamp}.json").write_text("{}", encoding="utf-8")
    instruments.save(_tiny_universe(), tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["universe.json"]


def test_a_legacy_timestamped_file_is_still_readable(tmp_path) -> None:
    # Before the first sync there is no universe.json, so the newest
    # timestamped file has to keep working.
    legacy = tmp_path / "universe_20260901_100000.json"
    instruments.save(_tiny_universe(), tmp_path)
    current = tmp_path / "universe.json"
    legacy.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
    current.unlink()
    back = instruments.load_latest(tmp_path)
    assert back is not None
    assert [i.symbol for i in back.fo_stocks] == ["RELIANCE"]


def test_no_staging_file_survives_a_successful_save(tmp_path) -> None:
    # The write goes via a temp file and os.replace, because this is now the
    # only copy: a truncated direct write would leave nothing to fall back on.
    instruments.save(_tiny_universe(), tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_an_unreadable_current_file_returns_none_rather_than_raising(tmp_path) -> None:
    (tmp_path / "universe.json").write_text("{not json", encoding="utf-8")
    assert instruments.load_latest(tmp_path) is None


def test_load_latest_is_none_when_nothing_was_ever_saved(tmp_path) -> None:
    assert instruments.load_latest(tmp_path / "empty") is None


def test_scannable_equities_excludes_indices() -> None:
    universe = instruments.Universe(
        captured_at=datetime(2026, 9, 8, 10, 0, tzinfo=IST),
        equities=[instruments.Instrument("RELIANCE", instruments.KIND_EQUITY),
                  instruments.Instrument("NIFTY", instruments.KIND_FO_INDEX)])
    assert [i.symbol for i in universe.scannable_equities] == ["RELIANCE"]


# --- point-in-time discipline --------------------------------------------

def test_truncate_excludes_the_bar_stamped_at_the_cutoff() -> None:
    # A bar is stamped at its START, so the bar labelled 10:00 covers 10:00
    # to 10:05 and its close is the 10:05 price. Keeping it turned every
    # "10:00" entry into a 10:05 entry: an audit measured 16 actionable
    # setups against 12 at a true cutoff, 8 of them manufactured by that one
    # bar and 4 genuine ones deleted.
    frame = _session("2026-09-08", 100.0, bars=20, step=1.0)
    bars = scan_data.BarSet(intraday={"T.NS": frame}, daily={}, requested=1,
                            failed=[])
    cutoff = pd.Timestamp("2026-09-08 10:00", tz=IST)
    kept = scan_data.truncate(bars, cutoff).intraday["T.NS"]
    assert kept.index[-1] == pd.Timestamp("2026-09-08 09:55", tz=IST)
    assert cutoff not in set(kept.index), "the cutoff bar must not survive"
    # Stronger than the two above, which lean on a sorted index that
    # truncate does not itself guarantee.
    assert all(ts < cutoff for ts in kept.index)


def test_truncate_cuts_daily_bars_on_date() -> None:
    # The cutoff date needs a row of its own here. Without one, "<= cutoff
    # date" and "< cutoff date" keep exactly the same rows and the boundary
    # this test is named for goes unmeasured. The cut is strictly before the
    # cutoff date: the replay date's daily bar is already complete when a
    # replay runs, so keeping it would hand the scan the very session it is
    # supposed to be blind to.
    daily = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0],
         "Low": [1.0, 2.0, 3.0], "Close": [1.0, 2.0, 3.0],
         "Volume": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-09-07", tz=IST),
                                pd.Timestamp("2026-09-08", tz=IST),
                                pd.Timestamp("2026-09-09", tz=IST)]))
    bars = scan_data.BarSet(intraday={}, daily={"T.NS": daily}, requested=1,
                            failed=[])
    kept = scan_data.truncate(bars, pd.Timestamp("2026-09-08 10:00", tz=IST))
    surviving = list(kept.daily["T.NS"].index.date)
    assert surviving == [date(2026, 9, 7)]
    assert date(2026, 9, 8) not in surviving, (
        "the cutoff date's own daily bar must not survive")


def test_append_log_rotates_a_file_with_an_older_column_set(tmp_path,
                                                            monkeypatch) -> None:
    # Appending new fields under old column names silently misaligns every
    # row. Adding the 'replayed' column did exactly that, putting True under
    # 'symbol', which corrupts the log the file exists to be.
    import scan_intraday
    path = tmp_path / "scan_log.csv"
    path.write_text("run_date,run_time,symbol\n2026-09-08,10:00:00,HAL\n",
                    encoding="utf-8")
    monkeypatch.setattr(config, "SCAN_LOG_CSV", path)
    scan_intraday.append_log([{name: "" for name in scan_intraday._CSV_FIELDS}])
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == scan_intraday._CSV_FIELDS
    superseded = list(tmp_path.glob("*.superseded*"))
    assert len(superseded) == 1, "the old file must be kept, not overwritten"
    assert "HAL" in superseded[0].read_text(encoding="utf-8")


def test_append_log_appends_when_the_header_already_matches(tmp_path,
                                                            monkeypatch) -> None:
    import scan_intraday
    path = tmp_path / "scan_log.csv"
    monkeypatch.setattr(config, "SCAN_LOG_CSV", path)
    row = {name: "" for name in scan_intraday._CSV_FIELDS}
    scan_intraday.append_log([row])
    scan_intraday.append_log([row])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "one header and two rows"
    assert list(tmp_path.glob("*.superseded*")) == []


def test_parse_as_of_accepts_a_bare_time_and_a_full_timestamp() -> None:
    import scan_intraday
    assert scan_intraday.parse_as_of("2026-09-04 10:00").date() == date(2026, 9, 4)
    assert scan_intraday.parse_as_of("2026-09-04T10:00").hour == 10
    today = scan_intraday.parse_as_of("10:35")
    assert (today.hour, today.minute, today.second) == (10, 35, 0)
    for bad in ("", "nonsense", "25:99", "2026-13-40 10:00"):
        with pytest.raises(ValueError):
            scan_intraday.parse_as_of(bad)


# --- setups.measure ------------------------------------------------------

def _daily(rows: list[tuple], start: str = "2026-09-04") -> pd.DataFrame:
    """Daily bars from (date, open, high, low, close, volume) rows."""
    index = pd.DatetimeIndex([pd.Timestamp(d, tz=IST) for d, *_ in rows])
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": low, "Close": c, "Volume": v}
         for _, o, h, low, c, v in rows], index=index)


def test_measure_excludes_todays_partial_daily_bar() -> None:
    intraday = _session("2026-09-08", 100.0, bars=20, volume=1000.0)
    daily = _daily([
        ("2026-09-07", 90.0, 95.0, 88.0, 94.0, 1_000_000.0),   # previous session
        ("2026-09-08", 100.0, 119.0, 99.0, 118.0, 500_000.0),  # today, partial
    ])
    reading = setups.measure("T", "T.NS", intraday, daily, 0.5,
                             datetime(2026, 9, 8, 13, 0, tzinfo=IST))
    assert reading is not None
    # 94.0 is yesterday's close. Using today's 118.0 would make day_change
    # negative on a rising session and corrupt relative strength with it.
    assert reading.prev_close == pytest.approx(94.0)
    assert reading.day_change_pct is not None and reading.day_change_pct > 0.0
    assert reading.cpr is not None
    # CPR must come from yesterday's 95/88/94, not today's 119/99/118.
    assert reading.cpr.pivot == pytest.approx((95.0 + 88.0 + 94.0) / 3.0)


def test_measure_anchors_prev_close_on_the_clock_not_the_last_bar() -> None:
    # The cutoff falls before the replay date's first intraday bar, so the
    # last surviving session is D-1 while the clock says D. Anchoring the
    # daily filter on that last bar instead of on the clock kept D's own
    # completed daily bar, and its close became prev_close: a replay of
    # yesterday reading today's finished session. prev_close must be D-1's
    # 174.0, not D's 274.0, and day_change_pct and relative strength with it.
    intraday = _session("2026-09-07", 100.0, bars=20)
    daily = _daily([("2026-09-07", 170.0, 175.0, 168.0, 174.0, 1e6),
                    ("2026-09-08", 180.0, 285.0, 179.0, 274.0, 1e6)])
    reading = setups.measure("T", "T.NS", intraday, daily, None,
                             datetime(2026, 9, 8, 9, 15, tzinfo=IST))
    assert reading is not None
    assert reading.prev_close == pytest.approx(174.0)
    # The CPR shares the anchor, so it must come from 175/168/174 as well.
    assert reading.cpr is not None
    assert reading.cpr.pivot == pytest.approx((175.0 + 168.0 + 174.0) / 3.0)


def test_measure_drops_daily_context_rather_than_leaking_it() -> None:
    # A non-timestamped daily index used to be swallowed, leaving the frame
    # unfiltered so prev_close became today's own partial close.
    intraday = _session("2026-09-08", 100.0, bars=20)
    daily = _daily([("2026-09-07", 90.0, 95.0, 88.0, 94.0, 1e6),
                    ("2026-09-08", 100.0, 119.0, 99.0, 118.0, 5e5)])
    daily.index = [0, 1]
    reading = setups.measure("T", "T.NS", intraday, daily, 0.5,
                             datetime(2026, 9, 8, 13, 0, tzinfo=IST))
    assert reading is not None
    assert reading.prev_close is None, "must fail closed, not leak 118.0"
    assert reading.day_change_pct is None
    assert reading.cpr is None


def test_measure_reports_the_session_extremes_and_time_left() -> None:
    intraday = _session("2026-09-08", 100.0, bars=20, step=1.0)
    reading = setups.measure("T", "T.NS", intraday, None, None,
                             datetime(2026, 9, 8, 13, 0, tzinfo=IST))
    assert reading is not None
    # _session ascends by 1.0 from 100.0 with a +/-0.5 span on each bar.
    assert reading.day_low == pytest.approx(99.5)
    assert reading.day_high == pytest.approx(119.5)
    assert reading.minutes_left == 150
    assert reading.bars_left > 0
    assert reading.turnover_20d is None, "no daily frame means unknown depth"


def test_measure_returns_none_without_a_usable_session() -> None:
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    assert setups.measure("T", "T.NS", empty, None, None,
                          datetime(2026, 9, 8, 13, 0, tzinfo=IST)) is None


# --- when the intraday scan may re-run itself -----------------------------
#
# The auto-refresh is a real re-scan, not a repaint, so these rules decide
# whether a timer is worth its cost. Every one of them returns a REASON
# too: a toggle the user switched on that then quietly does nothing is
# worse than having no toggle.

def _refresh(**over):
    """auto_refresh_interval with a healthy live session as the baseline."""
    kwargs = dict(enabled=True, requested=60, replaying=False,
                  want_options=False, in_trading_hours=True,
                  feed_running=True, feed_age=120.0, last_scan_seconds=0.0)
    kwargs.update(over)
    return scan_data.auto_refresh_interval(**kwargs)


def test_a_healthy_live_session_refreshes_at_the_requested_interval() -> None:
    interval, note = _refresh()
    assert interval == 60
    assert note == ""


def test_the_toggle_being_off_is_silent() -> None:
    # Off by choice needs no explanation; every other refusal does.
    interval, note = _refresh(enabled=False)
    assert interval is None and note == ""


def test_a_replay_is_never_refreshed() -> None:
    interval, note = _refresh(replaying=True)
    assert interval is None
    assert "past" in note


def test_option_contracts_block_the_timer() -> None:
    # One NSE chain request per setup, repeated forever, is rude to a third
    # party rather than merely slow.
    interval, note = _refresh(want_options=True)
    assert interval is None
    assert "NSE" in note


def test_nothing_refreshes_outside_trading_hours() -> None:
    interval, note = _refresh(in_trading_hours=False)
    assert interval is None
    assert "closed" in note


def test_a_stopped_feed_blocks_the_timer() -> None:
    # Without the feed each refresh re-downloads 216 symbols at 3 a second.
    interval, note = _refresh(feed_running=False)
    assert interval is None
    assert "not running" in note


def test_a_stale_feed_blocks_the_timer() -> None:
    interval, note = _refresh(feed_age=config.SCAN_LIVE_MAX_AGE_SECONDS + 1)
    assert interval is None
    assert "too old" in note


def test_an_unknowable_feed_age_blocks_the_timer() -> None:
    interval, note = _refresh(feed_age=float("nan"))
    assert interval is None
    assert "too old" in note


def test_a_feed_at_exactly_the_age_limit_still_refreshes() -> None:
    # The limit is what a HEALTHY feed peaks at, so the boundary must pass
    # or the timer stops for the same reason the live path used to.
    interval, _ = _refresh(feed_age=float(config.SCAN_LIVE_MAX_AGE_SECONDS))
    assert interval == 60


def test_the_interval_is_raised_above_the_measured_scan_time() -> None:
    # A 30s timer over a 22s scan leaves the page permanently mid-scan,
    # with reruns queueing behind each other.
    interval, note = _refresh(requested=30, last_scan_seconds=22.0)
    assert interval == 44
    assert "raised to 44s" in note


def test_a_fast_scan_does_not_raise_the_interval() -> None:
    interval, note = _refresh(requested=60, last_scan_seconds=5.0)
    assert interval == 60
    assert note == ""


# --- the pivot ladder ----------------------------------------------------
#
# Placement only. The ladder enters no gate and no score, so these tests
# are about the arithmetic being the standard construction and about the
# stop preferring a real level only when it sits inside the volatility
# window - the constraint that a previous attempt at structural stops
# violated, making every clean breakout unactionable.

def test_the_pivot_ladder_is_the_standard_construction() -> None:
    # H 100, L 90, C 98 -> P 96, and each level by the textbook formula.
    lad = indicators.pivot_ladder(100.0, 90.0, 98.0)
    assert lad is not None
    assert lad.pivot == pytest.approx(96.0)
    assert lad.r1 == pytest.approx(102.0)     # 2P - L
    assert lad.s1 == pytest.approx(92.0)      # 2P - H
    assert lad.r2 == pytest.approx(106.0)     # P + (H - L)
    assert lad.s2 == pytest.approx(86.0)      # P - (H - L)
    assert lad.r3 == pytest.approx(112.0)     # H + 2(P - L)
    assert lad.s3 == pytest.approx(82.0)      # L - 2(H - P)


def test_the_levels_are_ordered_outward_from_the_pivot() -> None:
    lad = indicators.pivot_ladder(100.0, 90.0, 98.0)
    assert lad.s3 < lad.s2 < lad.s1 < lad.pivot < lad.r1 < lad.r2 < lad.r3


def test_unusable_previous_sessions_give_no_ladder() -> None:
    # A ladder from a zero low puts S3 at a negative price, which would
    # then be the nearest "level" beneath everything.
    assert indicators.pivot_ladder(0.0, 0.0, 0.0) is None
    assert indicators.pivot_ladder(100.0, 90.0, 0.0) is None
    assert indicators.pivot_ladder(90.0, 100.0, 95.0) is None   # high < low
    assert indicators.pivot_ladder(None, 90.0, 95.0) is None


def test_below_and_above_walk_the_whole_ladder_not_just_one_side() -> None:
    # After a strong move up, R1 and even R2 sit BELOW the price and are
    # then the nearest real structure beneath it. Treating only S1-S3 as
    # support would reach past them to something far away.
    lad = indicators.pivot_ladder(100.0, 90.0, 98.0)
    assert lad.below(104.0)[0] == pytest.approx(102.0)   # R1, not S1
    assert lad.below(99.0)[0] == pytest.approx(96.0)     # the pivot itself
    assert lad.above(99.0)[0] == pytest.approx(102.0)
    # Nearest first, in both directions.
    assert list(lad.below(104.0)) == sorted(lad.below(104.0), reverse=True)
    assert list(lad.above(80.0)) == sorted(lad.above(80.0))


def test_a_price_outside_the_ladder_has_nothing_on_one_side() -> None:
    lad = indicators.pivot_ladder(100.0, 90.0, 98.0)
    assert lad.above(999.0) == ()
    assert lad.below(1.0) == ()


def test_the_ladder_is_not_used_for_stop_placement() -> None:
    # A DECISION, recorded as a test. Placing stops on pivots was built
    # and then measured - `python -m pivot_measurement` reproduces it.
    # Conditioning on the session, Mantel-Haenszel over 8,687 sessions
    # gives an odds ratio of 1.100 (p 0.616), and above 1.0 means the
    # pivot stop is hit MORE often. An unpaired reading of -2.19 pp was
    # selection, not signal. Since a pivot stop is always tighter than
    # the volatility stop it replaces, shipping it would have raised the
    # stop-hit rate for an effect no test can find. This fails if anyone
    # wires it back in.
    import inspect

    assert "pivots" not in inspect.signature(
        levels_mod.build_levels).parameters
    assert "pivots" not in inspect.signature(
        levels_mod._structural_levels).parameters


def test_a_qualifying_pivot_is_still_not_used_as_the_stop() -> None:
    # This test has to prove the level would REALLY have qualified,
    # otherwise "the stop is the volatility one" is true for the boring
    # reason that no level was in range. So the window is computed the way
    # build_levels computes it and the pivot's gap is checked to fall
    # inside it. What the test then shows is that a level which passes
    # every structural criterion is nonetheless not used - because the
    # measurement said it should not be, and because build_levels has no
    # parameter through which it could see the ladder at all.
    entry, atr, bars = 100.0, 0.6, 20
    lad = indicators.pivot_ladder(101.0, 97.0, 98.4)      # pivot at 98.80
    assert lad is not None and lad.pivot == pytest.approx(98.80)
    sigma = levels_mod.expected_remaining_range(atr, bars)
    base = sigma * config.SCAN_STOP_FRACTION
    gap = entry - lad.pivot
    # The level really is inside the acceptance window.
    assert (1.0 - config.SCAN_STRUCTURE_BAND) * base <= gap <= base, (
        gap, base)
    trade = levels_mod.build_levels("LONG", entry=entry, atr_per_bar=atr,
                                    bars_left=bars)
    assert trade is not None
    assert trade.stop_source == "volatility"
    assert trade.stop != pytest.approx(lad.pivot)


def test_intraday_structure_is_still_preferred_over_volatility() -> None:
    # The pre-existing structural stop must be untouched by all this. Its
    # own rationale is NOT what the pivot measurement tested, so it stays
    # exactly as it was.
    trade = levels_mod.build_levels("LONG", entry=100.0, atr_per_bar=0.6,
                                    bars_left=20, opening_low=98.75)
    assert trade is not None
    assert trade.stop_source == "opening-range low"
    assert trade.stop == pytest.approx(98.75)


def test_the_scan_does_not_compute_a_ladder_it_never_shows() -> None:
    # It was on Readings for a while, built for every symbol on every scan
    # and read by nothing - the display path builds its own from the daily
    # store. Recomputing it 216 times a scan to throw it away is the kind
    # of thing that survives only because no test objects.
    assert "pivots" not in setups.Readings.__dataclass_fields__


def test_the_displayed_ladder_excludes_todays_partial_bar() -> None:
    # The UI calls these "yesterday's" levels. Today's daily bar is still
    # forming, so a ladder built from it drifts through the session and
    # the label is false - which it was, because the daily store does
    # carry today's bar.
    import instrument_report

    index = pd.DatetimeIndex([
        pd.Timestamp("2026-09-08 00:00", tz=IST),
        pd.Timestamp(datetime.now(IST).date(), tz=IST),
    ])
    frame = pd.DataFrame(
        {"High": [110.0, 500.0], "Low": [90.0, 400.0],
         "Close": [100.0, 450.0]}, index=index)
    ladder = instrument_report._previous_session_pivots(frame)
    assert ladder is not None
    # Built from 8 Sep (H110 L90 C100), so the pivot is 100 - not anything
    # derived from today's 400-500 range.
    assert ladder.pivot == pytest.approx(100.0)


def test_no_complete_session_means_no_levels_rather_than_todays() -> None:
    import instrument_report

    index = pd.DatetimeIndex([pd.Timestamp(datetime.now(IST).date(), tz=IST)])
    frame = pd.DataFrame({"High": [110.0], "Low": [90.0], "Close": [100.0]},
                         index=index)
    assert instrument_report._previous_session_pivots(frame) is None


def test_a_ladder_whose_lower_rungs_go_negative_is_refused() -> None:
    # A range wide relative to its own price drives S3 below zero:
    # (100, 1, 1) gave S3 = -131, and the display would have shown it as a
    # level at -485%. The docstring already claimed this was refused.
    assert indicators.pivot_ladder(100.0, 1.0, 1.0) is None


def test_a_close_outside_the_previous_range_is_refused() -> None:
    # With no low <= close <= high check the rungs interleave:
    # (100, 90, 500) put S1 at 360 and R2 at 240, so "supports" sat above
    # "resistances" and the documented ordering was violated. The guards
    # were copied from central_pivot_range, which is order-insensitive
    # because it takes a max and a min; a seven-rung ladder is not.
    assert indicators.pivot_ladder(100.0, 90.0, 500.0) is None
    assert indicators.pivot_ladder(100.0, 90.0, 50.0) is None
    # A close exactly on either bound is still valid.
    assert indicators.pivot_ladder(100.0, 90.0, 100.0) is not None
    assert indicators.pivot_ladder(100.0, 90.0, 90.0) is not None


# --- when the first scan may run itself ----------------------------------
#
# Requiring a button press meant the intraday table was simply absent for
# anyone opening the tab mid-session, and a Streamlit restart clears
# session_state so a scan run before it is gone too. The FIRST scan now
# runs itself - but only when the result will be current and cheap, since
# a scan is a full sweep of the universe.

def _blocked(monkeypatch, feed=None, in_hours=True, as_of=None,
             want_options=False, scope=None):
    """app.autoscan_blocked with a healthy live session as the baseline."""
    import app

    state = {"running": True, "bars": 12, "age_seconds": 120.0}
    state.update(feed or {})
    monkeypatch.setattr(app, "live_feed_state", lambda: state)
    monkeypatch.setattr(app, "live_bars_in_hours", lambda: in_hours)
    if scope is None:
        scope = "F&O single stocks (fastest)"
    return app.autoscan_blocked(as_of, want_options, scope)


def test_a_healthy_live_session_may_scan_itself(monkeypatch) -> None:
    assert _blocked(monkeypatch) == ""


def test_a_replay_is_never_started_for_you(monkeypatch) -> None:
    why = _blocked(monkeypatch, as_of=datetime(2026, 9, 9, 10, 0))
    assert "replay" in why


def test_option_chains_are_never_fetched_automatically(monkeypatch) -> None:
    # One NSE request per setup. Triggering that unasked is rude to a
    # third party, not merely slow.
    assert "NSE" in _blocked(monkeypatch, want_options=True)


def test_nothing_scans_itself_outside_market_hours(monkeypatch) -> None:
    assert "closed" in _blocked(monkeypatch, in_hours=False)


def test_a_stopped_feed_blocks_the_automatic_scan(monkeypatch) -> None:
    # Without the feed a scan downloads 216 symbols at 3 a second, which
    # is not something to start on someone's behalf.
    assert "not running" in _blocked(monkeypatch, feed={"running": False})


def test_a_feed_with_no_completed_bar_yet_blocks_it(monkeypatch) -> None:
    # Right after the open the first bucket has not closed, so there is
    # nothing to scan from.
    assert "completed bar" in _blocked(monkeypatch, feed={"bars": 0})


def test_a_stale_feed_blocks_the_automatic_scan(monkeypatch) -> None:
    why = _blocked(monkeypatch,
                   feed={"age_seconds": config.SCAN_LIVE_MAX_AGE_SECONDS + 1})
    assert "too old" in why


def test_an_unknowable_bar_age_blocks_the_automatic_scan(monkeypatch) -> None:
    assert "too old" in _blocked(monkeypatch,
                                 feed={"age_seconds": float("nan")})


def test_a_wide_scope_is_never_scanned_automatically(monkeypatch) -> None:
    # "All listed equities" is now the DEFAULT in the dropdown, and the
    # first scan runs itself. Those two together would start a ~13 minute
    # download of bars for the ~2,350 symbols the live feed does not carry,
    # just because someone opened the tab.
    import app

    why = _blocked(monkeypatch,
                   scope="All listed equities (~2,570 - downloads the illiquid tail)")
    assert "download" in why


def test_the_feed_covered_scopes_still_scan_themselves(monkeypatch) -> None:
    # The feed streams every F&O underlying plus the ~1,000 most traded,
    # so these cost at most a handful of downloads.
    import app

    for scope in app._AUTOSCAN_SCOPES:
        assert _blocked(monkeypatch, scope=scope) == "", scope


def test_the_default_scope_is_all_listed_equities() -> None:
    # The selectbox has no explicit index, so the FIRST key is the default.
    import app

    assert list(app._SCAN_SCOPES)[0].startswith("All listed equities")


def test_every_autoscan_scope_name_matches_a_real_option() -> None:
    # A typo here would silently disable the automatic scan, since nothing
    # would ever be a member of _AUTOSCAN_SCOPES.
    import app

    assert app._AUTOSCAN_SCOPES
    for scope in app._AUTOSCAN_SCOPES:
        assert scope in app._SCAN_SCOPES, scope


def test_the_widest_scope_is_not_an_autoscan_scope() -> None:
    # It reaches ~1,580 symbols the feed does not carry, at 3 req/s.
    import app

    assert list(app._SCAN_SCOPES)[0] not in app._AUTOSCAN_SCOPES


def test_the_cached_barset_is_never_mutated_in_place() -> None:
    # THE ASSUMPTION app.load_scan_bars RESTS ON. It uses st.cache_resource
    # rather than st.cache_data, because pickling a BarSet holding
    # thousands of DataFrames raised UnserializableReturnValueError in
    # production and took the tab down mid-refresh. cache_resource hands
    # every session the SAME object, so anything editing one in place would
    # corrupt every other reader.
    import copy

    index = pd.DatetimeIndex([pd.Timestamp("2026-09-10 09:15", tz=IST),
                              pd.Timestamp("2026-09-10 09:20", tz=IST)])
    frame = pd.DataFrame({c: [1.0, 2.0] for c in
                          ("Open", "High", "Low", "Close", "Volume")},
                         index=index)
    bars = scan_data.BarSet(intraday={"A": frame, "B": frame},
                            daily={"A": frame, "B": frame},
                            requested=2, failed=[], source="live feed",
                            live_symbols=2, live_age_seconds=60.0)
    before = copy.deepcopy(bars.intraday), copy.deepcopy(bars.failed)

    cut = scan_data.truncate(bars, pd.Timestamp("2026-09-10 09:18", tz=IST))
    # truncate must have built a NEW BarSet, leaving the original intact.
    assert cut is not bars
    assert cut.intraday is not bars.intraday
    assert set(bars.intraday) == set(before[0])
    assert len(bars.intraday["A"]) == 2, "the source frame was truncated"
    assert bars.failed == before[1], "the source failed list was extended"
    # And the truncation really did happen on the copy.
    assert len(cut.intraday["A"]) == 1


def test_carrying_the_source_forward_does_not_edit_the_original() -> None:
    frame = pd.DataFrame({c: [1.0] for c in
                          ("Open", "High", "Low", "Close", "Volume")},
                         index=pd.DatetimeIndex(
                             [pd.Timestamp("2026-09-10 09:15", tz=IST)]))
    bars = scan_data.BarSet(intraday={"A": frame}, daily={"A": frame},
                            requested=1, failed=[], source="live feed",
                            live_symbols=1, live_age_seconds=30.0)
    rebuilt = scan_data._carry_source(bars, {}, {}, failed=["A"])
    assert rebuilt is not bars
    assert bars.failed == [], "the original failed list was mutated"
    assert bars.intraday, "the original intraday dict was emptied"
    # The provenance fields must survive the rebuild.
    assert rebuilt.source == "live feed"
    assert rebuilt.live_symbols == 1
    assert rebuilt.live_age_seconds == 30.0
