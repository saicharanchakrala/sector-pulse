"""The per-trade risk budget: charges inside it, a ceiling on it, options held to it.

Run with: .venv\\Scripts\\python -m pytest test_risk_budget.py -q

THREE DEFECTS THIS PINS.

  1. Charges were priced AFTER sizing, so a stop-out lost the budget plus
     its round trip - a median of about 1,100 against a stated 1,000 in
     the 2026-09-22/23 scan logs. The sizer now solves
     quantity * risk_per_share + cost(quantity) <= budget.
  2. The risk percent had no upper bound, and NaN walked through every
     `<= 0.0` guard into int(budget // nan). Above SCAN_MAX_RISK_PCT, or
     on NaN, build_levels now returns None.
  3. An option was always suggested one lot at a time, whatever that lot
     cost against the budget. pick_contract now sizes lots so the whole
     premium plus charges fits, and suggests nothing when one lot does not.
"""
from __future__ import annotations

import math
import random

import pytest

import config
import indicators
import levels
import options_chain
import scan_intraday
import setups
import trade_costs
from levels import LONG, SHORT

NAN = float("nan")


def _cost(entry: float, quantity: int) -> float:
    """The round trip the sizer prices, at a flat exit."""
    return trade_costs.equity_intraday_cost(entry, entry, quantity).total


# --- charges inside the budget ---------------------------------------------

def test_the_whole_loss_at_the_stop_fits_the_budget_on_every_build() -> None:
    """The invariant, over a wide random space - and it is the LARGEST fit.

    Every uncapped size is maximal as well as safe: one more share would
    overshoot. Without the second half a sizer that returned 1 share
    everywhere would pass.
    """
    random.seed(24)
    checked = uncapped = 0
    for _ in range(3_000):
        capital = random.choice([25_000.0, 100_000.0, 500_000.0, 2e6])
        risk_pct = random.choice([0.25, 0.5, 1.0, 1.5, 2.0])
        entry = random.uniform(20.0, 12_000.0)
        trade = levels.build_levels(
            random.choice([LONG, SHORT]), entry,
            entry * random.uniform(0.0003, 0.01), random.randint(5, 110),
            capital=capital, risk_pct=risk_pct)
        if trade is None:
            continue
        checked += 1
        budget = capital * risk_pct / 100.0
        rps = trade.risk_per_share
        assert trade.cost_rupees == pytest.approx(_cost(entry, trade.quantity))
        assert trade.lot_risk + trade.cost_rupees <= budget, trade
        assert trade.risk_with_costs <= budget
        if not trade.capital_capped:
            uncapped += 1
            more = trade.quantity + 1
            assert more * rps + _cost(entry, more) > budget, trade
    assert checked > 2_000, checked
    assert uncapped > 1_000, uncapped


def test_lot_risk_stays_price_risk_and_the_new_field_adds_the_charges() -> None:
    trade = levels.build_levels(LONG, 500.0, 1.2, 60)
    assert trade is not None
    assert trade.lot_risk == pytest.approx(trade.risk_per_share
                                           * trade.quantity)
    assert trade.risk_with_costs == pytest.approx(trade.lot_risk
                                                  + trade.cost_rupees)


def test_charges_take_shares_away_relative_to_the_old_rule() -> None:
    """At the defaults the old size overshot; the new one does not."""
    trade = levels.build_levels(LONG, 500.0, 1.2, 60)
    assert trade is not None
    budget = levels.risk_budget(config.SCAN_CAPITAL,
                                config.SCAN_RISK_PCT_PER_TRADE)
    old = int(budget // trade.risk_per_share)
    assert trade.quantity < old
    assert old * trade.risk_per_share + _cost(500.0, old) > budget


def test_charges_can_refuse_the_last_share() -> None:
    """A 990-rupee stop fits one share on price, and not once charged.

    20,000 a share: 0.03% brokerage is 6 a leg, and with STT, slippage and
    the rest one share costs about 29 to round-trip, so 990 + 29 > 1,000.
    """
    atr = 990.0 / (config.SCAN_STOP_FRACTION * 5.0)     # sqrt(25) bars
    assert int(1_000.0 // 990.0) == 1
    assert 990.0 + _cost(20_000.0, 1) > 1_000.0
    assert levels.build_levels(LONG, 20_000.0, atr, 25) is None


def test_the_sizer_refuses_rather_than_guessing_when_nothing_fits() -> None:
    assert levels._quantity_inside_budget(100.0, 2.5, 1_000.0, 0) == 0
    assert levels._quantity_inside_budget(100.0, 2_000.0, 1_000.0, 1) == 0


# --- the ceiling on risk per trade ------------------------------------------

def test_a_twenty_five_percent_risk_is_refused() -> None:
    """The typo this exists for: 25 meant as 2.5 would size at a quarter."""
    assert levels.build_levels(LONG, 100.0, 1.0, 25, risk_pct=25.0) is None
    assert levels.risk_budget(100_000.0, 25.0) == 0.0


def test_the_ceiling_itself_is_allowed_and_a_hair_above_is_not() -> None:
    top = config.SCAN_MAX_RISK_PCT
    assert top == pytest.approx(2.0)
    assert levels.build_levels(LONG, 100.0, 1.0, 25, risk_pct=top) is not None
    assert levels.build_levels(LONG, 100.0, 1.0, 25,
                               risk_pct=top + 1e-6) is None


def test_the_ceiling_is_read_at_call_time(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MAX_RISK_PCT", 0.5)
    assert levels.build_levels(LONG, 100.0, 1.0, 25, risk_pct=1.0) is None
    assert levels.risk_budget(100_000.0, 1.0) == 0.0


def test_the_cli_refuses_a_risk_percent_above_the_ceiling() -> None:
    with pytest.raises(SystemExit):
        scan_intraday._parse_args(["--risk-pct", "2.5"])


@pytest.mark.parametrize("flag,value", [
    ("--risk-pct", "nan"), ("--capital", "nan"), ("--capital", "inf"),
])
def test_the_cli_refuses_non_numbers_that_float_accepts(flag, value) -> None:
    with pytest.raises(SystemExit):
        scan_intraday._parse_args([flag, value])


def test_the_cli_accepts_the_ceiling_itself() -> None:
    args = scan_intraday._parse_args(
        ["--risk-pct", str(config.SCAN_MAX_RISK_PCT)])
    assert args.risk_pct == pytest.approx(config.SCAN_MAX_RISK_PCT)


# --- NaN and infinity return None instead of raising ------------------------

@pytest.mark.parametrize("override", [
    {"atr_per_bar": NAN}, {"entry": NAN}, {"capital": NAN},
    {"risk_pct": NAN}, {"reward_risk": NAN}, {"leverage": NAN},
    {"stop_fraction": NAN}, {"band": NAN}, {"bars_left": NAN},
    {"atr_per_bar": math.inf}, {"entry": math.inf}, {"capital": math.inf},
])
def test_non_finite_inputs_return_none_rather_than_raising(override) -> None:
    kwargs = {"direction": LONG, "entry": 100.0, "atr_per_bar": 1.0,
              "bars_left": 25}
    kwargs.update(override)
    assert levels.build_levels(**kwargs) is None


def test_a_nan_range_is_no_range() -> None:
    assert levels.expected_remaining_range(NAN, 25) == 0.0


def _directional_reading(**kwargs) -> setups.Readings:
    """A LONG-agreeing reading: above VWAP and above the opening range."""
    orb = indicators.OpeningRange(high=101.0, low=99.0, bars=3, minutes=15)
    defaults = {
        "symbol": "TEST", "ticker": "TEST.NS", "last": 102.0, "prev_close": 100.0,
        "day_change_pct": 2.0, "vwap": 100.5, "vwap_distance_pct": 1.49,
        "opening_range": orb, "cpr": None, "atr_bar": 1.0, "rvol": 2.0,
        "relative_strength": 1.5, "turnover_20d": 1e9, "oi_change_pct": 5.0,
        "futures_share": 0.85, "derivatives_turnover": 1e9, "day_high": 102.5,
        "day_low": 98.8, "minutes_left": 150, "bars_left": 25}
    defaults.update(kwargs)
    return setups.Readings(**defaults)


def test_a_nan_atr_blocks_the_setup_instead_of_raising() -> None:
    setup = setups.evaluate(_directional_reading(atr_bar=NAN))
    assert setup.direction == LONG
    assert setup.levels is None and not setup.actionable
    failed = [r for r in setup.reasons if "could not build levels" in r]
    assert failed and "[FAIL]" in failed[0]
    assert "ceiling" not in failed[0], "the ceiling was not the cause"


def test_a_nan_atr_has_no_approach_distance() -> None:
    inside = _directional_reading(last=100.5, vwap=100.0, atr_bar=NAN)
    assert setups.approach(inside) is None


def test_a_risk_percent_above_the_ceiling_blocks_the_setup_and_says_why() -> None:
    setup = setups.evaluate(_directional_reading(), risk_pct=25.0)
    assert not setup.actionable
    assert any("25% risk per trade is above the" in r and "[FAIL]" in r
               for r in setup.reasons), setup.reasons


# --- options held to the same budget ----------------------------------------

def _contract(**kwargs) -> options_chain.Contract:
    """A liquid contract: 1% spread, deep open interest and volume."""
    defaults = {"symbol": "X", "expiry": "29-Sep-2026", "strike": 100.0,
                "side": options_chain.CALL, "last_price": 2.0, "bid": 1.99,
                "ask": 2.01, "open_interest": 5_000, "oi_change": 100,
                "volume": 2_000, "implied_volatility": 18.0}
    defaults.update(kwargs)
    return options_chain.Contract(**defaults)


def test_lots_are_derived_from_the_budget() -> None:
    """2.00 premium, lot 100: 200 a lot plus charges, against 1,000.

    By hand: charges are a flat 47.20 (two 20-rupee orders plus GST) and
    about 2.37 a lot on premium turnover (slippage 0.5% a leg dominates),
    so four lots lose about 857 and five about 1,059.
    """
    assert options_chain.lots_within_budget(2.0, 100, 1_000.0) == 4
    assert options_chain.option_loss(2.0, 4, 100) == pytest.approx(857.0,
                                                                   abs=2.0)
    assert options_chain.option_loss(2.0, 4, 100) <= 1_000.0
    assert options_chain.option_loss(2.0, 5, 100) > 1_000.0


def test_the_worst_case_is_the_whole_premium_plus_charges() -> None:
    charges = trade_costs.options_cost(2.0, 2.0, 3, 100).total
    assert options_chain.option_loss(2.0, 3, 100) == pytest.approx(
        2.0 * 300 + charges)


def test_pick_contract_sizes_the_position_against_the_budget() -> None:
    pick = options_chain.pick_contract([_contract()], 100.0, LONG, 100)
    assert pick.contract is not None
    assert pick.lots == 4 and pick.quantity == 400
    assert pick.budget == pytest.approx(
        config.SCAN_CAPITAL * config.SCAN_RISK_PCT_PER_TRADE / 100.0)
    assert pick.total_at_risk <= pick.budget
    assert pick.premium_outlay == pytest.approx(2.0 * 400)
    assert pick.charges == pytest.approx(pick.total_at_risk
                                         - pick.premium_outlay)
    assert any("4 lot(s)" in r and "[PASS]" in r for r in pick.reasons)


def test_a_bigger_budget_buys_more_lots() -> None:
    small = options_chain.pick_contract([_contract()], 100.0, LONG, 100,
                                        capital=100_000.0, risk_pct=1.0)
    large = options_chain.pick_contract([_contract()], 100.0, LONG, 100,
                                        capital=100_000.0, risk_pct=2.0)
    assert large.lots > small.lots
    assert large.total_at_risk <= 2_000.0


def test_zero_lots_returns_no_contract_and_says_why() -> None:
    """10.00 x 500 is 5,000 of premium against a 1,000 budget."""
    expensive = _contract(last_price=10.0, bid=9.95, ask=10.05)
    pick = options_chain.pick_contract([expensive], 100.0, LONG, 500)
    assert pick.contract is None and pick.lots == 0
    assert pick.refused_by_budget
    assert pick.total_at_risk == 0.0
    headline = pick.reasons[0]
    assert "risk budget refuses" in headline
    assert "5,000" in headline and "1,000 budget" in headline


def test_the_budget_does_not_walk_out_to_a_cheaper_strike() -> None:
    """A far out-of-the-money option fits BECAUSE it is unlikely to pay."""
    at_money = _contract(strike=100.0, last_price=10.0, bid=9.95, ask=10.05)
    lottery = _contract(strike=110.0, last_price=0.5, bid=0.498, ask=0.502)
    # The cheaper strike must be a REAL alternative - liquid, and small
    # enough to fit - or refusing it would prove nothing about walking.
    assert options_chain.quality_reasons(lottery, 1, 500)[0]
    assert options_chain.lots_within_budget(lottery.mid, 500, 1_000.0) >= 1
    pick = options_chain.pick_contract([at_money, lottery], 100.0, LONG, 500)
    assert pick.contract is None and pick.refused_by_budget


def test_a_risk_percent_above_the_ceiling_refuses_every_contract() -> None:
    pick = options_chain.pick_contract([_contract()], 100.0, LONG, 100,
                                       risk_pct=5.0)
    assert pick.contract is None and pick.refused_by_budget


def test_an_illiquid_chain_is_not_reported_as_a_budget_refusal() -> None:
    thin = _contract(open_interest=1, volume=1)
    pick = options_chain.pick_contract([thin], 100.0, LONG, 100)
    assert pick.contract is None and not pick.refused_by_budget
    assert "no contract cleared the quality gates" in pick.reasons[0]


def test_a_short_view_buys_a_put_sized_the_same_way() -> None:
    put = _contract(side=options_chain.PUT)
    pick = options_chain.pick_contract([put], 100.0, SHORT, 100)
    assert pick.contract is put and pick.lots == 4


def _print_options_header(monkeypatch, capsys, **kwargs) -> str:
    """print_options up to its first chain, with no network touched."""
    monkeypatch.setattr(options_chain, "open_session", lambda: None)
    monkeypatch.setattr(options_chain, "fetch_chain",
                        lambda *args, **kw: ([], None, ""))
    setup = setups.evaluate(_directional_reading())
    scan_intraday.print_options([setup], {"TEST": 50}, **kwargs)
    return " ".join(capsys.readouterr().out.split())


def test_the_cli_option_budget_line_comes_from_risk_budget(
        monkeypatch, capsys) -> None:
    """The printed budget is levels.risk_budget's, not a second formula."""
    monkeypatch.setattr(levels, "risk_budget", lambda capital, pct: 1_234.0)
    out = _print_options_header(monkeypatch, capsys)
    assert "fits the 1,234-rupee per-trade budget" in out


def test_the_cli_option_budget_line_says_when_the_budget_is_refused(
        monkeypatch, capsys) -> None:
    out = _print_options_header(monkeypatch, capsys, risk_pct=5.0)
    assert "No per-trade budget" in out
    assert "0-rupee" not in out


def test_the_lot_sizer_refuses_nonsense() -> None:
    assert options_chain.lots_within_budget(NAN, 100, 1_000.0) == 0
    assert options_chain.lots_within_budget(2.0, 0, 1_000.0) == 0
    assert options_chain.lots_within_budget(2.0, 100, 0.0) == 0
