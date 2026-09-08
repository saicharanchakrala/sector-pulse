"""Tests for the target-weight rebalancing maths, loaders, and cadence gates.

Runs under pytest, or standalone with `python test_rebalance.py`.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import config

from allocator import (OVERSHOOT_FRACTION, actionable_drift, compute_rows,
                       fundable_units, max_abs_drift, plan_orders,
                       untracked_value)
from holdings import (HoldingsError, TargetsError, parse_number, load_holdings,
                      load_targets)
from import_tradebook import build_positions, read_trades, write_holdings
from portfolio_models import Holding, Order, Plan
from schedule_rules import (add_months, evaluate_cadence,
                            load_last_contribution, record_contribution)

EQUAL_PAIR = [
    Holding(symbol="A", quantity=10, avg_cost=90.0, last_price=100.0),
    Holding(symbol="B", quantity=10, avg_cost=110.0, last_price=100.0),
]


def _raises(exc_type: type[BaseException], func: Callable[..., object],
            *args: object, **kwargs: object) -> None:
    """Assert that calling func raises exc_type."""
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} from {func.__name__}")


def _write(directory: Path, name: str, text: str) -> Path:
    """Write a temp file and return its path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _assert_no_overshoot(rows: list, orders: list[Order]) -> None:
    """Assert every order stays inside its deficit plus half a unit price."""
    by_symbol = {row.symbol: row for row in rows}
    for order in orders:
        bound = (by_symbol[order.symbol].deficit
                 + order.last_price * OVERSHOOT_FRACTION)
        assert order.amount <= bound, (
            f"{order.symbol} spent {order.amount} against a bound of {bound}")


# --- compute_rows -------------------------------------------------------

def test_compute_rows_weights_and_deficits() -> None:
    rows = {row.symbol: row for row in
            compute_rows(EQUAL_PAIR, {"A": 50.0, "B": 50.0}, cash=1000.0)}
    assert rows["A"].value == 1000.0
    assert rows["A"].actual_weight == 50.0
    assert rows["A"].drift_pp == 0.0
    # Target is measured against the post-contribution total of 3000.
    assert rows["A"].deficit == 500.0
    assert rows["A"].excess == 0.0


def test_compute_rows_skewed_target_splits_deficit_and_excess() -> None:
    rows = {row.symbol: row for row in
            compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=1000.0)}
    assert rows["A"].deficit == 1250.0
    assert rows["B"].deficit == 0.0
    assert rows["B"].excess == 250.0
    assert rows["A"].drift_pp == -25.0
    assert rows["B"].drift_pp == 25.0


def test_untracked_holding_is_excluded_from_the_target_base() -> None:
    # A alone is tracked at 100%, so it is exactly on target even though B is
    # half the book. B's drift is reported as its share of the whole book.
    rows = {row.symbol: row for row in compute_rows(EQUAL_PAIR, {"A": 100.0})}
    assert rows["A"].actual_weight == 100.0
    assert rows["A"].drift_pp == 0.0
    assert rows["A"].deficit == 0.0
    assert rows["B"].untracked is True
    assert rows["B"].target_weight == 0.0
    assert rows["B"].actual_weight == 50.0
    assert rows["B"].excess == 1000.0
    assert rows["B"].deficit == 0.0


def test_untracked_value_totals_the_off_plan_holdings() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 100.0})
    assert untracked_value(rows) == 1000.0
    assert untracked_value(compute_rows(EQUAL_PAIR,
                                        {"A": 50.0, "B": 50.0})) == 0.0


def test_compute_rows_includes_target_symbol_not_yet_held() -> None:
    rows = {row.symbol: row for row in
            compute_rows(EQUAL_PAIR, {"A": 40.0, "B": 40.0, "C": 20.0},
                         cash=0.0)}
    assert rows["C"].quantity == 0.0
    assert rows["C"].value == 0.0
    assert rows["C"].deficit == 400.0


def test_compute_rows_sorted_by_deficit_desc() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=1000.0)
    assert [row.symbol for row in rows] == ["A", "B"]


def test_compute_rows_handles_empty_portfolio() -> None:
    rows = compute_rows([], {"A": 100.0}, cash=1000.0)
    assert rows[0].actual_weight == 0.0
    assert rows[0].deficit == 1000.0


def test_pnl_pct_matches_cost_basis() -> None:
    assert abs(EQUAL_PAIR[0].pnl_pct - 11.111) < 0.01
    assert abs(EQUAL_PAIR[1].pnl_pct + 9.0909) < 0.01


def test_max_abs_drift_ignores_untracked_rows_by_default() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=0.0)
    assert max_abs_drift(rows) == 25.0
    assert max_abs_drift([]) == 0.0
    # An off-plan holding must not hold the band permanently breached.
    with_untracked = compute_rows(
        EQUAL_PAIR + [Holding("C", 10, 100.0, 100.0)],
        {"A": 50.0, "B": 50.0})
    assert max_abs_drift(with_untracked) == 0.0
    assert abs(max_abs_drift(with_untracked, tracked_only=False) - 33.33) < 0.01


# --- plan_orders --------------------------------------------------------

def test_fill_mode_splits_equal_gaps_and_spends_all_cash() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 50.0, "B": 50.0}, cash=1000.0)
    orders, leftover = plan_orders(rows, 1000.0, mode="fill")
    assert sum(order.units for order in orders) == 10
    assert sum(order.amount for order in orders) == 1000.0
    assert leftover == 0.0


def test_fill_mode_waterfalls_into_the_largest_gap() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=1000.0)
    orders, leftover = plan_orders(rows, 1000.0, mode="fill")
    assert [order.symbol for order in orders] == ["A"]
    assert orders[0].units == 10
    assert leftover == 0.0


def test_fill_mode_stays_inside_the_overshoot_bound() -> None:
    # Deficits are A 575 and B 425 against a 1000 contribution, so both fill
    # nearly exactly and the guard is what stops the last unit overshooting.
    rows = compute_rows(EQUAL_PAIR, {"A": 52.5, "B": 47.5}, cash=1000.0)
    by_symbol = {row.symbol: row for row in rows}
    assert by_symbol["A"].deficit == 575.0
    assert by_symbol["B"].deficit == 425.0
    orders, leftover = plan_orders(rows, 1000.0, mode="fill")
    amounts = {order.symbol: order.amount for order in orders}
    assert amounts == {"A": 600.0, "B": 400.0}
    assert leftover == 0.0
    _assert_no_overshoot(rows, orders)


def test_whole_units_only_and_leftover_reported() -> None:
    holdings = [Holding("A", 1, 300.0, 300.0)]
    rows = compute_rows(holdings, {"A": 100.0}, cash=1000.0)
    orders, leftover = plan_orders(rows, 1000.0, mode="fill")
    assert orders[0].units == 3
    assert orders[0].amount == 900.0
    assert leftover == 100.0


def test_spread_mode_touches_every_deficit() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 60.0, "B": 40.0}, cash=1000.0)
    orders, _ = plan_orders(rows, 1000.0, mode="spread")
    assert {order.symbol for order in orders} == {"A", "B"}
    _assert_no_overshoot(rows, orders)


def test_spread_mode_caps_each_share_at_its_own_deficit() -> None:
    # C's 200 gap is pruned by min_order_value, so the surviving deficits sum
    # to 1800 against a 2000 contribution. Uncapped shares would overshoot.
    holdings = [Holding("A", 10, 100.0, 100.0), Holding("B", 10, 100.0, 100.0),
                Holding("C", 10, 100.0, 100.0)]
    rows = compute_rows(holdings, {"A": 38.0, "B": 38.0, "C": 24.0}, cash=2000.0)
    by_symbol = {row.symbol: row for row in rows}
    assert by_symbol["A"].deficit == 900.0
    assert by_symbol["C"].deficit == 200.0
    orders, leftover = plan_orders(rows, 2000.0, mode="spread",
                                   min_order_value=500.0)
    amounts = {order.symbol: order.amount for order in orders}
    assert amounts == {"A": 900.0, "B": 900.0}
    assert leftover == 200.0
    _assert_no_overshoot(rows, orders)


def test_min_order_value_excludes_small_gaps() -> None:
    holdings = [Holding("A", 10, 100.0, 100.0), Holding("B", 10, 100.0, 100.0)]
    rows = compute_rows(holdings, {"A": 60.0, "B": 40.0}, cash=1000.0)
    orders, _ = plan_orders(rows, 1000.0, mode="fill", min_order_value=500.0)
    assert [order.symbol for order in orders] == ["A"]


def test_dribble_retry_does_not_strand_the_contribution() -> None:
    # Both gaps are large but 800 split two ways gives 400 each, under the 500
    # minimum. Retiring both at once would return no orders at all.
    holdings = [Holding("A", 10, 100.0, 100.0), Holding("B", 10, 100.0, 100.0),
                Holding("C", 200, 100.0, 100.0)]
    rows = compute_rows(holdings, {"A": 33.34, "B": 33.33, "C": 33.33},
                        cash=800.0)
    orders, leftover = plan_orders(rows, 800.0, mode="fill",
                                   min_order_value=500.0)
    assert len(orders) == 1, "the contribution must concentrate, not vanish"
    assert orders[0].amount == 800.0
    assert leftover == 0.0
    _assert_no_overshoot(rows, orders)


def test_single_candidate_below_min_order_places_nothing() -> None:
    rows = compute_rows([Holding("A", 10, 100.0, 100.0)], {"A": 100.0},
                        cash=300.0)
    orders, leftover = plan_orders(rows, 300.0, mode="fill",
                                   min_order_value=500.0)
    assert orders == []
    assert leftover == 300.0


def test_cash_is_always_conserved() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 52.5, "B": 47.5}, cash=1000.0)
    for mode in ("fill", "spread"):
        for min_order in (0.0, 250.0, 500.0):
            orders, leftover = plan_orders(rows, 1000.0, mode=mode,
                                           min_order_value=min_order)
            deployed = sum(order.amount for order in orders)
            assert abs(deployed + leftover - 1000.0) < 1e-9, (mode, min_order)
            assert leftover >= 0.0


def test_zero_price_symbol_is_skipped() -> None:
    rows = compute_rows([Holding("A", 0, 0.0, 0.0)], {"A": 100.0}, cash=1000.0)
    orders, leftover = plan_orders(rows, 1000.0, mode="fill")
    assert orders == []
    assert leftover == 1000.0


def test_no_cash_means_no_orders() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=0.0)
    assert plan_orders(rows, 0.0) == ([], 0.0)


def test_fully_on_target_produces_no_orders() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 50.0, "B": 50.0}, cash=0.0)
    orders, leftover = plan_orders(rows, 0.0, mode="fill")
    assert orders == []
    assert leftover == 0.0


def test_unknown_mode_raises() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 50.0, "B": 50.0}, cash=100.0)
    _raises(ValueError, plan_orders, rows, 100.0, "sideways")


def test_weight_after_uses_the_deployed_total_not_the_contribution() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=1000.0)
    orders, _ = plan_orders(rows, 1000.0, mode="fill")
    assert abs(orders[0].weight_after - 66.667) < 0.01
    # With cash stranded, the base must shrink to what was actually spent.
    single = compute_rows([Holding("A", 1, 300.0, 300.0)], {"A": 100.0},
                          cash=1000.0)
    orders, leftover = plan_orders(single, 1000.0, mode="fill")
    assert leftover == 100.0
    assert abs(orders[0].weight_after - 100.0) < 1e-9


# --- loaders ------------------------------------------------------------

def test_parse_number_handles_grouping_and_blanks() -> None:
    assert parse_number("1,234.50") == 1234.5
    assert parse_number("") == 0.0
    assert parse_number(None) == 0.0
    assert parse_number("-") == 0.0
    assert parse_number("(500.00)") == -500.0
    assert parse_number("Rs 1,00,000") == 100000.0


def test_load_holdings_reads_zerodha_headers_and_ignores_extra_quantities() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "h.csv",
                      "Symbol,Quantity Available,Quantity Discrepant,"
                      "Quantity Long Term,Average Price,Previous Closing Price\n"
                      "GOLDBEES,\"1,063\",99,500,66.00,82.00\n")
        held = load_holdings(path)
    assert len(held) == 1
    assert held[0].quantity == 1063.0
    assert held[0].avg_cost == 66.0
    assert held[0].last_price == 82.0


def test_load_holdings_reads_kite_headers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "h.csv",
                      "Instrument,Qty.,Avg. cost,LTP,Cur. val,P&L\n"
                      "NIFTYBEES,612,265.00,285.00,174420.00,12240.00\n")
        held = load_holdings(path)
    assert held[0].symbol == "NIFTYBEES"
    assert held[0].quantity == 612.0
    assert held[0].avg_cost == 265.0
    assert held[0].last_price == 285.0


def test_load_holdings_does_not_mistake_a_name_column_for_the_symbol() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "h.csv",
                      "Sector Name,Exchange Name,Symbol,Quantity,Average Price,Price\n"
                      "Financials,NSE,BANKBEES,10,400.00,420.00\n")
        held = load_holdings(path)
    assert held[0].symbol == "BANKBEES"
    assert held[0].last_price == 420.0


def test_load_holdings_skips_zero_quantity_and_rejects_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skipped = _write(Path(tmp), "z.csv",
                         "Symbol,Quantity Available,Average Price,LTP\n"
                         "A,0,10,10\nB,5,10,10\n")
        assert [h.symbol for h in load_holdings(skipped)] == ["B"]
        dupes = _write(Path(tmp), "d.csv",
                       "Symbol,Quantity Available,Average Price,LTP\n"
                       "A,5,10,10\nA,6,10,10\n")
        _raises(HoldingsError, load_holdings, dupes)


def test_load_holdings_rejects_missing_file_and_bad_headers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _raises(HoldingsError, load_holdings, Path(tmp) / "nope.csv")
        junk = _write(Path(tmp), "j.csv", "foo,bar\n1,2\n")
        _raises(HoldingsError, load_holdings, junk)


def test_load_targets_parses_comments_and_validates_sum() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = _write(Path(tmp), "t.yaml",
                      "# comment\n\nNIFTYBEES: 60.0  # inline\nGOLDBEES: 40\n")
        assert load_targets(good) == {"NIFTYBEES": 60.0, "GOLDBEES": 40.0}
        bad_sum = _write(Path(tmp), "b.yaml", "A: 60\nB: 30\n")
        _raises(TargetsError, load_targets, bad_sum)


def test_load_targets_rejects_malformed_input() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _raises(TargetsError, load_targets, root / "missing.yaml")
        _raises(TargetsError, load_targets, _write(root, "e.yaml", "# only\n"))
        _raises(TargetsError, load_targets, _write(root, "n.yaml", "A 100\n"))
        _raises(TargetsError, load_targets, _write(root, "d.yaml", "A: 50\nA: 50\n"))
        _raises(TargetsError, load_targets, _write(root, "x.yaml", "A: abc\n"))
        _raises(TargetsError, load_targets,
                _write(root, "r.yaml", "A: 130\nB: -30\n"))


# --- cadence ------------------------------------------------------------

ON_TARGET = compute_rows(EQUAL_PAIR, {"A": 50.0, "B": 50.0}, cash=0.0)
FAR_OFF = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=0.0)


def test_add_months_clamps_to_the_month_length() -> None:
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2026, 1, 15), 1) == date(2026, 2, 15)
    assert add_months(date(2026, 12, 15), 1) == date(2027, 1, 15)
    assert add_months(date(2028, 1, 31), 1) == date(2028, 2, 29)


def test_first_run_invests_but_warns_about_recording() -> None:
    action, reasons, next_due = evaluate_cadence(date(2026, 9, 4), None, ON_TARGET)
    assert action == "INVEST"
    assert next_due is None
    assert "first run" in reasons[0]
    assert any("--record" in reason for reason in reasons)


def test_scheduled_contribution_due_on_the_same_day_next_month() -> None:
    action, _, next_due = evaluate_cadence(
        date(2026, 9, 4), date(2026, 8, 1), ON_TARGET)
    assert action == "INVEST"
    assert next_due == date(2026, 9, 1)


def test_monthly_cadence_does_not_drift_backwards() -> None:
    # A 30-day interval would walk Jan 1 -> Jan 31 -> Mar 2. Calendar months
    # must keep landing on the first.
    assert add_months(date(2026, 1, 1), 1) == date(2026, 2, 1)
    action, _, next_due = evaluate_cadence(
        date(2026, 1, 31), date(2026, 1, 1), ON_TARGET)
    assert action == "HOLD"
    assert next_due == date(2026, 2, 1)


def test_holds_inside_the_interval_when_drift_is_small() -> None:
    action, reasons, next_due = evaluate_cadence(
        date(2026, 9, 4), date(2026, 9, 1), ON_TARGET)
    assert action == "HOLD"
    assert next_due == date(2026, 10, 1)
    assert any("inside the" in reason for reason in reasons)


def test_daily_run_the_day_after_a_buy_holds() -> None:
    action, _, _ = evaluate_cadence(
        date(2026, 9, 2), date(2026, 9, 1), FAR_OFF)
    assert action == "HOLD"


def test_band_breach_invests_off_cycle_once_spacing_is_met() -> None:
    # Spacing is passed explicitly so this tests the gate, not whatever the
    # shipped MIN_DAYS_BETWEEN_BUYS happens to be.
    action, reasons, _ = evaluate_cadence(
        date(2026, 9, 15), date(2026, 9, 1), FAR_OFF, min_days_between_buys=7)
    assert action == "INVEST"
    assert any("breaches" in reason for reason in reasons)


def test_the_band_accelerates_while_drift_is_wide_then_settles() -> None:
    # The shipped behaviour, locked in: while drift breaches the band the
    # off-cycle path fires at the spacing floor, and once the gap closes the
    # cadence falls back to the monthly schedule on its own. FAR_OFF never
    # closes (nothing is bought here), so this checks the accelerated leg.
    last, day = date(2026, 9, 4), date(2026, 9, 5)
    buys = []
    while day <= date(2027, 9, 4):
        action, _, _ = evaluate_cadence(day, last, FAR_OFF)
        if action == "INVEST":
            buys.append(day)
            last = day
        day += timedelta(days=1)
    gaps = [(buys[i] - buys[i - 1]).days for i in range(1, len(buys))]
    assert gaps and min(gaps) == config.MIN_DAYS_BETWEEN_BUYS, sorted(set(gaps))
    assert len(buys) > 12, f"a breaching band must beat monthly, got {len(buys)}"

    # And with drift inside the band, the same year yields a monthly cadence.
    last, day, calm = date(2026, 9, 4), date(2026, 9, 5), []
    while day <= date(2027, 9, 4):
        action, _, _ = evaluate_cadence(day, last, ON_TARGET)
        if action == "INVEST":
            calm.append(day)
            last = day
        day += timedelta(days=1)
    assert 12 <= len(calm) <= 13, f"on target should be monthly, got {len(calm)}"


def test_band_breach_still_respects_minimum_spacing() -> None:
    action, reasons, _ = evaluate_cadence(
        date(2026, 9, 4), date(2026, 9, 1), FAR_OFF)
    assert action == "HOLD"
    assert any("minimum spacing" in reason for reason in reasons)


def test_untracked_holding_alone_never_triggers_an_off_cycle_buy() -> None:
    rows = compute_rows(EQUAL_PAIR + [Holding("C", 10, 100.0, 100.0)],
                        {"A": 50.0, "B": 50.0})
    action, _, _ = evaluate_cadence(date(2026, 9, 15), date(2026, 9, 1), rows,
                                    min_days_between_buys=7)
    assert action == "HOLD"


def test_future_dated_last_contribution_holds() -> None:
    action, reasons, _ = evaluate_cadence(
        date(2026, 9, 4), date(2026, 10, 1), FAR_OFF)
    assert action == "HOLD"
    assert any("future" in reason for reason in reasons)


# --- contributions log --------------------------------------------------

def _invest_plan(asof: date) -> Plan:
    """Build a minimal recordable Plan."""
    order = Order(symbol="A", units=5, last_price=100.0, amount=500.0,
                  deficit_before=900.0, weight_after=50.0)
    return Plan(asof=asof, action="INVEST", cash=500.0, portfolio_value=2000.0,
                orders=[order])


def test_record_then_load_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "contributions.csv"
        assert load_last_contribution(log) is None
        record_contribution(log, _invest_plan(date(2026, 9, 1)))
        assert load_last_contribution(log) == date(2026, 9, 1)
        record_contribution(log, _invest_plan(date(2026, 10, 1)))
        assert load_last_contribution(log) == date(2026, 10, 1)


def test_a_hold_cannot_be_recorded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "contributions.csv"
        hold = Plan(asof=date(2026, 9, 20), action="HOLD")
        _raises(ValueError, record_contribution, log, hold)
        assert not log.exists()


def test_non_invest_rows_do_not_move_the_clock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), "contributions.csv",
                     "date,action,symbol,units,last_price,amount,"
                     "portfolio_value,mode\n"
                     "2026-09-01,INVEST,A,5,100.0,500.0,2000.0,fill\n"
                     "2026-09-20,HOLD,,0,,0.0,2000.0,fill\n")
        assert load_last_contribution(log) == date(2026, 9, 1)
        action, _, next_due = evaluate_cadence(
            date(2026, 10, 4), load_last_contribution(log), ON_TARGET)
        assert action == "INVEST"
        assert next_due == date(2026, 10, 1)


def test_unparseable_dates_are_ignored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), "contributions.csv",
                     "date,action,symbol,units,last_price,amount,"
                     "portfolio_value,mode\n"
                     "not-a-date,INVEST,A,5,100.0,500.0,2000.0,fill\n"
                     "2026-09-01,INVEST,A,5,100.0,500.0,2000.0,fill\n")
        assert load_last_contribution(log) == date(2026, 9, 1)


# --- candidate eligibility ----------------------------------------------

def test_fundable_units_needs_min_order_affordable_and_no_overshoot() -> None:
    row = compute_rows([Holding("A", 5, 420.0, 420.0)], {"A": 100.0},
                       cash=5000.0)[0]
    # 2 units (840) is the smallest order clearing a 500 minimum.
    assert fundable_units(row, 5000.0, 500.0) == 2
    assert fundable_units(row, 1000.0, 500.0) == 2
    # 840 is unaffordable out of 700, so the row can host no valid order.
    assert fundable_units(row, 700.0, 500.0) == 0
    # No minimum means a single unit is enough.
    assert fundable_units(row, 700.0, 0.0) == 1
    # An unpriced or on-target row is never fundable.
    flat = compute_rows([Holding("B", 1, 0.0, 0.0)], {"B": 100.0}, cash=100.0)[0]
    assert fundable_units(flat, 100.0, 0.0) == 0


def test_expensive_survivor_cannot_strand_the_contribution() -> None:
    # S2 at 420 holds the WIDEST gap, so the waterfall prefers it, yet 700
    # cannot buy the 2 units (840) needed to clear a 500 minimum there. Only
    # the fundability gate can see that; retiring the smallest gap would keep
    # S2 and strand the lot. The cheap S0 must be funded instead.
    holdings = [Holding("S0", 1, 66.0, 66.0), Holding("S1", 50, 150.0, 150.0),
                Holding("S2", 5, 420.0, 420.0), Holding("S3", 200, 66.0, 66.0)]
    targets = {"S0": 23.63, "S1": 28.99, "S2": 34.0, "S3": 13.38}
    rows = compute_rows(holdings, targets, cash=700.0)
    by_symbol = {row.symbol: row for row in rows}
    # Precondition: without this ordering the test does not exercise the gate.
    assert by_symbol["S2"].deficit > by_symbol["S0"].deficit, (
        "the expensive row must hold the wider gap for this test to bite")
    assert fundable_units(by_symbol["S2"], 700.0, 500.0) == 0
    assert fundable_units(by_symbol["S0"], 700.0, 500.0) == 8
    orders, leftover = plan_orders(rows, 700.0, mode="fill",
                                   min_order_value=500.0)
    assert orders, "a valid order existed, the contribution must not vanish"
    assert [order.symbol for order in orders] == ["S0"]
    assert orders[0].amount == 660.0
    assert leftover == 40.0
    _assert_no_overshoot(rows, orders)


def test_negative_cash_raises_rather_than_losing_it() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 50.0, "B": 50.0}, cash=0.0)
    _raises(ValueError, plan_orders, rows, -500.0)


# --- actionable drift ---------------------------------------------------

def test_actionable_drift_ignores_drift_that_buying_cannot_close() -> None:
    # GHOST is a target with no price, so A sits at 100% of tracked value
    # against an 80% target. No purchase can fix that.
    rows = compute_rows([Holding("A", 10, 100.0, 100.0)],
                        {"A": 80.0, "GHOST": 20.0})
    assert max_abs_drift(rows) == 20.0
    assert actionable_drift(rows) == 0.0
    # A genuine shortfall is still actionable.
    assert actionable_drift(FAR_OFF) == 25.0


def test_unbuyable_target_never_triggers_an_off_cycle_buy() -> None:
    rows = compute_rows([Holding("A", 10, 100.0, 100.0)],
                        {"A": 80.0, "GHOST": 20.0})
    action, reasons, _ = evaluate_cadence(
        date(2026, 9, 20), date(2026, 9, 1), rows)
    assert action == "HOLD"
    assert any("inside the" in reason for reason in reasons)


# --- CLI reporting and validation ---------------------------------------

def _plan_with(rows: list, cash: float, min_order: float,
               unpriced: "list[str] | None" = None) -> Plan:
    """Build a Plan carrying just what _no_orders_reason inspects."""
    return Plan(asof=date(2026, 9, 4), action="INVEST", cash=cash, rows=rows,
                min_order_value=min_order, unpriced=list(unpriced or []))


def test_no_orders_reason_names_the_real_blocker() -> None:
    from plan_investment import _no_orders_reason

    too_dear = compute_rows([Holding("A", 1, 100.0, 100.0)], {"A": 100.0},
                            cash=50.0)
    assert "cannot buy one unit" in _no_orders_reason(
        _plan_with(too_dear, 50.0, 0.0))

    small_gaps = compute_rows([Holding("A", 100, 10.0, 10.0)], {"A": 100.0},
                              cash=100.0)
    assert "minimum order value" in _no_orders_reason(
        _plan_with(small_gaps, 100.0, 500.0))

    granular = compute_rows([Holding("A", 5, 420.0, 420.0)], {"A": 100.0},
                            cash=700.0)
    assert "needs at least" in _no_orders_reason(
        _plan_with(granular, 700.0, 500.0))

    unpriced = compute_rows([Holding("A", 0, 0.0, 0.0)], {"A": 100.0},
                            cash=1000.0)
    assert "no usable price" in _no_orders_reason(
        _plan_with(unpriced, 1000.0, 0.0, ["A"]))


def test_cli_rejects_nonsense_amounts() -> None:
    import contextlib
    import io
    import sys
    from plan_investment import _parse_args

    original = sys.argv
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            for bad in (["p", "--amount", "0"], ["p", "--amount", "-100"],
                        ["p", "--min-order", "-1"]):
                sys.argv = bad
                _raises(SystemExit, _parse_args)
        sys.argv = ["p", "--amount", "20000", "--min-order", "0"]
        args = _parse_args()
        assert args.amount == 20000.0
        assert args.min_order == 0.0
    finally:
        sys.argv = original


def test_load_holdings_prefers_a_symbol_column_over_a_bare_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "h.csv",
                      "Name,Trading Symbol,Quantity,Average Price,LTP\n"
                      "Nippon Nifty ETF,NIFTYBEES,10,250.00,275.00\n")
        held = load_holdings(path)
    assert held[0].symbol == "NIFTYBEES"


# --- tradebook import ---------------------------------------------------

TRADEBOOK_HEADER = ("symbol,isin,trade_date,exchange,segment,series,trade_type,"
                    "auction,quantity,price,trade_id,order_id,"
                    "order_execution_time")


def _trade_row(symbol: str, date_str: str, side: str, qty: float, price: float,
               trade_id: str = "", executed: str = "", exchange: str = "NSE",
               segment: str = "EQ", order_id: str = "ORD") -> str:
    """Build one tradebook CSV line."""
    return (f"{symbol},INF000,{date_str},{exchange},{segment},EQ,{side},false,"
            f"{qty:.6f},{price:.6f},{trade_id},{order_id},{executed}")


def _tradebook(directory: Path, name: str, rows: list[str]) -> Path:
    """Write a tradebook CSV and return its path."""
    return _write(directory, name, TRADEBOOK_HEADER + "\n" + "\n".join(rows) + "\n")


def test_weighted_average_cost_survives_a_sell() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 3, 100.0, "t1", "2025-01-01T10:00:00"),
            _trade_row("A", "2025-02-01", "buy", 7, 110.0, "t2", "2025-02-01T10:00:00"),
            _trade_row("A", "2025-03-01", "sell", 4, 150.0, "t3", "2025-03-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
    position = positions["A"]
    # Pool is 300 + 770 = 1070 over 10 units, so the average is 107.
    assert position.quantity == 6.0
    assert abs(position.avg_cost - 107.0) < 1e-9, "a sell must not move the average"
    assert abs(position.realised - 4 * (150.0 - 107.0)) < 1e-9
    assert abs(position.cost_pool - 6 * 107.0) < 1e-9


def test_later_buys_blend_against_the_post_sale_pool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 10, 100.0, "t1", "2025-01-01T10:00:00"),
            _trade_row("A", "2025-02-01", "sell", 5, 200.0, "t2", "2025-02-01T10:00:00"),
            _trade_row("A", "2025-03-01", "buy", 5, 120.0, "t3", "2025-03-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
    # 5 units at 100 plus 5 at 120 leaves 10 units averaging 110.
    assert positions["A"].quantity == 10.0
    assert abs(positions["A"].avg_cost - 110.0) < 1e-9


def test_full_exit_leaves_no_position_and_records_realised_pnl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        book = _tradebook(root, "tb.csv", [
            _trade_row("GONE", "2025-01-01", "buy", 25, 169.98, "t1", "2025-01-01T10:00:00"),
            _trade_row("GONE", "2025-02-01", "buy", 25, 169.25, "t2", "2025-02-01T10:00:00"),
            _trade_row("GONE", "2025-03-01", "sell", 50, 330.65, "t3", "2025-03-01T10:00:00"),
            _trade_row("KEEP", "2025-01-01", "buy", 10, 50.0, "t4", "2025-01-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
        assert positions["GONE"].quantity == 0.0
        assert positions["GONE"].realised > 0
        out = root / "holdings.csv"
        written = write_holdings(out, positions, {"KEEP": 55.0})
        assert [p.symbol for p in written] == ["KEEP"]
        held = load_holdings(out)
    assert [h.symbol for h in held] == ["KEEP"]


def test_written_holdings_round_trip_through_the_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        book = _tradebook(root, "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 3, 100.0, "t1", "2025-01-01T10:00:00"),
            _trade_row("A", "2025-02-01", "buy", 7, 110.0, "t2", "2025-02-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
        out = root / "holdings.csv"
        write_holdings(out, positions, {"A": 130.0})
        held = load_holdings(out)
    assert held[0].quantity == 10.0
    assert held[0].avg_cost == 107.0
    assert held[0].last_price == 130.0


def test_missing_price_writes_a_blank_ltp_that_still_loads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        book = _tradebook(root, "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 5, 100.0, "t1", "2025-01-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
        out = root / "holdings.csv"
        write_holdings(out, positions, {})
        held = load_holdings(out)
    assert held[0].last_price == 0.0
    assert held[0].quantity == 5.0


def test_duplicate_trade_ids_across_files_are_counted_once() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [_trade_row("A", "2025-01-01", "buy", 10, 100.0, "t1",
                           "2025-01-01T10:00:00")]
        first = _tradebook(root, "one.csv", rows)
        second = _tradebook(root, "two.csv", rows)
        positions = build_positions(read_trades([str(first), str(second)]))
    assert positions["A"].quantity == 10.0, "overlapping exports must not double count"


def test_file_order_on_the_command_line_does_not_change_the_result() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        early = _tradebook(root, "early.csv", [
            _trade_row("A", "2025-01-01", "buy", 10, 100.0, "t1", "2025-01-01T10:00:00"),
        ])
        late = _tradebook(root, "late.csv", [
            _trade_row("A", "2025-06-01", "sell", 4, 150.0, "t2", "2025-06-01T10:00:00"),
        ])
        forwards = build_positions(read_trades([str(early), str(late)]))
        backwards = build_positions(read_trades([str(late), str(early)]))
    assert forwards["A"].quantity == backwards["A"].quantity == 6.0
    assert abs(forwards["A"].avg_cost - backwards["A"].avg_cost) < 1e-9
    assert abs(forwards["A"].realised - backwards["A"].realised) < 1e-9


def test_selling_more_than_the_tradebooks_show_is_flagged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 5, 100.0, "t1", "2025-01-01T10:00:00"),
            _trade_row("A", "2025-02-01", "sell", 12, 150.0, "t2", "2025-02-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
    assert positions["A"].oversold == 7.0, "missing opening history must be flagged"
    assert positions["A"].quantity == 0.0


def test_read_trades_rejects_files_that_are_not_tradebooks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _raises(HoldingsError, read_trades, [str(root / "nope.csv")])
        junk = _write(root, "junk.csv", "Symbol,Quantity\nA,5\n")
        _raises(HoldingsError, read_trades, [str(junk)])
        empty = _tradebook(root, "empty.csv", [])
        _raises(HoldingsError, read_trades, [str(empty)])


def test_unknown_trade_types_and_zero_quantities_are_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 10, 100.0, "t1", "2025-01-01T10:00:00"),
            _trade_row("A", "2025-01-02", "bonus", 5, 0.0, "t2", "2025-01-02T10:00:00"),
            _trade_row("A", "2025-01-03", "buy", 0, 100.0, "t3", "2025-01-03T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
    assert positions["A"].quantity == 10.0


def test_colliding_trade_ids_across_exchanges_are_not_merged() -> None:
    # trade_id is a per-exchange sequence, so the NSE and BSE ranges overlap.
    # Merging on it alone would silently drop one of these trades.
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 100, 10.0, "5000001",
                       "2025-01-01T10:00:00", exchange="NSE"),
            _trade_row("A", "2025-01-01", "buy", 100, 20.0, "5000001",
                       "2025-01-01T11:00:00", exchange="BSE"),
        ])
        positions = build_positions(read_trades([str(book)]))
    assert positions["A"].quantity == 200.0
    assert abs(positions["A"].avg_cost - 15.0) < 1e-9


def test_genuinely_conflicting_trades_raise_instead_of_overwriting() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 100, 10.0, "t1",
                       "2025-01-01T10:00:00"),
            _trade_row("A", "2025-01-01", "buy", 999, 77.0, "t1",
                       "2025-01-01T10:00:00"),
        ])
        _raises(HoldingsError, read_trades, [str(book)])


def test_rows_without_a_trade_id_still_dedupe_across_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [_trade_row("A", "2025-01-01", "buy", 10, 100.0, "",
                           "2025-01-01T10:00:00")]
        first = _tradebook(root, "one.csv", rows)
        second = _tradebook(root, "two.csv", rows)
        positions = build_positions(read_trades([str(first), str(second)]))
    assert positions["A"].quantity == 10.0, "blank ids must not bypass dedup"


def test_blank_execution_time_replays_after_the_same_day_buys() -> None:
    # A timeless row placed first would look like a sell with nothing to sell.
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 100, 10.0, "t1",
                       "2025-01-01T09:30:00"),
            _trade_row("A", "2025-01-01", "sell", 50, 12.0, "t2", ""),
        ])
        positions = build_positions(read_trades([str(book)]))
    assert positions["A"].quantity == 50.0
    assert positions["A"].oversold == 0.0
    assert abs(positions["A"].realised - 100.0) < 1e-9


def test_buys_replay_before_sells_at_an_identical_timestamp() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "sell", 50, 12.0, "t2",
                       "2025-01-01T10:00:00"),
            _trade_row("A", "2025-01-01", "buy", 100, 10.0, "t1",
                       "2025-01-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
    assert positions["A"].quantity == 50.0
    assert positions["A"].oversold == 0.0


def test_non_iso_trade_dates_are_rejected_not_string_sorted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "09-01-2025", "buy", 100, 10.0, "t1", ""),
        ])
        _raises(HoldingsError, read_trades, [str(book)])


def test_zero_or_blank_prices_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 100, 10.0, "t1",
                       "2025-01-01T10:00:00"),
            _trade_row("A", "2025-01-02", "buy", 100, 0.0, "t2",
                       "2025-01-02T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
    # A zero-priced buy would otherwise halve the average to 5.00.
    assert positions["A"].quantity == 100.0
    assert abs(positions["A"].avg_cost - 10.0) < 1e-9


def test_non_equity_segments_are_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 10, 100.0, "t1",
                       "2025-01-01T10:00:00"),
            _trade_row("NIFTY25JANFUT", "2025-01-02", "buy", 50, 23000.0, "t2",
                       "2025-01-02T10:00:00", segment="FO"),
        ])
        positions = build_positions(read_trades([str(book)]))
    assert set(positions) == {"A"}


def test_realised_is_reported_for_a_still_open_position() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        book = _tradebook(Path(tmp), "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 100, 10.0, "t1",
                       "2025-01-01T10:00:00"),
            _trade_row("A", "2025-06-01", "sell", 40, 15.0, "t2",
                       "2025-06-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
    assert positions["A"].is_open is True
    assert abs(positions["A"].realised - 200.0) < 1e-9


def test_fractional_quantity_is_refused_rather_than_rounded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        book = _tradebook(root, "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 0.5, 100.0, "t1",
                       "2025-01-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
        _raises(HoldingsError, write_holdings, root / "h.csv", positions, {})


def test_incomplete_history_refuses_to_write_unless_forced() -> None:
    from import_tradebook import _refusal
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        book = _tradebook(root, "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 100, 10.0, "t1",
                       "2025-01-01T10:00:00"),
            _trade_row("A", "2025-02-01", "sell", 150, 15.0, "t2",
                       "2025-02-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
        out = root / "holdings.csv"
        assert _refusal(positions, out, force=False) is not None
        assert _refusal(positions, out, force=True) is None


def test_an_empty_result_refuses_to_clobber_a_populated_holdings_file() -> None:
    from import_tradebook import _refusal
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        existing = _write(root, "holdings.csv",
                          "Symbol,Quantity Available,Average Price,LTP\n"
                          "NIFTYBEES,641,275.48,273.23\n")
        book = _tradebook(root, "tb.csv", [
            _trade_row("A", "2025-01-01", "buy", 10, 10.0, "t1",
                       "2025-01-01T10:00:00"),
            _trade_row("A", "2025-02-01", "sell", 10, 12.0, "t2",
                       "2025-02-01T10:00:00"),
        ])
        positions = build_positions(read_trades([str(book)]))
        assert not [p for p in positions.values() if p.is_open]
        assert _refusal(positions, existing, force=False) is not None
        assert _refusal(positions, existing, force=True) is None


def _main() -> int:
    """Run every test_* function in this module and report the tally."""
    tests = {name: obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)}
    failures = 0
    for name, func in tests.items():
        try:
            func()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc or '(bare assert, no detail)'}")
        except Exception as exc:  # surface loader/maths errors as failures
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
