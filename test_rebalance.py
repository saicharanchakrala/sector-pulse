"""Tests for the target-weight rebalancing maths, loaders, and cadence gates.

Runs under pytest, or standalone with `python test_rebalance.py`.
"""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

from allocator import compute_rows, max_abs_drift, plan_orders
from holdings import (HoldingsError, TargetsError, load_holdings, load_targets,
                      _to_float)
from portfolio_models import Holding
from schedule_rules import evaluate_cadence

EQUAL_PAIR = [
    Holding(symbol="A", quantity=10, avg_cost=90.0, last_price=100.0),
    Holding(symbol="B", quantity=10, avg_cost=110.0, last_price=100.0),
]


def _raises(exc_type, func, *args, **kwargs) -> None:
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


def test_compute_rows_flags_untracked_holding() -> None:
    rows = {row.symbol: row for row in compute_rows(EQUAL_PAIR, {"A": 100.0})}
    assert rows["B"].untracked is True
    assert rows["B"].target_weight == 0.0
    assert rows["A"].untracked is False


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


def test_max_abs_drift() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=0.0)
    assert max_abs_drift(rows) == 25.0
    assert max_abs_drift([]) == 0.0


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


def test_fill_mode_does_not_overshoot_the_target() -> None:
    # A is short by only 100, so a 1000 contribution cannot all go there.
    holdings = [Holding("A", 10, 100.0, 100.0), Holding("B", 10, 100.0, 100.0)]
    rows = compute_rows(holdings, {"A": 52.5, "B": 47.5}, cash=1000.0)
    orders, leftover = plan_orders(rows, 1000.0, mode="fill")
    by_symbol = {order.symbol: order for order in orders}
    assert by_symbol["A"].amount <= rows[0].deficit + 100.0
    assert leftover + sum(order.amount for order in orders) == 1000.0


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


def test_min_order_value_excludes_small_gaps() -> None:
    holdings = [Holding("A", 10, 100.0, 100.0), Holding("B", 10, 100.0, 100.0)]
    rows = compute_rows(holdings, {"A": 60.0, "B": 40.0}, cash=1000.0)
    orders, _ = plan_orders(rows, 1000.0, mode="fill", min_order_value=500.0)
    assert [order.symbol for order in orders] == ["A"]


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


def test_weight_after_reflects_the_purchase() -> None:
    rows = compute_rows(EQUAL_PAIR, {"A": 75.0, "B": 25.0}, cash=1000.0)
    orders, _ = plan_orders(rows, 1000.0, mode="fill")
    assert abs(orders[0].weight_after - 66.667) < 0.01


# --- loaders ------------------------------------------------------------

def test_to_float_handles_grouping_and_blanks() -> None:
    assert _to_float("1,234.50") == 1234.5
    assert _to_float("") == 0.0
    assert _to_float(None) == 0.0
    assert _to_float("-") == 0.0
    assert _to_float("(500.00)") == -500.0
    assert _to_float("Rs 1,00,000") == 100000.0


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
    assert held[0].last_price == 285.0


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


def test_first_run_invests() -> None:
    action, reasons, next_due = evaluate_cadence(date(2026, 9, 4), None, ON_TARGET)
    assert action == "INVEST"
    assert next_due is None
    assert "first run" in reasons[0]


def test_scheduled_contribution_due_after_the_interval() -> None:
    action, _, next_due = evaluate_cadence(
        date(2026, 9, 4), date(2026, 8, 1), ON_TARGET)
    assert action == "INVEST"
    assert next_due == date(2026, 8, 31)


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
    action, reasons, _ = evaluate_cadence(
        date(2026, 9, 15), date(2026, 9, 1), FAR_OFF)
    assert action == "INVEST"
    assert any("breaches" in reason for reason in reasons)


def test_band_breach_still_respects_minimum_spacing() -> None:
    action, reasons, _ = evaluate_cadence(
        date(2026, 9, 4), date(2026, 9, 1), FAR_OFF)
    assert action == "HOLD"
    assert any("minimum spacing" in reason for reason in reasons)


def test_future_dated_last_contribution_holds() -> None:
    action, reasons, _ = evaluate_cadence(
        date(2026, 9, 4), date(2026, 10, 1), FAR_OFF)
    assert action == "HOLD"
    assert any("future" in reason for reason in reasons)


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
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # surface loader/maths errors as failures
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
