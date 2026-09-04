"""Target-weight contribution planner CLI.

Compares holdings against target weights and decides whether today is a buying
day, then splits the contribution into whole-unit orders against the largest
rupee shortfalls. Never places orders.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import date

import config
from allocator import compute_rows, max_abs_drift, plan_orders
from holdings import HoldingsError, TargetsError, load_holdings, load_targets
from portfolio_models import Holding, Order, Plan
from schedule_rules import (evaluate_cadence, load_last_contribution,
                            record_contribution)

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Not financial advice. The target weights are yours; this tool only does "
    "the arithmetic of moving your holdings toward them."
)


def _parse_args() -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(
        description="Plan where a contribution should go using target weights.")
    parser.add_argument("--amount", type=float, default=config.MONTHLY_CONTRIBUTION,
                        help="rupees to deploy (default: %(default)s)")
    parser.add_argument("--mode", choices=("fill", "spread"),
                        default=config.DEFAULT_ALLOCATION_MODE,
                        help="fill = waterfall into the biggest gap, "
                             "spread = proportional (default: %(default)s)")
    parser.add_argument("--holdings", default=str(config.HOLDINGS_CSV),
                        help="holdings CSV path")
    parser.add_argument("--targets", default=str(config.TARGETS_YAML),
                        help="target weights file path")
    parser.add_argument("--min-order", type=float, default=config.MIN_ORDER_VALUE,
                        help="skip orders below this rupee value "
                             "(default: %(default)s)")
    parser.add_argument("--offline", action="store_true",
                        help="use last prices from the CSV instead of yfinance")
    parser.add_argument("--force", action="store_true",
                        help="plan orders even when the cadence says HOLD")
    parser.add_argument("--record", action="store_true",
                        help="log this contribution as executed today")
    parser.add_argument("--json", action="store_true",
                        help="print the plan as JSON instead of a report")
    return parser.parse_args()


def fetch_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch live prices, degrading to an empty dict if yfinance is unusable."""
    try:
        from quotes import fetch_last_prices
    except ImportError as exc:
        logger.warning("Price fetching unavailable: %s", exc)
        return {}
    return fetch_last_prices(symbols)


def _resolve_prices(held: list[Holding], targets: dict[str, float],
                    offline: bool) -> tuple[list[Holding], str]:
    """Merge live prices over CSV prices and add zero-quantity target symbols."""
    symbols = sorted({holding.symbol for holding in held} | set(targets))
    prices: dict[str, float] = {}
    source = "csv"
    if not offline:
        prices = fetch_prices(symbols)
        source = "yfinance" if prices else "csv (yfinance unavailable)"
    by_symbol = {holding.symbol: holding for holding in held}
    resolved: list[Holding] = []
    for symbol in symbols:
        holding = by_symbol.get(symbol)
        price = prices.get(symbol, holding.last_price if holding else 0.0)
        if holding is None:
            resolved.append(Holding(symbol=symbol, quantity=0.0, avg_cost=0.0,
                                    last_price=price))
        else:
            resolved.append(dataclasses.replace(holding, last_price=price))
    return resolved, source


def build_plan(args: argparse.Namespace, today: date) -> Plan:
    """Load inputs, decide the cadence, and allocate the contribution."""
    targets = load_targets(args.targets)
    held = load_holdings(args.holdings)
    priced, source = _resolve_prices(held, targets, args.offline)
    rows = compute_rows(priced, targets, cash=args.amount)
    action, reasons, next_due = evaluate_cadence(
        today, load_last_contribution(config.CONTRIBUTIONS_CSV), rows)
    orders: list[Order] = []
    leftover = args.amount
    if action == "INVEST" or args.force:
        orders, leftover = plan_orders(rows, args.amount, mode=args.mode,
                                       min_order_value=args.min_order)
    if action == "HOLD" and args.force:
        reasons.append("NOTE: --force used, the orders below are a dry run")
    return Plan(asof=today, action=action, reasons=reasons, cash=args.amount,
                portfolio_value=sum(row.value for row in rows), mode=args.mode,
                rows=rows, orders=orders, leftover=leftover, next_due=next_due,
                price_source=source)


def _print_drift_table(plan: Plan) -> None:
    """Print the actual-versus-target table, widest shortfall first."""
    header = (f"{'SYMBOL':<12}{'QTY':>7}{'LTP':>10}{'VALUE':>12}"
              f"{'ACTUAL':>9}{'TARGET':>9}{'DRIFT':>9}{'SHORT BY':>12}"
              f"{'P/L':>9}")
    print(header)
    print("-" * len(header))
    for row in plan.rows:
        flag = " *" if row.untracked else ""
        short = f"{row.deficit:,.0f}" if row.deficit > 0 else "-"
        print(f"{row.symbol + flag:<12}{row.quantity:>7,.0f}"
              f"{row.last_price:>10,.2f}{row.value:>12,.0f}"
              f"{row.actual_weight:>8.2f}%{row.target_weight:>8.2f}%"
              f"{row.drift_pp:>+8.2f}%{short:>12}{row.pnl_pct:>+8.1f}%")
    if any(row.untracked for row in plan.rows):
        print("\n  * held but absent from the targets file, target treated as 0%")


def _print_orders(plan: Plan) -> None:
    """Print the buy list, or say why there is nothing to buy."""
    if not plan.orders:
        print("\nNo orders. Either nothing is below target, or the cash cannot "
              "buy a whole unit without overshooting a target.")
        return
    header = (f"{'BUY':<12}{'UNITS':>7}{'LTP':>10}{'AMOUNT':>12}"
              f"{'WEIGHT AFTER':>15}")
    print()
    print(header)
    print("-" * len(header))
    for order in plan.orders:
        print(f"{order.symbol:<12}{order.units:>7}{order.last_price:>10,.2f}"
              f"{order.amount:>12,.0f}{order.weight_after:>14.2f}%")
    print("-" * len(header))
    print(f"{'TOTAL':<12}{'':>7}{'':>10}{plan.deployed:>12,.0f}")


def print_report(plan: Plan) -> None:
    """Print the full human-readable plan."""
    print(f"\nCONTRIBUTION PLAN - {plan.asof.isoformat()}")
    print(f"Portfolio {plan.portfolio_value:,.0f}  |  contribution "
          f"{plan.cash:,.0f}  |  mode {plan.mode}  |  prices {plan.price_source}")
    print(f"Largest drift {max_abs_drift(plan.rows):.2f}pp  |  band "
          f"{config.REBALANCE_BAND_PP:.2f}pp")
    print(f"\nDECISION: {plan.action}")
    for reason in plan.reasons:
        print(f"  {reason}")
    if plan.next_due and plan.action == "HOLD":
        print(f"  next scheduled contribution: {plan.next_due.isoformat()}")
    print()
    _print_drift_table(plan)
    if plan.action == "INVEST" or plan.orders:
        _print_orders(plan)
        print(f"\nUndeployed cash: {plan.leftover:,.0f}")
    print(f"\n{DISCLAIMER}")


def _plan_to_dict(plan: Plan) -> dict:
    """Convert a Plan to JSON-safe primitives."""
    payload = dataclasses.asdict(plan)
    payload["asof"] = plan.asof.isoformat()
    payload["next_due"] = plan.next_due.isoformat() if plan.next_due else None
    payload["deployed"] = plan.deployed
    return payload


def main() -> int:
    """Entry point. Returns 2 when the input files are unusable, else 0."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        plan = build_plan(args, date.today())
    except (HoldingsError, TargetsError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.json:
        print(json.dumps(_plan_to_dict(plan), indent=2))
    else:
        print_report(plan)
    config.LAST_PLAN_JSON.write_text(
        json.dumps(_plan_to_dict(plan), indent=2), encoding="utf-8")
    if args.record:
        if plan.action == "INVEST" and plan.orders:
            record_contribution(config.CONTRIBUTIONS_CSV, plan)
            print(f"\nRecorded to {config.CONTRIBUTIONS_CSV.name}")
        else:
            print("\nNothing recorded: no orders were planned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
