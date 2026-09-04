"""Pure drift and whole-unit allocation maths. No network, no I/O."""
from __future__ import annotations

import logging
import math

from portfolio_models import DriftRow, Holding, Order

logger = logging.getLogger(__name__)

# A unit is only bought when at least this fraction of its price still fits
# inside the symbol's remaining deficit, so a buy cannot blow past the target.
OVERSHOOT_FRACTION = 0.5


def compute_rows(holdings: list[Holding], targets: dict[str, float],
                 cash: float = 0.0) -> list[DriftRow]:
    """Compare actual against target weights, sorted by rupee deficit desc."""
    by_symbol = {holding.symbol: holding for holding in holdings}
    portfolio_value = sum(holding.value for holding in holdings)
    total_after = portfolio_value + max(0.0, cash)
    rows: list[DriftRow] = []
    for symbol in sorted(set(by_symbol) | set(targets)):
        holding = by_symbol.get(symbol)
        value = holding.value if holding else 0.0
        target_weight = targets.get(symbol, 0.0)
        target_value = target_weight / 100.0 * total_after
        actual_weight = (value / portfolio_value * 100.0) if portfolio_value else 0.0
        rows.append(DriftRow(
            symbol=symbol,
            quantity=holding.quantity if holding else 0.0,
            last_price=holding.last_price if holding else 0.0,
            value=value,
            actual_weight=actual_weight,
            target_weight=target_weight,
            drift_pp=actual_weight - target_weight,
            deficit=max(0.0, target_value - value),
            excess=max(0.0, value - target_value),
            pnl_pct=holding.pnl_pct if holding else 0.0,
            untracked=symbol not in targets,
        ))
    rows.sort(key=lambda row: (-row.deficit, row.drift_pp))
    return rows


def max_abs_drift(rows: list[DriftRow]) -> float:
    """Largest absolute drift in percentage points across all rows."""
    return max((abs(row.drift_pp) for row in rows), default=0.0)


def _buyable(rows: list[DriftRow], min_order_value: float) -> list[DriftRow]:
    """Rows that can receive money: priced, under target, worth a real order."""
    buyable: list[DriftRow] = []
    for row in rows:
        if row.deficit <= 0.0:
            continue
        if row.last_price <= 0.0:
            logger.warning("Skipping %s: no usable last price", row.symbol)
            continue
        if row.deficit < min_order_value:
            continue
        buyable.append(row)
    return buyable


def _greedy_units(rows: list[DriftRow], cash: float,
                  units: "dict[str, int] | None" = None) -> tuple[dict[str, int], float]:
    """Buy one unit at a time into the largest remaining deficit until cash runs out."""
    counts = dict(units) if units else {row.symbol: 0 for row in rows}
    remaining = {row.symbol: row.deficit - counts.get(row.symbol, 0) * row.last_price
                 for row in rows}
    prices = {row.symbol: row.last_price for row in rows}
    cash_left = cash
    while True:
        best_symbol = None
        best_gap = 0.0
        for symbol, price in prices.items():
            if price <= 0.0 or price > cash_left:
                continue
            gap = remaining[symbol]
            if gap < price * OVERSHOOT_FRACTION:
                continue
            if gap > best_gap:
                best_gap = gap
                best_symbol = symbol
        if best_symbol is None:
            return counts, cash_left
        counts[best_symbol] = counts.get(best_symbol, 0) + 1
        remaining[best_symbol] -= prices[best_symbol]
        cash_left -= prices[best_symbol]


def _spread_units(rows: list[DriftRow], cash: float) -> tuple[dict[str, int], float]:
    """Split cash across every deficit in proportion, then top up with the remainder."""
    total_deficit = sum(row.deficit for row in rows)
    if total_deficit <= 0.0:
        return {row.symbol: 0 for row in rows}, cash
    counts: dict[str, int] = {}
    spent = 0.0
    for row in rows:
        share = cash * row.deficit / total_deficit
        units = int(math.floor(share / row.last_price))
        counts[row.symbol] = units
        spent += units * row.last_price
    return _greedy_units(rows, cash - spent, counts)


def plan_orders(rows: list[DriftRow], cash: float, mode: str = "fill",
                min_order_value: float = 0.0) -> tuple[list[Order], float]:
    """Turn a cash amount into whole-unit buy orders against the biggest deficits."""
    if mode not in {"fill", "spread"}:
        raise ValueError(f"Unknown allocation mode: {mode!r}")
    if cash <= 0.0:
        return [], max(0.0, cash)
    candidates = _buyable(rows, min_order_value)
    total_after = sum(row.value for row in rows) + cash
    for _ in range(len(candidates) + 1):
        if not candidates:
            return [], cash
        allocate = _greedy_units if mode == "fill" else _spread_units
        counts, leftover = allocate(candidates, cash)
        dribbles = {row.symbol for row in candidates
                    if 0 < counts.get(row.symbol, 0) * row.last_price < min_order_value}
        if not dribbles:
            orders = _build_orders(candidates, counts, total_after)
            return orders, leftover
        # Re-run without the symbols that only attracted a token amount, so
        # their cash goes to a gap large enough to be worth an order.
        candidates = [row for row in candidates if row.symbol not in dribbles]
    return [], cash


def _build_orders(rows: list[DriftRow], counts: dict[str, int],
                  total_after: float) -> list[Order]:
    """Build sorted Order records from per-symbol unit counts."""
    orders: list[Order] = []
    for row in rows:
        units = counts.get(row.symbol, 0)
        if units <= 0:
            continue
        amount = units * row.last_price
        weight_after = ((row.value + amount) / total_after * 100.0
                        if total_after else 0.0)
        orders.append(Order(symbol=row.symbol, units=units,
                            last_price=row.last_price, amount=amount,
                            deficit_before=row.deficit,
                            weight_after=weight_after))
    orders.sort(key=lambda order: -order.amount)
    return orders
