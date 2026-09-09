"""Pure drift and whole-unit allocation maths. No network, no I/O."""
from __future__ import annotations

import logging
import math
from typing import Callable

from portfolio_models import DriftRow, Holding, Order

logger = logging.getLogger(__name__)

# A unit is only bought when at least this fraction of its price still fits
# inside the symbol's remaining deficit, so a buy cannot blow past the target.
OVERSHOOT_FRACTION = 0.5

Allocator = Callable[[list[DriftRow], float, "dict[str, int] | None"],
                     "tuple[dict[str, int], float]"]


def compute_rows(holdings: list[Holding], targets: dict[str, float],
                 cash: float = 0.0) -> list[DriftRow]:
    """Compare actual against target weights, sorted by rupee deficit desc.

    Target weights describe the TRACKED portfolio, so tracked weights are
    measured against tracked value only. A holding absent from the targets file
    sits outside the plan: including its value in the base would dilute every
    tracked weight and leave the drift permanently short of target, which would
    hold the rebalance band breached forever. Untracked rows are reported with
    their share of the whole book instead, and are excluded from the band.
    """
    by_symbol = {holding.symbol: holding for holding in holdings}
    book_value = sum(holding.value for holding in holdings)
    tracked_value = sum(holding.value for holding in holdings
                        if holding.symbol in targets)
    total_after = tracked_value + max(0.0, cash)
    rows: list[DriftRow] = []
    for symbol in sorted(set(by_symbol) | set(targets)):
        holding = by_symbol.get(symbol)
        value = holding.value if holding else 0.0
        untracked = symbol not in targets
        if untracked:
            actual_weight = (value / book_value * 100.0) if book_value else 0.0
            target_weight = 0.0
            deficit = 0.0
            excess = value
        else:
            actual_weight = (value / tracked_value * 100.0) if tracked_value else 0.0
            target_weight = targets[symbol]
            target_value = target_weight / 100.0 * total_after
            deficit = max(0.0, target_value - value)
            excess = max(0.0, value - target_value)
        rows.append(DriftRow(
            symbol=symbol,
            quantity=holding.quantity if holding else 0.0,
            last_price=holding.last_price if holding else 0.0,
            value=value,
            actual_weight=actual_weight,
            target_weight=target_weight,
            drift_pp=actual_weight - target_weight,
            deficit=deficit,
            excess=excess,
            pnl_pct=holding.pnl_pct if holding else 0.0,
            untracked=untracked,
        ))
    rows.sort(key=lambda row: (-row.deficit, row.drift_pp))
    return rows


def max_abs_drift(rows: list[DriftRow], tracked_only: bool = True) -> float:
    """Largest absolute drift in percentage points, over actionable rows only.

    Two kinds of row are excluded by default, both for the same reason: their
    drift cannot be closed by buying, so counting them would hold the
    rebalance band permanently breached and fire an off-cycle buy forever.

    - Untracked holdings: their drift is their whole portfolio weight, and
      only a sale can reduce it.
    - Unpriced rows: with no price there is nothing to buy, so a target
      symbol the data source cannot resolve would otherwise sit at -100pp.
    """
    considered = ([row for row in rows
                   if not row.untracked and row.last_price > 0.0]
                  if tracked_only else rows)
    return max((abs(row.drift_pp) for row in considered), default=0.0)


def actionable_drift(rows: list[DriftRow]) -> float:
    """Largest drift that buying can actually close, in percentage points.

    This is the metric the rebalance band must gate on. Only a row that is
    below target AND has a usable price can absorb a contribution; a row that
    is over target, untracked, or unpriced cannot be corrected by buying it.
    If nothing is buyable, an off-cycle contribution would place no orders and
    record nothing, so the band would stay breached and fire again forever.
    """
    considered = [row for row in rows
                  if not row.untracked and row.last_price > 0.0
                  and row.deficit > 0.0]
    return max((abs(row.drift_pp) for row in considered), default=0.0)


def untracked_value(rows: list[DriftRow]) -> float:
    """Total rupee value held in symbols that are absent from the targets file."""
    return sum(row.value for row in rows if row.untracked)


def fundable_units(row: DriftRow, cash: float, min_order_value: float) -> int:
    """Smallest whole-unit order for this row that is actually placeable, else 0.

    An order has to clear three hurdles at once: it must reach
    min_order_value, it must be affordable, and it must not breach the
    overshoot bound. A symbol that cannot clear all three with the full
    contribution can never host a valid order, so it must not become a
    candidate at all.
    """
    if row.last_price <= 0.0 or row.deficit <= 0.0:
        return 0
    units = max(1, math.ceil(min_order_value / row.last_price))
    amount = units * row.last_price
    if amount > cash:
        return 0
    if amount > row.deficit + row.last_price * OVERSHOOT_FRACTION:
        return 0
    return units


def _buyable(rows: list[DriftRow], cash: float,
             min_order_value: float) -> list[DriftRow]:
    """Rows that can receive money: priced, under target, able to host an order."""
    buyable: list[DriftRow] = []
    for row in rows:
        if row.last_price <= 0.0:
            if row.deficit > 0.0:
                logger.warning("Skipping %s: no usable last price", row.symbol)
            continue
        if fundable_units(row, cash, min_order_value) > 0:
            buyable.append(row)
    return buyable


def _greedy_units(rows: list[DriftRow], cash: float,
                  units: "dict[str, int] | None" = None,
                  ) -> tuple[dict[str, int], float]:
    """Fill the largest remaining deficit first, in whole units, until cash runs out.

    Equivalent to buying one unit at a time into whichever gap is currently
    widest, but each pass buys the whole run of units that would go to the same
    symbol: enough to bring its gap down to the runner-up's. Where gaps are
    well separated that collapses thousands of passes into a handful. Where
    they are near-equal, as in an already-balanced portfolio, the run length
    is one and the loop still steps unit by unit, so this is a best-case
    speed-up rather than a guaranteed bound.
    """
    counts = dict(units) if units else {row.symbol: 0 for row in rows}
    remaining = {row.symbol: row.deficit - counts.get(row.symbol, 0) * row.last_price
                 for row in rows}
    prices = {row.symbol: row.last_price for row in rows}
    cash_left = cash
    while True:
        best_symbol = None
        best_gap = 0.0
        runner_up = 0.0
        for symbol, price in prices.items():
            if price <= 0.0 or price > cash_left:
                continue
            gap = remaining[symbol]
            if gap < price * OVERSHOOT_FRACTION:
                continue
            if gap > best_gap:
                best_symbol, best_gap, runner_up = symbol, gap, best_gap
            elif gap > runner_up:
                runner_up = gap
        if best_symbol is None:
            return counts, cash_left
        price = prices[best_symbol]
        # Units that keep this gap at or above the runner-up, so the next pass
        # legitimately re-picks. At least one, or the loop cannot progress.
        to_runner_up = max(1, int(math.floor((best_gap - runner_up) / price)))
        # Never breach the overshoot guard, and never overspend.
        by_gap = int(math.floor((best_gap - price * OVERSHOOT_FRACTION) / price)) + 1
        by_cash = int(math.floor(cash_left / price))
        step = min(to_runner_up, by_gap, by_cash)
        if step <= 0:
            return counts, cash_left
        counts[best_symbol] = counts.get(best_symbol, 0) + step
        remaining[best_symbol] -= step * price
        cash_left -= step * price


def _spread_units(rows: list[DriftRow], cash: float,
                  units: "dict[str, int] | None" = None,
                  ) -> tuple[dict[str, int], float]:
    """Split cash across every deficit in proportion, then top up with the remainder."""
    total_deficit = sum(row.deficit for row in rows)
    if total_deficit <= 0.0:
        return {row.symbol: 0 for row in rows}, cash
    counts: dict[str, int] = dict(units) if units else {}
    spent = 0.0
    for row in rows:
        # Cap the proportional share at the symbol's own deficit. _buyable may
        # have pruned rows, so the surviving shares can otherwise exceed the
        # gaps they are meant to close and overshoot the target.
        share = min(cash * row.deficit / total_deficit, row.deficit)
        already = counts.get(row.symbol, 0)
        extra = int(math.floor(share / row.last_price))
        counts[row.symbol] = already + extra
        spent += extra * row.last_price
    # Float rounding can push `spent` a hair above `cash`, so clamp rather
    # than hand a negative remainder to the top-up pass.
    return _greedy_units(rows, max(0.0, cash - spent), counts)


def _allocate(rows: list[DriftRow], cash: float, allocate: Allocator,
              min_order_value: float) -> tuple[dict[str, int], float]:
    """Allocate cash, retiring one dribble order at a time until none remain."""
    pool = list(rows)
    while len(pool) > 1:
        counts, leftover = allocate(pool, cash, None)
        dribbles = [row for row in pool
                    if 0 < counts.get(row.symbol, 0) * row.last_price < min_order_value]
        if not dribbles:
            return counts, leftover
        # Retire the dribble with the SMALLEST deficit: it is the least
        # important gap, and keeping the larger ones means the freed cash
        # still lands where the strategy wants it. Retiring by smallest
        # spend instead would select against cheap symbols, which are
        # precisely the ones able to absorb cash in min-order-sized chunks.
        weakest = min(dribbles, key=lambda row: row.deficit)
        pool = [row for row in pool if row.symbol != weakest.symbol]
    if not pool:
        return {}, cash
    # One candidate left: concentrate the contribution in that gap. Every
    # candidate cleared fundable_units against the full contribution, so this
    # always yields a placeable order; the guard is defence in depth.
    counts, leftover = allocate(pool, cash, None)
    only = pool[0]
    if counts.get(only.symbol, 0) * only.last_price < min_order_value:
        logger.warning("Could not place a valid order for %s", only.symbol)
        return {}, cash
    return counts, leftover


def plan_orders(rows: list[DriftRow], cash: float, mode: str = "fill",
                min_order_value: float = 0.0) -> tuple[list[Order], float]:
    """Turn a cash amount into whole-unit buy orders against the biggest deficits."""
    if mode not in {"fill", "spread"}:
        raise ValueError(f"Unknown allocation mode: {mode!r}")
    if cash < 0.0:
        raise ValueError(f"Contribution cannot be negative: {cash}")
    if cash == 0.0:
        return [], 0.0
    candidates = _buyable(rows, cash, min_order_value)
    if not candidates:
        return [], cash
    allocate: Allocator = _greedy_units if mode == "fill" else _spread_units
    counts, leftover = _allocate(candidates, cash, allocate, min_order_value)
    leftover = max(0.0, leftover)
    # Weights after the buy must be measured against the money actually
    # deployed, not the whole contribution, or every figure is understated.
    # Base is tracked value only, matching compute_rows.
    total_after = (sum(row.value for row in rows if not row.untracked)
                   + (cash - leftover))
    return _build_orders(candidates, counts, total_after), leftover


def _build_orders(rows: list[DriftRow], counts: dict[str, int],
                  total_after: float) -> list[Order]:
    """Build Order records from per-symbol unit counts, largest gap first."""
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
    orders.sort(key=lambda order: -order.deficit_before)
    return orders
