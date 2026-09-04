"""Data models for target-weight portfolio rebalancing."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Holding:
    """One ETF position as held in the demat account."""

    symbol: str
    quantity: float
    avg_cost: float
    last_price: float

    @property
    def value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.last_price

    @property
    def invested(self) -> float:
        """Total cost basis of the position."""
        return self.quantity * self.avg_cost

    @property
    def pnl_pct(self) -> float:
        """Unrealised profit or loss as a percentage of cost basis."""
        if self.invested == 0.0:
            return 0.0
        return (self.value / self.invested - 1.0) * 100.0


@dataclass(frozen=True)
class DriftRow:
    """Actual versus target comparison for one symbol."""

    symbol: str
    quantity: float
    last_price: float
    value: float
    actual_weight: float      # percent of current portfolio value
    target_weight: float      # percent, from targets.yaml
    drift_pp: float           # actual_weight - target_weight, percentage points
    deficit: float            # rupees below target at the post-contribution total
    excess: float             # rupees above target at the post-contribution total
    pnl_pct: float
    untracked: bool = False   # held but absent from targets.yaml


@dataclass(frozen=True)
class Order:
    """A whole-unit buy instruction for one symbol."""

    symbol: str
    units: int
    last_price: float
    amount: float             # units * last_price
    deficit_before: float
    weight_after: float       # percent of post-contribution total


@dataclass(frozen=True)
class Plan:
    """The full output of one rebalancing run."""

    asof: date
    action: str               # "INVEST" or "HOLD"
    reasons: list[str] = field(default_factory=list)
    cash: float = 0.0
    portfolio_value: float = 0.0
    mode: str = "fill"
    rows: list[DriftRow] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    leftover: float = 0.0
    next_due: date | None = None
    price_source: str = "csv"
    min_order_value: float = 0.0
    unpriced: list[str] = field(default_factory=list)

    @property
    def deployed(self) -> float:
        """Total rupees actually assigned to whole-unit orders."""
        return sum(order.amount for order in self.orders)
