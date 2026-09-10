"""Round-trip transaction costs for Zerodha intraday equity and options.

Every number a scanner reports about a target has to clear these costs or the
setup is a losing trade that looks like a winner. Intraday cost drag is the
whole game: an equity round trip runs near 0.06% of turnover, so a 0.1%
target is mostly broker revenue.

Rates live in config so they can be audited and corrected in one place.
They are as-published for Zerodha at the time of writing; verify against
zerodha.com/charges before acting on any breakeven figure computed here.
"""
from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised round-trip cost, in rupees, for one intraday position."""

    brokerage: float
    stt: float
    transaction: float
    stamp_duty: float
    sebi: float
    gst: float
    buy_turnover: float
    sell_turnover: float

    @property
    def total(self) -> float:
        """Total rupees lost to costs across both legs."""
        return (self.brokerage + self.stt + self.transaction
                + self.stamp_duty + self.sebi + self.gst)

    @property
    def breakeven_pct(self) -> float:
        """Percent the price must move, on the buy leg, just to break even.

        Measured against buy turnover because that is the capital committed.
        Returns 0.0 for an empty position rather than dividing by zero.
        """
        if self.buy_turnover <= 0.0:
            return 0.0
        return self.total / self.buy_turnover * 100.0


def _gst_on(brokerage: float, transaction: float, sebi: float) -> float:
    """GST applies to brokerage and exchange fees, never to STT or stamp duty."""
    return (brokerage + transaction + sebi) * config.COST_GST_PCT


def equity_intraday_cost(entry_price: float, exit_price: float,
                         quantity: int) -> CostBreakdown:
    """Cost of buying `quantity` at entry and selling the lot at exit.

    Direction-agnostic: a short sells first and buys back, but every rate
    below keys off the buy or sell leg rather than the order of the two, so
    the total is identical either way.
    """
    if quantity <= 0 or entry_price <= 0.0 or exit_price <= 0.0:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    turnover = buy_turnover + sell_turnover

    def leg_brokerage(value: float) -> float:
        """Percentage brokerage on one leg, subject to the per-order cap."""
        return min(value * config.COST_EQ_BROKERAGE_PCT,
                   config.COST_EQ_BROKERAGE_CAP)

    brokerage = leg_brokerage(buy_turnover) + leg_brokerage(sell_turnover)
    stt = sell_turnover * config.COST_EQ_STT_SELL_PCT
    transaction = turnover * config.COST_EQ_TXN_PCT
    stamp_duty = buy_turnover * config.COST_EQ_STAMP_BUY_PCT
    sebi = turnover * config.COST_SEBI_PCT
    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        transaction=transaction,
        stamp_duty=stamp_duty,
        sebi=sebi,
        gst=_gst_on(brokerage, transaction, sebi),
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
    )


def options_cost(entry_premium: float, exit_premium: float,
                 lots: int, lot_size: int) -> CostBreakdown:
    """Cost of one options round trip, charged on premium turnover.

    This is why cheap options are expensive: transaction charges and STT are
    percentages of premium, while brokerage is a flat 20 rupees per order.
    On a 5-rupee premium the flat fee alone can exceed the exchange fees
    several times over.
    """
    quantity = lots * lot_size
    if quantity <= 0 or entry_premium <= 0.0 or exit_premium <= 0.0:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    buy_turnover = entry_premium * quantity
    sell_turnover = exit_premium * quantity
    turnover = buy_turnover + sell_turnover

    brokerage = config.COST_OPT_BROKERAGE_FLAT * 2
    stt = sell_turnover * config.COST_OPT_STT_SELL_PCT
    transaction = turnover * config.COST_OPT_TXN_PCT
    stamp_duty = buy_turnover * config.COST_OPT_STAMP_BUY_PCT
    sebi = turnover * config.COST_SEBI_PCT
    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        transaction=transaction,
        stamp_duty=stamp_duty,
        sebi=sebi,
        gst=_gst_on(brokerage, transaction, sebi),
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
    )


def futures_cost(entry_price: float, exit_price: float,
                 lots: int, lot_size: int) -> CostBreakdown:
    """Cost of one equity-futures round trip, charged on NOTIONAL.

    The middle of the three stacks and by far the cheapest per rupee of
    exposure: brokerage is capped at 20 rupees a leg, and every percentage
    charge applies to the contract's notional rather than to a premium. On
    a six-lakh contract the capped brokerage is about 0.003% - which is
    why a future can clear its costs on a move an option cannot.

    STT is 0.02% and sell-side only, a fifth of what equity delivery pays
    on each leg and a fifth of what an option pays on premium.
    """
    quantity = lots * lot_size
    if quantity <= 0 or entry_price <= 0.0 or exit_price <= 0.0:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    turnover = buy_turnover + sell_turnover

    brokerage = sum(
        min(leg * config.COST_FUT_BROKERAGE_PCT,
            config.COST_FUT_BROKERAGE_CAP)
        for leg in (buy_turnover, sell_turnover))
    stt = sell_turnover * config.COST_FUT_STT_SELL_PCT
    transaction = turnover * config.COST_FUT_TXN_PCT
    stamp_duty = buy_turnover * config.COST_FUT_STAMP_BUY_PCT
    sebi = turnover * config.COST_SEBI_PCT
    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        transaction=transaction,
        stamp_duty=stamp_duty,
        sebi=sebi,
        gst=_gst_on(brokerage, transaction, sebi),
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
    )


def futures_breakeven_pct(price: float, lots: int, lot_size: int) -> float:
    """Round-trip breakeven as a percent move, assuming a flat exit."""
    return futures_cost(price, price, lots, lot_size).breakeven_pct


def equity_breakeven_pct(price: float, quantity: int) -> float:
    """Round-trip breakeven as a percent move, assuming a flat exit.

    Evaluated at exit == entry, which slightly understates the true figure
    (a profitable exit carries marginally more STT), so it is the floor a
    target has to clear rather than an exact answer.
    """
    return equity_intraday_cost(price, price, quantity).breakeven_pct


def options_breakeven_pct(premium: float, lots: int, lot_size: int) -> float:
    """Round-trip breakeven as a percent of premium, assuming a flat exit."""
    return options_cost(premium, premium, lots, lot_size).breakeven_pct
