"""Entry, stop, target and size for one intraday setup.

The geometry is deliberately mechanical, because a level you cannot restate
as arithmetic is a level you cannot audit after the trade:

  sigma    the plausible move left in the session: bar ATR * sqrt(bars left)
  stop     half that sigma, or a structural level when one sits near it
  target   the reward-to-risk multiple of the stop distance, which at the
           default half-sigma stop lands on sigma itself
  risk     entry to stop, per share
  size     risk budget in rupees divided by risk per share
  target   entry plus the reward-to-risk multiple of risk

Nothing here forecasts. It turns one price, one volatility reading and one
structural level into a position whose downside is known before entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import config
import trade_costs

LONG = "LONG"
SHORT = "SHORT"


@dataclass(frozen=True)
class TradeLevels:
    """A fully specified intraday position, before any gate is applied."""

    direction: str
    entry: float
    stop: float
    target: float
    quantity: int
    lot_risk: float                 # rupees at risk if the stop fills exactly
    reward_risk: float              # target distance / stop distance
    breakeven_pct: float            # round-trip cost as a percent move
    cost_rupees: float              # round-trip cost for this exact size
    expected_range: float           # plausible remaining move, in rupees
    stop_source: str                # which level set the stop
    capital_capped: bool            # True when funding, not risk, set the size

    @property
    def risk_per_share(self) -> float:
        """Absolute entry-to-stop distance."""
        return abs(self.entry - self.stop)

    @property
    def target_distance(self) -> float:
        """Absolute entry-to-target distance."""
        return abs(self.target - self.entry)

    @property
    def target_pct(self) -> float:
        """Target distance as a percent of entry."""
        if self.entry <= 0.0:
            return 0.0
        return self.target_distance / self.entry * 100.0

    @property
    def stop_pct(self) -> float:
        """Stop distance as a percent of entry."""
        if self.entry <= 0.0:
            return 0.0
        return self.risk_per_share / self.entry * 100.0

    @property
    def cost_multiple(self) -> float:
        """How many times the round-trip cost the target is worth.

        Below about 1.0 the trade cannot win after charges even when the
        price call is correct. Returns 0.0 when the cost is unknown or zero,
        which fails the cost gate closed rather than passing it on missing
        data the way an infinite multiple would.
        """
        if self.breakeven_pct <= 0.0:
            return 0.0
        return self.target_pct / self.breakeven_pct

    @property
    def required_win_rate(self) -> float:
        """Fraction of trades that must hit target, not stop, to break even.

        Solves p*reward - (1-p)*risk = costs. This is the number that makes
        a reward-to-risk ratio meaningful or meaningless: 2:1 sounds like an
        edge, but it only says a 33% win rate breaks even *before costs*.
        Once costs are a large fraction of the money at risk - which is
        exactly what happens when the stop is tight - the real bar climbs
        fast. Returns 1.0 when costs exceed the whole reward, meaning the
        trade cannot break even at any win rate.
        """
        if self.lot_risk <= 0.0:
            return 1.0
        reward = self.lot_risk * self.reward_risk
        denominator = reward + self.lot_risk
        if denominator <= 0.0:
            return 1.0
        return min(1.0, (self.lot_risk + self.cost_rupees) / denominator)

    @property
    def reachable(self) -> bool:
        """Whether the target sits inside the plausible remaining move."""
        return self.expected_range > 0.0 and self.target_distance <= self.expected_range


def expected_remaining_range(atr_per_bar: float, bars_left: int) -> float:
    """Plausible remaining move as ATR scaled by the square root of time.

    Assumes bar-to-bar moves accumulate like a random walk, so range grows
    with sqrt(bars) rather than linearly. That is an assumption, not a
    measured property of NSE intraday data, and it is the single loosest
    link in the reachability gate: a trending day exceeds it and a dead
    afternoon falls short. It is here to reject targets that need a move
    twice the size of anything the day has offered, not to predict range.
    """
    if atr_per_bar <= 0.0 or bars_left <= 0:
        return 0.0
    return atr_per_bar * math.sqrt(bars_left)


def _structural_levels(direction: str, opening_low: "float | None",
                       opening_high: "float | None", day_low: "float | None",
                       day_high: "float | None") -> list[tuple[float, str]]:
    """Every candidate structural stop for this direction, preferred first.

    Returns all of them rather than just the first. Returning only the
    opening range meant that when it sat on the wrong side of the entry the
    session low was never tried, so the fallback could not fire.
    """
    if direction == LONG:
        pairs = ((opening_low, "opening-range low"), (day_low, "session low"))
    else:
        pairs = ((opening_high, "opening-range high"), (day_high, "session high"))
    return [(level, name) for level, name in pairs
            if level is not None and level > 0.0]


def build_levels(direction: str, entry: float, atr_per_bar: float,
                 bars_left: int, opening_low: "float | None" = None,
                 opening_high: "float | None" = None,
                 day_low: "float | None" = None,
                 day_high: "float | None" = None,
                 capital: float = config.SCAN_CAPITAL,
                 risk_pct: float = config.SCAN_RISK_PCT_PER_TRADE,
                 reward_risk: float = config.SCAN_REWARD_RISK,
                 stop_fraction: float = config.SCAN_STOP_FRACTION,
                 band: float = config.SCAN_STRUCTURE_BAND,
                 leverage: float = config.SCAN_MIS_LEVERAGE
                 ) -> "TradeLevels | None":
    """Assemble levels for one direction, or None if the inputs cannot support them.

    Returns None rather than a degraded position when entry or ATR is
    missing: a stop derived from a zero ATR would sit on top of the entry
    and size the position at the full risk budget divided by nearly nothing.
    """
    if direction not in (LONG, SHORT):
        raise ValueError(f"direction must be {LONG} or {SHORT}, got {direction!r}")
    if entry is None or entry <= 0.0 or atr_per_bar is None or atr_per_bar <= 0.0:
        return None
    if capital <= 0.0 or risk_pct <= 0.0 or reward_risk <= 0.0:
        return None
    if leverage < 1.0 or band < 0.0 or band >= 1.0 or stop_fraction <= 0.0:
        return None

    # Risk is bounded into [min_multiple, atr_multiple] ATRs. Taking the
    # further of structure and the ATR band instead - the obvious reading of
    # "beyond the noise" - made every clean breakout unactionable: after a
    # 1.5% move off the opening range the structural stop sits 1.5% away, so
    # a 2:1 target needs 3%, which the reachability gate correctly rejects.
    # Clamping keeps the structural level when it falls inside the band and
    # ignores it when it does not.
    sigma = expected_remaining_range(atr_per_bar, bars_left)
    if sigma <= 0.0:
        return None
    base = sigma * stop_fraction
    if base <= 0.0:
        return None
    # A structural level is preferred when it sits just inside the volatility
    # distance, since a real level is a better stop than an arithmetic one.
    # The window is [(1-band)*base, base] and deliberately never exceeds
    # base: a wider stop scales the target past sigma, which the reachability
    # gate then rejects. Accepting structure out to 1.4*base made every
    # structural stop in (base, 1.4*base] a guaranteed rejection, which left
    # the feature all but unusable and shipped the suite red.
    distance, stop_source = base, "volatility"
    for level, name in _structural_levels(direction, opening_low, opening_high,
                                          day_low, day_high):
        gap = abs(entry - level)
        on_protective_side = (level < entry if direction == LONG
                              else level > entry)
        if on_protective_side and (1.0 - band) * base <= gap <= base:
            distance, stop_source = gap, name
            break
    stop = entry - distance if direction == LONG else entry + distance
    if (direction == LONG and stop >= entry) or (direction == SHORT and stop <= entry):
        return None

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0.0:
        return None
    budget = capital * risk_pct / 100.0
    by_risk = int(budget // risk_per_share)
    # Funding cap: risk-based sizing on a tight stop asks for a position many
    # times the account. MIS leverage relaxes it but does not remove it.
    affordable = int((capital * leverage) // entry)
    quantity = min(by_risk, affordable)
    capital_capped = quantity < by_risk
    if quantity <= 0:
        return None

    if direction == LONG:
        target = entry + reward_risk * risk_per_share
    else:
        target = entry - reward_risk * risk_per_share
        if target <= 0.0:
            return None

    cost = trade_costs.equity_intraday_cost(entry, entry, quantity)
    return TradeLevels(
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        quantity=quantity,
        lot_risk=risk_per_share * quantity,
        reward_risk=reward_risk,
        breakeven_pct=cost.breakeven_pct,
        cost_rupees=cost.total,
        expected_range=sigma,
        stop_source=stop_source,
        capital_capped=capital_capped,
    )
