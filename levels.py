"""Entry, stop, target and size for one intraday setup.

The geometry is deliberately mechanical, because a level you cannot restate
as arithmetic is a level you cannot audit after the trade:

  plausible move
           what is left in the session: bar ATR * sqrt(bars left). It was
           called `sigma`, which it is NOT - it measures about 1.4
           standard deviations. See expected_remaining_range. Do not call
           it a "band" either: `band` is already a PARAMETER of
           build_levels, meaning the structural window fraction.
  stop     half a plausible move, or a structural level near one
  target   the reward-to-risk multiple of the stop distance, which at the
           default half-move stop lands on one full plausible move
  risk     entry to stop, per share
  size     the largest share count whose loss at the stop PLUS its
           round-trip charges fits the rupee risk budget, capped by what
           capital times MIS leverage can fund. Charges used to be priced
           after sizing: in the scan logs of 2026-09-22 and 23 the median
           stop-out lost about 1,100 (p90 about 1,200) against a 1,000
           budget, and 97% of the 7,079 rows with levels exceeded it.
  target   entry plus the reward-to-risk multiple of risk

Nothing here forecasts. It turns one price, one volatility reading and one
structural level into a position whose downside is known before entry.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import config
import trade_costs

logger = logging.getLogger(__name__)

LONG = "LONG"
SHORT = "SHORT"

# Upper bound on the sizing walk in _quantity_inside_budget. NOT a tuning
# knob: each step provably lands at or above the answer and strictly below
# the previous guess. Over 20,000 random inputs in the space the scan
# produces - prices 20 to 12,000, stops 0.02% to 3%, budgets 250 to
# 20,000 - the walk never needed more than six steps (the brokerage cap is
# the only non-linear term, and it is worth at most about 47 rupees). That
# figure is for that space only: far outside it, at sub-rupee prices with
# stops a few millionths of the price, a review measured 19 steps and a
# log-uniform sweep 20. The bound exists so that an input nobody
# anticipated fails closed with a warning instead of spinning.
_SIZING_MAX_STEPS = 64


def _finite_positive(value) -> bool:
    """True for a real, finite number above zero; False for None, NaN or inf.

    WHY NOT `value <= 0.0`. Every comparison with NaN is False, so a guard
    written that way waves NaN straight through - and int(budget // nan)
    then raised deep inside the sizer instead of build_levels returning
    None as its docstring promises. A NaN ATR is not hypothetical: it is
    what a symbol whose bars are all missing produces.
    """
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def risk_budget(capital: float, risk_pct: float) -> float:
    """Rupees one trade may lose at its stop, charges included; 0.0 if refused.

    The single place the per-trade budget is computed, so that equity
    sizing here and option sizing in options_chain cannot disagree about
    what a trade is allowed to lose.

    FAILS CLOSED, returning 0.0 rather than a clamped figure, when either
    input is missing, non-finite or not positive, or when risk_pct is above
    config.SCAN_MAX_RISK_PCT. The ceiling is read at call time, not bound
    as a default, so a test or a config change moves it everywhere at once.
    """
    if not (_finite_positive(capital) and _finite_positive(risk_pct)):
        return 0.0
    if float(risk_pct) > float(config.SCAN_MAX_RISK_PCT):
        return 0.0
    return float(capital) * float(risk_pct) / 100.0


@dataclass(frozen=True)
class TradeLevels:
    """A fully specified intraday position, before any gate is applied."""

    direction: str
    entry: float
    stop: float
    target: float
    quantity: int
    lot_risk: float                 # PRICE risk: rupees lost to the move alone
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
    def risk_with_costs(self) -> float:
        """Everything lost if the stop fills exactly: the move plus charges.

        THIS is the number the sizer holds inside the risk budget, and the
        one the portfolio cap adds up. lot_risk stays PRICE risk only,
        because required_win_rate solves p*reward - (1-p)*risk = costs and
        needs the move and the charges as separate terms; folding the
        charges into lot_risk would count them twice there.

        The charges are priced at a flat exit (exit == entry), the same
        call that produces cost_rupees and breakeven_pct. At the real stop
        price a LONG pays marginally less (a smaller sell leg) and a SHORT
        marginally more (a larger buy-back leg) - for a stop 0.5% away on a
        1,00,000 ticket the short's difference is about 13 paise.
        """
        return self.lot_risk + self.cost_rupees

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

        THE ASSUMPTION, WHICH THIS SYSTEM BREAKS. The formula treats every
        non-winner as a full loss of `risk`. That is right for a trade held
        strictly to its stop or its target, and wrong for this one: the
        intraday scanner squares off at the close, so a position that
        reaches neither level exits at whatever the market is then.

        Measured over 93 sessions and 2,436 signals: 280 reached the
        target, 782 stopped out, and 1,374 - 56%, the majority - ended at
        NEITHER, averaging +0.24R rather than -1R. Counting those as full
        losses overstates the bar this column reports, which for a tool
        whose purpose is refusing trades means refusing too many.

        It is left as it is, because the arithmetic is correct for what it
        states and the alternative is a fitted constant that would move
        with every regime. What changed is that the assumption is now
        written down, here and in the column's own tooltip.
        """
        if self.lot_risk <= 0.0:
            return 1.0
        reward = self.lot_risk * self.reward_risk
        denominator = reward + self.lot_risk
        if denominator <= 0.0:
            return 1.0
        return min(1.0, (self.lot_risk + self.cost_rupees) / denominator)

    # Relative slack on the target-versus-range comparison. NOT a
    # judgement about how much overshoot is acceptable: it exists because
    # the two quantities are the SAME NUMBER by construction and floating
    # point cannot be relied on to say so.
    #
    # With the default volatility stop at half a sigma and a 2:1 target,
    # target_distance = 2 * 0.5 * sigma = sigma = expected_range exactly.
    # Measured over 4,000 randomised volatility-stopped trades: 100% land
    # mathematically on the line, and a bare `<=` refused 48.8% of them -
    # a coin flip per symbol, silently suppressing about half of every
    # scan's actionable setups. setups._reach_reason already documents the
    # intent, that this "is an identity, not a test" and "can only fail
    # when there is no remaining move at all".
    _REACH_TOLERANCE = 1e-9

    @property
    def reachable(self) -> bool:
        """Whether the target sits inside the plausible remaining move."""
        if self.expected_range <= 0.0:
            return False
        slack = self.expected_range * (1.0 + self._REACH_TOLERANCE)
        return self.target_distance <= slack


def expected_remaining_range(atr_per_bar: float, bars_left: int) -> float:
    """Plausible remaining move as ATR scaled by the square root of time.

    Assumes bar-to-bar moves accumulate like a random walk, so range grows
    with sqrt(bars) rather than linearly. That is an assumption, not a
    measured property of NSE intraday data, and it is the single loosest
    link in the reachability gate: a trending day exceeds it and a dead
    afternoon falls short. It is here to reject targets that need a move
    twice the size of anything the day has offered, not to predict range.

    THIS IS A BAND, NOT A SIGMA, AND THE CODE USED TO CALL IT ONE. ATR is
    a mean absolute TRUE RANGE - a high-to-low span covering both
    directions - while a sigma is the standard deviation of the signed
    move. This returns the first; everything downstream was reading the
    second.

    Measured two ways, which is why a range is quoted rather than a point:

        per-observation median ratio        1.37
        per-symbol median ratio             1.50   (IQR 1.39 - 1.58)
        terminal move inside it            88.6%   (a 1-sigma band: 68.3%)

    So one band is about 1.4 standard deviations, and the published
    geometry restates as:

                        as written   in true sigma   touch probability
        stop              0.5 move     0.69 - 0.75      45% - 49%
        target            1.0 move     1.37 - 1.50      13% - 17%

    The touch figures previously quoted - 62% and 32% - are the values at
    r = 1.0, which is the assumption this measurement rejects. Both were
    overstated.

    WHAT THIS DOES NOT RESCUE, because reading it as good news is the
    obvious mistake. The breakeven hit rate is stop / (stop + target), and
    at half a band against one band that is 0.5b / 1.5b = 1/3 whatever b
    measures - the unit cancels exactly. The empirical check against it is
    a raw count with no unit in it at all: 280 targets against 782 stops is
    26.4%, against the 33.3% required. Correcting the unit makes the
    Varsity comparison less lopsided than it was relayed; it moves the
    expectancy not at all.
    """
    # Finite-and-positive rather than `<= 0`, which NaN passes: a NaN ATR
    # returned a NaN range, and every downstream `range <= 0.0` guard then
    # waved it through as well.
    if not (_finite_positive(atr_per_bar) and _finite_positive(bars_left)):
        return 0.0
    return atr_per_bar * math.sqrt(bars_left)


def _structural_levels(direction: str, opening_low: "float | None",
                       opening_high: "float | None", day_low: "float | None",
                       day_high: "float | None") -> list[tuple[float, str]]:
    """Every candidate structural stop for this direction, preferred first.

    Returns all of them rather than just the first. Returning only the
    opening range meant that when it sat on the wrong side of the entry the
    session low was never tried, so the fallback could not fire.

    THE PIVOT LADDER IS DELIBERATELY ABSENT, and this is a measurement
    rather than an oversight. Adding S1-S3/R1-R3 here was built and then
    tested; run `python -m pivot_measurement` to reproduce every figure.

    Conditioning on the session - which holds the date, the symbol and the
    session's own path fixed - finds nothing. Mantel-Haenszel over 8,687
    sessions gives a common odds ratio of 1.100, p 0.616, and above 1.0
    means a pivot stop is hit MORE often. Paired session means agree:
    +0.445 pp, CI [-1.299, +2.218].

    An unpaired estimate over the same data read -2.19 pp and looked
    convincing. It was selection: control stops, which are placed at
    random and so cannot be influenced by a pivot at all, are hit 40.75%
    in sessions that have a pivot in the usable window against 42.72% in
    sessions that do not. That -1.97 pp gap is almost the whole apparent
    effect - pivot-bearing sessions are simply quieter.

    Two cautions for anyone re-running this. Pairing does NOT hold
    distance constant by itself; over the full 0.15-2.0 move range the
    same paired estimator returns +6.27 pp, a pure distance artefact. It
    is the [0.30, 0.50] move WINDOW that makes it honest. And a
    permutation test that shuffles the pivot label within distance buckets
    only is invalid here - it treats 79,725 observations from 14,474
    sessions as exchangeable, and returns p 0.02 for an estimate whose
    date-blocked interval spans zero.

    What is fairly excluded is a benefit larger than about 1.3 pp on a
    ~40% hit rate. A smaller edge would not be detected. Since a pivot
    stop is always TIGHTER than the volatility stop it would replace,
    shipping it on that basis would raise the stop-hit rate for an effect
    no test can find. indicators.pivot_ladder is kept for DISPLAY only.
    """
    if direction == LONG:
        pairs = ((opening_low, "opening-range low"), (day_low, "session low"))
    else:
        pairs = ((opening_high, "opening-range high"), (day_high, "session high"))
    return [(level, name) for level, name in pairs
            if level is not None and level > 0.0]


def _quantity_inside_budget(entry: float, risk_per_share: float,
                            budget: float, ceiling: int) -> int:
    """Largest share count whose stop-out loss INCLUDING charges fits the budget.

    Solves q * risk_per_share + cost(q) <= budget for a whole number q, with
    cost(q) = trade_costs.equity_intraday_cost(entry, entry, q).total - the
    same call that prices cost_rupees, so the stored fields satisfy
    lot_risk + cost_rupees <= budget exactly. `ceiling` must be at or above
    the answer; the price-only size budget // risk_per_share always is,
    because charges can only take shares away.

    WHY A WALK AND NOT ONE DIVISION. Cost is not linear in q: brokerage is
    0.03% a leg but capped at 20 rupees an order, so the per-share charge
    FALLS as the ticket grows. The per-share loss risk_per_share +
    cost(q)/q is therefore non-increasing in q, which gives the walk its
    guarantee: from any guess above the answer, budget divided by the
    per-share loss at that guess is still at or above the answer (the loss
    per share there is no larger than at the answer), so the next guess
    min(guess - 1, that estimate) never undershoots and always shrinks. The
    first guess that fits is the exact largest q. Measured over 20,000
    random inputs (prices 20 to 12,000, stops 0.02% to 3%, budgets 250 to
    20,000): two cost evaluations in the typical case, six at worst, and
    q + 1 overshot the budget in every one.

    Returns 0 when not one share fits, which the caller turns into None.
    """
    quantity = ceiling
    for _ in range(_SIZING_MAX_STEPS):
        if quantity <= 0:
            return 0
        cost = trade_costs.equity_intraday_cost(entry, entry, quantity).total
        if quantity * risk_per_share + cost <= budget:
            return quantity
        per_share_loss = risk_per_share + cost / quantity
        quantity = min(quantity - 1, int(budget // per_share_loss))
    logger.warning("Sizing did not settle within %d steps (entry %s, risk "
                   "per share %s, budget %s); refusing the position rather "
                   "than guessing a size", _SIZING_MAX_STEPS, entry,
                   risk_per_share, budget)
    return 0


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

    Also None - failing closed rather than trimming - when any numeric
    input is NaN or infinite, when risk_pct is above
    config.SCAN_MAX_RISK_PCT, and when not a single share fits the budget
    once its round-trip charges are counted.
    """
    if direction not in (LONG, SHORT):
        raise ValueError(f"direction must be {LONG} or {SHORT}, got {direction!r}")
    # FINITE AND POSITIVE, not `<= 0.0`, which NaN passes. See
    # _finite_positive: a NaN entry or ATR used to reach int(budget // nan)
    # and raise instead of returning None.
    if not (_finite_positive(entry) and _finite_positive(atr_per_bar)):
        return None
    if not (_finite_positive(capital) and _finite_positive(risk_pct)
            and _finite_positive(reward_risk)):
        return None
    # The per-trade ceiling. risk_budget applies it too; checking it here
    # as well makes the refusal explicit at the top of the function rather
    # than a side effect of a zero budget much further down.
    if risk_pct > config.SCAN_MAX_RISK_PCT:
        return None
    if not (_finite_positive(leverage) and _finite_positive(stop_fraction)):
        return None
    if not math.isfinite(band):
        return None
    if leverage < 1.0 or band < 0.0 or band >= 1.0:
        return None

    # Risk is bounded into [min_multiple, atr_multiple] ATRs. Taking the
    # further of structure and the ATR band instead - the obvious reading of
    # "beyond the noise" - made every clean breakout unactionable: after a
    # 1.5% move off the opening range the structural stop sits 1.5% away, so
    # a 2:1 target needs 3%, which the reachability gate correctly rejects.
    # Clamping keeps the structural level when it falls inside the band and
    # ignores it when it does not.
    # NOT named `band`: this function already has a parameter by that
    # name (SCAN_STRUCTURE_BAND, the structural window FRACTION), and
    # shadowing it silently collapses the window to [(1 - plausible move)
    # * base, base]. The test suite caught exactly that.
    plausible_move = expected_remaining_range(atr_per_bar, bars_left)
    if not _finite_positive(plausible_move):
        return None
    base = plausible_move * stop_fraction
    if base <= 0.0:
        return None
    # A structural level is preferred when it sits just inside the volatility
    # distance, since a real level is a better stop than an arithmetic one.
    # The window is [(1-band)*base, base] and deliberately never exceeds
    # base: a wider stop scales the target past one plausible move, which the reachability
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
    if not _finite_positive(risk_per_share):
        return None
    budget = risk_budget(capital, risk_pct)
    if budget <= 0.0:
        return None
    # CHARGES COME OUT OF THE BUDGET, not on top of it. The old rule was
    # budget // risk_per_share with the round trip priced afterwards, so
    # a stop-out lost the budget PLUS its charges: a median of about 1,100
    # against a stated 1,000 in the 2026-09-22/23 scan logs. The
    # price-only size is kept as the walk's starting ceiling, because
    # charges can only remove shares.
    by_price = int(budget // risk_per_share)
    by_risk = _quantity_inside_budget(entry, risk_per_share, budget, by_price)
    # Funding cap: risk-based sizing on a tight stop asks for a position many
    # times the account. MIS leverage relaxes it but does not remove it.
    # Fewer shares than by_risk always still fit the budget, because the
    # stop-out loss including charges only grows with quantity.
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
        expected_range=plausible_move,
        stop_source=stop_source,
        capital_capped=capital_capped,
    )
