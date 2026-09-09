"""Turn one symbol's bars into a gated, ranked intraday setup.

The shape mirrors decision.py on purpose: every gate records its own
PASS/FAIL line, a setup is actionable only when all of them pass, and the
rank score is a bounded weighted sum of readings rather than an opaque
model output. If you cannot read the reasons list and rebuild the verdict by
hand, this module is wrong.

Direction is not predicted. It is read off two agreeing structural facts -
which side of VWAP the price sits, and which side of the opening range it
has broken. When those two disagree the symbol is chop and gets no setup,
which is the common case and is meant to be.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

import config
import indicators
import levels as levels_mod
from levels import LONG, SHORT, TradeLevels

logger = logging.getLogger(__name__)

NO_SETUP = "NO SETUP"


@dataclass(frozen=True)
class Readings:
    """Every measured input behind one symbol's verdict."""

    symbol: str
    ticker: str
    last: float
    prev_close: "float | None"
    day_change_pct: "float | None"
    vwap: "float | None"
    vwap_distance_pct: "float | None"
    opening_range: "indicators.OpeningRange | None"
    cpr: "indicators.CentralPivotRange | None"
    atr_bar: "float | None"
    rvol: "float | None"
    relative_strength: "float | None"
    turnover_20d: "float | None"
    oi_change_pct: "float | None"          # derivatives positioning, F&O only
    futures_share: "float | None"          # futures vs options turnover mix
    derivatives_turnover: "float | None"   # rupees, futures + option premium
    day_high: "float | None"
    day_low: "float | None"
    minutes_left: int
    bars_left: int
    session_bar_count: int = 0        # bars traded today at the scan instant
    range_closed: bool = True         # has the session passed its opening range



@dataclass(frozen=True)
class Setup:
    """One symbol's intraday verdict, its levels and its full audit trail."""

    readings: Readings
    direction: str
    levels: "TradeLevels | None"
    passed: bool
    rank_score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        """Convenience passthrough for reporting."""
        return self.readings.symbol

    @property
    def actionable(self) -> bool:
        """Whether every gate passed and levels exist."""
        return self.passed and self.levels is not None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a value into [low, high]."""
    return max(low, min(high, value))


def _bars_per_session(frame: pd.DataFrame) -> int:
    """Typical bar count in a full session, used to convert minutes to bars."""
    try:
        counts: dict = {}
        for stamp in frame.index:
            counts[stamp.date()] = counts.get(stamp.date(), 0) + 1
    except AttributeError:
        return 0
    if not counts:
        return 0
    return int(pd.Series(list(counts.values())).median())


def measure(symbol: str, ticker: str, intraday: pd.DataFrame,
            daily: "pd.DataFrame | None", benchmark_change_pct: "float | None",
            now, fo_state=None) -> "Readings | None":
    """Compute every reading for one symbol, or None without a usable session.

    `fo_state` is the derivatives snapshot for this underlying when one
    exists. Names without listed derivatives simply carry None, which the
    reporting distinguishes from a reading of zero: no open interest is not
    the same fact as flat open interest.
    """
    today = indicators.session_bars(intraday)
    if today is None:
        return None
    closes = today["Close"].dropna()
    if closes.empty:
        return None
    last = float(closes.iloc[-1])
    if last <= 0.0 or math.isnan(last):
        return None

    prev_close = turnover = None
    cpr = None
    if daily is not None and not daily.empty:
        prior = daily.dropna(subset=["Close"])
        # Exclude today's partial daily bar: its high and low are incomplete,
        # and a CPR built from them would move during the session. Failing
        # closed on an unexpected index is deliberate - swallowing the error
        # left the frame unfiltered, so prev_close became today's own partial
        # close and every reading derived from it leaked the future.
        try:
            # Anchored on the CLOCK, not on the last surviving bar. The old
            # anchor slid to D-1 whenever no bar of the replay date survived
            # the cutoff, which let D's completed daily bar become
            # prev_close - a straight leak into day_change_pct and from
            # there into the relative-strength gate.
            cutoff_day = now.astimezone(indicators.IST).date()
            prior = prior.loc[[ts for ts in prior.index
                               if ts.date() < cutoff_day]]
        except AttributeError:
            logger.warning("%s: daily index is not timestamped, so today's "
                           "partial bar cannot be excluded; dropping daily "
                           "context rather than leaking it", symbol)
            prior = prior.iloc[0:0]
        if not prior.empty:
            prev_close = float(prior["Close"].iloc[-1])
            if "Volume" in prior.columns:
                volume = float(prior["Volume"].tail(20).mean())
                if not math.isnan(volume):
                    turnover = volume * last
            if {"High", "Low"} <= set(prior.columns):
                prev_high = float(prior["High"].iloc[-1])
                prev_low = float(prior["Low"].iloc[-1])
                cpr = indicators.central_pivot_range(prev_high, prev_low,
                                                     prev_close)

    vwap = indicators.vwap(today)
    day_change = indicators.percent_change(last, prev_close) if prev_close else None
    minutes_left = indicators.minutes_left_in_session(now)
    per_session = _bars_per_session(intraday)
    session_minutes = _session_length_minutes()
    bars_left = 0
    if per_session > 0 and session_minutes > 0:
        bars_left = int(minutes_left / (session_minutes / per_session))

    return Readings(
        symbol=symbol,
        ticker=ticker,
        last=last,
        prev_close=prev_close,
        day_change_pct=day_change,
        vwap=vwap,
        vwap_distance_pct=indicators.percent_change(last, vwap) if vwap else None,
        opening_range=indicators.opening_range(intraday),
        cpr=cpr,
        atr_bar=indicators.atr(intraday),
        rvol=indicators.relative_volume(intraday),
        relative_strength=indicators.relative_strength(day_change,
                                                       benchmark_change_pct),
        turnover_20d=turnover,
        oi_change_pct=(fo_state.oi_change_pct if fo_state is not None else None),
        futures_share=(fo_state.futures_share if fo_state is not None else None),
        derivatives_turnover=(fo_state.derivatives_turnover_rupees
                              if fo_state is not None else None),
        day_high=_extreme(today, "High", high=True),
        day_low=_extreme(today, "Low", high=False),
        minutes_left=minutes_left,
        bars_left=bars_left,
        session_bar_count=len(today),
        range_closed=indicators.opening_range_closed(intraday),
    )


def _extreme(bars: pd.DataFrame, column: str, high: bool) -> "float | None":
    """Session high or low, or None when the column is missing or all NaN."""
    if column not in bars.columns:
        return None
    values = bars[column].dropna()
    if values.empty:
        return None
    value = float(values.max() if high else values.min())
    return None if math.isnan(value) else value


def _session_length_minutes() -> int:
    """Length of the NSE equity session in minutes, from config."""
    open_h, open_m = config.SCAN_SESSION_OPEN
    close_h, close_m = config.SCAN_SESSION_CLOSE
    return (close_h * 60 + close_m) - (open_h * 60 + open_m)


def choose_direction(reading: Readings) -> tuple[str, str]:
    """Direction from VWAP side and opening-range break, plus a reason line.

    Both facts must agree. Requiring agreement is what keeps the scanner
    quiet: on most symbols on most days price is chopping around VWAP inside
    the opening range, and that is not a setup at any price.
    """
    orb = reading.opening_range
    if reading.vwap is None:
        return NO_SETUP, "no VWAP (volume missing this session) [FAIL]"
    if orb is None:
        return NO_SETUP, "no opening range (session start missing) [FAIL]"
    if not reading.range_closed:
        # Distinct from a genuine no-break: the comparison itself is not yet
        # defined, because the latest bar is one of the bars defining the
        # range. Reporting this as "no agreement" would present arithmetic
        # as a market reading.
        return NO_SETUP, (
            f"the opening range has not closed yet - only "
            f"{reading.session_bar_count} bar(s) traded, all inside the first "
            f"{orb.minutes} minutes, so no break is computable [N/A]")
    above_vwap = reading.last > reading.vwap
    broke_up = reading.last > orb.high
    broke_down = reading.last < orb.low
    if above_vwap and broke_up:
        return LONG, (f"above VWAP {reading.vwap:.2f} and broke the opening "
                      f"range high {orb.high:.2f} [PASS]")
    if not above_vwap and broke_down:
        return SHORT, (f"below VWAP {reading.vwap:.2f} and broke the opening "
                       f"range low {orb.low:.2f} [PASS]")
    inside = not broke_up and not broke_down
    detail = ("still inside the opening range" if inside
              else "broken the opening range against its VWAP side")
    return NO_SETUP, f"no agreement: price is {detail} [FAIL]"


def _liquidity_reason(reading: Readings) -> tuple[bool, str]:
    """Turnover gate: intraday needs depth to get out, not just to get in."""
    if reading.turnover_20d is None:
        return False, "no volume history, so depth is unknown [FAIL]"
    ok = reading.turnover_20d >= config.SCAN_MIN_TURNOVER
    return ok, (f"20d turnover {reading.turnover_20d:,.0f} vs floor "
                f"{config.SCAN_MIN_TURNOVER:,.0f} [{'PASS' if ok else 'FAIL'}]")


def _rvol_reason(reading: Readings) -> tuple[bool, str]:
    """Participation gate, failing closed when the baseline is missing."""
    if reading.rvol is None:
        return False, "no relative-volume baseline (needs 2+ sessions) [FAIL]"
    ok = reading.rvol >= config.SCAN_RVOL_MIN
    return ok, (f"relative volume {reading.rvol:.2f}x vs floor "
                f"{config.SCAN_RVOL_MIN:.2f}x [{'PASS' if ok else 'FAIL'}]")


def _strength_reason(reading: Readings, direction: str) -> tuple[bool, str]:
    """Relative-strength gate: the move should lead the index, not follow it."""
    value = reading.relative_strength
    if value is None:
        return False, "no benchmark comparison available [FAIL]"
    ok = value > 0.0 if direction == LONG else value < 0.0
    want = "outperform" if direction == LONG else "underperform"
    return ok, (f"relative strength {value:+.2f}pp vs Nifty, needs to "
                f"{want} [{'PASS' if ok else 'FAIL'}]")


def _time_reason(reading: Readings) -> tuple[bool, str]:
    """Session-time gate: a target needs runway to be reached at all."""
    ok = reading.minutes_left >= config.SCAN_MIN_MINUTES_LEFT
    return ok, (f"{reading.minutes_left} min left vs minimum "
                f"{config.SCAN_MIN_MINUTES_LEFT} [{'PASS' if ok else 'FAIL'}]")


def _cost_reason(trade: TradeLevels) -> tuple[bool, str]:
    """Cost gate: the target must be worth several round trips, not one."""
    multiple = trade.cost_multiple
    ok = multiple >= config.SCAN_COST_MULTIPLE
    shown = "inf" if math.isinf(multiple) else f"{multiple:.1f}"
    return ok, (f"target {trade.target_pct:.2f}% is {shown}x the "
                f"{trade.breakeven_pct:.3f}% round-trip cost, needs "
                f"{config.SCAN_COST_MULTIPLE:.0f}x "
                f"[{'PASS' if ok else 'FAIL'}]")


def _oi_reason(reading: Readings) -> tuple[bool, str]:
    """Derivatives positioning: is open interest building or unwinding?

    Rising open interest on a directional move is read as fresh positioning
    for either side - new longs on a rally, new shorts on a decline - while
    falling OI is an unwind of existing positions and conventionally less
    durable. Returned as INFO unless SCAN_REQUIRE_OI_CONFIRMATION is set,
    because the convention has no backtest here and a veto built on it would
    look like rigour while being a guess.
    """
    change = reading.oi_change_pct
    required = config.SCAN_REQUIRE_OI_CONFIRMATION
    if change is None:
        line = "no listed derivatives, so open interest says nothing"
        return (not required), f"{line} [{'FAIL' if required else 'INFO'}]"
    building = change >= config.SCAN_MIN_OI_CHANGE_PCT
    read = "building" if building else "unwinding"
    share = reading.futures_share
    mix = (f", {share * 100:.0f}% of derivatives turnover in futures"
           if share is not None else "")
    verdict = ("PASS" if building else "FAIL") if required else "INFO"
    return (building or not required), (
        f"open interest {change:+.2f}% ({read}){mix} [{verdict}]")


def _win_rate_reason(trade: TradeLevels) -> tuple[bool, str]:
    """The gate that costs actually bind: what win rate this trade needs.

    A reward-to-risk ratio on its own says nothing about profitability. This
    converts levels plus real charges into the hit rate required to break
    even, and rejects setups where the broker has already taken the edge.
    """
    needed = trade.required_win_rate
    ok = needed <= config.SCAN_MAX_WIN_RATE
    return ok, (f"needs a {needed * 100:.1f}% win rate to break even after "
                f"{trade.cost_rupees:,.0f} of costs against {trade.lot_risk:,.0f} "
                f"at risk, ceiling {config.SCAN_MAX_WIN_RATE * 100:.0f}% "
                f"[{'PASS' if ok else 'FAIL'}]")


def _reach_reason(trade: TradeLevels) -> tuple[bool, str]:
    """Confirm the target sits inside the plausible remaining move.

    Honest framing: with the default half-sigma stop and a 2:1 target this
    is an identity, not a test - the target lands exactly on sigma by
    construction, so it can only fail when there is no remaining move at
    all. It is reported as an invariant rather than a PASS so the trail does
    not imply a check that cannot bind. A structural stop tighter than the
    volatility distance is the one case that leaves real headroom.
    """
    if trade.expected_range <= 0.0:
        return False, "no remaining move left in the session [FAIL]"
    headroom = trade.expected_range - trade.target_distance
    if headroom > 0.01 * trade.expected_range:
        detail = (f"{headroom:.2f} inside the {trade.expected_range:.2f} "
                  f"plausible remaining move")
    else:
        detail = (f"exactly on the {trade.expected_range:.2f} plausible "
                  f"remaining move, which the 2:1 target guarantees")
    return trade.reachable, (
        f"target {trade.target_distance:.2f} away sits {detail} "
        f"[{'OK' if trade.reachable else 'FAIL'}]")


def score_setup(reading: Readings, trade: TradeLevels) -> float:
    """Bounded 0..1 rank score from five independent readings.

    Each term is scaled per symbol, never against the rest of the scan, so
    one symbol's score does not shift when another is added or removed.
    Weights are a judgement call and have no backtest behind them.
    """
    # No `or` fallbacks here: score_setup runs only after every gate passed,
    # so rvol and relative_strength are known non-None. A default would
    # advertise a contract this function does not have.
    rvol = _clamp((reading.rvol - 1.0) / 2.0)
    strength = _clamp(abs(reading.relative_strength) / 2.0)
    stretch = _clamp(abs(reading.vwap_distance_pct or 0.0) / 1.5)
    cost = _clamp((trade.cost_multiple - 1.0) / 5.0)
    headroom = 0.0
    if trade.target_distance > 0.0:
        headroom = _clamp(trade.expected_range / trade.target_distance - 1.0)
    # A name with no derivatives scores 0.0 on this term rather than a
    # neutral 0.5: absent data must not flatter a symbol against one that
    # has the data and shows an unwind.
    positioning = 0.0
    if reading.oi_change_pct is not None:
        positioning = _clamp(reading.oi_change_pct / 10.0)
    return round(0.25 * rvol + 0.20 * strength + 0.15 * stretch
                 + 0.15 * cost + 0.10 * headroom + 0.15 * positioning, 4)


def evaluate(reading: Readings, capital: float = config.SCAN_CAPITAL,
             risk_pct: float = config.SCAN_RISK_PCT_PER_TRADE) -> Setup:
    """Run every gate for one symbol and return its complete verdict."""
    reasons: list[str] = []
    price_ok = reading.last >= config.SCAN_MIN_PRICE
    reasons.append(f"price {reading.last:.2f} vs floor "
                   f"{config.SCAN_MIN_PRICE:.2f} "
                   f"[{'PASS' if price_ok else 'FAIL'}]")
    liquid_ok, liquid_line = _liquidity_reason(reading)
    reasons.append(liquid_line)
    direction, direction_line = choose_direction(reading)
    reasons.append(direction_line)
    time_ok, time_line = _time_reason(reading)
    reasons.append(time_line)

    if direction == NO_SETUP:
        return Setup(reading, NO_SETUP, None, False, 0.0, reasons)

    rvol_ok, rvol_line = _rvol_reason(reading)
    reasons.append(rvol_line)
    strength_ok, strength_line = _strength_reason(reading, direction)
    reasons.append(strength_line)
    oi_ok, oi_line = _oi_reason(reading)
    reasons.append(oi_line)

    orb = reading.opening_range
    trade = levels_mod.build_levels(
        direction=direction,
        entry=reading.last,
        atr_per_bar=reading.atr_bar or 0.0,
        bars_left=reading.bars_left,
        opening_low=orb.low if orb else None,
        opening_high=orb.high if orb else None,
        day_low=reading.day_low,
        day_high=reading.day_high,
        capital=capital,
        risk_pct=risk_pct,
    )
    if trade is None:
        reasons.append("could not build levels: no ATR or the stop collapsed "
                       "onto the entry [FAIL]")
        return Setup(reading, direction, None, False, 0.0, reasons)

    reasons.append(f"stop {trade.stop:.2f} from the {trade.stop_source}, "
                   f"{trade.stop_pct:.2f}% away, sizing {trade.quantity} "
                   f"shares for {trade.lot_risk:,.0f} at risk [PASS]")
    if trade.capital_capped:
        reasons.append(f"size capped by funding at {trade.quantity} shares "
                       f"({trade.entry * trade.quantity:,.0f} notional), so "
                       f"less than the full risk budget is deployed [INFO]")
    cost_ok, cost_line = _cost_reason(trade)
    reasons.append(cost_line)
    win_ok, win_line = _win_rate_reason(trade)
    reasons.append(win_line)
    reach_ok, reach_line = _reach_reason(trade)
    reasons.append(reach_line)

    passed = all((price_ok, liquid_ok, time_ok, rvol_ok, strength_ok,
                  oi_ok, cost_ok, win_ok, reach_ok))
    score = score_setup(reading, trade) if passed else 0.0
    return Setup(reading, direction, trade, passed, score, reasons)


def rank(setups: list[Setup]) -> list[Setup]:
    """Actionable setups first by descending score, then everything else."""
    return sorted(setups, key=lambda s: (s.actionable, s.rank_score),
                  reverse=True)


def top_setup(setups: list[Setup]) -> "Setup | None":
    """Highest-scoring actionable setup, or None when nothing cleared."""
    ready = [s for s in setups if s.actionable]
    if not ready:
        return None
    return max(ready, key=lambda s: s.rank_score)


def explain(setup: Setup) -> str:
    """Plain-language account of what the rule measured, in one paragraph."""
    reading = setup.readings
    if setup.direction == NO_SETUP:
        failed = [r for r in setup.reasons if "[FAIL]" in r]
        head = (f"{reading.symbol} has no intraday setup right now because "
                f"its structure does not line up")
        return f"{head}: {failed[0].replace(' [FAIL]', '')}." if failed else f"{head}."
    trade = setup.levels
    if trade is None or not setup.passed:
        failed = [r.replace(" [FAIL]", "") for r in setup.reasons if "[FAIL]" in r]
        return (f"{reading.symbol} points {setup.direction.lower()} but is not "
                f"actionable: " + "; ".join(failed) + ".")
    side = "buy" if setup.direction == LONG else "sell short"
    return (
        f"{reading.symbol} reads {setup.direction} at {trade.entry:.2f}. The "
        f"mechanical plan is to {side} with a stop at {trade.stop:.2f} "
        f"({trade.stop_pct:.2f}% away, set by the {trade.stop_source}) and to "
        f"exit into {trade.target:.2f} ({trade.target_pct:.2f}%), which is "
        f"{trade.reward_risk:.1f} times the risk. At {trade.quantity} shares "
        f"that puts {trade.lot_risk:,.0f} rupees at risk to make "
        f"{trade.lot_risk * trade.reward_risk:,.0f} before costs. Volume is "
        f"running {reading.rvol:.2f}x its usual pace for this time of day and "
        f"the stock is {reading.relative_strength:+.2f}pp against the Nifty"
        + (f", and open interest is {reading.oi_change_pct:+.2f}% so "
           f"derivatives positioning is "
           f"{'building' if reading.oi_change_pct >= 0 else 'unwinding'}. "
           if reading.oi_change_pct is not None else ". ")
        + 
        f"After {trade.cost_rupees:,.0f} of round-trip charges this has to "
        f"work {trade.required_win_rate * 100:.1f}% of the time just to break "
        f"even, which is the number that decides whether the "
        f"{trade.reward_risk:.1f}:1 ratio means anything. "
        f"These are levels derived from today's range and volatility, not a "
        f"forecast: the rule has no backtest behind it, so treat the score as "
        f"a description of what was measured."
    )
