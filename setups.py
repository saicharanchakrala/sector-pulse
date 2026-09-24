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
import dataclasses
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
    # One C-level pass over the index rather than a Python loop building a
    # dict a Timestamp at a time. Profiled at 1.58s across 180 calls.
    if frame is None or frame.empty:
        return 0
    if not isinstance(frame.index, pd.DatetimeIndex):
        return 0
    counts = pd.Series(frame.index.date).value_counts()
    if counts.empty:
        return 0
    return int(counts.median())


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
        # TYPE-CHECKED, NOT COERCED. pd.DatetimeIndex([0, 1]) succeeds - it
        # reads plain integers as nanoseconds since 1970 - so coercing an
        # untimestamped index produces 1970 dates that are all "before
        # today", the filter passes everything, and prev_close becomes
        # today's own partial close. That is the exact leak this block
        # exists to prevent, and it is what an earlier version of this
        # optimisation reintroduced;
        # test_measure_drops_daily_context_rather_than_leaking_it caught it.
        if isinstance(prior.index, pd.DatetimeIndex):
            # Anchored on the CLOCK, not on the last surviving bar. The old
            # anchor slid to D-1 whenever no bar of the replay date survived
            # the cutoff, which let D's completed daily bar become
            # prev_close - a straight leak into day_change_pct and from
            # there into the relative-strength gate.
            cutoff_day = now.astimezone(indicators.IST).date()
            prior = prior[prior.index.date < cutoff_day]
        else:
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
        opening_range=indicators.opening_range(intraday, session=today),
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
        range_closed=indicators.opening_range_closed(intraday,
                                                     session=today),
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


def _price_reason(reading: Readings) -> tuple[bool, str]:
    """The penny-stock floor. Extracted so approach() shares it.

    It was the one gate with no helper, so approach() hand-rolled a copy -
    directly against the rule stated beside it, that a second copy of a
    rule which agrees today and drifts in a month is worse than not
    reporting it at all.
    """
    ok = reading.last >= config.SCAN_MIN_PRICE
    return ok, (f"price {reading.last:.2f} vs floor "
                f"{config.SCAN_MIN_PRICE:.2f} "
                f"[{'PASS' if ok else 'FAIL'}]")


def _room_reason(reading: Readings,
                 trade: TradeLevels) -> tuple[bool, str]:
    """Is as much plausible move left as the day has already made?

    THE ONE GATE THAT IS NOT ABOUT EVIDENCE. Every other gate asks whether
    the move is real; this one asks whether there is any of it left. The
    relative-strength gate requires today's outperformance, so a long
    cannot pass until it is already up more than the index - which means
    the scanner reports moves in progress and the entry is always behind
    the start. Measured 2026-09-18 across 78 cleared setups, the median
    had moved 3.16% and asked for 3.10% more, and 46% needed the remaining
    session to travel further than the whole morning had.

    FAILS OPEN on a missing previous close. Without it the ratio is not
    computable, and refusing a setup for want of a daily bar would turn a
    data gap into a verdict - which is the opposite of what this is for.

    The threshold has no backtest behind it, which is why it is a config
    value with its measurement recorded beside it rather than a literal.
    """
    prev = reading.prev_close
    floor = float(config.SCAN_MIN_ROOM_RATIO)
    if floor <= 0.0:
        return True, ("room to move not enforced (SCAN_MIN_ROOM_RATIO is "
                      "0) [INFO]")
    if prev is None:
        return True, "no previous close, so room to move is unknown [INFO]"
    if prev <= 0.0:
        return True, ("previous close is not a usable price, so room to "
                      "move is unknown [INFO]")

    # SIGNED, in the trade's own direction. abs() conflated move made
    # TOWARDS the target with move made against it, so a long sitting 4%
    # BELOW its previous close - the entire gap above it as room, passing
    # relative strength because the index fell further - was refused as a
    # chase, and a short carrying adverse move was let through. The gate
    # exists to stop chasing; unsigned it did the opposite on exactly the
    # gap-down recoveries and reversal shorts it most needed to get right.
    spent = ((reading.last - prev) if trade.direction == LONG
             else (prev - reading.last))
    if spent <= 0.0:
        return True, ("the day's move so far is against this direction, so "
                      "the whole plausible range is still ahead [PASS]")
    ratio = trade.expected_range / spent
    ok = ratio >= floor
    return ok, (f"{trade.expected_range:.2f} plausible move left against "
                f"{spent:.2f} already made in this direction today, ratio "
                f"{ratio:.2f} vs floor {floor:.2f} "
                f"[{'PASS' if ok else 'FAIL'}] [ENTRY]")


@dataclass(frozen=True)
class Approach:
    """A name sitting near a trigger it has NOT yet broken.

    WHY THIS EXISTS. choose_direction requires the opening range to be
    BROKEN, so a symbol still inside it is NO_SETUP with no levels and no
    score - correctly, because nothing has happened yet. The consequence
    is that the scanner can only ever report moves already under way, and
    the only honest way to see a name earlier is to say so explicitly:
    this is the level, this is how far away it is, and these are the gates
    that would still block it if it got there.

    `ready` means every gate OTHER than the break already passes. It is
    not a prediction that the break will happen, and most will not - that
    is the price of being early rather than a flaw in the measure.
    """

    symbol: str
    side: str
    trigger: float
    last: float
    distance: float
    distance_atr: float
    ready: bool
    blockers: list = field(default_factory=list)


def approach(reading: Readings,
             max_atr: "float | None" = None) -> "Approach | None":
    """The pre-break state of one symbol, or None when it does not apply.

    None rather than a neutral Approach for every case where the question
    is not yet defined: no VWAP, no opening range, a range that has not
    closed, a range already broken, or no volatility to measure distance
    in. Each of those is a different fact from "far from its trigger".

    THE SIDE COMES FROM WHERE THE BREAK LANDS, not from where price sits
    now. An earlier version read the current VWAP side and justified it as
    "a break the scanner would refuse anyway" - which is false whenever
    VWAP sits between price and the range edge, because breaking that
    edge CROSSES VWAP. Measured: last 100.0, VWAP 100.5, range 98-101 was
    reported as a SHORT 1.00 ATR from 98.0, while choose_direction would
    in fact have accepted the LONG that was 0.50 ATR away at 101.0 - the
    panel named the wrong trigger and hid the right one, on a geometry
    that is common mid-session.

    So a break of the high qualifies when the high is at or above VWAP,
    and a break of the low when the low is at or below VWAP. Both can
    qualify; the nearer is reported.
    """
    orb = reading.opening_range
    if orb is None or reading.vwap is None or not reading.range_closed:
        return None
    atr = reading.atr_bar
    # isfinite, not just truthiness: NaN is truthy and fails every `<=`,
    # so `not atr or atr <= 0.0` let a NaN ATR through to distance / nan.
    if atr is None or not math.isfinite(atr) or atr <= 0.0:
        return None
    last = reading.last
    if last > orb.high or last < orb.low:
        return None                      # already broken: evaluate() owns it

    candidates = []
    if orb.high >= reading.vwap:        # breaking the high lands above VWAP
        candidates.append((LONG, orb.high, orb.high - last))
    if orb.low <= reading.vwap:         # breaking the low lands below VWAP
        candidates.append((SHORT, orb.low, last - orb.low))
    if not candidates:
        return None
    side, trigger, distance = min(candidates, key=lambda c: c[2])
    ceiling = (config.SCAN_APPROACH_MAX_ATR if max_atr is None else max_atr)
    distance_atr = distance / atr
    if distance_atr > ceiling:
        return None

    # EVERY GATE evaluate() WOULD RUN, against levels built AT THE
    # TRIGGER. An earlier version checked five of the eleven inputs to
    # `passed` and still called the result "every gate other than the
    # break" - which made `ready` a promise the code did not test.
    # Reproducible in the default config: a name 0.60 ATR from its trigger
    # with no blockers, which on breaking failed the room gate at ratio
    # 0.39, because reaching the trigger is itself more move spent. Those
    # rows sorted to the TOP of the panel.
    blockers = []
    price_ok, price_line = _price_reason(reading)
    if not price_ok:
        blockers.append(price_line.split(" [")[0])
    for ok, line in (_liquidity_reason(reading),
                     _time_reason(reading),
                     _rvol_reason(reading),
                     _strength_reason(reading, side)):
        if not ok:
            blockers.append(line.split(" [")[0])

    # Provisional levels at the trigger, so the gates that need a trade -
    # cost, win rate, reach, room - are judged on the setup that would
    # actually exist rather than skipped.
    at_trigger = dataclasses.replace(reading, last=trigger)
    trade = levels_mod.build_levels(
        direction=side, entry=trigger, atr_per_bar=atr,
        bars_left=reading.bars_left,
        opening_low=orb.low, opening_high=orb.high,
        day_low=reading.day_low, day_high=reading.day_high)
    if trade is None:
        blockers.append("levels cannot be built at the trigger")
    else:
        for ok, line in (_oi_reason(at_trigger),
                         _cost_reason(trade),
                         _win_rate_reason(trade),
                         _reach_reason(trade),
                         _room_reason(at_trigger, trade)):
            if not ok:
                blockers.append(line.split(" [")[0])

    return Approach(symbol=reading.symbol, side=side, trigger=trigger,
                    last=last, distance=distance, distance_atr=distance_atr,
                    ready=not blockers, blockers=blockers)


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
    price_ok, price_line = _price_reason(reading)
    reasons.append(price_line)
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
    # `atr_bar or 0.0` did not catch NaN, which is truthy. build_levels now
    # refuses NaN itself; mapping it to 0.0 here as well keeps the intent
    # readable at the call site.
    atr = reading.atr_bar
    trade = levels_mod.build_levels(
        direction=direction,
        entry=reading.last,
        atr_per_bar=atr if atr is not None and math.isfinite(atr) else 0.0,
        bars_left=reading.bars_left,
        opening_low=orb.low if orb else None,
        opening_high=orb.high if orb else None,
        day_low=reading.day_low,
        day_high=reading.day_high,
        capital=capital,
        risk_pct=risk_pct,
    )
    if trade is None:
        # The ceiling is named only when it is the cause. Listing it among
        # every possible reason made the trail claim it on a NaN ATR too.
        if (risk_pct is not None and math.isfinite(risk_pct)
                and risk_pct > config.SCAN_MAX_RISK_PCT):
            why = (f"the {risk_pct:g}% risk per trade is above the "
                   f"{config.SCAN_MAX_RISK_PCT:g}% ceiling")
        else:
            why = ("no usable ATR, the stop collapsed onto the entry, or not "
                   "one share fits the budget once charges are counted")
        reasons.append(f"could not build levels: {why} [FAIL]")
        return Setup(reading, direction, None, False, 0.0, reasons)

    reasons.append(f"stop {trade.stop:.2f} from the {trade.stop_source}, "
                   f"{trade.stop_pct:.2f}% away, sizing {trade.quantity} "
                   f"shares for {trade.lot_risk:,.0f} of price risk plus "
                   f"{trade.cost_rupees:,.0f} of charges = "
                   f"{trade.risk_with_costs:,.0f} lost at the stop [PASS]")
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
    room_ok, room_line = _room_reason(reading, trade)
    reasons.append(room_line)

    passed = all((price_ok, liquid_ok, time_ok, rvol_ok, strength_ok,
                  oi_ok, cost_ok, win_ok, reach_ok, room_ok))
    score = score_setup(reading, trade) if passed else 0.0
    return Setup(reading, direction, trade, passed, score, reasons)


def rank(setups: list[Setup]) -> list[Setup]:
    """Actionable setups first by descending score, then everything else."""
    return sorted(setups, key=lambda s: (s.actionable, s.rank_score),
                  reverse=True)


# Lead-ins of the audit line a portfolio cap writes. Stable text on
# purpose: scan_intraday.first_blocker and scan_publish.log_row store the
# first 120 characters of the failing reason as `blocked_by`, and the
# figures after the lead-in differ row to row (as they do for every other
# gate), so a PREFIX match on these is how the scan log groups capped
# rows. capped_by_portfolio recognises a capped setup the same way.
PORTFOLIO_RISK_CAP = "portfolio risk cap"
PORTFOLIO_NOTIONAL_CAP = "portfolio notional cap"

# Relative slack on the cap comparisons, for the same reason as
# TradeLevels._REACH_TOLERANCE: five setups each held to exactly the
# budget sum to exactly the cap in real arithmetic, and a float sum must
# not refuse the fifth over a rounding error. One part in a billion of a
# 5,000 cap is five millionths of a rupee.
_CAP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PortfolioExposure:
    """What one scan snapshot commits in total, against the two caps."""

    risk_used: float            # rupees lost if every kept setup stops out
    risk_cap: float
    notional_used: float        # rupees of position across the kept setups
    notional_cap: float
    kept: int                   # actionable setups inside the caps
    blocked: int                # actionable setups the caps turned away

    @property
    def risk_used_pct_of_cap(self) -> float:
        """Share of the combined risk cap in use, 0.0 when there is no cap."""
        if self.risk_cap <= 0.0:
            return 0.0
        return self.risk_used / self.risk_cap * 100.0


def _cap_settings(capital: "float | None", max_risk_pct: "float | None",
                  leverage: "float | None"
                  ) -> tuple[float, float, float, float, float]:
    """Resolve the cap inputs and the two caps, reading config at CALL time.

    Returns (capital, max_risk_pct, leverage, risk_cap, notional_cap).

    None defaults rather than config values bound into the signature,
    because a default argument is evaluated once at import: a test that
    monkeypatches config, or a CLI run with --capital, would otherwise be
    capped against the capital the module happened to import with.

    FAILS CLOSED on an unusable setting. NaN compares False with
    everything, so a NaN cap would block nothing at all; an unusable cap
    is returned as 0.0 instead, which blocks every actionable setup and
    says so in each audit line.
    """
    capital = float(config.SCAN_CAPITAL if capital is None else capital)
    max_risk_pct = float(config.SCAN_MAX_OPEN_RISK_PCT
                         if max_risk_pct is None else max_risk_pct)
    leverage = float(config.SCAN_MIS_LEVERAGE if leverage is None
                     else leverage)
    risk_cap = capital * max_risk_pct / 100.0
    notional_cap = capital * leverage
    if not (math.isfinite(risk_cap) and risk_cap > 0.0):
        risk_cap = 0.0
    if not (math.isfinite(notional_cap) and notional_cap > 0.0):
        notional_cap = 0.0
    return capital, max_risk_pct, leverage, risk_cap, notional_cap


def capped_by_portfolio(setup: Setup) -> bool:
    """Whether a portfolio cap, rather than a gate, is what blocked this setup."""
    return any(reason.startswith((PORTFOLIO_RISK_CAP, PORTFOLIO_NOTIONAL_CAP))
               and "[FAIL]" in reason for reason in (setup.reasons or []))


def apply_portfolio_caps(ranked: list[Setup],
                         capital: "float | None" = None,
                         max_risk_pct: "float | None" = None,
                         leverage: "float | None" = None) -> list[Setup]:
    """Hold one scan's actionable setups inside a combined risk and funding cap.

    WHY. Every setup is sized on its own against its own stop, so the
    per-trade budget bounds each row and nothing bounded the total. On
    2026-09-22 the scan log marked 569 setups taken, each risking 1,000 of
    1,00,000 - a book that, entered in full, loses more than five times the
    account on one bad afternoon.

    THE RULE. Walk the ACTIONABLE setups in the order given (rank order,
    best first) and keep each while two running totals stay inside their
    caps:

        risk      sum of risk_with_costs      <= capital * max_risk_pct / 100
        notional  sum of entry * quantity     <= capital * leverage

    Risk is counted WITH charges, because that is what a stop-out actually
    takes. The first setup that would breach either cap CLOSES the book:
    it and every lower-ranked actionable setup are turned non-actionable
    with an audit line, even one small enough to squeeze into what is left.
    Back-filling would let position size rather than rank decide which
    trades are taken, and it would favour exactly the funding-capped names
    that deploy less than the full budget; closing keeps the kept set equal
    to the top of the ranking, which is the statement a reader can check.

    The audit line reads "<lead-in>: ... [FAIL] [ENTRY]". The lead-in is
    PORTFOLIO_RISK_CAP or PORTFOLIO_NOTIONAL_CAP, naming the cap the
    closing setup breached (risk first when it breached both), so
    scan_intraday.first_blocker records it as `blocked_by`. [ENTRY] marks
    it as a reason not to OPEN the trade - the same tag the room-to-move
    gate uses - so the position watch does not tell someone already in
    the trade that it "no longer clears the gates" because other setups
    ranked above it.

    NOT ENFORCED, AND WORTH SAYING PLAINLY: the cap bounds what ONE scan
    snapshot marks actionable. The scanner never learns which suggestions
    were entered, so positions already open are not subtracted, and a
    symbol capped in one 30-second pass can be the one kept in the next.
    Anything that unions snapshots over a session can therefore hold more
    than one snapshot's worth. THE SCAN LOG IS SUCH A UNION: its `taken`
    column is true for a setup that was actionable in ANY pass that day
    (scan_publish.remember lets a taken row replace a blocked one and
    never the reverse), so a session's `taken` set can hold far more than
    the risk cap allows at any one instant. Read `taken` as "was inside
    the cap in at least one snapshot", never as "the book the cap held".

    Pure: nothing is mutated. Non-actionable setups pass through as the
    same objects, capped ones are copies via dataclasses.replace with
    passed=False and the extra reason. rank_score is kept, so the copy
    still shows what the rule scored before the cap turned it away. The
    input order is preserved, which is still a valid rank order.
    """
    (capital, max_risk_pct, leverage,
     risk_cap, notional_cap) = _cap_settings(capital, max_risk_pct, leverage)
    used_risk = used_notional = 0.0
    kept = 0
    closed_by = ""                  # which cap closed the book, once one has
    closed_at = 0                   # 1-based actionable rank that closed it
    position = 0
    out: list[Setup] = []
    for setup in ranked:
        if not setup.actionable:
            out.append(setup)
            continue
        position += 1
        trade = setup.levels
        risk = trade.risk_with_costs
        notional = trade.entry * trade.quantity
        if not closed_by:
            over_risk = used_risk + risk > risk_cap * (1.0 + _CAP_TOLERANCE)
            over_notional = (used_notional + notional
                             > notional_cap * (1.0 + _CAP_TOLERANCE))
            if not (over_risk or over_notional):
                used_risk += risk
                used_notional += notional
                kept += 1
                out.append(setup)
                continue
            closed_by = (PORTFOLIO_RISK_CAP if over_risk
                         else PORTFOLIO_NOTIONAL_CAP)
            closed_at = position
        reason = (
            f"{closed_by}: the {kept} higher-ranked setup(s) kept already "
            f"lose {used_risk:,.0f} at their stops against a combined cap "
            f"of {risk_cap:,.0f} ({max_risk_pct:g}% of {capital:,.0f}) and "
            f"hold {used_notional:,.0f} of the {notional_cap:,.0f} notional "
            f"allowed ({leverage:g}x); the book closed at actionable rank "
            f"{closed_at} and is not back-filled, so this setup's "
            f"{risk:,.0f} at risk on {notional:,.0f} of notional is not "
            f"taken [FAIL] [ENTRY]")
        out.append(dataclasses.replace(setup, passed=False,
                                       reasons=[*setup.reasons, reason]))
    blocked = position - kept
    if blocked:
        logger.info("Portfolio caps kept %d of %d actionable setups (%s "
                    "closed the book at rank %d): %.0f of %.0f risk, %.0f "
                    "of %.0f notional", kept, position, closed_by,
                    closed_at, used_risk, risk_cap, used_notional,
                    notional_cap)
    return out


def portfolio_exposure(ranked: list[Setup],
                       capital: "float | None" = None,
                       max_risk_pct: "float | None" = None,
                       leverage: "float | None" = None) -> PortfolioExposure:
    """Totals of a capped scan, for the report and the dashboard.

    Reads the result of apply_portfolio_caps rather than re-running it, so
    the figures on screen are the ones the cap actually enforced: `kept`
    is what is still actionable, `blocked` is what carries a portfolio cap
    line.
    """
    *_, risk_cap, notional_cap = _cap_settings(capital, max_risk_pct,
                                               leverage)
    actionable = [s for s in ranked if s.actionable]
    return PortfolioExposure(
        risk_used=sum(s.levels.risk_with_costs for s in actionable),
        risk_cap=risk_cap,
        notional_used=sum(s.levels.entry * s.levels.quantity
                          for s in actionable),
        notional_cap=notional_cap,
        kept=len(actionable),
        blocked=sum(1 for s in ranked if capped_by_portfolio(s)),
    )


def _cap_lead_in(reasons) -> str:
    """The portfolio-cap lead-in among these reasons, or "" when none is one.

    The same test capped_by_portfolio applies - a reason that STARTS with a
    lead-in and carries [FAIL] - returning which cap it names, so a caller
    can say which cap closed the book rather than only that one did.
    """
    for reason in reasons or []:
        text = str(reason)
        if "[FAIL]" not in text:
            continue
        for lead_in in (PORTFOLIO_RISK_CAP, PORTFOLIO_NOTIONAL_CAP):
            if text.startswith(lead_in):
                return lead_in
    return ""


def closing_cap(ranked: list[Setup]) -> str:
    """Which cap closed the book in a capped ranking, or "" if none closed it.

    Read off the first capped setup's audit line: apply_portfolio_caps
    writes the SAME lead-in on every setup after the book closes, so the
    first one found is the answer for all of them.
    """
    for setup in ranked:
        lead_in = _cap_lead_in(setup.reasons)
        if lead_in:
            return lead_in
    return ""


def cap_config_problem(risk_pct: "float | None" = None,
                       capital: "float | None" = None,
                       max_risk_pct: "float | None" = None,
                       leverage: "float | None" = None) -> str:
    """Why the combined caps cannot hold one full-size trade; "" when they can.

    THE MISCONFIGURATION THIS NAMES. apply_portfolio_caps fails closed, so
    an unusable cap - NaN, zero, negative - blocks every actionable setup.
    So, in practice, does a risk cap set BELOW the per-trade risk: with
    SCAN_MAX_OPEN_RISK_PCT at 1.5 and 2% risked per trade, the top setup
    alone breaches the cap and closes the book on everything after it.
    Either way the scan comes back empty, and an empty scan is also what a
    quiet market looks like - so without this the fault read as "nothing
    set up today", which is the one reading it must never get.

    Used by scan_intraday._parse_args to refuse the run outright, by the
    dashboard to show an error rather than an empty table, and by
    cap_blocked_all_text to name the cause. Config is read at call time,
    as _cap_settings does.

    A plain `<` on the configured percents: a cap EQUAL to the per-trade
    risk holds exactly one full-size trade, which is tight but legal. A
    per-trade risk that is not a finite positive number is not judged
    here; the sizer and the CLI already refuse it on their own terms.
    """
    (capital, max_risk_pct, leverage,
     risk_cap, notional_cap) = _cap_settings(capital, max_risk_pct, leverage)
    per_trade = float(config.SCAN_RISK_PCT_PER_TRADE if risk_pct is None
                      else risk_pct)
    if risk_cap <= 0.0:
        return (f"the combined risk cap is unusable ({max_risk_pct:g}% of "
                f"{capital:,.0f}, config.SCAN_MAX_OPEN_RISK_PCT), and an "
                f"unusable cap blocks every setup")
    if notional_cap <= 0.0:
        return (f"the funding cap is unusable ({leverage:g}x of "
                f"{capital:,.0f}, config.SCAN_MIS_LEVERAGE), and an "
                f"unusable cap blocks every setup")
    if (math.isfinite(per_trade) and per_trade > 0.0
            and max_risk_pct < per_trade):
        return (f"the combined risk cap of {max_risk_pct:g}% "
                f"(config.SCAN_MAX_OPEN_RISK_PCT) is below the "
                f"{per_trade:g}% one trade may lose, so the top full-size "
                f"setup breaches it on its own and closes the book")
    return ""


def cap_blocked_all_text(held: int, closed_by: str = PORTFOLIO_RISK_CAP,
                         risk_pct: "float | None" = None,
                         capital: "float | None" = None,
                         max_risk_pct: "float | None" = None,
                         leverage: "float | None" = None) -> str:
    """The empty-scan message for when a portfolio cap, not the gates, emptied it.

    The CLI and the dashboard used to call every empty scan a quiet market
    ("the expected answer", "not a fault"). When `held` setups cleared
    every gate and a cap turned all of them away, that is false: a
    working cap always keeps the top setup, because no single setup can
    lose more than the per-trade budget or fund more than the notional
    cap. So this names the cap that closed the book, its value in percent
    and rupees, and the per-trade risk it was compared with, and says in
    so many words that it is a configuration fault.

    Plain sentences with rupee figures and no markup, so the CLI can wrap
    it and the dashboard can show it as it is. `closed_by` is a lead-in
    (see closing_cap); the per-trade budget comes from levels.risk_budget,
    the one place it is computed.
    """
    (capital, max_risk_pct, leverage,
     risk_cap, notional_cap) = _cap_settings(capital, max_risk_pct, leverage)
    per_trade = float(config.SCAN_RISK_PCT_PER_TRADE if risk_pct is None
                      else risk_pct)
    budget = levels_mod.risk_budget(capital, per_trade)
    which = ("funding (notional) cap" if closed_by == PORTFOLIO_NOTIONAL_CAP
             else "combined risk cap")
    budget_text = (f"{budget:,.0f} rupees" if budget > 0.0
                   else "nothing, because the sizer refuses that percent")
    problem = cap_config_problem(per_trade, capital, max_risk_pct, leverage)
    cause = (f"The cause: {problem}." if problem else
             "The top setup alone did not fit, which a working "
             "configuration never produces.")
    return (
        f"Nothing is actionable because the {which} held back all {held} "
        f"setup(s) that cleared every gate. The risk cap is "
        f"{max_risk_pct:g}% of {capital:,.0f} = {risk_cap:,.0f} rupees "
        f"(config.SCAN_MAX_OPEN_RISK_PCT) and the funding cap "
        f"{notional_cap:,.0f} ({leverage:g}x); one trade may lose up to "
        f"{per_trade:g}% = {budget_text} at its stop. {cause} This is a "
        f"configuration fault, not a quiet market: fix the cap and re-run.")


@dataclass(frozen=True)
class PublishedExposure:
    """Combined exposure rebuilt from a PUBLISHED scan table, and what it shows."""

    exposure: PortfolioExposure
    held: tuple                 # symbols a cap line held back, in table order
    closed_by: str              # lead-in of the cap that closed the book, or ""
    cap_seen: bool              # at least one row carries a portfolio-cap line
    over_cap: bool              # actionable rows exceed a cap beyond rounding
    unpriced: int               # actionable rows lacking entry, stop or quantity
    uncharged: int              # actionable rows lacking cost_rupees


def _cell_number(value) -> "float | None":
    """A finite float from a published cell; None for None, NaN, inf or text."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cell_flag(value) -> bool:
    """A published boolean cell as a bool, with a missing cell read as False.

    NOT bool(value): NaN is truthy, and a column absent from some rows of
    an older table arrives as NaN, which bool() would read as actionable.
    """
    if value is None or (isinstance(value, float) and value != value):
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def published_exposure(rows, capital: "float | None" = None,
                       max_risk_pct: "float | None" = None,
                       leverage: "float | None" = None) -> PublishedExposure:
    """portfolio_exposure for a published table, rebuilt from its columns.

    WHY. The dashboard's main live path reads the table the feed publishes
    (scan_publish.row_for, one dict per row), not Setup objects, so it had
    no combined-risk block at all - and its help text claimed the feed
    applies the cap, which is true only of a feed redeployed with the cap
    in it. This rebuilds the same totals from the columns the table does
    carry, so the claim can be checked instead of asserted. `rows` is an
    iterable of dicts - table.to_dict("records") - not the DataFrame
    itself, whose truth value pandas refuses to give.

    THE ARITHMETIC. For every row with `actionable` true, the loss at the
    stop is abs(entry - stop) * quantity + cost_rupees - risk_with_costs
    rebuilt, since lot_risk is quantity times the stop distance - and the
    notional is entry * quantity. Rows are read in the order given; the
    feed writes its table in rank order, so `held` is in rank order too.

    ROUNDING. row_for rounds entry, stop and cost to two places, so the
    rebuilt loss can differ from the one the cap enforced by up to one
    paisa a share plus a paisa a row, and the notional by half a paisa a
    share. `over_cap` is set only past that allowance, so a table the cap
    did hold is never reported as breaching it; it is left False when the
    cap itself is unusable, because the table cannot be judged against it
    (cap_config_problem names that fault instead).

    ROBUST TO OLDER TABLES, because the feed and the UI deploy separately.
    No `reasons` column means no cap line can be seen. A row with no
    `cost_rupees` counts price risk only and is tallied in `uncharged`, so
    the caller can say the total UNDERSTATES the loss. An actionable row
    missing entry, stop or quantity is left out of the sums, tallied in
    `unpriced`, and still counted as kept.
    """
    *_, risk_cap, notional_cap = _cap_settings(capital, max_risk_pct,
                                               leverage)
    risk_used = notional_used = 0.0
    risk_slack = notional_slack = 0.0
    kept = unpriced = uncharged = 0
    held: list = []
    closed_by = ""
    for row in (rows if rows is not None else []):
        lead_in = _cap_lead_in(str(row.get("reasons") or "").split(" | "))
        if lead_in:
            held.append(row.get("symbol"))
            closed_by = closed_by or lead_in
        if not _cell_flag(row.get("actionable")):
            continue
        kept += 1
        entry = _cell_number(row.get("entry"))
        stop = _cell_number(row.get("stop"))
        quantity = _cell_number(row.get("quantity"))
        if entry is None or stop is None or quantity is None:
            unpriced += 1
            continue
        cost = _cell_number(row.get("cost_rupees"))
        if cost is None:
            uncharged += 1
            cost = 0.0
        risk_used += abs(entry - stop) * quantity + cost
        notional_used += entry * quantity
        risk_slack += 0.01 * quantity + 0.01
        notional_slack += 0.005 * quantity
    over_risk = (risk_cap > 0.0 and risk_used
                 > risk_cap * (1.0 + _CAP_TOLERANCE) + risk_slack)
    over_notional = (notional_cap > 0.0 and notional_used
                     > notional_cap * (1.0 + _CAP_TOLERANCE) + notional_slack)
    exposure = PortfolioExposure(
        risk_used=risk_used, risk_cap=risk_cap, notional_used=notional_used,
        notional_cap=notional_cap, kept=kept, blocked=len(held))
    return PublishedExposure(
        exposure=exposure, held=tuple(held), closed_by=closed_by,
        cap_seen=bool(held), over_cap=bool(over_risk or over_notional),
        unpriced=unpriced, uncharged=uncharged)


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
        # Everything from the first " [" off, not just " [FAIL]": the room
        # gate and the portfolio cap also carry an [ENTRY] tag, which is
        # for the position watch, not for a sentence.
        failed = [r.split(" [")[0] for r in setup.reasons if "[FAIL]" in r]
        return (f"{reading.symbol} points {setup.direction.lower()} but is not "
                f"actionable: " + "; ".join(failed) + ".")
    side = "buy" if setup.direction == LONG else "sell short"
    return (
        f"{reading.symbol} reads {setup.direction} at {trade.entry:.2f}. The "
        f"mechanical plan is to {side} with a stop at {trade.stop:.2f} "
        f"({trade.stop_pct:.2f}% away, set by the {trade.stop_source}) and to "
        f"exit into {trade.target:.2f} ({trade.target_pct:.2f}%), which is "
        f"{trade.reward_risk:.1f} times the risk. At {trade.quantity} shares "
        f"that puts {trade.lot_risk:,.0f} rupees at risk on the move - "
        f"{trade.risk_with_costs:,.0f} once charges are counted, which is "
        f"what the size is held to - to make "
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
