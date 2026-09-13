"""Assess every instrument across four holding periods, not just intraday.

THE FLAW THIS FIXES. The scanner only ever assessed intraday, and its gates
are intraday gates. Chief among them: "45 minutes left vs minimum" - which
correctly refuses a same-session trade near the close, and has nothing
whatever to say about a ten-day or one-year view. At 15:22 the scanner
reported nothing actionable across 216 names, which reads as "no
opportunities" when it actually meant "too late for MY horizon only".

So each horizon now gets its own inputs, its own gates and its own clock:

    intraday   entry to today's close      intraday bars, session-gated
    short      about 10 sessions           daily bars, no session gate
    mid        about 63 sessions           daily bars, no session gate
    long       about 252 sessions          daily bars, no session gate

WHAT THE RANKING IS AND IS NOT. Read the honesty note below before using
the output. Every one of these four horizons has been measured on this
project's own data and none showed predictive skill; at short, mid and long
the model's selection was measurably WORSE than buying the same universe
equally weighted (spread -0.26%, -0.25% and -7.36% respectively). What the
ranking here therefore is: a consistent, cost-aware ordering of what the
data currently looks like. What it is not: a forecast, or evidence that the
top of the list will outperform the bottom. The columns that survived
measurement are the cost ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import bar_store
import config
import trade_costs

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Sessions held, and the round-trip cost that horizon actually pays.
# Intraday is MIS: brokerage capped at Rs 20 a leg, STT on the sell only.
# Everything overnight is CNC delivery: zero brokerage but STT on BOTH
# legs, which is dearer in absolute terms and cheaper as a share of a
# larger expected move.
#
# The intraday figure is COMPUTED from the live cost stack rather than
# transcribed from it. It used to be the literal 0.082, which was correct
# when written and silently wrong the moment slippage was added - the same
# way a hand-copied constant is always wrong one edit later. It is quoted
# at a one-lakh ticket because brokerage is capped per order, so cost per
# rupee falls as the ticket grows; see trade_costs.
_INTRADAY_COST_PCT = trade_costs.equity_breakeven_pct(1000.0, 100)

# Delivery has no function in trade_costs to call, so this half stays
# hand-derived: 0.1% STT on BOTH legs plus stamp duty and exchange fees,
# with no brokerage on CNC. Slippage is added on the same terms as
# everywhere else rather than being quietly omitted from the one stack
# that happens to lack a calculator.
_DELIVERY_STATUTORY_PCT = 0.23
_DELIVERY_COST_PCT = round(
    _DELIVERY_STATUTORY_PCT + 2 * config.COST_SLIPPAGE_PCT * 100.0, 4)

HORIZONS = {
    "intraday": {"sessions": 1, "cost_pct": _INTRADAY_COST_PCT, "daily": False},
    "short": {"sessions": 10, "cost_pct": _DELIVERY_COST_PCT, "daily": True},
    "mid": {"sessions": 63, "cost_pct": _DELIVERY_COST_PCT, "daily": True},
    "long": {"sessions": 252, "cost_pct": _DELIVERY_COST_PCT, "daily": True},
}

HONESTY = (
    "**This list is not a prediction, and we tested that properly.** We ran "
    "the same method on about 875,000 past examples. It picked winners no "
    "better than the identical method fed deliberately scrambled answers - "
    "and over three months and over a year, the scrambled version actually "
    "did better. Simply buying every stock equally beat our picks at all "
    "three holding periods, by 0.06, 0.09 and 5.5 percentage points. Over "
    "a year our ranking scored 0.47 where a coin flip is 0.50. So read the "
    "order as 'what the numbers look like today', never as 'these are the "
    "good ones'."
)

# The same finding for anyone who wants the figures rather than the plain
# summary. Kept beside it so the two can never drift apart.
#
# WHAT THESE FIGURES MEASURE, precisely, because it is easy to over-read
# them. They come from the gradient-boosted pipeline in forecast_horizons,
# which selects by calibrated probability. That is NOT the composite that
# orders the tables here: rank() is a hand-weighted heuristic, it has
# never been measured on its own, and neither module imports the other. So
# these numbers establish that a model free to find any pattern it liked
# found none. They do not certify the ordering on screen, which is if
# anything the weaker of the two.
HONESTY_DETAIL = (
    "Out-of-sample results on a survivorship-corrected universe of 2,254 "
    "companies. Short: -0.21% excess over the index, selection spread "
    "-0.26%, permutation p 0.55. Mid: +1.65% excess but equal-weight gave "
    "+1.90%, spread -0.25%, permutation p 0.55. Long: +3.64% excess against "
    "equal-weight +11.01%, spread -7.36%, permutation p 1.00 - all 30 "
    "shuffled-label runs beat the real one. Ranking accuracy (AUC) was "
    "0.5130, 0.4740 and 0.4671 where 0.50 is a coin flip."
)


@dataclass
class Assessment:
    """One symbol at one horizon: what is measurable, and what it costs."""

    symbol: str
    horizon: str
    price: float
    direction: str
    trend: float                  # return over the horizon's own lookback, %
    relative: float               # same, minus the benchmark, %
    volatility: float             # annualised, %
    drawdown: float               # from the trailing high, %
    position_in_range: float      # 0..1 within the trailing window
    expected_move: float          # volatility scaled to the horizon, %
    cost_pct: float               # round-trip charges, %
    cost_multiple: float          # expected move over cost
    score: float = 0.0
    reasons: list = field(default_factory=list)
    blocked: str = ""

    @property
    def pays_for_itself(self) -> bool:
        """Whether the plausible move covers the round trip at least 3x."""
        return self.cost_multiple >= 3.0

    @property
    def stop_distance_pct(self) -> float:
        """Half the plausible move, in percent.

        Half, so that an exit at the full plausible move gives 2:1 reward
        to risk - the same convention the intraday sizer uses, and the one
        the required-win-rate arithmetic assumes. Any other split would
        make the two halves of this project disagree about what a trade
        is.
        """
        move = self.expected_move
        return move / 2.0 if move == move else float("nan")

    @property
    def stop_price(self) -> float:
        """Where the stop would sit, in rupees, or NaN if unknowable.

        Placed against `direction`, which describes whether the instrument
        is currently stronger or weaker than the index. It is NOT a
        prediction, so this is "the stop for that position" rather than
        "the stop for the trade you should take".
        """
        gap = self.stop_distance_pct
        if gap != gap or self.price <= 0:
            return float("nan")
        step = self.price * gap / 100.0
        return self.price - step if self.direction == "LONG" else self.price + step

    @property
    def target_price(self) -> float:
        """Where the exit would sit: the plausible move away from here."""
        move = self.expected_move
        if move != move or self.price <= 0:
            return float("nan")
        step = self.price * move / 100.0
        return self.price + step if self.direction == "LONG" else self.price - step


def reachable(days_to_expiry: "int | None") -> list:
    """Daily horizons that fit before a contract expires, longest last.

    A September future cannot be held for 252 sessions, so offering a
    long-term verdict on one is not a cautious estimate - it is an answer
    to a question that cannot be asked. Sessions are converted from
    calendar days at five per seven, which is close enough for a bound
    that only ever excludes whole horizons.

    None means no expiry, so every daily horizon applies.
    """
    daily = [h for h in HORIZONS if HORIZONS[h].get("daily")]
    if days_to_expiry is None:
        return daily
    if days_to_expiry <= 0:
        return []
    sessions_left = int(days_to_expiry * 5 / 7)
    return [h for h in daily if _sessions_for(h) <= sessions_left]


def _sessions_for(horizon: str) -> int:
    return int(HORIZONS[horizon]["sessions"])


def _lookback_for(horizon: str) -> int:
    """Trailing window used to describe a horizon.

    Deliberately the same length as the holding period: describing a
    one-year view from a five-day trend would be measuring something the
    horizon cannot act on.
    """
    return max(5, _sessions_for(horizon))


def _annualised_volatility(closes: np.ndarray) -> float:
    """Annualised daily volatility, in percent, or NaN if unmeasurable.

    Non-positive closes are DROPPED rather than clamped. Clamping the
    divisor to 1e-9 turned one bad price into a return of about 1e11, and
    since every rank here is cross-sectional that single value took the top
    of the volatility column for the whole run.
    """
    usable = closes[np.isfinite(closes) & (closes > 0)]
    if usable.size < 10:
        return float("nan")
    returns = np.diff(usable) / usable[:-1]
    returns = returns[np.isfinite(returns)]
    if returns.size < 9:
        return float("nan")
    return float(np.std(returns) * np.sqrt(252) * 100.0)


# The longest lookback, in sessions, for which intraday bars are a better
# volatility estimate than daily ones. Written once: it decides both which
# horizons ask for the intraday store and which ones use it.
FINE_LOOKBACK_MAX = 20
# How much of the assessed universe the intraday store must cover, and how
# current it must be, before it is used at all. Below either, the whole run
# uses the daily estimate - see the note in assess_daily.
FINE_MIN_COVERAGE = 0.9
FINE_MAX_STALE_DAYS = 7
# How many completed SESSIONS the daily store may be missing before the
# horizons say so. Sessions, not calendar days: four calendar days on a
# Sunday is two missed sessions, and counting days let exactly that pass
# unremarked. One missing session is worth stating - trend, drawdown and
# position-in-range all come from this store and two of them are ranking
# columns.
DAILY_MAX_STALE_SESSIONS = 1


def realised_volatility(fine: pd.DataFrame, sessions: int) -> float:
    """Annualised volatility from intraday bars, in percent, or NaN.

    WHY THIS BEATS THE DAILY ESTIMATE AT THE SHORT HORIZON. Ten daily
    returns cannot support a volatility estimate, which is why the daily
    path silently widened its window to sixty sessions - so the "plausible
    move" for a two-week hold was borrowed from two months of data. Ten
    sessions of intraday bars is roughly 1,250 returns at the configured
    three minutes, over the window actually being assessed.

    THE OVERNIGHT GAP IS INCLUDED, and leaving it out would have been the
    trap. Realised variance from intraday returns alone misses every
    close-to-open move, and for Indian equities those are a real share of
    total variance - so the estimate would come out low, the plausible
    move with it, and the cost-multiple gate would let through setups
    whose move cannot actually cover the round trip. Each session's
    variance is therefore the sum of squared in-session returns PLUS the
    squared overnight return, which is the standard construction.
    """
    if fine is None or "Close" not in fine.columns or fine.empty:
        return float("nan")
    closes = fine["Close"].dropna()
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if len(closes) < 20:
        return float("nan")
    try:
        days = closes.index.date
    except AttributeError:
        return float("nan")
    ordered = sorted(set(days))
    window = ordered[-sessions:] if sessions > 0 else ordered
    # Seeded from the session BEFORE the window, so the first session in it
    # gets an overnight term like every other. Seeding from inside the
    # window left one session in ten missing its gap - a downward bias, in
    # the estimator whose whole argument is that the gap must be counted.
    previous_close = None
    if window:
        earlier = closes[days < window[0]]
        if not earlier.empty:
            previous_close = float(earlier.iloc[-1])
    variances = []
    for day in window:
        block = closes[days == day]
        if len(block) < 5:
            # Too few bars to say anything about the session's own path,
            # but its close still anchors the next session's gap.
            previous_close = float(block.iloc[-1])
            continue
        values = block.to_numpy(dtype=float)
        inside = np.diff(np.log(values))
        variance = float(np.sum(inside ** 2))
        if previous_close and previous_close > 0:
            gap = float(np.log(values[0] / previous_close))
            variance += gap ** 2
        previous_close = float(values[-1])
        # Finite, NOT positive. A session that genuinely did not move -
        # a halt, an illiquid name, a frozen feed - has variance zero, and
        # that is a measurement rather than a gap in the data. Dropping
        # such sessions averaged only the moving ones and biased the whole
        # estimate upward: five flat sessions among five volatile ones came
        # out at 29.68% where counting them gives about 21%.
        if np.isfinite(variance):
            variances.append(variance)
    if len(variances) < 3:
        # Three sessions is the floor for an average worth annualising by
        # 252. Below it the caller falls back to the daily estimate.
        return float("nan")
    daily_variance = float(np.mean(variances))
    return float(np.sqrt(daily_variance * 252.0) * 100.0)


def _benchmark_move(dates, benchmark, lookback: int) -> float:
    """Benchmark return over the SAME DATES the stock's trend was measured.

    Positional indexing compared different spans. `bench[-(lookback+1)]` is
    the benchmark's 64th-from-last ROW, which is only the stock's window
    start when the two series have identical trading calendars - and they
    do not, whenever either has a suspension, a listing gap or a missing
    bar. Reindexing onto the stock's dates and carrying the last known
    close forward makes the two spans the same span.
    """
    if benchmark is None or "Close" not in benchmark.columns:
        return float("nan")
    closes = benchmark["Close"].dropna()
    if closes.empty or len(dates) < lookback + 1:
        return float("nan")
    # The benchmark must actually REACH the end of the stock's window.
    # ffill has no staleness limit, so an index frame that stops early
    # would carry its last known close forward and be compared against
    # the stock's current price - a fabricated excess return for every
    # symbol at once. Checking the extent is what the positional version
    # was doing, crudely, with `bench.size >= lookback + 1`.
    try:
        if closes.index.max() < dates[-1]:
            return float("nan")
        aligned = closes.reindex(dates, method="ffill")
    except Exception:
        return float("nan")
    first, last = aligned.iloc[-(lookback + 1)], aligned.iloc[-1]
    if not (first == first and last == last):
        return float("nan")
    if first <= 0 or last <= 0:
        return float("nan")
    return (float(last) / float(first) - 1.0) * 100.0


def assess_daily(symbol: str, frame: pd.DataFrame, horizon: str,
                 benchmark: "pd.DataFrame | None",
                 live_price: "float | None" = None,
                 cost_pct: "float | None" = None,
                 fine: "pd.DataFrame | None" = None) -> "Assessment | None":
    """One symbol at one daily horizon, from cached daily bars.

    No session-time gate: whether 40 minutes remain today is irrelevant to
    a position meant to be held for ten sessions or a year.

    `live_price` re-anchors the readings that answer "where is it NOW" -
    the price, the stop, the exit, the drawdown and the range position.

    `fine` is that symbol's intraday bars, used for volatility at the
    short horizons only (lookback <= FINE_LOOKBACK_MAX) and ignored
    elsewhere. Pass None to use the daily estimate. The caller passes it
    for every symbol in a run or for none - see _usable_fine.
    Without it those come from the newest daily close, which during a
    session is yesterday's: measured at 1.10% away on RELIANCE, enough to
    turn a 2:1 stop-and-target into 6.7:1 for anyone acting on it.

    `trend` and `relative` deliberately stay on completed bars even when a
    live price is given. `relative` subtracts the benchmark's move over the
    same span, so advancing this stock's end point while leaving the index
    on its last close would invent excess return from a timing mismatch.
    Volatility is a property of the window and does not move on one tick.
    """
    if frame is None or "Close" not in frame.columns:
        return None
    series = frame["Close"].dropna()
    # Non-positive and non-finite closes are removed ONCE, here, so every
    # reading below is computed on the same trustworthy prices. Cleaning
    # inside the volatility helper alone left position_in_range and
    # drawdown - both of them scored - reading the raw array, where a
    # single zero sets `low` and therefore the whole range position.
    series = series[np.isfinite(series) & (series > 0)]
    return _assess_series(symbol, series.to_numpy(dtype=float), series.index,
                          horizon, benchmark, live_price=live_price,
                          cost_pct=cost_pct, fine=fine)


def _assess_series(symbol: str, closes, dates, horizon: str,
                   benchmark: "pd.DataFrame | None",
                   live_price: "float | None" = None,
                   cost_pct: "float | None" = None,
                   fine: "pd.DataFrame | None" = None) -> "Assessment | None":
    """The arithmetic, over already-cleaned closes and their dates.

    Separated from assess_daily so the universe sweep can slice closes
    straight out of one long frame instead of building a DataFrame per
    symbol - measured at 6.7s for 2,285 of them, against 0.5s to read the
    rows they came from. Both callers reach this function, so there is one
    implementation of every reading rather than two that agree today.

    `closes` must already have had nulls, non-finite values and
    non-positive prices removed, and `dates` must be the stamps that
    survived alongside them.
    """
    lookback = _lookback_for(horizon)
    if closes.size < lookback + 5:
        return None
    price = float(closes[-1])
    if price <= 0:
        return None
    # The anchor for everything positional. `trend` keeps using the close
    # below, so the benchmark comparison stays span-for-span honest.
    anchor = price
    if live_price is not None and np.isfinite(live_price) and live_price > 0:
        anchor = float(live_price)
    window = closes[-(lookback + 1):]
    # The FIRST close of the window is a divisor, and only the last was
    # being checked. A zero start gives an infinite trend and a negative
    # one flips its sign, and either then leads the cross-sectional rank.
    base = float(window[0])
    if not (base > 0) or not np.isfinite(base):
        return None
    trend = (price / base - 1.0) * 100.0
    if not np.isfinite(trend):
        return None
    # The short horizon prefers realised volatility from intraday bars,
    # because ten daily returns cannot support an estimate and the daily
    # path has to borrow sixty sessions to get one. Falls back to the
    # daily figure when no intraday bars were supplied - and the CALLER
    # decides that for the whole run rather than per symbol, because these
    # two estimators do not share a level and mixing them inside one
    # cross-sectional rank would let data availability move a symbol's
    # position and flip its cost gate.
    volatility = float("nan")
    if fine is not None and lookback <= FINE_LOOKBACK_MAX:
        volatility = realised_volatility(fine, lookback)
    if volatility != volatility:
        volatility = _annualised_volatility(closes[-max(lookback, 60):])
    trailing = closes[-lookback:]
    high, low = float(trailing.max()), float(trailing.min())
    span = max(high - low, 1e-9)

    relative = trend
    bench_move = _benchmark_move(dates, benchmark, lookback)
    if bench_move == bench_move:
        relative = trend - bench_move

    sessions = _sessions_for(horizon)
    # Volatility scaled to the horizon by root time, which is the same
    # sigma convention the intraday sizer uses.
    expected = (volatility / np.sqrt(252.0) * np.sqrt(sessions)
                if volatility == volatility else float("nan"))
    # `cost_pct` overrides the horizon's default so a derivative can be
    # priced on its own stack. A future's round trip is about 0.034% of
    # notional against the 0.23% a cash delivery pays, and the cost
    # multiple - which is a hard gate - is meaningless if the wrong one is
    # used.
    cost = (float(HORIZONS[horizon]["cost_pct"]) if cost_pct is None
            else float(cost_pct))
    multiple = (expected / cost) if cost > 0 and expected == expected else 0.0

    direction = "LONG" if relative > 0 else "SHORT"
    reasons = [
        f"{lookback}-session trend {trend:+.2f}% "
        f"({'ahead of' if relative > 0 else 'behind'} the index by "
        f"{abs(relative):.2f}pp)",
        f"annualised volatility {volatility:.1f}%, so a {sessions}-session "
        f"move of about {expected:.2f}% is plausible",
        f"round trip costs {cost:.3f}%, which the plausible move covers "
        f"{multiple:.1f}x [{'PASS' if multiple >= 3.0 else 'FAIL'}]",
    ]
    # Clamped, because a live price CAN sit outside the trailing range
    # while the last close could not. Uncapped, a new high read as a range
    # position of 1.04 and a "below recent peak" of +3.8%.
    drawdown = (min(0.0, (anchor / high - 1.0) * 100.0) if high > 0
                else float("nan"))
    position = min(1.0, max(0.0, (anchor - low) / span))
    return Assessment(
        symbol=symbol, horizon=horizon, price=anchor, direction=direction,
        trend=trend, relative=relative, volatility=volatility,
        drawdown=drawdown, position_in_range=position,
        expected_move=expected, cost_pct=cost, cost_multiple=multiple,
        reasons=reasons,
        blocked="" if multiple >= 3.0 else "expected move too small for costs")


# Daily-horizon gates. Deliberately fewer and blunter than the intraday
# set: a position held for weeks is not helped by a breakout reading taken
# over three minutes,
# and inventing gates to look thorough would be nine ways of saying the
# same thing. Each returns (passed, line) so the report can show the number
# behind every verdict rather than just its outcome.
MIN_TURNOVER = 50_000_000.0        # Rs 5 crore a day, same floor as intraday
MIN_PRICE = 20.0
MIN_COST_MULTIPLE = 3.0
MAX_DRAWDOWN = -35.0               # deep below the trailing high, %


def gate_lines(assessment: "Assessment", turnover: "float | None"
               ) -> tuple:
    """(passed, reasons) for one daily-horizon assessment."""
    reasons = []
    checks = []

    ok = assessment.price >= MIN_PRICE
    checks.append(ok)
    reasons.append(f"price {assessment.price:,.2f} vs floor "
                   f"{MIN_PRICE:,.2f} [{'PASS' if ok else 'FAIL'}]")

    if turnover is None:
        reasons.append("20-session turnover unknown, so liquidity fails "
                       "closed rather than passing on missing data [FAIL]")
        checks.append(False)
    else:
        ok = turnover >= MIN_TURNOVER
        checks.append(ok)
        reasons.append(f"20-session turnover {turnover:,.0f} vs floor "
                       f"{MIN_TURNOVER:,.0f} [{'PASS' if ok else 'FAIL'}]")

    ok = assessment.cost_multiple >= MIN_COST_MULTIPLE
    checks.append(ok)
    reasons.append(
        f"plausible {_sessions_for(assessment.horizon)}-session move "
        f"{assessment.expected_move:.2f}% covers the {assessment.cost_pct:.3f}% "
        f"round trip {assessment.cost_multiple:.1f}x, needs "
        f"{MIN_COST_MULTIPLE:.0f}x [{'PASS' if ok else 'FAIL'}]")

    # THIS GATE IS LONG-ONLY, AND SO IS THE SYSTEM. `direction` is set to
    # SHORT precisely when relative < 0 (see _direction_from), and this
    # then requires relative > 0 - so every SHORT fails here by
    # construction, always, whatever the rest of its readings say. That is
    # not a bug in the gate: the daily horizons were built to find
    # outperformers, and nothing here has ever been measured on the short
    # side. It IS a bug to present those rows as ranked candidates with a
    # stop and an exit, which is why to_frame now blanks them.
    ok = assessment.relative > 0
    checks.append(ok)
    if assessment.direction == "SHORT":
        reasons.append(
            f"relative to the index {assessment.relative:+.2f}pp over "
            f"{_lookback_for(assessment.horizon)} sessions - this system "
            f"only evaluates LONG ideas, so a weaker-than-index reading is "
            f"NO OPINION rather than a short suggestion [FAIL]")
    else:
        reasons.append(
            f"relative to the index {assessment.relative:+.2f}pp over "
            f"{_lookback_for(assessment.horizon)} sessions, needs to be "
            f"ahead [{'PASS' if ok else 'FAIL'}]")

    deep = assessment.drawdown == assessment.drawdown and         assessment.drawdown < MAX_DRAWDOWN
    checks.append(not deep)
    reasons.append(
        f"{assessment.drawdown:+.2f}% from the trailing high, floor "
        f"{MAX_DRAWDOWN:.0f}% [{'FAIL' if deep else 'PASS'}]")

    if assessment.volatility != assessment.volatility:
        reasons.append("volatility not computable, so the move estimate "
                       "cannot be trusted [FAIL]")
        checks.append(False)

    return all(checks), reasons


def turnover_20d(frame: "pd.DataFrame | None") -> "float | None":
    """Median rupee turnover over the last 20 sessions, or None."""
    if frame is None or frame.empty:
        return None
    if "Volume" not in frame.columns or "Close" not in frame.columns:
        return None
    tail = frame.tail(20)
    values = (tail["Close"] * tail["Volume"]).dropna()
    if values.empty:
        return None
    return float(values.median())


def rank(assessments: list, top: int = 20) -> list:
    """Score cross-sectionally, then return the best `top`.

    Scored against the other names in the SAME run rather than on absolute
    thresholds, because "strong relative trend" only means anything next to
    what the rest of the market did. Names whose plausible move cannot
    cover costs are ranked last whatever else they show.

    SHORT rows sort to the bottom, and that is deliberate rather than a
    sign error: score is monotone in `relative`, and a SHORT is by
    definition a negative `relative`. Since gate_lines is long-only by
    construction, a SHORT can never become actionable, so ranking it above
    a tradeable LONG would promote a row the system refuses to act on.
    to_frame blanks its levels for the same reason.
    """
    usable = [a for a in assessments if a is not None]
    if not usable:
        return []
    # No `multiple` column. cost_multiple is expected_move / cost_pct, and
    # cost_pct is a constant per horizon, so it is volatility times a
    # constant - measured at a ratio of exactly 2.173913 across 372 names,
    # spearman(multiple, -volatility) = -1.000000 (the ratio is 0.866108
    # at short and 4.347826 at long - constant at every horizon). Scoring
    # both put 0.10 + 0.20 x rank(multiple) into the total: a POSITIVE
    # loading on volatility, so the penalty this weight exists to apply
    # was inverted. Controlling for the other two columns, the old
    # composite's marginal correlation with volatility was +1.0.
    #
    # Two things this does NOT achieve, stated so they are not over-read.
    # The TOTAL correlation with volatility stays positive at short and
    # mid (+0.05 and +0.16, down from +0.49 and +0.55), because relative
    # strength and range position both co-vary with volatility; only the
    # marginal loading is corrected. And cost coverage now reaches the
    # ordering solely through the `pays_for_itself` partition below, which
    # 2,469 of 2,470 names clear - so the ranking carries almost no cost
    # content. That is still the right trade, since the alternative was a
    # column that inverted the penalty, but it means the measured part of
    # this module is the cost COLUMNS in the table, not the order.
    frame = pd.DataFrame([{
        "i": i, "relative": a.relative,
        "position": a.position_in_range,
        "volatility": -(a.volatility if a.volatility == a.volatility else 0.0),
    } for i, a in enumerate(usable)])
    # The old proportions, renormalised over the three survivors. Not
    # re-chosen: nothing in this module has shown predictive skill, so new
    # numbers would imply a basis for preferring them that does not exist.
    for column, weight in (("relative", 0.643), ("position", 0.214),
                           ("volatility", 0.143)):
        ranks = frame[column].rank(pct=True).fillna(0.0)
        frame[f"w_{column}"] = ranks * weight
    frame["score"] = frame[[c for c in frame.columns
                            if c.startswith("w_")]].sum(axis=1)
    for _, row in frame.iterrows():
        usable[int(row["i"])].score = round(float(row["score"]), 4)
    payable = [a for a in usable if a.pays_for_itself]
    rest = [a for a in usable if not a.pays_for_itself]
    payable.sort(key=lambda a: a.score, reverse=True)
    rest.sort(key=lambda a: a.score, reverse=True)
    return (payable + rest)[:top]


def live_prices_for(symbols: list) -> dict:
    """Live last-traded price per symbol, or {} when there is no session.

    One batched quote sweep rather than a call per symbol: Kite caps a
    quote at 500 instruments, so the whole universe is a handful of
    requests. Returns {} on any failure - the horizons then fall back to
    the last daily close, which is stale but not wrong, and the UI says
    which anchor it used.
    """
    if not symbols:
        return {}
    try:
        import market_source
        return market_source.last_prices(list(symbols))
    except Exception as exc:
        logger.warning("No live prices, so the horizons stay anchored to "
                       "the last daily close: %s", exc)
        return {}


def _measurable_sessions(frame) -> int:
    """Sessions in `frame` that realised_volatility can actually use.

    Its own preconditions, not a proxy for them: a session needs at least
    five bars to say anything about its own path, and the estimator wants
    three such sessions before it will annualise an average.
    """
    if frame is None or frame.empty or "Close" not in frame.columns:
        return 0
    try:
        days = frame["Close"].dropna().index.date
    except AttributeError:
        return 0
    counts = {}
    for day in days:
        counts[day] = counts.get(day, 0) + 1
    return sum(1 for n in counts.values() if n >= 5)


def last_completed_session(now: "datetime | None" = None) -> date:
    """The most recent weekday whose session has finished.

    Weekdays only, so it does not know about exchange holidays - which
    makes it overcount the gap on one, and warning on a holiday is the
    safe direction to be wrong in.
    """
    now = now or datetime.now(IST)
    close_hour, close_minute = config.SCAN_SESSION_CLOSE
    day = now.date()
    finished_today = (now.hour, now.minute) >= (close_hour, close_minute)
    if day.weekday() >= 5 or not finished_today:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def daily_store_lag(long_frame=None, now: "datetime | None" = None) -> tuple:
    """(newest session in the daily store, completed sessions missing).

    Measured from the DATA, not the file's timestamp: a rebuild that folds
    nothing still refreshes the mtime, so mtime records when someone last
    ran a command rather than how current the bars are.

    SESSIONS, not calendar days. Measured on 2026-09-13, a Sunday: the
    store's newest bar was 2026-09-09, which is four calendar days and TWO
    missed sessions. A day-based threshold of four let that pass.

    Returns (None, None) when there is nothing to measure. Pass the long
    frame a sweep has already loaded to make this free.
    """
    try:
        # Only None means "fetch it yourself". An empty frame that was
        # PASSED means the caller looked and found nothing, and quietly
        # reloading the real store behind their back would answer a
        # different question from the one asked.
        if long_frame is None:
            long_frame = bar_store.load_long("day")
        if long_frame is None or len(long_frame) == 0:
            return None, None
        newest = pd.DatetimeIndex(long_frame["stamp"]).max()
        if newest is pd.NaT or newest != newest:
            return None, None
        newest = newest.date()
        missing = 0
        day = last_completed_session(now)
        while day > newest:
            if day.weekday() < 5:
                missing += 1
            day -= timedelta(days=1)
        return newest, missing
    except Exception as exc:
        logger.warning("Could not read the daily store's age: %s", exc)
    return None, None


def _usable_fine(fine: dict, symbols: list) -> dict:
    """The intraday store if it can be used for the whole run, else {}.

    ALL OR NOTHING, and that is the point. Realised volatility and the
    60-session daily estimate do not share a level, so handing one to the
    symbols that happen to have intraday bars and the other to the rest
    puts two different measurements in the same rank(pct=True) column and
    the same cost-multiple gate. A symbol's position - and whether its
    setup passes - would then depend on whether its bars had been cached.

    COVERAGE MEANS MEASURABLE, not merely present. Measured on the real
    3-minute store the day it was first built: 121 symbols held 14
    sessions and RELIANCE held exactly one, written by a single
    backfill_today. A one-session frame is not empty, so counting frames
    would have called it covered - and then the estimator returns NaN for
    it and that symbol alone falls back to the daily figure, which is the
    mixing this function exists to stop.

    What remains is bounded rather than eliminated: up to
    (1 - FINE_MIN_COVERAGE) of the population can still fall back
    individually. Dropping those symbols from the tables altogether would
    be worse than a slightly noisier volatility rank for a tenth of them.

    Two other ways to be unusable: too little of the universe covered at
    all, or a store that has stopped being topped up. The second is easy
    to reach by accident, since the intraday store is fed by whatever
    interval the live feed fetches, and that follows config.
    """
    if not fine or not symbols:
        return {}
    usable = {s: fine[s] for s in symbols
              if _measurable_sessions(fine.get(s)) >= 3}
    if len(usable) < FINE_MIN_COVERAGE * len(symbols):
        logger.info("Intraday bars can be measured for only %d/%d symbols, "
                    "so the short horizon uses the daily volatility "
                    "estimate for all of them rather than two estimators "
                    "in one ranking.", len(usable), len(symbols))
        return {}
    fine = usable
    latest = None
    for frame in fine.values():
        if frame is None or frame.empty:
            continue
        try:
            stamp = frame.index[-1].date()
        except Exception:
            continue
        latest = stamp if latest is None or stamp > latest else latest
    if latest is None:
        return {}
    behind = (date.today() - latest).days
    if behind > FINE_MAX_STALE_DAYS:
        logger.warning("The %s store's newest bar is %d days old, so the "
                       "short horizon uses the daily volatility estimate. "
                       "Is the feed still fetching that interval?",
                       config.HORIZON_SHORT_INTERVAL, behind)
        return {}
    return fine


def assess_universe(symbols: list, horizons: "list | None" = None,
                    benchmark_symbol: str = "NIFTY 50",
                    top: int = 20, progress=None,
                    live: "dict | None" = None) -> dict:
    """Every symbol across the daily horizons, ranked, top `top` each.

    `progress` is called as progress(horizon, done, total) so a UI can fill
    its table as results arrive rather than waiting for the whole sweep.

    `live` maps symbol to its current price. Pass it to anchor the levels
    on now rather than on the last daily close; pass None during a replay
    or outside market hours, where the close IS the right anchor.
    """
    wanted = [h for h in (horizons or list(HORIZONS))
              if HORIZONS.get(h, {}).get("daily")]
    if not wanted:
        return {}
    longest = max(_lookback_for(h) for h in wanted)
    # Calendar days for the deepest lookback, with slack for holidays.
    start = date.today() - timedelta(days=int(longest * 7 / 5) + 30)
    # ONE long frame plus positional bounds, rather than a DataFrame per
    # symbol. Measured on this store: 5.71s to build 2,285 frames against
    # 0.62s for the long read and its bounds, for the same rows.
    long_frame = bar_store.load_long(
        "day", symbols=list(symbols) + [benchmark_symbol], start=start)
    bounds = bar_store.slice_bounds(long_frame)
    if not bounds:
        logger.warning("No consolidated daily bars; run bar_store first")
        return {}
    all_closes = long_frame["Close"].to_numpy(dtype=float)
    all_dates = pd.DatetimeIndex(long_frame["stamp"])
    # The daily store's own age, from the bars just loaded. The intraday
    # store is REFUSED when it goes stale because the daily estimate
    # stands behind it; nothing stands behind this one, so it warns and
    # the caller puts the age on screen.
    newest, behind = daily_store_lag(long_frame)
    if behind is not None and behind > DAILY_MAX_STALE_SESSIONS:
        logger.warning(
            "The daily store's newest session is %s - %d completed "
            "session(s) missing. Trend, drawdown and position in range "
            "come from it, so they describe that date rather than today. "
            "Rebuild with: python -m bar_store --intervals day",
            newest, behind)

    def cleaned(symbol: str):
        """(closes, dates) for one symbol, filtered exactly as assess_daily
        filters them: nulls, non-finite values and non-positive prices out,
        their stamps out with them."""
        span = bounds.get(symbol)
        if span is None:
            return None
        begin, stop = span
        closes = all_closes[begin:stop]
        keep = np.isfinite(closes) & (closes > 0)
        return closes[keep], all_dates[begin:stop][keep]

    benchmark = None
    bench = cleaned(benchmark_symbol)
    if bench is not None:
        # _benchmark_move reindexes onto the stock's dates, so the
        # benchmark alone still arrives as a frame - one, not 2,285.
        benchmark = pd.DataFrame({"Close": bench[0]}, index=bench[1])
    prepared = {}
    for symbol in symbols:
        ready = cleaned(symbol)
        if ready is not None:
            prepared[symbol] = ready
    # Intraday bars for the short horizons only - mid and long never
    # consult them - and loaded once rather than per symbol. Bounded by
    # date: realised_volatility reads ten sessions and the store holds
    # years of them.
    fine = {}
    if any(_lookback_for(h) <= FINE_LOOKBACK_MAX for h in wanted):
        try:
            fine = bar_store.load(
                config.HORIZON_SHORT_INTERVAL, symbols=list(symbols),
                start=date.today() - timedelta(days=FINE_MAX_STALE_DAYS + 30))
        except Exception as exc:
            logger.warning("No %s bars, so the short horizon uses the daily "
                           "volatility estimate: %s",
                           config.HORIZON_SHORT_INTERVAL, exc)
        fine = _usable_fine(fine, symbols)
    out = {}
    for horizon in wanted:
        results = []
        total = len(symbols)
        for done, symbol in enumerate(symbols, start=1):
            ready = prepared.get(symbol)
            if ready is None:
                continue
            try:
                assessment = _assess_series(
                    symbol, ready[0], ready[1], horizon, benchmark,
                    live_price=(live or {}).get(symbol),
                    fine=fine.get(symbol))
            except Exception as exc:
                logger.warning("%s at %s: %s", symbol, horizon, exc)
                continue
            if assessment is not None:
                results.append(assessment)
            if progress is not None and done % 25 == 0:
                progress(horizon, done, total)
        out[horizon] = rank(results, top=top)
        if progress is not None:
            progress(horizon, total, total)
    return out


def to_frame(assessments: list) -> pd.DataFrame:
    """Ranked assessments as a display table, cost columns to the front."""
    if not assessments:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Symbol": a.symbol,
        # A SHORT row here is a stock weaker than the index, which this
        # long-only system has no opinion about - so it gets no levels.
        # Printing a stop and an exit beside a verdict the gates can never
        # pass is how HDFCLIFE was read as a trade ticket.
        "View": ("weaker than index (not evaluated)"
                 if a.direction == "SHORT" else a.direction),
        "Price": round(a.price, 2),
        "Stop loss at": (None if a.direction == "SHORT"
                         else round(a.stop_price, 2)),
        "Exit price": (None if a.direction == "SHORT"
                       else round(a.target_price, 2)),
        "Cost %": round(a.cost_pct, 3),
        "Plausible move %": round(a.expected_move, 2),
        "Move vs fees": f"{a.cost_multiple:.1f}x",
        "Pays for itself?": "yes" if a.pays_for_itself else "no",
        "Trend %": round(a.trend, 2),
        "Beat index by": round(a.relative, 2),
        "Volatility %": round(a.volatility, 1),
        "Below recent peak": round(a.drawdown, 2),
        "Where in range": round(a.position_in_range, 2),
        "Our ranking": a.score,
    } for a in assessments])
