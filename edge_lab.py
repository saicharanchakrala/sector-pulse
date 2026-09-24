"""A harness for testing intraday rules against measured price behaviour.

Built because the first rule failed in a way that could only be diagnosed by
measurement. Over a year of bars - 54,397 signals, 229 sessions, 210 names,
`python -m edge_lab` - its direction call matched a coin tossed at the same
bars to within 0.003 R in every stop/target cell, and every cell lost money
once costs were charged - so the problem was the signal, not the geometry.
Guessing a second rule would repeat the mistake.

The contract is deliberately narrow so that rules are comparable and cannot
cheat:

    def my_rule(ctx):
        '''Return "LONG", "SHORT" or None for the bar at ctx.i.'''

`ctx` exposes only what had printed by bar `i`. Arrays are pre-sliced;
scalars like the previous close and the ATR come from prior sessions, and
the opening range is derived from the sliced arrays rather than handed in,
so it cannot exist before the bars that form it have printed. The harness
then measures what happened after the signal - which is the part the rule
never sees.

One exception, and it is the caller's: `benchmark_change` is whatever dict
the caller passes. kite_bars.benchmark_day_changes fills it with each
session's FULL-day change, so a rule reading ctx.benchmark_change or
ctx.relative_strength off that dict is reading the index's outcome for a
day it is still trading. Treat those two as opt-in lookahead unless the
caller supplies a point-in-time series; every other field is safe by
construction.

Four disciplines, all enforced here rather than left to each rule:

  one signal per direction per session, taken on the first bar it fires, so
    one move cannot be counted as many wins;
  a bar that spans both stop and target counts as a stop, because 5-minute
    bars carry no intra-bar ordering and assuming otherwise flatters results;
  unresolved positions are marked out at the close, not discarded and not
    scored as flat, because a rule that leaves most trades open must be
    judged on what that pays;
  every result is set against a coin tossed for direction at the same bars,
    scored under these same rules, because a closed-form baseline cannot
    see the session end or the tie rule and misreads chance as skill.

Sample-size honesty: every path now loads from Kite, which reaches years -
the 5-minute set in bar_cache spans 249 sessions across 210 names - so a
result here is not short of rows. It is still short of independence:
signals within a session are correlated, so treat a result that survives
only at one parameter setting as noise.
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import trade_costs

logger = logging.getLogger(__name__)

# UNITS. Every column suffixed `_sigma` here - mfe_sigma, mae_sigma,
# close_move_sigma - is in PLAUSIBLE-MOVE units, not standard deviations.
# One plausible move is
# levels.expected_remaining_range (ATR * sqrt(bars)) and measures roughly
# 1.4 true sigma. The names are kept because they are written into saved
# CSVs that would silently stop matching, but a figure read from them is
# not a sigma and must be divided by about 1.4 before being compared with
# any textbook one-sigma rule.
LONG = "LONG"
SHORT = "SHORT"

CACHE_DIR = config.PROJECT_ROOT / "bar_cache"
# The bar size this harness measures on. Separate from the live scanner's
# config.SCAN_BAR_INTERVAL on purpose - the harness exists to compare bar
# sizes, so it cannot inherit the one currently in production - but the
# opening-range bar count MUST be derived from whichever size a run uses.
# It was `// 5` against a 5m default: a 3m run would then have taken its
# opening range over 3 bars, which is 9 minutes against the 5m arm's 15,
# and any comparison between the two would have been measuring the range
# length as much as the bar size.
LAB_BAR_MINUTES = 5
BARS_PER_ORB = max(1, config.SCAN_OPENING_RANGE_MINUTES // LAB_BAR_MINUTES)
# Prior sessions in the relative-volume baseline. Matches the live
# scanner's 20-day window; also bounds the per-session precompute, which is
# what keeps the harness linear in sessions rather than quadratic.
RVOL_BASELINE_SESSIONS = 20
DEFAULT_STOPS = (0.25, 0.5, 0.75, 1.0, 1.5)
DEFAULT_TARGETS = (0.25, 0.5, 0.75, 1.0, 1.5)

# Round-trip equity cost as a fraction of turnover. COMPUTED from the live
# cost stack rather than transcribed out of it, so a rate change in config
# cannot leave this copy behind - it was the hand-written 0.000824, which
# silently excluded slippage once that was added.
#
# Cost per rupee is HYPERBOLIC in ticket size, because brokerage is capped
# at 20 rupees an order, so a single constant is only right near one
# ticket. That ticket is stated here rather than assumed: a tighter stop
# buys more shares and pays a LOWER fraction than this.
COST_TICKET = 100_000.0
COST_PRICE = 1000.0
COST_FRACTION = trade_costs.equity_breakeven_pct(
    COST_PRICE, int(COST_TICKET / COST_PRICE)) / 100.0


@dataclass
class Context:
    """Everything knowable at bar `i` of one session. Nothing beyond it."""

    symbol: str
    day: object
    i: int                       # index of the current bar within the session
    n: int                       # bars in the full session
    open_: np.ndarray            # session arrays, sliced to [:i+1]
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vwap: np.ndarray             # running session VWAP, same length
    atr: float                   # Wilder ATR from PRIOR sessions only
    prev_close: float            # prior session's close
    prev_high: float
    prev_low: float
    rvol: float                  # cumulative volume vs prior sessions at bar i
    benchmark_change: "float | None"   # caller-supplied; see module docstring

    @property
    def price(self) -> float:
        """The latest close."""
        return float(self.close[-1])

    @property
    def bars_left(self) -> int:
        """Bars remaining in the session after this one."""
        return self.n - (self.i + 1)

    @property
    def sigma(self) -> float:
        """Remaining move implied by sqrt-of-time scaling.

        Retained as the unit of measurement so results are comparable with
        the live rule, NOT as an endorsement: measurement showed realised
        range runs well under it.
        """
        return self.atr * math.sqrt(max(1, self.bars_left))

    @property
    def orb_closed(self) -> bool:
        """Whether the opening range has finished forming."""
        return self.i >= BARS_PER_ORB

    @property
    def orb_high(self) -> float:
        """Opening range high, NaN until the range has actually closed.

        Computed from the sliced session arrays, so the value cannot
        contain a bar that has not printed by `i`. It used to be handed in
        precomputed, which meant that at i=0 it already carried the highs
        of bars 1 and 2 and a rule that forgot ctx.orb_closed was reading
        the future. NaN rather than a number because every comparison
        against NaN is False, so such a rule now simply does not fire.
        """
        if not self.orb_closed:
            return float("nan")
        return float(np.max(self.high[:BARS_PER_ORB]))

    @property
    def orb_low(self) -> float:
        """Opening range low, NaN until the range has actually closed."""
        if not self.orb_closed:
            return float("nan")
        return float(np.min(self.low[:BARS_PER_ORB]))

    @property
    def day_change_pct(self) -> "float | None":
        """Percent change against the prior close."""
        if self.prev_close <= 0:
            return None
        return (self.price / self.prev_close - 1.0) * 100.0

    @property
    def relative_strength(self) -> "float | None":
        """Day change less the benchmark's."""
        change = self.day_change_pct
        if change is None or self.benchmark_change is None:
            return None
        return change - self.benchmark_change

    @property
    def vwap_distance_pct(self) -> "float | None":
        """Percent distance of price from the running VWAP."""
        line = float(self.vwap[-1])
        if not np.isfinite(line) or line <= 0:
            return None
        return (self.price / line - 1.0) * 100.0

    @property
    def cpr(self):
        """Previous session's central pivot range as (bottom, pivot, top)."""
        if min(self.prev_high, self.prev_low, self.prev_close) <= 0:
            return None
        pivot = (self.prev_high + self.prev_low + self.prev_close) / 3.0
        bc = (self.prev_high + self.prev_low) / 2.0
        tc = 2 * pivot - bc
        return (min(tc, bc), pivot, max(tc, bc))

    @property
    def gap_pct(self) -> "float | None":
        """Opening gap against the prior close."""
        if self.prev_close <= 0:
            return None
        return (float(self.open_[0]) / self.prev_close - 1.0) * 100.0

    @property
    def minutes_into_session(self) -> int:
        """Minutes elapsed since the open at this bar."""
        return self.i * 5


def _session_true_ranges(high, low, close) -> np.ndarray:
    """True ranges inside one session; the first bar keeps its own span."""
    out = np.empty(len(high))
    out[0] = high[0] - low[0]
    if len(high) > 1:
        prev = close[:-1]
        out[1:] = np.maximum.reduce([high[1:] - low[1:],
                                     np.abs(high[1:] - prev),
                                     np.abs(low[1:] - prev)])
    return out


def _wilder(values: np.ndarray, period: int) -> "float | None":
    """Wilder's smoothing, or None below `period` observations."""
    if len(values) < period:
        return None
    out = float(values[:period].mean())
    for value in values[period:]:
        out = (out * (period - 1) + float(value)) / period
    return out


def _span_days(period: str, default: int = 60) -> int:
    """Calendar days from a '60d' or '3mo' style span.

    See market_source.calendar_days: "60d" is sixty trading sessions, which
    is about ninety calendar days, not sixty.
    """
    import market_source
    return market_source.calendar_days(period, default)


def load_bars(symbols: list[str], period: str = "60d",
              interval: str = f"{LAB_BAR_MINUTES}m", refresh: bool = False
              ) -> dict[str, pd.DataFrame]:
    """Fetch and cache OHLCV frames, one parquet file per symbol.

    Parquet rather than pickle: the cache is data, and data should not be
    able to execute on load.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # "kite_" in the tag on purpose. bar_cache still holds files named
    # SYMBOL__5m_60d.parquet from the yfinance era - auto_adjust=True
    # back-adjusted Yahoo bars, with the index named Datetime where
    # kite_client names it Date. They carry no provenance marker, so the
    # only way not to read them as Kite data is to not look there.
    tag = f"kite_{interval}_{period}"
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        path = CACHE_DIR / f"{symbol.replace('/', '_')}__{tag}.parquet"
        if path.exists() and not refresh:
            try:
                out[symbol] = pd.read_parquet(path)
                continue
            except Exception as exc:
                logger.warning("Unreadable cache %s: %s", path.name, exc)
        missing.append(symbol)
    if missing:
        # Kite returns one flat frame per symbol, so the private
        # scan_data._unpack this used to borrow is not needed on this path.
        import market_source
        span = _span_days(period)
        end = date.today()
        try:
            by_symbol = market_source.bars(
                missing, end - timedelta(days=span), end,
                interval=market_source.kite_interval(interval))
        except market_source.NoSession as exc:
            logger.warning("%s", exc)
            by_symbol = {}
        for symbol in missing:
            frame = by_symbol.get(symbol)
            if frame is None or frame.empty:
                continue
            path = CACHE_DIR / f"{symbol.replace('/', '_')}__{tag}.parquet"
            try:
                frame.to_parquet(path)
            except Exception as exc:
                logger.warning("Could not cache %s: %s", symbol, exc)
            out[symbol] = frame
    return out


def _prepare(frame: pd.DataFrame) -> "tuple | None":
    """Split one symbol's frame into per-session arrays plus prior context."""
    needed = {"High", "Low", "Close"}
    if frame is None or frame.empty or not needed <= set(frame.columns):
        return None
    frame = frame.dropna(subset=["High", "Low", "Close"]).sort_index()
    if frame.empty:
        return None
    try:
        days = np.array([t.date() for t in frame.index])
    except AttributeError:
        return None
    unique = sorted(set(days))
    sessions = {}
    for day in unique:
        mask = days == day
        sessions[day] = {
            "open": (frame["Open"].values[mask] if "Open" in frame.columns
                     else frame["Close"].values[mask]),
            "high": frame["High"].values[mask],
            "low": frame["Low"].values[mask],
            "close": frame["Close"].values[mask],
            "volume": (frame["Volume"].values[mask]
                       if "Volume" in frame.columns
                       else np.ones(int(mask.sum()))),
        }
        sessions[day]["tr"] = _session_true_ranges(
            sessions[day]["high"], sessions[day]["low"], sessions[day]["close"])
    return unique, sessions


def first_touch(high_path: np.ndarray, low_path: np.ndarray, direction: str,
                entry: float, stop_distance: float,
                target_distance: float) -> str:
    """Which level the forward path reached first."""
    if stop_distance <= 0 or target_distance <= 0:
        return "NEITHER"
    if direction == LONG:
        stop_hits = np.flatnonzero(low_path <= entry - stop_distance)
        target_hits = np.flatnonzero(high_path >= entry + target_distance)
    else:
        stop_hits = np.flatnonzero(high_path >= entry + stop_distance)
        target_hits = np.flatnonzero(low_path <= entry - target_distance)
    stop_at = stop_hits[0] if stop_hits.size else None
    target_at = target_hits[0] if target_hits.size else None
    if stop_at is None and target_at is None:
        return "NEITHER"
    if stop_at is None:
        return "TARGET"
    if target_at is None:
        return "STOP"
    # A bar containing both carries no ordering, so assume the stop.
    return "STOP" if stop_at <= target_at else "TARGET"


def _prior_levels(sessions: dict, prior_days: list) -> "tuple | None":
    """Wilder ATR over prior sessions plus the last prior session's levels.

    None when there is nothing to size against - no prior sessions, or no
    usable ATR from them. Fail closed: a session that cannot be sized is
    skipped rather than measured against a guessed range.
    """
    if not prior_days:
        return None
    prior_tr = np.concatenate([sessions[d]["tr"] for d in prior_days])
    atr = _wilder(prior_tr, config.SCAN_ATR_BARS)
    if atr is None or atr <= 0:
        return None
    last = sessions[prior_days[-1]]
    return (atr, float(last["close"][-1]), float(last["high"].max()),
            float(last["low"].min()))


def _rvol_baseline(sessions: dict, prior_days: list) -> np.ndarray:
    """Median cumulative-volume curve across the recent prior sessions.

    Computed once per session rather than per bar: per bar made the cost
    grow with sessions squared, which is unusable past a few dozen
    sessions. The window also matches the live scanner's 20-day baseline
    instead of quietly widening with history.
    """
    curves = [np.cumsum(sessions[d]["volume"])
              for d in prior_days[-RVOL_BASELINE_SESSIONS:]]
    width = max((len(c) for c in curves), default=0)
    if not width:
        return np.zeros(0)
    padded = np.full((len(curves), width), np.nan)
    for row, curve in enumerate(curves):
        padded[row, :len(curve)] = curve
    with np.errstate(invalid="ignore"):
        return np.nanmedian(padded, axis=0)


def _session_vwap(bars: dict) -> tuple:
    """Running session VWAP and cumulative volume, bar by bar."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    cum_volume = np.cumsum(bars["volume"])
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.cumsum(typical * bars["volume"]) / cum_volume
    return vwap, cum_volume


@dataclass
class _Session:
    """One session's precomputed inputs for the bar loop.

    Everything here is either the session's own bars, which get sliced per
    bar before a rule sees them, or a scalar from PRIOR sessions.
    """

    symbol: str
    day: object
    bars: dict
    n: int
    atr: float
    prev_close: float
    prev_high: float
    prev_low: float
    vwap: np.ndarray
    cum_volume: np.ndarray
    baseline: np.ndarray
    benchmark_change: "float | None"

    def context(self, i: int) -> Context:
        """The context for bar `i`, sliced so no later bar is reachable."""
        bars = self.bars
        at_i = float(self.baseline[i]) if i < len(self.baseline) else 0.0
        rvol = (float(self.cum_volume[i]) / at_i
                if at_i > 0 and np.isfinite(at_i) else 0.0)
        return Context(
            symbol=self.symbol, day=self.day, i=i, n=self.n,
            open_=bars["open"][:i + 1], high=bars["high"][:i + 1],
            low=bars["low"][:i + 1], close=bars["close"][:i + 1],
            volume=bars["volume"][:i + 1], vwap=self.vwap[:i + 1],
            atr=self.atr, prev_close=self.prev_close,
            prev_high=self.prev_high, prev_low=self.prev_low,
            rvol=rvol, benchmark_change=self.benchmark_change)


def _prepare_session(symbol: str, day, sessions: dict, prior_days: list,
                     change: "float | None") -> "_Session | None":
    """One session's inputs, or None when it cannot be measured honestly."""
    bars = sessions[day]
    n = len(bars["close"])
    if n <= BARS_PER_ORB + 1:
        return None
    levels = _prior_levels(sessions, prior_days)
    if levels is None:
        return None
    atr, prev_close, prev_high, prev_low = levels
    vwap, cum_volume = _session_vwap(bars)
    return _Session(symbol=symbol, day=day, bars=bars, n=n, atr=atr,
                    prev_close=prev_close, prev_high=prev_high,
                    prev_low=prev_low, vwap=vwap, cum_volume=cum_volume,
                    baseline=_rvol_baseline(sessions, prior_days),
                    benchmark_change=change)


def _signal_row(ctx: Context, bars: dict, direction: str,
                stops: tuple, targets: tuple) -> dict:
    """One signal plus the forward path the rule never saw.

    The forward slice starts at i+1, so the entry bar's own extremes never
    count as an excursion.

    THE COIN-FLIP TWIN. Every geometry is also scored for the OPPOSITE
    direction on the same forward path, under the same session end and the
    same tie rule, and kept under `flipped`. A trader who tossed a coin for
    direction at this bar would take each side half the time, so the
    average of the two sides is exactly what a coin flip pays here. That
    is the baseline summarise compares against. The closed-form
    stop / (stop + target) it replaces assumes unlimited time and a
    continuous path; this harness counts hits only before the close and
    gives a bar spanning both levels to the stop, and both of those favour
    the nearer level: at the shipped geometry a driftless walk hits about
    24%, nine points below the formula's 33%, so the formula read pure
    chance as nine points worse than chance.
    """
    i, entry, sigma = ctx.i, ctx.price, ctx.sigma
    forward_high = bars["high"][i + 1:]
    forward_low = bars["low"][i + 1:]
    session_close = float(bars["close"][-1])
    if direction == LONG:
        mfe = float(forward_high.max()) - entry
        mae = entry - float(forward_low.min())
        close_move = session_close - entry
    else:
        mfe = entry - float(forward_low.min())
        mae = float(forward_high.max()) - entry
        close_move = entry - session_close
    row = {
        "symbol": ctx.symbol, "day": ctx.day, "i": i,
        "direction": direction, "entry": entry, "sigma": sigma,
        "bars_left": ctx.bars_left, "rvol": ctx.rvol,
        "relative_strength": ctx.relative_strength,
        "mfe_sigma": max(0.0, mfe) / sigma,
        "mae_sigma": max(0.0, mae) / sigma,
        "close_move_sigma": close_move / sigma,
    }
    opposite = SHORT if direction == LONG else LONG
    flipped: dict = {}
    for stop in stops:
        for target in targets:
            row[(stop, target)] = first_touch(
                forward_high, forward_low, direction, entry,
                sigma * stop, sigma * target)
            flipped[(stop, target)] = first_touch(
                forward_high, forward_low, opposite, entry,
                sigma * stop, sigma * target)
    row["flipped"] = flipped
    return row


def _session_rows(rule, session: _Session, stops: tuple, targets: tuple,
                  one_per_direction: bool) -> list[dict]:
    """Measured signals from one session, at most one per direction.

    The last bar is never a signal bar: there would be no forward path to
    measure it against.
    """
    rows: list[dict] = []
    fired: set[str] = set()
    for i in range(session.n - 1):
        if one_per_direction and len(fired) == 2:
            break
        ctx = session.context(i)
        try:
            direction = rule(ctx)
        except Exception as exc:
            logger.warning("%s rule raised on %s bar %d: %s",
                           session.symbol, session.day, i, exc)
            break
        if direction not in (LONG, SHORT):
            continue
        if one_per_direction and direction in fired:
            continue
        fired.add(direction)
        rows.append(_signal_row(ctx, session.bars, direction, stops, targets))
    return rows


def run_rule(rule, frames: dict[str, pd.DataFrame],
             benchmark_change: "dict | None" = None,
             stops: tuple = DEFAULT_STOPS, targets: tuple = DEFAULT_TARGETS,
             warmup: int = 10, one_per_direction: bool = True) -> list[dict]:
    """Evaluate one rule over every session in `frames`.

    Returns a row per signal, carrying the forward excursions the rule never
    saw and a first-touch verdict for each (stop, target) pair.
    """
    changes = benchmark_change or {}
    rows: list[dict] = []
    for symbol, frame in frames.items():
        prepared = _prepare(frame)
        if prepared is None:
            continue
        unique, sessions = prepared
        if len(unique) <= warmup:
            continue
        for position, day in enumerate(unique):
            if position < warmup:
                continue
            session = _prepare_session(symbol, day, sessions,
                                       unique[:position], changes.get(day))
            if session is None:
                continue
            rows.extend(_session_rows(rule, session, stops, targets,
                                      one_per_direction))
    return rows


def _tally(rows: list[dict], key: tuple, stop: float, reward_risk: float,
           flipped: bool) -> tuple:
    """(hits, stops, open, summed R) for one geometry on one side.

    AN OPEN TRADE IS MARKED AT THE CLOSE, in R: its close move over the
    stop distance. It used to count as 0R, which contradicted this
    module's own docstring and forecast_diagnostics, where valuing
    unresolved trades at zero is named as the error that once produced a
    false edge. It matters most at the shipped geometry, where most
    signals reach neither level. The value is bounded by construction: a
    trade that touched neither level closed between them, so it lies in
    (-1, reward_risk).

    `flipped` reads the coin-flip twin instead, whose close move is the
    rule's with the sign reversed.
    """
    hits = stopped = open_ = 0
    summed = 0.0
    for row in rows:
        if key not in row:
            continue
        verdicts = row.get("flipped", {}) if flipped else row
        verdict = verdicts.get(key)
        if verdict is None:
            continue
        if verdict == "TARGET":
            hits += 1
        elif verdict == "STOP":
            stopped += 1
        else:
            open_ += 1
        summed += _outcome_r(verdict, row, stop, reward_risk, flipped)
    return hits, stopped, open_, summed


def _outcome_r(verdict: str, row: dict, stop: float, reward_risk: float,
               flipped: bool) -> float:
    """One trade's result in R: +reward_risk, -1, or marked at the close.

    The single scoring rule for _tally and edge_by_session, so the grid
    and the interval printed under it cannot score a trade two ways.
    """
    if verdict == "TARGET":
        return reward_risk
    if verdict == "STOP":
        return -1.0
    move = float(row["close_move_sigma"])
    return (-move if flipped else move) / stop


def edge_by_session(rows: list[dict], stop: float, target: float) -> dict:
    """The rule's edge over the coin at one geometry, with its interval.

    Per signal the edge is half the rule side minus the twin side, which
    averages to exactly the grid's `edge_r`. The 95% interval resamples
    whole SESSIONS through forecast_stats.block_bootstrap, because every
    signal on a day shares that day's move: 54,397 signals from 229
    sessions is 229 observations, not 54,397. Signals without a twin are
    left out rather than scored against nothing.
    """
    import forecast_stats

    key = (stop, target)
    reward_risk = target / stop
    values, groups = [], []
    for row in rows:
        own = row.get(key)
        twin = row.get("flipped", {}).get(key)
        if own is None or twin is None:
            continue
        values.append((_outcome_r(own, row, stop, reward_risk, False)
                       - _outcome_r(twin, row, stop, reward_risk, True)) / 2)
        groups.append(str(row["day"]))
    if not values:
        return {"stop": stop, "target": target, "signals": 0, "sessions": 0,
                "edge_r": None, "low": None, "high": None}
    mean, low, high, _ = forecast_stats.block_bootstrap(values, groups)
    estimable = low == low and high == high          # NaN is the refusal
    return {"stop": stop, "target": target, "signals": len(values),
            "sessions": len(set(groups)), "edge_r": float(mean),
            "low": float(low) if estimable else None,
            "high": float(high) if estimable else None}


def summarise(rows: list[dict], stops: tuple = DEFAULT_STOPS,
              targets: tuple = DEFAULT_TARGETS) -> dict:
    """Aggregate signals into excursion stats and a geometry grid.

    Expectancy is net of costs. Costs are charged in units of risk, because
    that is what makes them comparable across geometries: a wider stop risks
    more rupees for the same turnover, so the same charge is a smaller
    fraction of R.

    Every cell carries its own coin-flip baseline: `random_rate` and
    `random_gross_r` are what tossing a coin for direction at the same bars
    would have paid, and `edge_pp` / `edge_r` are the rule minus that.
    `edge_r` is the one to read. It is in R, it counts the trades that
    reached neither level, and it is gross, so costs cannot make a rule
    with real direction skill look like one without it. A positive
    `edge_r` in one cell of a 25-cell grid is not evidence on its own:
    pick the best of 25 noisy numbers and one will look good.
    """
    if not rows:
        return {"signals": 0, "grid": [], "excursions": {}}
    mfe = pd.Series([r["mfe_sigma"] for r in rows])
    mae = pd.Series([r["mae_sigma"] for r in rows])
    excursions = {
        "mfe": {f"p{p}": float(mfe.quantile(p / 100)) for p in (10, 25, 50, 75, 90)},
        "mae": {f"p{p}": float(mae.quantile(p / 100)) for p in (10, 25, 50, 75, 90)},
        "mfe_mean": float(mfe.mean()), "mae_mean": float(mae.mean()),
        "mfe_over_mae": float(mfe.mean() / mae.mean()) if mae.mean() else None,
    }
    grid = []
    for stop in stops:
        # Median stop distance as a fraction of entry. It depends only on
        # the stop, so it is computed once per stop rather than once per
        # (stop, target) cell.
        stop_fracs = [r["sigma"] * stop / r["entry"] for r in rows
                      if r["entry"] > 0]
        median_stop_frac = float(np.median(stop_fracs)) if stop_fracs else 0.0
        cost_in_r = (COST_FRACTION / median_stop_frac
                     if median_stop_frac > 0 else 0.0)
        for target in targets:
            reward_risk = target / stop
            hits, stopped, open_, summed = _tally(
                rows, (stop, target), stop, reward_risk, flipped=False)
            total = hits + stopped + open_
            resolved = hits + stopped
            if total == 0:
                continue
            gross = summed / total
            hit_rate = (hits / resolved) if resolved else None
            # The coin-flip baseline: the rule's side and the opposite side
            # pooled, which is a fair coin's expectation on the same bars.
            # See _signal_row. Only when EVERY row scored here carries its
            # twin: a row without one would leave the two sides describing
            # different sets of rows, so the baseline reads None instead.
            # Checked per row rather than by comparing totals, which one
            # rule-only row plus one twin-only row would satisfy.
            key = (stop, target)
            twinned = all(key in r.get("flipped", {})
                          for r in rows if key in r)
            c_hits, c_stopped, c_open, c_summed = _tally(
                rows, key, stop, reward_risk, flipped=True)
            c_total = c_hits + c_stopped + c_open
            c_resolved = c_hits + c_stopped
            pooled_resolved = resolved + c_resolved
            random_rate = ((hits + c_hits) / pooled_resolved
                           if twinned and pooled_resolved else None)
            random_gross = ((summed + c_summed) / (total + c_total)
                            if twinned else None)
            grid.append({
                "stop": stop, "target": target, "reward_risk": reward_risk,
                "signals": total, "resolved": resolved,
                "hit_rate": hit_rate,
                "random_rate": random_rate,
                "edge_pp": ((hit_rate - random_rate) * 100
                            if hit_rate is not None
                            and random_rate is not None else None),
                "gross_r": gross,
                "random_gross_r": random_gross,
                "edge_r": (gross - random_gross
                           if random_gross is not None else None),
                "cost_r": cost_in_r,
                "net_r": gross - cost_in_r,
            })
    return {"signals": len(rows), "grid": grid, "excursions": excursions,
            "sessions": len({r["day"] for r in rows}),
            "symbols": len({r["symbol"] for r in rows}),
            "longs": sum(1 for r in rows if r["direction"] == LONG),
            "shorts": sum(1 for r in rows if r["direction"] == SHORT)}


def best_geometry(summary: dict) -> "dict | None":
    """The (stop, target) pair with the highest net expectancy."""
    grid = [g for g in summary.get("grid", []) if g.get("net_r") is not None]
    return max(grid, key=lambda g: g["net_r"]) if grid else None


def _net_r_desc(row: dict) -> float:
    """Sort key putting the best net expectancy first, unscored rows last.

    An explicit None check rather than `or`, which treated a net_r of
    exactly 0.0 as missing and sorted a break-even geometry below every
    losing one.
    """
    value = row.get("net_r")
    return float("inf") if value is None else -float(value)


def print_report(name: str, summary: dict) -> None:
    """Print one rule's measured result."""
    if not summary.get("signals"):
        print(f"\n{name}: no signals")
        return
    exc = summary["excursions"]
    print(f"\n{'=' * 78}")
    print(f"{name}")
    print(f"  {summary['signals']:,} signals | {summary['sessions']} sessions | "
          f"{summary['symbols']} names | {summary['longs']:,} long / "
          f"{summary['shorts']:,} short")
    print(f"  MFE/MAE mean ratio {exc['mfe_over_mae']:.3f}  "
          f"(above 1.0 means price favours the signal)")
    print(f"  {'':<5}" + "".join(f"{f'p{p}':>8}" for p in (10, 25, 50, 75, 90))
          + f"{'mean':>8}")
    for label in ("mfe", "mae"):
        print(f"  {label.upper():<5}"
              + "".join(f"{exc[label][f'p{p}']:>8.3f}" for p in (10, 25, 50, 75, 90))
              + f"{exc[f'{label}_mean']:>8.3f}")
    best = best_geometry(summary)
    print("  coin = tossing a coin for direction at the same bars. "
          "edgeR = grossR - coinR, the one to read.")
    print("  hitgap = hit% - coin% over RESOLVED trades only. Not evidence: "
          "open trades can give it back by the close.")
    print(f"  {'stop':>5}{'tgt':>6}{'R:R':>5}{'resolved':>9}{'hit%':>7}"
          f"{'coin%':>7}{'hitgap':>7}{'grossR':>8}{'coinR':>8}{'edgeR':>8}"
          f"{'netR':>8}")
    print("  " + "-" * 78)

    def cell(value, spec: str, width: int, scale: float = 1.0) -> str:
        """One right-aligned number, or n/a when the cell has none.

        The sign flag has to precede the width in a format spec, so a
        signed spec like "+.3f" becomes ">+8.3f", never ">8+.3f". The
        value is rounded to the printed precision first, so a tiny
        negative prints as +0.000 rather than -0.000.
        """
        if value is None:
            return f"{'n/a':>{width}}"
        sign, rest = ("+", spec[1:]) if spec.startswith("+") else ("", spec)
        places = int(rest[1:-1])
        shown = round(value * scale, places) + 0.0
        return f"{shown:>{sign}{width}{rest}}"

    for row in sorted(summary["grid"], key=_net_r_desc):
        if row["hit_rate"] is None:
            continue
        mark = "  <<" if best and row is best else ""
        print(f"  {row['stop']:>5.2f}{row['target']:>6.2f}"
              f"{row['reward_risk']:>5.1f}{row['resolved']:>9,}"
              f"{cell(row['hit_rate'], '.1f', 7, 100.0)}"
              f"{cell(row.get('random_rate'), '.1f', 7, 100.0)}"
              f"{cell(row.get('edge_pp'), '+.1f', 7)}"
              f"{cell(row['gross_r'], '+.3f', 8)}"
              f"{cell(row.get('random_gross_r'), '+.3f', 8)}"
              f"{cell(row.get('edge_r'), '+.3f', 8)}"
              f"{cell(row['net_r'], '+.3f', 8)}{mark}")
    if best:
        print(f"  << marks the best of {len(summary['grid'])} cells, chosen "
              f"after seeing them. That choice is not evidence of an edge.")


def reference_breakout(ctx: Context) -> "str | None":
    """The structural core of the shipped rule, and nothing else.

    LONG when price is above the session VWAP AND above the opening-range
    high; SHORT on the mirror; None otherwise, including before the range
    has closed. The live scanner adds relative-volume, strength, turnover
    and reachability gates on top, so this measures the direction call
    those gates filter, not the full scanner. It is here so the README's
    headline figures can be reproduced from the repository rather than
    from a script that was never committed.
    """
    if not ctx.orb_closed:
        return None
    line = float(ctx.vwap[-1])
    if not np.isfinite(line):
        return None
    # The same comparisons as setups.direction, ties included: price AT
    # the VWAP is "not above" it there, so it counts on the SHORT side.
    above_vwap = ctx.price > line
    if above_vwap and ctx.price > ctx.orb_high:
        return LONG
    if not above_vwap and ctx.price < ctx.orb_low:
        return SHORT
    return None


def main(argv: "list[str] | None" = None) -> int:
    """Measure reference_breakout over the cached one-year 5-minute bars.

    Reads only the local bar cache, through forecast_diagnostics'
    loader so the cache naming lives in one place rather than here as
    well. Never fetches. The benchmark index is dropped: it is not a
    tradeable name and the rule reads no benchmark field.
    """
    import argparse

    import forecast_diagnostics

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="measure only the first N symbols (0 = all)")
    args = parser.parse_args(argv)
    frames = forecast_diagnostics.load_frames()
    frames.pop(forecast_diagnostics.BENCHMARK, None)
    if args.limit > 0:
        frames = dict(list(frames.items())[:args.limit])
    if not frames:
        print("No cached 5-minute bars found under bar_cache.")
        return 1
    rows = run_rule(reference_breakout, frames,
                    warmup=forecast_diagnostics.WARMUP)
    print_report("reference_breakout: VWAP side + opening-range break",
                 summarise(rows))
    # The shipped geometry in this harness's units: the live stop fraction
    # of one plausible move, and the live reward-to-risk multiple of it.
    stop = config.SCAN_STOP_FRACTION
    shipped = edge_by_session(rows, stop, stop * config.SCAN_REWARD_RISK)
    if shipped["edge_r"] is not None:
        interval = ("not estimable (too few sessions)"
                    if shipped["low"] is None else
                    f"95% interval {shipped['low']:+.3f} to "
                    f"{shipped['high']:+.3f} R")
        print(f"\n  Shipped geometry (stop {shipped['stop']:.2f}, target "
              f"{shipped['target']:.2f}): edge over the coin "
              f"{shipped['edge_r']:+.4f} R per signal, {interval}, "
              f"resampling {shipped['sessions']} whole sessions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
