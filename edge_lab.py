"""A harness for testing intraday rules against measured price behaviour.

Built because the first rule failed in a way that could only be diagnosed by
measurement. Across 3,537 signals its adverse excursion exceeded its
favourable excursion at every percentile, and every stop/target pair lost
money - so the problem was the signal, not the geometry. Guessing a second
rule would repeat the mistake.

The contract is deliberately narrow so that rules are comparable and cannot
cheat:

    def my_rule(ctx):
        '''Return "LONG", "SHORT" or None for the bar at ctx.i.'''

`ctx` exposes only what had printed by bar `i`. Arrays are pre-sliced;
scalars like the previous close and the ATR come from prior sessions. There
is no field on the context that contains a future price, so a rule cannot
look ahead even by accident. The harness then measures what happened after
the signal - which is the part the rule never sees.

Three disciplines, all enforced here rather than left to each rule:

  one signal per direction per session, taken on the first bar it fires, so
    one move cannot be counted as many wins;
  a bar that spans both stop and target counts as a stop, because 5-minute
    bars carry no intra-bar ordering and assuming otherwise flatters results;
  unresolved positions are marked out at the close, not discarded, because a
    rule that leaves most trades open must be judged on what that pays.

Sample-size honesty: 5-minute bars reach about 59 sessions. Signals within a
session are correlated, so an edge measured here is provisional. Treat a
result that survives only at one parameter setting as noise.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import instruments

logger = logging.getLogger(__name__)

LONG = "LONG"
SHORT = "SHORT"

CACHE_DIR = config.PROJECT_ROOT / "bar_cache"
BARS_PER_ORB = max(1, config.SCAN_OPENING_RANGE_MINUTES // 5)
# Prior sessions in the relative-volume baseline. Matches the live
# scanner's 20-day window; also bounds the per-session precompute, which is
# what keeps the harness linear in sessions rather than quadratic.
RVOL_BASELINE_SESSIONS = 20
DEFAULT_STOPS = (0.25, 0.5, 0.75, 1.0, 1.5)
DEFAULT_TARGETS = (0.25, 0.5, 0.75, 1.0, 1.5)

# Round-trip equity cost as a fraction of turnover, measured in trade_costs:
# 82.45 rupees on 1,00,000 of buy turnover.
COST_FRACTION = 0.000824


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
    orb_high: float              # opening range, None until it has closed
    orb_low: float
    rvol: float                  # cumulative volume vs prior sessions at bar i
    benchmark_change: "float | None"   # index % change on the day so far

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


def load_bars(symbols: list[str], period: str = "60d",
              interval: str = "5m", refresh: bool = False
              ) -> dict[str, pd.DataFrame]:
    """Fetch and cache OHLCV frames, one parquet file per symbol.

    Parquet rather than pickle: the cache is data, and data should not be
    able to execute on load.
    """
    import scan_data  # local: scan_data imports yfinance, which is slow

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{interval}_{period}"
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
        import yfinance as yf
        tickers = [instruments.to_ticker(s) for s in missing]
        by_ticker: dict[str, pd.DataFrame] = {}
        for batch in scan_data.chunks(tickers, config.SCAN_BATCH_SIZE):
            raw = yf.download(batch, period=period, interval=interval,
                              auto_adjust=True, progress=False,
                              group_by="column")
            if raw is not None and not raw.empty:
                by_ticker.update(scan_data._unpack(raw, batch))
        for symbol in missing:
            frame = by_ticker.get(instruments.to_ticker(symbol))
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
            bars = sessions[day]
            n = len(bars["close"])
            if n <= BARS_PER_ORB + 1:
                continue
            prior_days = unique[:position]
            prior_tr = np.concatenate([sessions[d]["tr"] for d in prior_days])
            atr = _wilder(prior_tr, config.SCAN_ATR_BARS)
            if atr is None or atr <= 0:
                continue
            last_prior = sessions[prior_days[-1]]
            prev_close = float(last_prior["close"][-1])
            prev_high = float(last_prior["high"].max())
            prev_low = float(last_prior["low"].min())
            # Relative-volume baseline: the MEDIAN cumulative-volume curve
            # across the most recent prior sessions, computed once per
            # session rather than per bar. Recomputing it per bar made the
            # cost grow with sessions squared, which is unusable past a few
            # dozen sessions. The window also matches the live scanner's
            # 20-day baseline instead of quietly widening with history.
            recent = prior_days[-RVOL_BASELINE_SESSIONS:]
            curves = [np.cumsum(sessions[d]["volume"]) for d in recent]
            width = max((len(c) for c in curves), default=0)
            baseline_curve = np.zeros(width)
            if curves and width:
                padded = np.full((len(curves), width), np.nan)
                for row, curve in enumerate(curves):
                    padded[row, :len(curve)] = curve
                with np.errstate(invalid="ignore"):
                    baseline_curve = np.nanmedian(padded, axis=0)
            typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
            cum_volume = np.cumsum(bars["volume"])
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap = np.cumsum(typical * bars["volume"]) / cum_volume
            orb_high = float(bars["high"][:BARS_PER_ORB].max())
            orb_low = float(bars["low"][:BARS_PER_ORB].min())
            fired: set[str] = set()
            for i in range(n - 1):
                if one_per_direction and len(fired) == 2:
                    break
                baseline = (float(baseline_curve[i])
                            if i < len(baseline_curve) else 0.0)
                rvol = (float(cum_volume[i]) / baseline
                        if baseline > 0 and np.isfinite(baseline) else 0.0)
                ctx = Context(
                    symbol=symbol, day=day, i=i, n=n,
                    open_=bars["open"][:i + 1], high=bars["high"][:i + 1],
                    low=bars["low"][:i + 1], close=bars["close"][:i + 1],
                    volume=bars["volume"][:i + 1], vwap=vwap[:i + 1],
                    atr=atr, prev_close=prev_close, prev_high=prev_high,
                    prev_low=prev_low, orb_high=orb_high, orb_low=orb_low,
                    rvol=rvol, benchmark_change=changes.get(day))
                try:
                    direction = rule(ctx)
                except Exception as exc:
                    logger.warning("%s rule raised on %s %s bar %d: %s",
                                   symbol, day, symbol, i, exc)
                    break
                if direction not in (LONG, SHORT):
                    continue
                if one_per_direction and direction in fired:
                    continue
                fired.add(direction)
                sigma = ctx.sigma
                if sigma <= 0:
                    continue
                entry = ctx.price
                forward_high = bars["high"][i + 1:]
                forward_low = bars["low"][i + 1:]
                if forward_high.size == 0:
                    continue
                if direction == LONG:
                    mfe = float(forward_high.max()) - entry
                    mae = entry - float(forward_low.min())
                    close_move = float(bars["close"][-1]) - entry
                else:
                    mfe = entry - float(forward_low.min())
                    mae = float(forward_high.max()) - entry
                    close_move = entry - float(bars["close"][-1])
                row = {
                    "symbol": symbol, "day": day, "i": i,
                    "direction": direction, "entry": entry, "sigma": sigma,
                    "bars_left": ctx.bars_left, "rvol": rvol,
                    "relative_strength": ctx.relative_strength,
                    "mfe_sigma": max(0.0, mfe) / sigma,
                    "mae_sigma": max(0.0, mae) / sigma,
                    "close_move_sigma": close_move / sigma,
                }
                for stop in stops:
                    for target in targets:
                        row[(stop, target)] = first_touch(
                            forward_high, forward_low, direction, entry,
                            sigma * stop, sigma * target)
                rows.append(row)
    return rows


def summarise(rows: list[dict], stops: tuple = DEFAULT_STOPS,
              targets: tuple = DEFAULT_TARGETS) -> dict:
    """Aggregate signals into excursion stats and a geometry grid.

    Expectancy is net of costs. Costs are charged in units of risk, because
    that is what makes them comparable across geometries: a wider stop risks
    more rupees for the same turnover, so the same charge is a smaller
    fraction of R.
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
        for target in targets:
            verdicts = [r[(stop, target)] for r in rows if (stop, target) in r]
            hits = verdicts.count("TARGET")
            stopped = verdicts.count("STOP")
            open_ = verdicts.count("NEITHER")
            total = hits + stopped + open_
            resolved = hits + stopped
            if total == 0:
                continue
            reward_risk = target / stop
            gross = (hits * reward_risk - stopped) / total
            # Median stop distance as a fraction of entry, per geometry.
            stop_fracs = [r["sigma"] * stop / r["entry"] for r in rows
                          if r["entry"] > 0]
            median_stop_frac = float(np.median(stop_fracs)) if stop_fracs else 0.0
            cost_in_r = (COST_FRACTION / median_stop_frac
                         if median_stop_frac > 0 else 0.0)
            grid.append({
                "stop": stop, "target": target, "reward_risk": reward_risk,
                "signals": total, "resolved": resolved,
                "hit_rate": (hits / resolved) if resolved else None,
                "random_rate": stop / (stop + target),
                "edge_pp": ((hits / resolved) - stop / (stop + target)) * 100
                           if resolved else None,
                "gross_r": gross, "cost_r": cost_in_r,
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
    print(f"  {'stop':>5}{'tgt':>6}{'R:R':>5}{'resolved':>9}{'hit%':>7}"
          f"{'rand%':>7}{'edge':>7}{'grossR':>8}{'netR':>8}")
    print("  " + "-" * 62)
    for row in sorted(summary["grid"], key=lambda g: -(g["net_r"] or -9)):
        if row["hit_rate"] is None:
            continue
        mark = "  <<" if best and row is best else ""
        print(f"  {row['stop']:>5.2f}{row['target']:>6.2f}"
              f"{row['reward_risk']:>5.1f}{row['resolved']:>9,}"
              f"{row['hit_rate'] * 100:>7.1f}{row['random_rate'] * 100:>7.1f}"
              f"{row['edge_pp']:>+7.1f}{row['gross_r']:>+8.3f}"
              f"{row['net_r']:>+8.3f}{mark}")
