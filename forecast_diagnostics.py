"""Does the scanner's confidence score rank outcomes at all?

The one diagnostic worth running before improving anything. Every past
signal gets the PRODUCTION score (setups.evaluate, not a reimplementation)
and its true forward outcome. Then: sort by score, cut into ten groups, and
look at net R per group.

  monotone rising  -> the score carries information; trading only the top
                      slice can be positive even when the average is zero
  flat             -> the score carries nothing, and no amount of
                      selectivity, sizing or extra features rescues it

Two honesty requirements built in:

  * Signals inside one session move together, so the confidence interval is
    a block bootstrap resampling whole SESSION-DAYS, not individual signals.
    Treating 57k correlated signals as 57k independent draws is what makes
    a coin flip look like an edge.
  * Historical open interest is not in the cache, so the score's
    positioning term is 0.0 for every signal. It is 15% of the weight and
    constant, so it cannot affect the RANKING this test measures - but the
    absolute scores here run below live ones.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config          # noqa: E402
import edge_lab        # noqa: E402
import forecast_stats  # noqa: E402
import indicators      # noqa: E402
import setups          # noqa: E402
from setups import Readings  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "bar_cache"
OUT_DIR = ROOT / "forecast_cache"
OUT_DIR.mkdir(exist_ok=True)


SPAN = "20250908_20260908"          # the one-year 5-minute pull
BENCHMARK = "NIFTY 50"   # as load_frames spells it, underscores already swapped
WARMUP = 20                         # sessions before a symbol is measurable
MINUTES_PER_BAR = 5


def load_frames() -> dict[str, pd.DataFrame]:
    """Every symbol's one-year 5-minute frame from the cache."""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(CACHE.glob(f"kite__*__5minute__{SPAN}.parquet")):
        symbol = path.name.split("__")[1].replace("_", " ")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            print(f"  unreadable {path.name}: {exc}")
            continue
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        frames[symbol] = frame
    return frames


def benchmark_changes(frame: pd.DataFrame) -> dict:
    """Benchmark percent change per session, indexed BY TIMESTAMP.

    This returned one scalar per session - the index's FULL-DAY move - and
    handed it to every bar of that session. Since relative_strength is
    day_change_pct minus this, and relative_strength is both a hard gate
    (setups._strength_reason) and 20% of the score being tested here, every
    signal was scored against the outcome of the session it was still
    trading in. edge_lab's own docstring names this trap and warns that a
    caller supplying full-day changes is opting into lookahead; this file
    was that caller.

    Now: a Series per session of the index's move UP TO EACH BAR, so bar i
    sees only what the index had done by bar i. Aligned on timestamps
    rather than bar position, because 658 of 6,225 session-pairs in the
    cache have different bar counts and position-matching would read a
    later clock time off the benchmark.
    """
    if frame is None or frame.empty:
        return {}
    closes = frame["Close"].dropna()
    by_day: dict = {}
    for day, series in closes.groupby(closes.index.date):
        if len(series) < 2:
            continue
        first = float(series.iloc[0])
        if first > 0:
            by_day[day] = (series / first - 1.0) * 100.0
    return by_day


def benchmark_at(series, stamp) -> "float | None":
    """The benchmark's move as at `stamp`, or None when it is unknown.

    Looks the timestamp up rather than clamping to the last available bar:
    clamping would serve the index's closing move for any bar past the
    benchmark's own last one, which is the same lookahead by another route.
    """
    if series is None or len(series) == 0:
        return None
    upto = series.loc[series.index <= stamp]
    return float(upto.iloc[-1]) if len(upto) else None


def turnover_estimate(sessions: dict, prior_days: list) -> float:
    """Median daily rupee turnover over prior sessions.

    Estimated from the 5-minute bars themselves (sum of close x volume)
    rather than from daily bars, because only the intraday cache is loaded
    here. It feeds the liquidity gate, whose job is to exclude illiquid
    names; on the F&O universe every name clears it comfortably, so the
    approximation is not doing load-bearing work.
    """
    values = []
    for day in prior_days[-20:]:
        bars = sessions.get(day)
        if not bars:
            continue
        volume = np.asarray(bars["volume"], dtype=float)
        close = np.asarray(bars["close"], dtype=float)
        total = float(np.nansum(volume * close))
        if total > 0:
            values.append(total)
    return float(np.median(values)) if values else 0.0


def reading_from(ctx, turnover: float,
                 benchmark: "float | None" = None) -> Readings:
    """A production Readings built from one edge_lab Context.

    Every field that the gates or the score actually consult is filled from
    the context. The derivatives fields are None because historical OI is
    not cached - which fails the OI gate open (it is advisory by default)
    and zeroes the positioning term.
    """
    i = ctx.i
    highs = ctx.high[:i + 1]
    lows = ctx.low[:i + 1]
    vwap = float(ctx.vwap[i]) if not math.isnan(ctx.vwap[i]) else None
    orb = None
    if ctx.orb_closed and not math.isnan(ctx.orb_high) and not math.isnan(ctx.orb_low):
        orb = indicators.OpeningRange(
            high=float(ctx.orb_high), low=float(ctx.orb_low),
            bars=edge_lab.BARS_PER_ORB,
            minutes=edge_lab.BARS_PER_ORB * MINUTES_PER_BAR)
    price = ctx.price
    distance = ((price - vwap) / vwap * 100.0) if vwap else None
    change = None
    if ctx.prev_close:
        change = (price / ctx.prev_close - 1.0) * 100.0
    return Readings(
        symbol=ctx.symbol, ticker=ctx.symbol, last=price,
        prev_close=ctx.prev_close, day_change_pct=change,
        vwap=vwap, vwap_distance_pct=distance,
        opening_range=orb, cpr=None, atr_bar=ctx.atr,
        rvol=ctx.rvol,
        # NOT ctx.relative_strength: that reads Context.benchmark_change,
        # which the caller used to fill with the session's full-day move.
        # Computed here from the point-in-time benchmark instead, and left
        # None when the benchmark is unknown so the gate fails closed.
        relative_strength=(None if benchmark is None or change is None
                           else change - benchmark),
        turnover_20d=turnover, oi_change_pct=None,
        futures_share=None, derivatives_turnover=None,
        day_high=float(np.nanmax(highs)), day_low=float(np.nanmin(lows)),
        minutes_left=ctx.bars_left * MINUTES_PER_BAR,
        bars_left=ctx.bars_left,
        session_bar_count=i + 1, range_closed=ctx.orb_closed)


def outcome(ctx, session, direction: str, trade) -> tuple:
    """(gross R, cost R) for one signal, from the path it never saw.

    Unresolved trades are marked out at the close rather than dropped:
    valuing them at zero was one of the two errors that manufactured a
    false edge here before.
    """
    i = ctx.i
    forward_high = np.asarray(session.bars["high"][i + 1:], dtype=float)
    forward_low = np.asarray(session.bars["low"][i + 1:], dtype=float)
    if forward_high.size == 0:
        return None, None
    risk = trade.risk_per_share
    if risk <= 0 or trade.lot_risk <= 0:
        return None, None
    touch = edge_lab.first_touch(forward_high, forward_low, direction,
                                 trade.entry, risk, trade.target_distance)
    if touch == "TARGET":
        gross = trade.target_distance / risk
    elif touch == "STOP":
        gross = -1.0
    else:
        close = float(session.bars["close"][-1])
        move = (close - trade.entry) if direction == edge_lab.LONG \
            else (trade.entry - close)
        gross = move / risk
    return gross, trade.cost_rupees / trade.lot_risk


def candidate_bars(session) -> list:
    """Every bar where a direction is possible, in order.

    This used to return only the FIRST qualifying bar per direction, on the
    argument that a session takes one signal per direction so a later bar
    could never become one. That argument is wrong, and the error was
    measured: choose_direction needs only VWAP side plus an opening-range
    break, but an ACTIONABLE signal must also clear liquidity, relative
    volume, strength, time, cost, win-rate and reachability. When the first
    breaking bar fails one of those, the old code consumed the session and
    never looked at the later bar that did pass.

    Counted over 12 cached symbols and 2,748 sessions: 449 actionable
    signals sat at the first qualifying bar and 746 sat at a later one, so
    62% of what the scanner would actually show was invisible - and missing
    non-randomly, in exactly the variables being scored.

    So the full set is returned and the caller stops per direction at the
    first bar that is actionable, keeping the first break separately for
    the gates-ignored table.
    """
    n = session.n
    orb = edge_lab.BARS_PER_ORB
    if n <= orb + 1:
        return []
    high = np.asarray(session.bars["high"], dtype=float)
    low = np.asarray(session.bars["low"], dtype=float)
    close = np.asarray(session.bars["close"], dtype=float)
    vwap = np.asarray(session.vwap, dtype=float)
    orb_high = float(np.nanmax(high[:orb]))
    orb_low = float(np.nanmin(low[:orb]))
    if not np.isfinite(orb_high) or not np.isfinite(orb_low):
        return []
    idx = np.arange(n - 1)
    idx = idx[idx >= orb]
    if idx.size == 0:
        return []
    price = close[idx]
    ref = vwap[idx]
    with np.errstate(invalid="ignore"):
        qualifies = ((price > ref) & (price > orb_high)) |                     ((price <= ref) & (price < orb_low))
    return [int(i) for i in idx[qualifies]]


def collect() -> pd.DataFrame:
    """Every measurable signal with its production score and true outcome."""
    frames = load_frames()
    print(f"  loaded {len(frames)} symbol frames from cache")
    bench = frames.pop(BENCHMARK, None)
    changes = benchmark_changes(bench) if bench is not None else {}
    print(f"  benchmark sessions: {len(changes)}")
    # Fail loudly. Without the benchmark, relative_strength is None for
    # every signal, the strength gate fails closed, evaluate() never
    # reaches score_setup, and every score is 0.0 - which would render the
    # whole decile test vacuous while still printing a tidy table.
    if len(changes) < 100:
        raise SystemExit(
            f"benchmark {BENCHMARK!r} gave {len(changes)} sessions; "
            f"available: {sorted(frames)[:5]}...")

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    items = sorted(frames.items())
    if limit:
        items = items[:limit]
        print(f"  SMOKE RUN: first {len(items)} symbols only")
    rows = []
    skipped = defaultdict(int)
    for n_done, (symbol, frame) in enumerate(items, start=1):
        prepared = edge_lab._prepare(frame)
        if prepared is None:
            skipped["unprepared"] += 1
            continue
        unique, sessions = prepared
        if len(unique) <= WARMUP:
            skipped["too few sessions"] += 1
            continue
        # edge_lab's session dict carries only the OHLCV arrays, not the
        # timestamps, so they are rebuilt here. Needed to look the
        # benchmark up by clock time rather than by bar position - 658 of
        # 6,225 session-pairs in the cache have different bar counts, and
        # position-matching would read the index at a later time than the
        # stock's bar. Vectorised .date on the index, once per symbol.
        index = frame.index
        day_of = index.date
        stamps_by_day = {day: index[day_of == day] for day in unique}
        for position, day in enumerate(unique):
            if position < WARMUP:
                continue
            session = edge_lab._prepare_session(
                symbol, day, sessions, unique[:position], None)
            if session is None:
                skipped["unmeasurable session"] += 1
                continue
            bench_series = changes.get(day)
            turnover = turnover_estimate(sessions, unique[:position])
            stamps = stamps_by_day.get(day)
            fired_actionable, fired_break = set(), set()
            for i in candidate_bars(session):
                ctx = session.context(i)
                # Point-in-time benchmark for THIS bar. Context is frozen,
                # so the value is handed to reading_from rather than set on
                # it, and a bar with no benchmark reading gets None so the
                # strength gate fails closed instead of guessing.
                stamp = (stamps[i] if stamps is not None and i < len(stamps)
                         else None)
                bench = (benchmark_at(bench_series, stamp)
                         if stamp is not None else None)
                try:
                    reading = reading_from(ctx, turnover, bench)
                    setup = setups.evaluate(reading)
                except Exception as exc:
                    skipped[f"evaluate: {type(exc).__name__}"] += 1
                    continue
                if setup.direction == setups.NO_SETUP or setup.levels is None:
                    continue
                side = setup.direction
                # One row per direction per session, as the scanner takes.
                # The kept bar is the first ACTIONABLE one, falling back to
                # the first break when no bar in the session ever passes -
                # so the gates-ignored table still has a row and the
                # gate-passing table is no longer a biased subsample.
                if setup.actionable:
                    if side in fired_actionable:
                        continue
                    fired_actionable.add(side)
                else:
                    if side in fired_actionable or side in fired_break:
                        continue
                    fired_break.add(side)
                gross, cost = outcome(ctx, session, setup.direction, setup.levels)
                if gross is None:
                    continue
                # evaluate() returns rank_score 0.0 whenever ANY gate fails,
                # so 80% of candidates collapsed onto one value and eight of
                # the ten deciles were unrankable. Call score_setup directly
                # to get the score the formula actually yields, so the
                # ranking is defined across the whole candidate range.
                if reading.rvol is None or reading.relative_strength is None:
                    skipped["score needs rvol and strength"] += 1
                    continue
                raw = setups.score_setup(reading, setup.levels)
                rows.append({
                    "symbol": symbol, "day": day, "bar": i,
                    "direction": setup.direction,
                    "score": raw,
                    "prod_score": setup.rank_score,
                    "passed": bool(setup.actionable),
                    "gross_r": gross, "cost_r": cost,
                    "net_r": gross - cost,
                    "rvol": ctx.rvol,
                    "rel_strength": ctx.relative_strength,
                })
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(items)} symbols, {len(rows)} signals", flush=True)
    if skipped:
        print("  skipped:", dict(skipped))
    return pd.DataFrame(rows)


def block_bootstrap_mean(data: pd.DataFrame, column: str,
                         draws: int = 2000) -> tuple:
    """Mean and a 95% interval that resamples whole SESSION-DAYS.

    Resampling individual signals would treat correlated same-day trades as
    independent and shrink the interval to nothing. The day is the block.

    Delegates to forecast_stats, which is tested, rather than keeping the
    second copy this used to be. Agreement was verified to zero absolute
    difference across 40 randomised trials - but NOT unconditionally, and
    the difference is deliberate in both places it appears:

      * Fewer than five day-blocks now returns a NaN interval instead of a
        number. One day of signals cannot support a bootstrap interval,
        and the old body returned [obs, obs] for a single block, which
        reads as perfect precision. Per-decile calls with a thin decile
        will therefore show [nan, nan] where they used to show a range.
      * Rows with a NaN `day` used to be dropped silently by groupby and
        are now kept as their own block, which widens the interval rather
        than quietly discarding data.

    The p-value is dropped on the way out only to keep this function's
    existing three-value signature; callers wanting it use forecast_stats
    directly.
    """
    if data.empty or column not in data.columns:
        return float("nan"), float("nan"), float("nan")
    mean, low, high, _ = forecast_stats.block_bootstrap(
        data[column].to_numpy(dtype=float), data["day"].to_numpy(),
        draws=draws)
    return mean, low, high


def spearman(left: pd.Series, right: pd.Series) -> float:
    """Rank correlation, computed without scipy (not installed here).

    Spearman is just Pearson on the ranks, so ranking both sides and
    correlating is the definition rather than an approximation.
    """
    if len(left) < 3:
        return float("nan")
    return float(left.rank().corr(right.rank()))


def deciles(data: pd.DataFrame, label: str) -> None:
    """Net R by score decile, with a day-blocked interval on each."""
    print()
    print("=" * 86)
    print(f"{label}   n={len(data):,}  sessions={data['day'].nunique():,}  "
          f"symbols={data['symbol'].nunique():,}")
    print("=" * 86)
    if data.empty:
        print("  no signals")
        return
    if data["score"].nunique() < 10:
        print(f"  only {data['score'].nunique()} distinct scores; "
              f"using those as bins")
        data = data.assign(bucket=data["score"].rank(method="dense") - 1)
    else:
        data = data.assign(
            bucket=pd.qcut(data["score"].rank(method="first"), 10,
                           labels=False, duplicates="drop"))
    print(f"{'decile':>7} {'score range':>16} {'n':>7} {'win%':>7} "
          f"{'gross R':>9} {'cost R':>8} {'net R':>9} {'95% CI on net':>22}")
    print("-" * 86)
    for bucket, group in data.groupby("bucket"):
        obs, lo, hi = block_bootstrap_mean(group, "net_r", draws=600)
        wins = float((group["gross_r"] > 0).mean() * 100.0)
        print(f"{int(bucket) + 1:>7} "
              f"{group['score'].min():>7.4f}-{group['score'].max():<8.4f} "
              f"{len(group):>7,} {wins:>6.1f}% "
              f"{group['gross_r'].mean():>9.4f} "
              f"{group['cost_r'].mean():>8.4f} "
              f"{obs:>9.4f} "
              f"[{lo:>8.4f}, {hi:>8.4f}]")
    print("-" * 86)
    obs, lo, hi = block_bootstrap_mean(data, "net_r")
    print(f"{'ALL':>7} {'':>16} {len(data):>7,} "
          f"{float((data['gross_r'] > 0).mean() * 100.0):>6.1f}% "
          f"{data['gross_r'].mean():>9.4f} {data['cost_r'].mean():>8.4f} "
          f"{obs:>9.4f} [{lo:>8.4f}, {hi:>8.4f}]")

    # Rank correlation on the day means, so correlated same-day signals
    # cannot inflate the sample size behind the coefficient.
    per_day = data.groupby("day").apply(
        lambda g: pd.Series({"score": g["score"].mean(),
                             "net_r": g["net_r"].mean()}),
        include_groups=False)
    if len(per_day) > 5:
        print(f"  Spearman(score, net R) across {len(per_day)} session-days: "
              f"{spearman(per_day['score'], per_day['net_r']):+.4f}")
    print(f"  Spearman(score, net R) across all {len(data):,} signals: "
          f"{spearman(data['score'], data['net_r']):+.4f}"
          f"  (optimistic: signals are not independent)")

    top, bottom = data[data["bucket"] == data["bucket"].max()], \
        data[data["bucket"] == data["bucket"].min()]
    if not top.empty and not bottom.empty:
        spread = top["net_r"].mean() - bottom["net_r"].mean()
        print(f"  top decile minus bottom decile, net R: {spread:+.4f}")


if __name__ == "__main__":
    print("Collecting signals with production scores and true outcomes...")
    data = collect()
    if data.empty:
        print("NO SIGNALS COLLECTED - nothing to analyse")
        raise SystemExit(1)
    out = OUT_DIR / "decile_signals.parquet"
    try:
        data.to_parquet(out)
        print(f"\n  saved {len(data):,} signals to {out}")
    except Exception as exc:
        print(f"  could not save: {exc}")
    deciles(data, "ALL DIRECTIONAL CANDIDATES (score computed, gates ignored)")
    deciles(data[data["passed"]].copy(),
            "GATE-PASSING SIGNALS ONLY (what the scanner would show you)")
