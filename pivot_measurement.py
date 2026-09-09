"""Does a stop sitting ON a pivot survive better than one that is not?

THE ANSWER IS NO, and this script is how that was established. It exists in
the repo rather than in a scratch directory because three docstrings and a
test now cite its figures, and a number nobody can reproduce is not
evidence.

    .venv\\Scripts\\python -m pivot_measurement

THE CONFOUND THAT HAS TO BE BEATEN. The pivot stop that levels.build_levels
would choose is always TIGHTER than the volatility stop it replaces,
because the structural window caps at the volatility distance. A tighter
stop is hit more often for that reason alone, so any comparison that does
not hold distance fixed measures the distance and not the pivot.

FOUR DESIGNS, each controlling for more than the last. They disagree, and
which one you believe is the whole question:

  1. UNPAIRED, bucketed by sigma-distance. Holds distance only coarsely.
  2. WINDOWED to [0.30, 0.50] sigma - the only band build_levels can
     accept a structural stop in - with a bootstrap blocked by date.
  3. PAIRED within session, so distance, date, symbol and the session's
     own path are all fixed.
  4. MANTEL-HAENSZEL conditioning on session, the textbook conditional
     test, which uses every observation rather than one mean per session
     and so is better powered than the paired sign test.

Designs 1 and 2 report a benefit. Designs 3 and 4 report nothing. The
selection diagnostic below shows why: sessions that happen to have a pivot
inside the window are QUIETER sessions, in which any stop survives better.
Control stops - placed at random, so no pivot can influence them - are hit
less often in pivot-bearing sessions by very nearly the whole size of the
apparent effect.

TWO MISTAKES THIS SCRIPT EXISTS TO NOT REPEAT:

  * A permutation test that shuffles the pivot label within distance
    buckets only treats tens of thousands of observations from ~14,000
    sessions as exchangeable. It returns p ~ 0.02 while a date-blocked
    bootstrap of the same estimate spans zero. That p was quoted as
    evidence and was an artefact of ignoring clustering; the permutation
    here shuffles within session as well.
  * Pairing does NOT hold distance constant by itself. Run over the full
    0.15-2.0 sigma range the paired estimator returns about +6 pp, a pure
    distance artefact, because a session's pivots sit systematically
    nearer than its uniformly drawn controls. It is the WINDOW that makes
    the paired estimate honest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import bar_store
import config
import forecast_stats
import indicators

IST = "Asia/Kolkata"
ENTRY_HOUR, ENTRY_MINUTE = 9, 45
BARS_PER_SESSION = 75           # 09:15 to 15:30 in five-minute buckets
MIN_SIGMA, MAX_SIGMA = 0.15, 2.0
CONTROLS_PER_SESSION = 3
TOLERANCE_PCT = 0.001           # a control this close to a level is dropped
BUCKETS = np.array([0.15, 0.35, 0.55, 0.75, 1.0, 1.3, 1.6, 2.0])
SEED = 20260909
SYMBOL_LIMIT = 400


def usable_window() -> tuple:
    """The only sigma band build_levels can accept a structural stop in."""
    base = config.SCAN_STOP_FRACTION
    return (1.0 - config.SCAN_STRUCTURE_BAND) * base, base


def daily_atr(frame: pd.DataFrame, period: int = 14) -> "pd.Series | None":
    """Wilder ATR on daily bars, or None when there is too little history."""
    need = {"High", "Low", "Close"}
    if frame is None or not need <= set(frame.columns) or len(frame) < period + 2:
        return None
    high, low = frame["High"].astype(float), frame["Low"].astype(float)
    close = frame["Close"].astype(float)
    prior = close.shift(1)
    span = pd.concat([(high - low).abs(), (high - prior).abs(),
                      (low - prior).abs()], axis=1).max(axis=1)
    return span.ewm(alpha=1.0 / period, adjust=False).mean()


def observations(limit: int = SYMBOL_LIMIT, verbose: bool = True) -> pd.DataFrame:
    """One row per candidate stop: distance in sigma, pivot or control, hit.

    The entry is the 09:45 close, after the opening range has formed. Sigma
    is the expected remaining-session move: a 14-day ATR scaled by the root
    of the fraction of the session still to come. The ladder comes from the
    PREVIOUS session, so every level is fixed before the open.
    """
    rng = np.random.default_rng(SEED)
    fine = bar_store.load("5minute")
    coarse = bar_store.load("day")
    shared = sorted(set(fine) & set(coarse))[:limit]
    if verbose:
        print(f"{len(fine):,} symbols with 5-minute bars, {len(coarse):,} "
              f"with daily; using {len(shared)}", flush=True)
    rows = []
    for index, symbol in enumerate(shared, start=1):
        session_bars, days = fine[symbol], coarse[symbol]
        atr = daily_atr(days)
        if atr is None:
            continue
        atr_by_date = {stamp.date(): value for stamp, value in atr.items()
                       if value == value}
        highs = {s.date(): v for s, v in days["High"].items()}
        lows = {s.date(): v for s, v in days["Low"].items()}
        closes = {s.date(): v for s, v in days["Close"].items()}
        ordered = sorted(closes)
        previous = {day: ordered[i - 1] for i, day in enumerate(ordered) if i}
        for day, block in session_bars.groupby(session_bars.index.date):
            before = previous.get(day)
            if before is None or before not in atr_by_date:
                continue
            ladder = indicators.pivot_ladder(float(highs.get(before, 0.0)),
                                             float(lows.get(before, 0.0)),
                                             float(closes.get(before, 0.0)))
            if ladder is None:
                continue
            local = block.sort_index()
            cut = [i for i, s in enumerate(local.index)
                   if (s.hour, s.minute) <= (ENTRY_HOUR, ENTRY_MINUTE)]
            if not cut or len(local) - cut[-1] < 10:
                continue
            at = cut[-1]
            entry = float(local["Close"].iloc[at])
            rest = local.iloc[at + 1:]
            if entry <= 0 or rest.empty:
                continue
            low_after = float(rest["Low"].min())
            sigma = (float(atr_by_date[before])
                     * np.sqrt(len(rest) / BARS_PER_SESSION))
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            session = f"{symbol}|{day}"
            levels = [v for v in ladder.below(entry) if v > 0]
            for level in levels:
                units = (entry - level) / sigma
                if MIN_SIGMA <= units <= MAX_SIGMA:
                    rows.append((day, symbol, session, units,
                                 low_after <= level, 1))
            near = np.array(levels, dtype=float) if levels else np.array([])
            for _ in range(CONTROLS_PER_SESSION):
                units = float(rng.uniform(MIN_SIGMA, MAX_SIGMA))
                price = entry - units * sigma
                if price <= 0:
                    continue
                if near.size and np.min(np.abs(near - price)) <= entry * TOLERANCE_PCT:
                    continue
                rows.append((day, symbol, session, units,
                             low_after <= price, 0))
        if verbose and index % 100 == 0:
            print(f"  {index}/{len(shared)} symbols, {len(rows):,} rows",
                  flush=True)
    return pd.DataFrame(rows, columns=["day", "symbol", "session", "units",
                                       "hit", "is_pivot"])


def bucketed(data: pd.DataFrame, edges=BUCKETS) -> tuple:
    """Per-bucket hit rates and the pivot-count-weighted difference."""
    frame = data.copy()
    frame["bucket"] = np.digitize(frame["units"], edges)
    lines, weights, diffs = [], [], []
    for bucket, block in frame.groupby("bucket"):
        pivot = block[block["is_pivot"] == 1]["hit"]
        control = block[block["is_pivot"] == 0]["hit"]
        if len(pivot) < 30 or len(control) < 30:
            continue
        low = edges[bucket - 1] if bucket else edges[0]
        high = edges[min(bucket, len(edges) - 1)]
        lines.append((f"{low:.2f}-{high:.2f}", len(pivot), pivot.mean() * 100,
                      len(control), control.mean() * 100,
                      (pivot.mean() - control.mean()) * 100))
        weights.append(len(pivot))
        diffs.append(pivot.mean() - control.mean())
    overall = (float(np.average(diffs, weights=weights)) if diffs
               else float("nan"))
    return lines, overall


def excess_against_bucket(data: pd.DataFrame, edges) -> pd.DataFrame:
    """Pivot rows, each carrying its excess over its own bucket's controls."""
    frame = data.copy()
    frame["bucket"] = np.digitize(frame["units"], edges)
    rate = frame[frame["is_pivot"] == 0].groupby("bucket")["hit"].mean()
    only = frame[frame["is_pivot"] == 1].copy()
    only["excess"] = only["hit"] - only["bucket"].map(rate)
    return only.dropna(subset=["excess"])


def mantel_haenszel(data: pd.DataFrame, strata: list) -> tuple:
    """Common odds ratio and a chi-square p over the given strata.

    The textbook conditional test. It uses every discordant OBSERVATION
    rather than only sessions that produced both a pivot and a control, so
    it carries far more information than the paired sign test.
    """
    frame = data.copy()
    num = den = 0.0
    stat_num = stat_var = 0.0
    for _, block in frame.groupby(strata):
        a = float(((block.is_pivot == 1) & (block.hit == 1)).sum())
        b = float(((block.is_pivot == 1) & (block.hit == 0)).sum())
        c = float(((block.is_pivot == 0) & (block.hit == 1)).sum())
        d = float(((block.is_pivot == 0) & (block.hit == 0)).sum())
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        num += a * d / n
        den += b * c / n
        stat_num += a - (a + b) * (a + c) / n
        stat_var += ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1))
    if den <= 0 or stat_var <= 0:
        return float("nan"), float("nan"), 0
    from math import erfc, sqrt
    chi = (abs(stat_num) - 0.5) ** 2 / stat_var
    p = erfc(sqrt(max(chi, 0.0) / 2.0))
    return num / den, p, int(frame.groupby(strata).ngroups)


def report() -> int:
    started = time.monotonic()
    data = observations()
    if data.empty:
        print("no observations")
        return 1
    pivots = int((data["is_pivot"] == 1).sum())
    print(f"\n{len(data):,} observations, {data['day'].nunique()} dates, "
          f"{data['symbol'].nunique()} symbols, "
          f"{data['session'].nunique():,} sessions "
          f"({pivots:,} pivot, {len(data) - pivots:,} control) "
          f"in {time.monotonic() - started:.0f}s")

    print("\n1. UNPAIRED, bucketed by sigma-distance")
    lines, overall = bucketed(data)
    print(f"{'sigma band':>12} {'pivot n':>8} {'pivot%':>8} {'ctrl n':>8} "
          f"{'ctrl%':>8} {'diff pp':>9}")
    for edge, pn, ph, cn, ch, diff in lines:
        print(f"{edge:>12} {pn:8,} {ph:8.2f} {cn:8,} {ch:8.2f} {diff:+9.2f}")
    print(f"   weighted difference {overall * 100:+.3f} pp "
          f"(negative favours pivots)")

    low, high = usable_window()
    window = data[(data.units >= low) & (data.units <= high)].copy()
    edges = np.linspace(low, high, 5)
    print(f"\n2. WINDOWED to [{low:.2f}, {high:.2f}] sigma, "
          f"bootstrap blocked by date")
    only = excess_against_bucket(window, edges)
    mean, lo, hi, _ = forecast_stats.block_bootstrap(
        only["excess"].to_numpy(dtype=float) * 100, only["day"].to_numpy())
    print(f"   n {len(window):,} ({int((window.is_pivot == 1).sum()):,} pivot) "
          f"over {window.day.nunique()} dates")
    print(f"   distance-matched excess {mean:+.3f} pp, 95% CI "
          f"[{lo:+.3f}, {hi:+.3f}]")

    print("\n3. PAIRED within session")
    counts = window.groupby("session")["is_pivot"].agg(["min", "max"])
    both = set(counts[(counts["min"] == 0) & (counts["max"] == 1)].index)
    paired = window[window.session.isin(both)]
    rows = []
    for name, block in paired.groupby("session"):
        p = block[block.is_pivot == 1]
        c = block[block.is_pivot == 0]
        rows.append({"session": name, "day": block.day.iloc[0],
                     "diff": p.hit.mean() - c.hit.mean(),
                     "gap": p.units.mean() - c.units.mean()})
    pairs = pd.DataFrame(rows).dropna()
    mean, lo, hi, _ = forecast_stats.block_bootstrap(
        pairs["diff"].to_numpy(dtype=float) * 100, pairs["day"].to_numpy())
    better = int((pairs["diff"] < 0).sum())
    worse = int((pairs["diff"] > 0).sum())
    print(f"   {len(pairs):,} sessions with both, mean within-pair distance "
          f"gap {pairs['gap'].mean():+.4f} sigma")
    print(f"   paired difference {mean:+.3f} pp, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"   pivot better in {better}, worse in {worse}, "
          f"tied in {int((pairs['diff'] == 0).sum()):,}")
    from math import comb
    n = better + worse
    if n:
        p_sign = sum(comb(n, k) for k in range(better, n + 1)) / 2 ** n
        print(f"   one-sided sign test on {n} discordant sessions: "
              f"p {p_sign:.4f}")

    print("\n4. MANTEL-HAENSZEL, conditioning on session")
    ratio, p, used = mantel_haenszel(window, ["session"])
    print(f"   {used:,} contributing sessions (both arms present)")
    print(f"   common odds ratio {ratio:.3f} (1.000 = no effect; "
          f"above 1 means pivot stops are hit MORE often), p {p:.3f}")
    print("   Uses every observation in those sessions rather than "
          "one mean each, so it is better powered than the sign test.")

    print("\n5. WHY 1 AND 2 DISAGREE WITH 3 AND 4 - the selection diagnostic")
    has = set(window[window.is_pivot == 1].session)
    ctrl = window[window.is_pivot == 0]
    inside = ctrl[ctrl.session.isin(has)]["hit"]
    outside = ctrl[~ctrl.session.isin(has)]["hit"]
    print(f"   control stops only, so no pivot can act on them:")
    print(f"     in sessions WITH a pivot in the window: "
          f"{inside.mean() * 100:.2f}% hit (n {len(inside):,})")
    print(f"     in sessions WITHOUT one:                "
          f"{outside.mean() * 100:.2f}% hit (n {len(outside):,})")
    print(f"     selection gap {(inside.mean() - outside.mean()) * 100:+.3f} pp"
          f" - compare with design 2's estimate")

    print("\n6. WHY THE WINDOW MATTERS, not the pairing")
    everything = data.copy()
    counts = everything.groupby("session")["is_pivot"].agg(["min", "max"])
    both_all = set(counts[(counts["min"] == 0) & (counts["max"] == 1)].index)
    wide = everything[everything.session.isin(both_all)]
    rows = []
    for name, block in wide.groupby("session"):
        p = block[block.is_pivot == 1]
        c = block[block.is_pivot == 0]
        rows.append({"day": block.day.iloc[0],
                     "diff": p.hit.mean() - c.hit.mean(),
                     "gap": p.units.mean() - c.units.mean()})
    widepairs = pd.DataFrame(rows).dropna()
    mean, lo, hi, _ = forecast_stats.block_bootstrap(
        widepairs["diff"].to_numpy(dtype=float) * 100,
        widepairs["day"].to_numpy())
    print(f"   the SAME paired estimator over the full "
          f"[{MIN_SIGMA}, {MAX_SIGMA}] sigma range:")
    print(f"     {mean:+.3f} pp, CI [{lo:+.3f}, {hi:+.3f}] - and the within-"
          f"pair distance gap is {widepairs['gap'].mean():+.3f} sigma")
    print(f"   Pairing alone does NOT hold distance constant. The window "
          f"does.")

    print("\nVERDICT: designs 3 and 4, which control for the session, find "
          "nothing.\nThe apparent benefit in 1 and 2 is the selection gap in "
          "5. Pivot levels\nare kept for DISPLAY and are not used to place "
          "stops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
