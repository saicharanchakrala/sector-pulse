"""Intraday forecasting dataset: features at time t, outcomes that followed,
across a GRID of stop/target geometries.

Why a grid. With the shipped geometry (stop 0.5 sigma, target 1.0 sigma) the
target is reached 13.6% of the time and 44% of positions never resolve at
all. That base rate is set by the geometry, not by any prediction, so
hand-picking one pair would bake in an arbitrary choice and then invite me
to tune it until it looked good. Instead every row carries its outcome under
twelve geometries; the geometry is chosen on TRAIN only and the count of
twelve enters the multiple-testing correction.

Why it is cheap. The running maximum of the forward highs is monotone
non-decreasing and the running minimum of the forward lows is monotone
non-increasing, so "first bar that touches level L" is a binary search on
those two arrays. One O(n) pass per row, then O(log n) per geometry -
exactly equivalent to scanning the path for each geometry separately, and
about twenty times faster.

Everything else follows the measured lessons: true range within a session
so overnight gaps never widen the ATR; short prior sessions dropped from
the RVOL baseline rather than padded; a bar that spans both stop and target
counted as a stop, since 5-minute bars carry no intra-bar ordering;
unresolved positions marked out at the close rather than discarded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "bar_cache"
OUT_DIR = ROOT / "forecast_cache"
OUT_DIR.mkdir(exist_ok=True)


SP = OUT_DIR
SPAN = "20250908_20260908"
BENCHMARK = "NIFTY 50"
OUT = OUT_DIR / "ml_dataset.parquet"

ORB_BARS = 3
SAMPLE_EVERY = 4
MIN_BARS_LEFT = 9
ATR_PERIOD = 14
WARMUP = 20
COST_FRACTION = 0.000824

# Pre-registered. Twelve, and all twelve are counted later.
STOP_FRACTIONS = (0.25, 0.5, 0.75)
REWARD_RISKS = (0.75, 1.0, 1.5, 2.0)
GEOMETRIES = [(s, r) for s in STOP_FRACTIONS for r in REWARD_RISKS]


def tag(stop_fraction: float, reward: float) -> str:
    """Stable column suffix for one geometry."""
    return f"s{int(stop_fraction * 100)}r{int(reward * 100)}"


def sessions_from(frame: pd.DataFrame) -> tuple:
    """Split one symbol's frame into per-session numpy arrays."""
    if frame is None or frame.empty:
        return [], {}
    if any(c not in frame.columns for c in ("Open", "High", "Low", "Close", "Volume")):
        return [], {}
    frame = frame.sort_index()
    days = frame.index.to_series().dt.date.to_numpy()
    out, order = {}, []
    for day in pd.unique(days):
        block = frame.loc[days == day]
        if len(block) < ORB_BARS + MIN_BARS_LEFT + 2:
            continue
        out[day] = {
            "open": block["Open"].to_numpy(dtype=float),
            "high": block["High"].to_numpy(dtype=float),
            "low": block["Low"].to_numpy(dtype=float),
            "close": block["Close"].to_numpy(dtype=float),
            "volume": block["Volume"].to_numpy(dtype=float),
        }
        order.append(day)
    return order, out


def wilder_atr(order: list, sessions: dict, upto: int) -> float:
    """Mean true range per bar over prior sessions, gaps excluded."""
    ranges = []
    for day in order[max(0, upto - ATR_PERIOD - 1):upto]:
        bars = sessions[day]
        high, low, close = bars["high"], bars["low"], bars["close"]
        if len(close) < 2:
            continue
        prev = close[:-1]
        ranges.append(np.maximum(high[1:] - low[1:],
                                 np.maximum(np.abs(high[1:] - prev),
                                            np.abs(low[1:] - prev))))
    if not ranges:
        return float("nan")
    joined = np.concatenate(ranges)
    if joined.size < ATR_PERIOD:
        return float("nan")
    return float(np.mean(joined[-ATR_PERIOD * 5:]))


def volume_curve(order: list, sessions: dict, upto: int) -> np.ndarray:
    """Median cumulative-volume curve over the last 20 prior sessions."""
    curves, length = [], 0
    for day in order[max(0, upto - 20):upto]:
        cum = np.cumsum(sessions[day]["volume"])
        curves.append(cum)
        length = max(length, len(cum))
    if not curves:
        return np.array([])
    usable = [c for c in curves if len(c) >= length * 0.8]
    if not usable:
        return np.array([])
    padded = np.full((len(usable), length), np.nan)
    for row, cum in enumerate(usable):
        padded[row, :len(cum)] = cum
    return np.nanmedian(padded, axis=0)


def resolve(cmax: np.ndarray, neg_cmin: np.ndarray, close_price: float,
            long: bool, entry: float, stop_distance: float,
            target_distance: float) -> tuple:
    """(label, gross R) via binary search on the monotone running extremes.

    cmax is the running max of forward highs (non-decreasing) and neg_cmin
    is the negated running min of forward lows (also non-decreasing), so
    both admit searchsorted. A tie resolves to the stop.
    """
    n = cmax.size
    if stop_distance <= 0 or target_distance <= 0 or n == 0:
        return 0, 0.0
    if long:
        hit_target = np.searchsorted(cmax, entry + target_distance, "left")
        hit_stop = np.searchsorted(neg_cmin, -(entry - stop_distance), "left")
    else:
        hit_target = np.searchsorted(neg_cmin, -(entry - target_distance), "left")
        hit_stop = np.searchsorted(cmax, entry + stop_distance, "left")
    reward = target_distance / stop_distance
    if hit_target >= n and hit_stop >= n:
        move = (close_price - entry) if long else (entry - close_price)
        return 0, move / stop_distance
    if hit_stop >= n:
        return 1, reward
    if hit_target >= n:
        return 0, -1.0
    if hit_stop <= hit_target:
        return 0, -1.0
    return 1, reward


def session_rows(symbol: str, day, bars: dict, atr: float,
                 curve: np.ndarray, prev_close: float, prev_high: float,
                 prev_low: float, bench_at: np.ndarray) -> list:
    """Feature rows plus every geometry's outcome, for one session."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    open_, volume = bars["open"], bars["volume"]
    n = len(close)
    if n < ORB_BARS + MIN_BARS_LEFT + 1 or not np.isfinite(atr) or atr <= 0:
        return []
    typical = (high + low + close) / 3.0
    cum_vol = np.cumsum(volume)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.cumsum(typical * volume) / np.maximum(cum_vol, 1e-9)
    orb_high = float(np.max(high[:ORB_BARS]))
    orb_low = float(np.min(low[:ORB_BARS]))
    orb_width = max(orb_high - orb_low, 1e-9)
    cpr_pivot = (prev_high + prev_low + prev_close) / 3.0
    close_price = float(close[-1])

    rows = []
    for i in range(ORB_BARS, n - MIN_BARS_LEFT):
        if (i - ORB_BARS) % SAMPLE_EVERY:
            continue
        price = float(close[i])
        if not np.isfinite(price) or price <= 0:
            continue
        bars_left = n - (i + 1)
        sigma = atr * float(np.sqrt(max(1, bars_left)))
        if sigma <= 0 or not np.isfinite(sigma):
            continue

        cmax = np.maximum.accumulate(high[i + 1:])
        neg_cmin = np.maximum.accumulate(-low[i + 1:])

        day_high = float(np.max(high[:i + 1]))
        day_low = float(np.min(low[:i + 1]))
        day_range = max(day_high - day_low, 1e-9)
        vw = float(vwap[i]) if np.isfinite(vwap[i]) else price
        rvol = np.nan
        if curve.size > i and np.isfinite(curve[i]) and curve[i] > 0:
            rvol = float(cum_vol[i] / curve[i])
        window = slice(max(0, i - 6), i + 1)
        base6 = float(close[max(0, i - 6)])
        base12 = float(close[max(0, i - 12)])
        day_change = (price / prev_close - 1.0) * 100.0 if prev_close > 0 else 0.0
        bench = float(bench_at[min(i, bench_at.size - 1)])

        row = {
            "symbol": symbol, "day": day, "bar": i,
            "bars_left": bars_left, "price": price, "sigma": sigma,
            "vwap_dist_atr": (price - vw) / atr,
            "orb_pos": (price - orb_low) / orb_width,
            "orb_break_up": float(price > orb_high),
            "orb_break_down": float(price < orb_low),
            "range_pos": (price - day_low) / day_range,
            "day_range_atr": day_range / atr,
            "rvol": rvol,
            "mom6_atr": (price - base6) / atr,
            "mom12_atr": (price - base12) / atr,
            "day_change_pct": day_change,
            "gap_pct": ((float(open_[0]) / prev_close - 1.0) * 100.0
                        if prev_close > 0 else 0.0),
            "bench_change": bench,
            "rel_strength": day_change - bench,
            "bar_range_atr": float(high[i] - low[i]) / atr,
            "up_bars_7": int(np.sum(close[window] > open_[window])),
            "vol_burst": (float(np.mean(volume[window]))
                          / max(float(np.mean(volume[:i + 1])), 1e-9)),
            "prev_close_dist_atr": (price - prev_close) / atr,
            "cpr_dist_atr": (price - cpr_pivot) / atr,
            "frac_session_done": (i + 1) / n,
            "atr_pct": atr / price * 100.0,
        }
        for stop_fraction, reward in GEOMETRIES:
            risk = sigma * stop_fraction
            suffix = tag(stop_fraction, reward)
            long_label, long_r = resolve(cmax, neg_cmin, close_price, True,
                                         price, risk, risk * reward)
            short_label, short_r = resolve(cmax, neg_cmin, close_price, False,
                                           price, risk, risk * reward)
            cost_r = (COST_FRACTION * price) / risk
            row[f"L_{suffix}"] = long_label
            row[f"Lr_{suffix}"] = long_r
            row[f"S_{suffix}"] = short_label
            row[f"Sr_{suffix}"] = short_r
            row[f"cost_{suffix}"] = cost_r
        rows.append(row)
    return rows


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    paths = sorted(CACHE.glob(f"kite__*__5minute__{SPAN}.parquet"))
    bench_paths = [p for p in paths if BENCHMARK.replace(" ", "_") in p.name]
    if limit:
        paths = bench_paths + [p for p in paths if p not in bench_paths][:limit]
        print(f"SMOKE: {len(paths)} files")
    print(f"files: {len(paths)}")

    frames = {}
    for path in paths:
        symbol = path.name.split("__")[1].replace("_", " ")
        try:
            frames[symbol] = pd.read_parquet(path)
        except Exception as exc:
            print(f"  unreadable {path.name}: {exc}")

    bench = frames.pop(BENCHMARK, None)
    if bench is None:
        print(f"FATAL: {BENCHMARK} missing - relative strength impossible")
        return 1
    bench_order, bench_sessions = sessions_from(bench)
    bench_at = {}
    for day in bench_order:
        close = bench_sessions[day]["close"]
        first = float(close[0])
        if first > 0:
            bench_at[day] = (close / first - 1.0) * 100.0
    print(f"benchmark sessions: {len(bench_at)}")
    if len(bench_at) < 100:
        print("FATAL: benchmark too sparse")
        return 1

    chunks = []
    total = 0
    for done, (symbol, frame) in enumerate(sorted(frames.items()), start=1):
        order, sessions = sessions_from(frame)
        if len(order) <= WARMUP:
            continue
        rows = []
        for position, day in enumerate(order):
            if position < WARMUP or day not in bench_at:
                continue
            atr = wilder_atr(order, sessions, position)
            if not np.isfinite(atr) or atr <= 0:
                continue
            prior = sessions[order[position - 1]]
            rows.extend(session_rows(
                symbol, day, sessions[day], atr,
                volume_curve(order, sessions, position),
                float(prior["close"][-1]), float(np.max(prior["high"])),
                float(np.min(prior["low"])), bench_at[day]))
        if rows:
            chunks.append(pd.DataFrame(rows))
            total += len(rows)
        if done % 25 == 0:
            print(f"  {done}/{len(frames)} symbols, {total:,} rows", flush=True)

    if not chunks:
        print("NO ROWS")
        return 1
    data = pd.concat(chunks, ignore_index=True)
    del chunks

    for column in ("rvol", "vwap_dist_atr", "mom6_atr", "day_change_pct",
                   "range_pos", "vol_burst", "rel_strength"):
        data[f"xs_{column}"] = (data.groupby(["day", "bar"])[column]
                                .rank(pct=True).astype("float32"))
    data = data.replace([np.inf, -np.inf], np.nan)

    print(f"\nrows {len(data):,} | days {data.day.nunique()} | "
          f"symbols {data.symbol.nunique()}")
    print(f"{'geometry':>12} {'long hit':>9} {'short hit':>10} "
          f"{'unresolved':>11} {'cost R':>8} {'mean gross':>11}")
    for stop_fraction, reward in GEOMETRIES:
        suffix = tag(stop_fraction, reward)
        unres = float((data[f"Lr_{suffix}"].abs() < 1e-9).mean())
        unres = float(((data[f"Lr_{suffix}"] != -1.0)
                       & (data[f"L_{suffix}"] == 0)).mean())
        print(f"{'s' + str(stop_fraction) + ' r' + str(reward):>12} "
              f"{data[f'L_{suffix}'].mean():>9.4f} "
              f"{data[f'S_{suffix}'].mean():>10.4f} "
              f"{unres:>11.4f} {data[f'cost_{suffix}'].median():>8.4f} "
              f"{data[f'Lr_{suffix}'].mean():>11.4f}")
    target = OUT if not limit else OUT.with_suffix(".smoke.parquet")
    data.to_parquet(target)
    print(f"\nsaved -> {target}  ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
