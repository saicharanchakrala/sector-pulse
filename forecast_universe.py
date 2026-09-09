"""Point-in-time universe dataset: the survivorship fix.

The previous build used today's F&O list as the universe for the whole
decade, so names liquid in 2018 that shrank out since were absent - and
their absence is not random, they are the losers. Measured directly on the
partial pool: names that went on to be F&O-liquid in 2026 beat the index by
+14.95% per year while the rest managed +5.19%, so nearly TEN POINTS a year
of the apparent long-horizon edge was payment for knowing the future.

Here the universe is rebuilt at every date from information available then:
rank every listed name by trailing 20-session rupee turnover, keep the top
N above a turnover floor. A name that falls out in 2020 stops appearing
from 2020. Verified to churn: on the partial pool only 3 of 294 names were
present at all 477 dates, and A2ZINFRA and ALANKIT qualified in 2017 and
were gone by 2026.

MEMORY. This runs over ~2,750 symbols x ~2,650 dates. Holding every feature
as a full matrix at once costs about 2.3 GB and thrashes, so each feature is
computed, immediately narrowed to the sampled dates, stacked into the output
and then released. The benchmark columns are broadcast at stack time rather
than materialised as matrices of one repeated column. High and low are not
loaded at all - nothing here uses them.

RESIDUAL BIAS: Kite's master lists what is listed TODAY, so a fully
delisted company - acquired, gone private, expelled - is absent from the
pool as well. This removes the "dropped out of the F&O list" bias, not the
"vanished entirely" one. Results stay flattered; treat any measured edge as
an upper bound.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "bar_cache"
OUT_DIR = ROOT / "forecast_cache"
OUT_DIR.mkdir(exist_ok=True)


SP = OUT_DIR
BENCHMARK = "NIFTY_50"
OUT = OUT_DIR / "pit_dataset.parquet"

HORIZONS = {"short": 10, "mid": 63, "long": 252}
UNIVERSE_SIZE = 250
SAMPLE_EVERY = 5
MIN_TURNOVER = 2.0e7          # Rs 2 crore/day, so the tail stays tradeable
MIN_BARS = 300
DELIVERY_COST = 0.0023


def load_wide() -> tuple:
    """Wide date-by-symbol close and volume. High/low are not needed."""
    closes, volumes = {}, {}
    paths = sorted(CACHE.glob("kite__*__day__2016*.parquet"))
    print(f"deep daily files: {len(paths):,}", flush=True)
    skipped = 0
    for path in paths:
        symbol = path.name.split("__")[1]
        try:
            frame = pd.read_parquet(path, columns=["Close", "Volume"])
        except Exception:
            skipped += 1
            continue
        if frame is None or len(frame) < MIN_BARS:
            skipped += 1
            continue
        frame = frame.sort_index()
        frame.index = frame.index.normalize()
        frame = frame[~frame.index.duplicated(keep="last")]
        closes[symbol] = frame["Close"].astype("float32")
        volumes[symbol] = frame["Volume"].astype("float32")
    print(f"usable symbols: {len(closes):,}  (skipped {skipped:,})", flush=True)
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex_like(close)
    return close, volume


def forward_min_ratio(close: pd.DataFrame, span: int) -> pd.DataFrame:
    """Worst close over the next `span` sessions, as a ratio to today.

    Reversing, rolling, then reversing back gives a FORWARD window through
    the same code path as a trailing one, which is easier to verify than
    index arithmetic.
    """
    rolled = close.iloc[::-1].rolling(span, min_periods=span).min().iloc[::-1]
    return rolled.shift(-1) / close - 1.0


def main() -> int:
    close, volume = load_wide()
    if BENCHMARK not in close.columns:
        print(f"FATAL: {BENCHMARK} missing from the pool")
        return 1
    bench = close[BENCHMARK].astype("float64").copy()
    close = close.drop(columns=[BENCHMARK])
    volume = volume.drop(columns=[BENCHMARK], errors="ignore")
    print(f"matrix: {close.shape[0]:,} dates x {close.shape[1]:,} symbols",
          flush=True)

    # ---- point-in-time universe, from trailing turnover only ----
    turnover = (close * volume).rolling(20, min_periods=15).mean()
    rank = turnover.where(turnover >= MIN_TURNOVER).rank(
        axis=1, ascending=False, method="first")
    in_universe = rank <= UNIVERSE_SIZE
    del rank
    gc.collect()
    per_date = in_universe.sum(axis=1)
    print(f"universe per date: median {int(per_date.median())}, "
          f"min {int(per_date.min())}, max {int(per_date.max())} "
          f"(cap {UNIVERSE_SIZE}, floor Rs {MIN_TURNOVER / 1e7:.0f} cr)",
          flush=True)

    dates = close.index[::SAMPLE_EVERY]
    mom252_full = (close / close.shift(252) - 1.0) * 100.0
    eligible = in_universe.loc[dates] & mom252_full.loc[dates].notna()
    stacked = eligible.stack()
    index = stacked[stacked].index
    print(f"sampled dates {len(dates):,}  eligible cells "
          f"{len(index):,}", flush=True)
    del stacked, eligible, in_universe
    gc.collect()

    out = pd.DataFrame(index=index)
    out.index.names = ["date", "symbol"]

    def add(name: str, frame: pd.DataFrame) -> None:
        """Narrow to sampled dates, stack onto the output, release."""
        out[name] = frame.loc[dates].stack().reindex(index).astype("float32")
        del frame
        gc.collect()

    # ---- trailing features, one at a time ----
    add("mom252", mom252_full)
    del mom252_full
    gc.collect()
    for span in (5, 21, 63, 126):
        add(f"mom{span}", (close / close.shift(span) - 1.0) * 100.0)
    returns = close.pct_change()
    add("vol252", returns.rolling(252, min_periods=200).std()
        * np.sqrt(252) * 100.0)
    add("vol21", returns.rolling(21, min_periods=15).std()
        * np.sqrt(252) * 100.0)
    del returns
    gc.collect()
    add("volume_ratio", volume.rolling(20, min_periods=15).mean()
        / volume.rolling(252, min_periods=200).mean())
    add("turnover_log", np.log10(turnover.clip(lower=1.0)))
    add("price_log", np.log10(close.clip(lower=1e-6)))
    roll_max = close.rolling(252, min_periods=200).max()
    roll_min = close.rolling(252, min_periods=200).min()
    add("pos_52w", (close - roll_min) / (roll_max - roll_min))
    add("drawdown_from_high", (close / roll_max - 1.0) * 100.0)
    del roll_max, roll_min, turnover, volume
    gc.collect()

    out["mom252_ex21"] = out["mom252"] - out["mom21"]
    out["vol_ratio"] = out["vol21"] / out["vol252"]

    # ---- benchmark terms: broadcast a Series, never a matrix ----
    date_level = out.index.get_level_values("date")
    for span in (21, 63, 126, 252):
        bench_mom = (bench / bench.shift(span) - 1.0) * 100.0
        aligned = bench_mom.reindex(date_level).to_numpy()
        out[f"rel_mom{span}"] = (out[f"mom{span}"].to_numpy() - aligned)
        if span in (63, 252):
            out[f"bench_mom{span}"] = aligned

    # ---- labels: excess of the benchmark, net of delivery cost ----
    for name, span in HORIZONS.items():
        raw = (close.shift(-span) / close - 1.0) * 100.0
        add(f"raw_{name}", raw)
        del raw
        gc.collect()
        bench_move = (bench.shift(-span) / bench - 1.0) * 100.0
        aligned = bench_move.reindex(date_level).to_numpy()
        out[f"bench_{name}"] = aligned
        out[f"y_{name}"] = (out[f"raw_{name}"].to_numpy() - aligned
                            - DELIVERY_COST * 100.0)
        add(f"mdd_{name}", forward_min_ratio(close, span) * 100.0)
    add("price", close)
    del close
    gc.collect()

    data = out.reset_index()
    del out
    gc.collect()
    data["date"] = pd.to_datetime(data["date"]).dt.date

    for column in ("mom21", "mom63", "mom252", "mom252_ex21", "rel_mom63",
                   "rel_mom252", "vol252", "vol_ratio", "pos_52w",
                   "volume_ratio", "turnover_log", "drawdown_from_high"):
        data[f"xs_{column}"] = (data.groupby("date")[column]
                                .rank(pct=True).astype("float32"))
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_short"])

    print(f"\nrows {len(data):,} | dates {data.date.nunique():,} | "
          f"symbols {data.symbol.nunique():,}")
    print(f"names per date: median {int(data.groupby('date').size().median())}")
    print()
    print(f"{'horizon':>8} {'rows':>9} {'mean excess':>12} {'median':>9} "
          f"{'>0 share':>9} {'mean worst':>11}")
    for name in HORIZONS:
        sub = data[f"y_{name}"].dropna()
        if sub.empty:
            print(f"{name:>8} {'0':>9}")
            continue
        print(f"{name:>8} {len(sub):>9,} {sub.mean():>12.3f} "
              f"{sub.median():>9.3f} {float((sub > 0).mean()):>9.3f} "
              f"{data[f'mdd_{name}'].dropna().mean():>11.2f}")
    data.to_parquet(OUT)
    print(f"\nsaved -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print()
    print("Compare against the F&O-today run: long-horizon mean excess was")
    print("+14.62% there. Measured survivorship gap was +9.76%/yr, so a")
    print("figure near +5% here would confirm the bias was the bulk of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
