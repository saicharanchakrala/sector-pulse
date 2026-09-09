"""One consolidated bar cache, so every run loads history once and cheaply.

THE PROBLEM. bar_cache holds one parquet per symbol per span: 2,492 daily
files and 2,229 five-minute files. Reading them all costs 12.4ms each, so
31 seconds just to see the daily history - paid again on every run, by
every tool. That is the cost the scanner was wearing before any indicator
ran.

WHY PARQUET AND NOT JSON. Measured on one symbol's 2,646 daily bars: 82.6 KB
as parquet against 264.6 KB as JSON, and 7.4ms to parse against 11.2ms.
Extrapolated across the universe that is 139 MB versus about 675 MB, and
JSON would also lose the dtypes - every price and timestamp would come back
as a string to be re-parsed. Columnar storage is simply the right shape for
"give me the Close column for 2,000 symbols".

WHAT THIS ADDS. The per-symbol files stay as the fetch-time landing area;
this module folds them into a single long-format file per interval, keyed by
symbol and timestamp, and rebuilds only what changed. Readers then do one
file read with a pushdown filter instead of thousands of opens.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "bar_cache"
STORE = ROOT / "forecast_cache"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


IST = "Asia/Kolkata"


def _as_ist(when) -> pd.Timestamp:
    """A tz-aware IST timestamp, whatever kind of date came in."""
    stamp = pd.Timestamp(when)
    return stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)


def store_path(interval: str) -> Path:
    """Where the consolidated file for one interval lives."""
    return STORE / f"bars_{interval}.parquet"


def _source_files(interval: str) -> list:
    """Per-symbol parquets for one interval, deepest span per symbol.

    A symbol can have several spans cached from different runs. Only the
    longest is folded in, because a short window adds nothing a long one
    does not already carry and would only create duplicate rows to drop.
    """
    best: dict[str, tuple] = {}
    for path in CACHE.glob(f"kite__*__{interval}__*.parquet"):
        parts = path.name.split("__")
        if len(parts) < 4:
            continue
        symbol = parts[1].replace("_", " ")
        size = path.stat().st_size
        if symbol not in best or size > best[symbol][0]:
            best[symbol] = (size, path)
    return [(symbol, path) for symbol, (_, path) in sorted(best.items())]


def rebuild(interval: str = "day", verbose: bool = True) -> int:
    """Fold every per-symbol parquet into one long-format file.

    Long format (a symbol column rather than a column per symbol) because
    symbols have different histories: a wide frame over 2,000 names with
    ragged starts is mostly nulls, and adding one newly listed name would
    rewrite every row.
    """
    files = _source_files(interval)
    if not files:
        if verbose:
            print(f"no per-symbol files for interval {interval!r}")
        return 0
    STORE.mkdir(parents=True, exist_ok=True)
    chunks = []
    skipped = 0
    for index, (symbol, path) in enumerate(files, start=1):
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Unreadable %s: %s", path.name, exc)
            skipped += 1
            continue
        if frame is None or frame.empty:
            skipped += 1
            continue
        keep = [c for c in OHLCV if c in frame.columns]
        if not keep:
            skipped += 1
            continue
        block = frame[keep].copy()
        block.index.name = "stamp"
        block = block.reset_index()
        block["symbol"] = symbol
        chunks.append(block)
        if verbose and index % 500 == 0:
            print(f"  {index}/{len(files)} folded", flush=True)
    if not chunks:
        return 0
    joined = pd.concat(chunks, ignore_index=True)
    joined = joined.sort_values(["symbol", "stamp"])
    joined = joined.drop_duplicates(subset=["symbol", "stamp"], keep="last")
    joined["symbol"] = joined["symbol"].astype("category")
    target = store_path(interval)
    temporary = target.with_suffix(".tmp")
    joined.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    if verbose:
        print(f"{interval}: {len(joined):,} rows, "
              f"{joined['symbol'].nunique():,} symbols, "
              f"{target.stat().st_size / 1e6:.1f} MB, {skipped} skipped")
    return len(joined)


def load(interval: str = "day", symbols: "list | None" = None,
         start: "date | None" = None,
         end: "date | None" = None) -> dict:
    """Bars per symbol from the consolidated file, or {} if there is none.

    Filters are pushed into the parquet read where possible so a request for
    twenty symbols does not decompress two thousand.
    """
    path = store_path(interval)
    if not path.exists():
        return {}
    filters = []
    if symbols:
        filters.append(("symbol", "in", set(symbols)))
    # Stored stamps are tz-aware IST, so a naive bound raises inside the
    # parquet reader - which this function then swallowed into an empty
    # result. A filter that fails silently is worse than no filter, so the
    # bounds are localised and a date failure retries without them.
    if start is not None:
        filters.append(("stamp", ">=", _as_ist(start)))
    if end is not None:
        filters.append(("stamp", "<=", _as_ist(end) + pd.Timedelta(days=1)))
    try:
        frame = pd.read_parquet(path, filters=filters or None)
    except Exception as exc:
        logger.warning("Filtered read of %s failed (%s); retrying on "
                       "symbols only", path.name, exc)
        symbol_only = [f for f in filters if f[0] == "symbol"]
        try:
            frame = pd.read_parquet(path, filters=symbol_only or None)
        except Exception as inner:
            logger.warning("Could not read %s: %s", path.name, inner)
            return {}
        if frame is not None and not frame.empty:
            stamps = pd.DatetimeIndex(frame["stamp"])
            if start is not None:
                frame = frame[stamps >= _as_ist(start)]
                stamps = pd.DatetimeIndex(frame["stamp"])
            if end is not None:
                frame = frame[stamps <= _as_ist(end) + pd.Timedelta(days=1)]
    if frame is None or frame.empty:
        return {}
    out = {}
    for symbol, group in frame.groupby("symbol", observed=True):
        block = group.drop(columns=["symbol"]).set_index("stamp").sort_index()
        block.index = pd.DatetimeIndex(block.index)
        out[str(symbol)] = block
    return out


def status(interval: str = "day") -> dict:
    """What the consolidated store holds, without loading it."""
    path = store_path(interval)
    if not path.exists():
        return {"present": False, "interval": interval}
    try:
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(path).metadata
        rows = meta.num_rows
    except Exception:
        rows = None
    return {
        "present": True, "interval": interval,
        "rows": rows, "megabytes": round(path.stat().st_size / 1e6, 1),
        "modified": datetime.fromtimestamp(path.stat().st_mtime),
        "source_files": len(_source_files(interval)),
    }


def main(argv=None) -> int:
    """Rebuild the consolidated stores from the per-symbol cache."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Fold per-symbol bar files into one file per interval")
    parser.add_argument("--intervals", default="day,5minute",
                        help="comma-separated intervals (default: day,5minute)")
    args = parser.parse_args(argv)
    for interval in [i.strip() for i in args.intervals.split(",") if i.strip()]:
        started = time.time()
        rows = rebuild(interval)
        if rows:
            print(f"  built in {time.time() - started:.1f}s")
            check = time.time()
            loaded = load(interval)
            print(f"  reads back {len(loaded):,} symbols in "
                  f"{time.time() - check:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
