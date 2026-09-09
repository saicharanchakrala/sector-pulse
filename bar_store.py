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


def _span_end(span: str) -> str:
    """The end date out of a cache filename's span, or "" if unparseable.

    Spans look like "20260823_20260909.parquet". Returned as the raw
    eight-digit string because YYYYMMDD sorts correctly as text, so no
    date parsing - and no date-parsing failure mode - is needed.
    """
    stem = span.split(".")[0]
    tail = stem.rsplit("_", 1)[-1]
    return tail if len(tail) == 8 and tail.isdigit() else ""


def _source_files(interval: str) -> list:
    """Every per-symbol parquet for one interval, oldest write first.

    A symbol can have several spans cached from different runs, and ALL of
    them are folded in. Taking only the biggest file was wrong twice over:
    file size is not span length, and neither is freshness. A name cached
    over a year-long window in the spring and refreshed over a ten-day one
    this morning has a larger stale file and a smaller current one, so the
    fold kept the stale one and lost today. Measured on this cache: of the
    460 five-minute symbols holding more than one span, 216 had their
    latest bar in a file that was not the largest - so the store lost the
    current session for 47% of the intraday universe.

    Ordering puts the freshest data last, which is what makes
    `keep="last"` in rebuild() resolve collisions in favour of the newer
    figure. Overlap costs nothing: duplicate (symbol, stamp) rows are
    dropped there.

    The sort key is the span END PARSED FROM THE FILENAME first and mtime
    only as a tie-break, because mtime is the weaker signal - a restore, a
    copy or a sync rewrites it and would silently reorder the fold, while
    the span in `kite__RELIANCE__day__20160101_20260909.parquet` is a fact
    about the contents. A name whose span cannot be parsed falls back to
    mtime alone.
    """
    found: list[tuple] = []
    for path in CACHE.glob(f"kite__*__{interval}__*.parquet"):
        parts = path.name.split("__")
        if len(parts) < 4:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:                     # vanished between glob and stat
            continue
        found.append((parts[1].replace("_", " "), _span_end(parts[3]),
                      modified, path))
    found.sort(key=lambda row: (row[0], row[1], row[2]))
    return [(symbol, path) for symbol, _, _, path in found]


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
    # Stability is what makes `keep="last"` below resolve a duplicate
    # (symbol, stamp) in favour of the file folded latest - the freshest
    # write. A multi-column sort_values IGNORES `kind` and routes through
    # lexsort, which is already stable, so this argument is belt and
    # braces rather than the load-bearing part. It is spelled out so a
    # later move to a single-column sort cannot quietly lose the
    # guarantee.
    joined = joined.sort_values(["symbol", "stamp"], kind="stable")
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
        # STRICTLY before midnight of the following day, which is every bar
        # on `end` and nothing after it. `<=` on that same instant admitted
        # the next day's 00:00 daily bar: asked for end=2026-03-31 it
        # returned 2026-04-01. In a backtest that is tomorrow's close.
        filters.append(("stamp", "<", _as_ist(end) + pd.Timedelta(days=1)))
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
            try:
                stamps = pd.DatetimeIndex(frame["stamp"])
                # A file written before the stamps were localised holds
                # naive values, and comparing those with an IST bound
                # raises - out of the very retry that exists to survive a
                # date failure. Aligned here instead.
                if stamps.tz is None:
                    stamps = stamps.tz_localize(IST)
                else:
                    stamps = stamps.tz_convert(IST)
                if start is not None:
                    keep = stamps >= _as_ist(start)
                    frame, stamps = frame[keep], stamps[keep]
                if end is not None:
                    keep = stamps < _as_ist(end) + pd.Timedelta(days=1)
                    frame = frame[keep]
            except Exception as exc:
                # Returning every row for the requested symbols would hand
                # a backtest data from beyond its own cutoff, so this fails
                # loudly rather than over-serving.
                logger.warning("Could not apply date bounds to %s (%s); "
                               "refusing to return unbounded rows",
                               path.name, exc)
                return {}
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
