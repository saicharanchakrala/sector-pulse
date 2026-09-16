"""The handful of daily bars the intraday scan needs, published small.

WHY THIS EXISTS. setups.measure takes a `daily` frame and reads exactly
four things from it - the previous close, the previous high and low for
the central pivot range, and a twenty-session mean volume for turnover.
Everything else in the 559 MB consolidated daily store is unused by the
intraday path.

A container running the scan has no such store and should never have one:
it would be stale the moment the nightly fold ran, and baking 559 MB into
an image that is pulled on every scale-up is absurd for four numbers a
symbol. So this publishes the SLICE instead - the last few sessions per
symbol, about 2 MB for the whole universe.

WHY A SLICE AND NOT THE FOUR NUMBERS. It would be smaller still to compute
prev_close and the rest here and ship those. It would also move the
calculation - and with it the lookahead guard that decides which bars
count as "before today". measure() filters `prior.index.date < cutoff_day`
against the CLOCK, and that filter is the reason a partial daily bar
cannot become prev_close. Precomputing here would either duplicate that
logic or freeze its answer at publication time, and an artifact published
at 20:00 saying "yesterday's close" is wrong for a scan running the next
afternoon.

Shipping the bars keeps measure() byte-for-byte unchanged and keeps the
guard where the clock is.
"""
from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import config
import object_store

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
OBJECT = "daily_context.parquet"
LOCAL = config.PROJECT_ROOT / OBJECT

# measure() needs twenty sessions of volume from bars strictly BEFORE the
# scan's own date, so twenty-one is the arithmetic minimum. The slack
# covers holidays, a symbol that missed a session, and a scan running a
# day or two after the last publish.
TAIL_SESSIONS = 25

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def build(frames: "dict | None" = None):
    """The last TAIL_SESSIONS daily bars per symbol, as one long frame."""
    import pandas as pd

    if frames is None:
        import bar_store

        try:
            frames = bar_store.load("day")
        except Exception as exc:
            logger.warning("No daily store, so no context to publish: %s", exc)
            return pd.DataFrame()

    rows = []
    for symbol, frame in (frames or {}).items():
        if frame is None or frame.empty:
            continue
        if not set(COLUMNS) <= set(frame.columns):
            continue
        tail = frame.sort_index().tail(TAIL_SESSIONS)[COLUMNS].copy()
        tail["symbol"] = symbol
        rows.append(tail.reset_index())
    if not rows:
        return pd.DataFrame()
    joined = pd.concat(rows, ignore_index=True)
    stamp = joined.columns[0]
    return joined.rename(columns={stamp: "stamp"})


def publish(frames: "dict | None" = None) -> int:
    """Write the context where the container can read it. Returns rows."""
    table = build(frames)
    if table.empty:
        logger.warning("Nothing to publish")
        return 0
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    if object_store.enabled():
        object_store.put(OBJECT, payload)
    else:
        LOCAL.write_bytes(payload)
    return len(table)


def load(symbols: "list | None" = None) -> dict:
    """Daily context per symbol, in the shape measure() expects.

    `symbols` filters BEFORE the per-symbol frames are built. Without it
    this builds 2,520 DataFrames and a caller scanning the 216 F&O names
    discards 2,304 of them - which is precisely the defect measured in
    live_bars.load_today on 2026-09-16, 22.5 seconds of constructing
    frames nobody asked for. Same shape, same fix, applied before it
    could ship.

    Returns {} rather than raising when there is nothing published, so a
    caller falls back to whatever daily source it had before.
    """
    import pandas as pd

    try:
        if object_store.enabled():
            payload = object_store.get(OBJECT)
        else:
            payload = LOCAL.read_bytes() if LOCAL.exists() else None
    except Exception as exc:
        logger.warning("Could not read %s: %s", OBJECT, exc)
        return {}
    if not payload:
        return {}
    try:
        table = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        logger.warning("Unreadable %s: %s", OBJECT, exc)
        return {}
    if table.empty or "symbol" not in table.columns:
        return {}
    if symbols:
        table = table[table["symbol"].isin(set(symbols))]
        if table.empty:
            return {}

    out = {}
    for symbol, group in table.groupby("symbol", observed=True):
        block = group.drop(columns=["symbol"]).set_index("stamp").sort_index()
        block.index = pd.DatetimeIndex(block.index)
        if block.index.tz is None:
            block.index = block.index.tz_localize(IST)
        out[str(symbol)] = block[COLUMNS]
    return out


def age() -> "tuple | None":
    """(newest session, days behind) or None, so staleness is visible."""
    frames = load()
    if not frames:
        return None
    newest = max(f.index.max() for f in frames.values() if not f.empty)
    return newest.date(), (datetime.now(IST).date() - newest.date()).days


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = publish()
    if not rows:
        print("Nothing published - is the daily store present?")
        return 1
    where = object_store.describe() if object_store.enabled() else str(LOCAL)
    print(f"published {rows:,} rows to {where}/{OBJECT}")
    stamp = age()
    if stamp:
        print(f"  newest session {stamp[0]} ({stamp[1]} day(s) ago)")
    if "--show" in argv:
        frames = load()
        print(f"  {len(frames):,} symbols, "
              f"median {sorted(len(f) for f in frames.values())[len(frames) // 2]}"
              f" sessions each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
