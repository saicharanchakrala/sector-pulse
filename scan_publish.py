"""Run the intraday scan where the bars already are, and publish the table.

WHY IN THE FEED. Measured 2026-09-16, a scan of the 216 F&O underlyings on
the laptop cost 48 seconds, and only 5.8 of those were the scan itself:

    5.6s   live_bars.status()        reading a 5 MB object to count rows
    7.6s   token lookup
   29.4s   live_bars.combined()      re-reading bars over the network
    5.8s   measure + evaluate + rank

Every one of those first three is the UI fetching data the FEED already
holds in memory. The feed built today's bars tick by tick, prewarmed the
prior sessions at startup, and keeps both. Running the scan there costs
the 5.8 seconds and nothing else, and the UI reads a table instead of
assembling one.

WHAT THIS IS NOT. It is not a second copy of the scan. measure, evaluate
and rank are imported from setups and used exactly as the UI uses them -
the same gates, the same scoring, the same levels. Only the data path
changes. A second implementation that agreed today and drifted in a month
would be worse than the slow version it replaced.

STALENESS IS THE PRICE. The UI stops computing and starts reading, so what
it shows was true when the scan ran rather than when the page loaded. The
published table therefore carries its own timestamp, and a reader is
expected to refuse it past an age - the same contract live bars already
have through SCAN_LIVE_MAX_AGE_SECONDS.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import object_store

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# The published table, one per session, overwritten in place. Named by day
# so a reader cannot pick up yesterday's believing it is today's.
OBJECT_PREFIX = "setups_"

# Seconds between scans. Measured 2026-09-16 on the 216 F&O underlyings, a
# FULL loop iteration - splitting the stream frame, splicing, measuring,
# evaluating, ranking and publishing - took 14.0 seconds. The scan alone
# is 6.1 of that; the rest is turning one flat frame of every streamed
# symbol into per-symbol frames.
#
# Thirty therefore runs at roughly a 47% duty cycle, which leaves the CPU
# for the tick stream - this process's actual job. Do not lower it without
# measuring: a scan that overruns its own interval simply runs
# continuously, and the socket is what suffers.
DEFAULT_EVERY = 30


def object_name(when: "datetime | None" = None) -> str:
    day = (when or datetime.now(IST)).date()
    return f"{OBJECT_PREFIX}{day:%Y%m%d}.parquet"


def assemble(symbols: list, history: dict, today: dict,
             seed: "dict | None" = None) -> dict:
    """Prior sessions, the startup seed and today's bars, spliced per symbol.

    The same three sources and the same precedence as live_bars.combined -
    freshest wins on a duplicate stamp - but from objects already in hand
    rather than from storage.
    """
    import pandas as pd

    out = {}
    for symbol in symbols:
        parts = [f for f in (history.get(symbol),
                             (seed or {}).get(symbol),
                             today.get(symbol))
                 if f is not None and not f.empty]
        if not parts:
            continue
        joined = pd.concat(parts)
        joined = joined[~joined.index.duplicated(keep="last")].sort_index()
        out[symbol] = joined
    return out


def row_for(setup, now) -> dict:
    """Flatten ONE setup, actionable or not.

    NOT scan_intraday._row, and the difference matters. That one assumes
    levels exist - it reads trade.entry directly - because the scan log
    only ever records trades worth taking. A published table that dropped
    everything else would throw away the two groups the UI actually shows
    beside the winners: names that got a direction and were blocked by a
    gate, and the reasons they were blocked.

    That audit trail is the scanner's subtractive value. On 2026-09-16 a
    pass over 216 F&O underlyings produced 182 rows through _row and
    silently discarded 34, which is precisely what this avoids.
    """
    reading = setup.readings
    trade = setup.levels

    def number(value, places=2):
        # None, NOT "". An empty string in a numeric column makes the whole
        # column an object dtype, and parquet then refuses the table with
        # "Could not convert '' with type str: tried to convert to double".
        # None becomes NaN, which is what a missing number means anyway.
        return None if value is None else round(float(value), places)

    row = {
        "run_date": now.strftime("%Y-%m-%d"),
        "run_time": now.strftime("%H:%M:%S"),
        "symbol": reading.symbol,
        "direction": setup.direction,
        "passed": bool(setup.passed),
        "actionable": bool(setup.actionable),
        "score": number(setup.rank_score, 4),
        "rvol": number(reading.rvol, 3),
        "relative_strength": number(reading.relative_strength, 3),
        "oi_change_pct": number(reading.oi_change_pct, 3),
        "futures_share": number(reading.futures_share, 4),
        "vwap": number(reading.vwap),
        "turnover_20d": number(reading.turnover_20d, 0),
        # The audit trail, joined. Every gate records its own reason, and
        # a table that says NO without saying why is not usable.
        "reasons": " | ".join(setup.reasons or []),
    }
    row.update({
        "entry": number(getattr(trade, "entry", None)),
        "stop": number(getattr(trade, "stop", None)),
        "target": number(getattr(trade, "target", None)),
        "quantity": getattr(trade, "quantity", None) if trade else None,
        "stop_pct": number(getattr(trade, "stop_pct", None), 3),
        "target_pct": number(getattr(trade, "target_pct", None), 3),
        "breakeven_pct": number(getattr(trade, "breakeven_pct", None), 4),
        "cost_rupees": number(getattr(trade, "cost_rupees", None)),
        "required_win_rate": number(getattr(trade, "required_win_rate", None), 4),
    })
    return row


def run_once(symbols: list, history: dict, today: dict, daily: dict,
             seed: "dict | None" = None,
             now: "datetime | None" = None,
             benchmark_symbol: str = "NIFTY 50"):
    """Scan and return (table, assembled_count). Table may be empty.

    Imports setups lazily so this module can be read, and its assembly
    tested, without pulling the whole scanner in.
    """
    import pandas as pd

    import setups

    now = now or datetime.now(IST)
    combined = assemble(symbols, history, today, seed)
    if not combined:
        return pd.DataFrame(), 0

    bench = combined.get(benchmark_symbol)
    evaluated = []
    for symbol, frame in combined.items():
        if symbol == benchmark_symbol:
            continue
        try:
            reading = setups.measure(symbol, symbol, frame,
                                     daily.get(symbol), bench, now)
            if reading is not None:
                evaluated.append(setups.evaluate(reading))
        except Exception as exc:
            logger.debug("%s could not be evaluated: %s", symbol, exc)

    ranked = setups.rank(evaluated)
    rows = [row_for(setup, now) for setup in ranked]
    table = pd.DataFrame(rows)
    if not table.empty:
        table["scanned_at"] = now.isoformat(timespec="seconds")
    return table, len(combined)


def publish(table, when: "datetime | None" = None) -> int:
    """Write the table where a reader can pick it up. Returns rows."""
    if table is None or table.empty:
        return 0
    buffer = io.BytesIO()
    try:
        table.to_parquet(buffer, index=False)
        object_store.put(object_name(when), buffer.getvalue())
    except Exception as exc:
        logger.warning("Could not publish the scan: %s", exc)
        return 0
    return len(table)


def load(when: "datetime | None" = None):
    """The published table and its age in seconds, or None.

    Returns None rather than an empty frame when there is nothing, so a
    caller can tell "no scan yet" from "a scan that found nothing" - which
    are different facts and only one of them is a problem.
    """
    import pandas as pd

    try:
        payload = object_store.get(object_name(when))
    except object_store.StorageError:
        raise
    if not payload:
        return None
    try:
        table = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        logger.warning("Unreadable scan table: %s", exc)
        return None
    if table.empty or "scanned_at" not in table.columns:
        return table, float("nan")
    stamp = pd.Timestamp(table["scanned_at"].iloc[0])
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(IST)
    age = (datetime.now(IST) - stamp.to_pydatetime()).total_seconds()
    return table, max(0.0, age)


def loop(builder, symbols: list, history: dict, tokens: dict, daily: dict,
         stop: threading.Event, every: int = DEFAULT_EVERY,
         seed: "dict | None" = None) -> None:
    """Scan on a timer until `stop`. Intended as a daemon thread target.

    NEVER RAISES OUT. This runs beside the tick stream, and a scan that
    fails must not take the feed down with it - the feed's job is the
    socket, and bars still being written is worth more than a table.
    """
    import live_bars

    logger.info("Scan loop every %ds over %d symbols", every, len(symbols))
    while not stop.wait(every):
        started = time.monotonic()
        try:
            frame = builder.snapshot()
            today = live_bars.frames_from(frame, tokens)
            table, assembled = run_once(symbols, history, today, daily,
                                        seed=seed)
            rows = publish(table)
            logger.info("scan: %d setups from %d symbols in %.1fs",
                        rows, assembled, time.monotonic() - started)
        except Exception as exc:
            logger.warning("scan failed (%.1fs): %s",
                           time.monotonic() - started, exc)
