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

import config
import object_store

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# The published table, one per session, overwritten in place. Named by day
# so a reader cannot pick up yesterday's believing it is today's.
OBJECT_PREFIX = "setups_"

# Seconds between scans. Measured 2026-09-16 on the 216 F&O underlyings, a
# STEADY-STATE loop iteration is about 5.3 seconds:
#
#     0.80s  frames_from   splitting one flat frame of 2,497 streamed
#                          symbols into the 216 wanted
#     0.15s  assemble      splicing history, seed and today
#     3.73s  evaluate      measure, evaluate, rank
#     0.60s  publish       a 56 KB table to S3
#
# An earlier version of this comment said 14 seconds, measured on the
# FIRST iteration in a fresh process. That is misleading: the first S3
# put costs 7.2s against 0.6s for every one after it, because it pays the
# TLS handshake and credential resolution. A long-running feed pays that
# once at startup and never again.
#
# So thirty seconds runs at roughly an 18% duty cycle, leaving the CPU for
# the tick stream - this process's actual job. Do not lower it without
# measuring on a warm process: a scan that overruns its own interval
# simply runs continuously, and the socket is what suffers.
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

    # THE PRE-TRIGGER STATE, published for every row whether it broke or
    # not. All of it was already computed and then thrown away, because
    # the table only ever described setups that had triggered - so the
    # page could report what had moved and nothing about what had not.
    # getattr throughout, matching how the trade fields above are read: a
    # reading that does not carry one of these publishes a null rather
    # than taking the whole table down.
    orb = getattr(reading, "opening_range", None)
    last = getattr(reading, "last", None)
    prev = getattr(reading, "prev_close", None)
    moved = (None if last is None or not prev else last - prev)
    row.update({
        "last": number(last),
        "prev_close": number(prev),
        "day_change_pct": number(getattr(reading, "day_change_pct", None), 3),
        "moved_today": number(None if moved is None else abs(moved)),
        "or_high": number(getattr(orb, "high", None)),
        "or_low": number(getattr(orb, "low", None)),
        "atr_bar": number(getattr(reading, "atr_bar", None), 4),
        "minutes_left": getattr(reading, "minutes_left", None),
    })

    # Imported here, not at module scope: this module keeps setups lazy so
    # it can be read and its assembly tested without pulling the whole
    # scanner in. Guarded because a reading that predates the pre-trigger
    # fields - or a stand-in in a test - must publish "no approach
    # information" rather than take the whole table down.
    # NARROW, AND LOUD. A blanket `except Exception` at DEBUG meant a real
    # TypeError inside approach() produced all-None columns for every
    # symbol, render_approaching then found an empty frame and returned
    # silently, and the whole feature could be dead in production with no
    # signal anywhere above DEBUG.
    try:
        import setups as setups_mod
    except ImportError as exc:                    # the caller already has it
        logger.warning("setups unavailable, so no approach columns: %s", exc)
        near = None
    else:
        try:
            near = setups_mod.approach(reading)
        except (AttributeError, TypeError) as exc:
            logger.warning("approach() failed for %s, so the approaching "
                           "panel will be empty: %s", reading.symbol, exc)
            near = None
    row.update({
        "approach_side": None if near is None else near.side,
        "approach_trigger": None if near is None else number(near.trigger),
        "approach_distance": None if near is None else number(near.distance),
        "approach_atr": None if near is None else number(near.distance_atr, 3),
        # READY means every gate OTHER than the break already passes. It
        # is not a forecast that the break happens, and most will not.
        "approach_ready": None if near is None else bool(near.ready),
        "approach_blockers": (None if near is None
                              else " | ".join(near.blockers)),
    })
    return row


def benchmark_change_pct(intraday, daily, now) -> "float | None":
    """Today's percent change for the relative-strength benchmark.

    WHY THIS EXISTS RATHER THAN A FRAME. setups.measure takes a FLOAT for
    the benchmark, not a frame - it wants the day change, not the bars.
    An earlier version of run_once handed it `combined.get("NIFTY 50")`
    straight through, so the one code path that could have produced a
    number never did.

    None when either side is missing, which makes every relative-strength
    gate fail closed rather than compare each symbol against zero and read
    a falling market as strength. That is the safe direction, but it is
    not a harmless one: it blocks the whole scan, so the caller logs it.

    Anchored on the CLOCK, like measure's own daily cutoff. Anchoring on
    the last surviving bar instead lets today's completed daily bar become
    the previous close, which leaks the session's outcome into the gate
    that is supposed to judge it.
    """
    import indicators
    import pandas as pd

    # NEVER RAISES. The per-symbol loop in run_once has its own handler;
    # this call sits outside it, so a benchmark frame missing a Close
    # column would take down every scan for the rest of the session at one
    # bare "scan failed" a cycle.
    try:
        if now.tzinfo is None:
            # Exchange-local, matching what indicators documents for a
            # naive clock. astimezone() would read it as SYSTEM local and
            # shift the cutoff day on any machine not set to IST.
            now = now.replace(tzinfo=IST)
        if intraday is None or daily is None or intraday.empty or daily.empty:
            return None
        cutoff_day = now.astimezone(IST).date()

        # PINNED TO THE CLOCK'S DAY, not to the frame's last session.
        # session_bars() defaults to whatever session ends the frame, and
        # the daily cutoff below is anchored on the clock - so a benchmark
        # holding only prior sessions would read its last intraday close
        # against that same day's daily close and return 0.0. Not None:
        # 0.0, which passes every None check, silences the warning in
        # run_once, and makes each symbol's relative strength equal its own
        # day change. In a rising market that marks every rising name as
        # leading the index. This is reachable on every feed start before
        # the index's first 3-minute bar closes.
        session = indicators.session_bars(intraday, day=cutoff_day)
        if session is None or session.empty:
            return None
        closes = session["Close"].dropna()
        if closes.empty:
            return None

        prior = daily.dropna(subset=["Close"])
        if not isinstance(prior.index, pd.DatetimeIndex):
            logger.warning("Benchmark daily index is not timestamped; "
                           "refusing to compare against a partial bar")
            return None
        prior = prior[prior.index.date < cutoff_day]
        if prior.empty:
            return None
        return indicators.percent_change(float(closes.iloc[-1]),
                                         float(prior["Close"].iloc[-1]))
    except Exception as exc:
        logger.warning("Benchmark comparison unusable: %s", exc)
        return None


def run_once(symbols: list, history: dict, today: dict, daily: dict,
             seed: "dict | None" = None,
             now: "datetime | None" = None,
             benchmark_symbol: str = config.SCAN_BENCHMARK):
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

    # A FLOAT, not the frame. Every relative-strength gate fails closed
    # when this is None, so the whole scan reads "nothing cleared" - which
    # is indistinguishable from a quiet market unless it says so here.
    bench = benchmark_change_pct(combined.get(benchmark_symbol),
                                 daily.get(benchmark_symbol), now)
    if bench is None:
        logger.warning("No %s comparison available, so every "
                       "relative-strength gate will fail and nothing can be "
                       "actionable. Is the benchmark in the streamed "
                       "universe and the daily context?", benchmark_symbol)

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


_PARSED_LOCK = threading.Lock()
_PARSED: dict = {}


def load(when: "datetime | None" = None):
    """The published table and its age in seconds, or None.

    Returns None rather than an empty frame when there is nothing, so a
    caller can tell "no scan yet" from "a scan that found nothing" - which
    are different facts and only one of them is a problem.

    SKIPS THE DOWNLOAD WHEN THE OBJECT HAS NOT MOVED. The feed republishes
    roughly every half minute while Streamlit re-executes its whole script
    on every interaction, so most calls here are for bytes this process
    already parsed. A HEAD decides it. The AGE is still recomputed every
    time from the table's own stamp, because the caller refuses a table
    past an age and a cached age would freeze that judgement.
    """
    import pandas as pd

    name = object_name(when)
    table = None
    try:
        tag = object_store.version(name)
    except object_store.StorageError:
        # A failed HEAD is a real fault and must surface, not silently
        # downgrade to a full GET that will fail the same way.
        raise
    if tag is not None:
        with _PARSED_LOCK:
            remembered = _PARSED.get(name)
        if remembered is not None and remembered[0] == tag:
            table = remembered[1]

    if table is None:
        try:
            payload = object_store.get(name)
        except object_store.StorageError:
            raise
        if not payload:
            return None
        try:
            table = pd.read_parquet(io.BytesIO(payload))
        except Exception as exc:
            logger.warning("Unreadable scan table: %s", exc)
            return None
        if tag is not None:
            with _PARSED_LOCK:
                # ONE ENTRY. The key is date-stamped, so keeping every one
                # would grow a long-lived Streamlit process by a full
                # table per trading day and never release any of them.
                # Only the current day is ever read hot.
                _PARSED.clear()
                _PARSED[name] = (tag, table)
    # A SHALLOW COPY PER CALLER. Every Streamlit session in this process
    # would otherwise share one frame for as long as the object does not
    # move, so anything mutating it in place would corrupt the others.
    # Nothing does today - the reader sorts, which copies - but that is a
    # property of today's caller, not a guarantee this function can make.
    return _aged(table.copy(deep=False))


def _aged(table):
    """(table, seconds since it was scanned), or (table, nan) without a stamp."""
    import pandas as pd

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
