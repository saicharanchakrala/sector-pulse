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

import csv
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
    ranked = setups.apply_portfolio_caps(ranked)
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


_LOG_LOCK = threading.Lock()
_SESSION_LOG: dict = {}
_SESSION_LOG_DAY = None

_SEEN_LOCK = threading.Lock()
_FIRST_DIRECTION: dict = {}
_FIRST_CLEARED: dict = {}
_SEEN_DAY = None


def stamp_first_seen(table, when: "datetime | None" = None):
    """Add when each setup first appeared, and when it first cleared.

    WHY THE TABLE CANNOT ANSWER THIS ALONE. Every scan rebuilds the whole
    table from scratch, so `run_time` is when THIS pass ran - not when the
    setup arrived. A row that has been on screen since 09:45 looked
    identical to one that appeared four seconds ago, which is the
    difference between a setup that has held for an hour and one that has
    just printed.

    Two stamps, because they answer different questions:
      first_seen_at - when the symbol first showed this direction
      cleared_at    - when it was first ACTIONABLE, or blank if never

    ACTIONABLE NOW MEANS INSIDE THE COMBINED CAP. run_once passes the
    ranking through setups.apply_portfolio_caps before the rows are built,
    so a setup that passed every gate but was held back by the cap is not
    actionable and does not stamp cleared_at. The stamp therefore reads
    "first passed every gate while inside the combined risk cap", not
    "first passed every gate"; stamps written before the cap existed
    meant the latter.

    First wins for both, and `cleared_at` is never overwritten by a later
    block: the moment it qualified is the moment it would have been acted
    on, and a setup that clears at 10:42 and fails at 11:30 was still
    suggested at 10:42.
    """
    global _SEEN_DAY

    import pandas as pd

    if table is None or getattr(table, "empty", True):
        return table
    now = when or datetime.now(IST)
    stamp = now.strftime("%H:%M:%S")
    with _SEEN_LOCK:
        if _SEEN_DAY != now.date():
            _FIRST_DIRECTION.clear()
            _FIRST_CLEARED.clear()
            _SEEN_DAY = now.date()
        first_seen, cleared = [], []
        for row in table.to_dict("records"):
            key = (row.get("symbol"), row.get("direction"))
            if _missing(row.get("entry")):
                # No levels, so nothing was ever suggested for this row.
                first_seen.append(None)
                cleared.append(None)
                continue
            first_seen.append(_FIRST_DIRECTION.setdefault(key, stamp))
            if row.get("actionable"):
                cleared.append(_FIRST_CLEARED.setdefault(key, stamp))
            else:
                cleared.append(_FIRST_CLEARED.get(key))
        out = table.copy()
        out["first_seen_at"] = pd.Series(first_seen, index=out.index,
                                         dtype="object")
        out["cleared_at"] = pd.Series(cleared, index=out.index,
                                      dtype="object")
        # THE PLANNED EXIT, as a clock time rather than a countdown.
        # minutes_left answers "how long until the market shuts", which is
        # not the same question as "when am I out of this" - and read as
        # the latter it is the wrong number. There is no exit rule before
        # SCAN_EXIT_BY: a setup leaves on its target, its stop, or this
        # time, whichever comes first, so this is the only one of the
        # three that can be known in advance.
        hour, minute = getattr(config, "SCAN_EXIT_BY",
                               config.SCAN_SESSION_CLOSE)
        out["exit_by"] = f"{hour:02d}:{minute:02d}"
        return out


def log_object_name(when: "datetime | None" = None) -> str:
    """The session's scan log in the object store, named by day."""
    day = (when or datetime.now(IST)).date()
    return f"scan_log_{day:%Y%m%d}.csv"


def _log_fields() -> list:
    """The scan log's columns, taken from the one place that defines them.

    Imported rather than restated so a column added to the CLI's log
    cannot silently go unrecorded here - a missing key raises when the row
    is built instead of producing a file outcomes.py reads as malformed.
    """
    import scan_intraday

    return list(scan_intraday._CSV_FIELDS)


def _missing(value) -> bool:
    """Whether a published field carries no number.

    NaN AS WELL AS None, because these rows come out of a DataFrame via
    to_dict("records") and pandas represents a missing number as NaN. NaN
    is TRUTHY and is not None, so `if quantity` and `entry is None` both
    read it as present - which on 2026-09-21 made every scan log attempt
    die with "cannot convert float NaN to integer", and would have logged
    NO SETUP rows had it got past that. The unit tests passed because they
    built rows as plain dicts holding None.
    """
    return value is None or value != value


# --- the scan window -----------------------------------------------------

# What loop() logs while it idles because no symbol other than the
# benchmark has a bar from today's session in hand. A constant so the
# idle state is ONE state, logged once, rather than a fresh message every
# tick.
NO_SESSION_BARS = ("no symbol has a bar stamped today in the stream or the "
                   "seed - a holiday, or a feed that has not yet closed a "
                   "bar")


def window_bounds() -> tuple:
    """The first scannable minute and the session close, as IST times.

    The first minute is the opening range's close, not the bell: before
    SCAN_OPENING_RANGE_MINUTES have elapsed the latest bar is one of the
    bars defining the range, so the range brackets the price by
    construction and no breakout can be represented. app._session_bounds
    offers a replay the same first minute for the same reason.
    """
    from datetime import time as clock

    open_hour, open_minute = config.SCAN_SESSION_OPEN
    first = open_hour * 60 + open_minute + config.SCAN_OPENING_RANGE_MINUTES
    close_hour, close_minute = config.SCAN_SESSION_CLOSE
    return clock(first // 60, first % 60), clock(close_hour, close_minute)


def scan_window(now: datetime) -> "tuple[bool, str]":
    """Whether a scan at `now` can measure today's session, and why not.

    THE BUG THIS CLOSES. loop() scanned on a timer with no clock check,
    and setups.measure falls back to the last session it holds - so a
    scan before the open measured YESTERDAY's bars and row_for stamped
    them with TODAY's run_date. Measured 2026-09-24: 2,350 of the 9,800
    resolved outcomes carry a run_time before 09:30 (06:37, 08:08,
    09:21), and all 2,485 rows of scan_log_20260924.csv are stamped
    00:00-06:48. remember() kept each as the first sighting, adoption
    carried it across restarts, and outcomes.py scored yesterday's levels
    against today's bars.

    OPEN on a weekday from the opening range's close (see window_bounds)
    up to but not including the session close. At the close the session
    is over and nothing a scan finds can be traded.

    REUSES live_bars.in_session for the close and the weekend, so the
    feed's bar gate and this cannot disagree about trading hours. The
    weekend is also checked here only to name it in the reason. Like
    in_session this knows nothing of exchange HOLIDAYS; loop() covers
    those by refusing to scan without bars stamped today.

    A NAIVE CLOCK IS CLOSED, not assumed IST. Read as exchange-local, a
    naive UTC clock would open this window at 15:00 IST and hold it open
    until 21:00 - six hours of scanning a finished session. Closed is the
    direction that cannot publish anything stale.

    `now` is required, never defaulted, so a test freezes time by passing
    it. The reasons are fixed strings per state, so loop() can log a
    change of state once instead of logging every tick.
    """
    import live_bars

    if now.tzinfo is None:
        return False, "the clock carries no timezone"
    local = now.astimezone(IST)
    first, close = window_bounds()
    if local.weekday() >= 5:
        return False, "no session on a weekend"
    if local.time() < first:
        return False, f"before the opening range closes at {first:%H:%M}"
    if not live_bars.in_session(local):
        return False, f"the session closed at {close:%H:%M}"
    return True, f"inside the scan window {first:%H:%M}-{close:%H:%M}"


def scan_instant(row) -> "datetime | None":
    """When a published or logged row was scanned, in IST, or None.

    READ FROM THE ROW'S OWN run_date AND run_time, not from the clock of
    whoever is holding it, because those two columns are what outcomes.py
    resolves from and calibrate judges - a guard on anything else could
    pass a row stamped 06:37. Both writers stamp HH:MM:SS; HH:MM is also
    read, because hand-written rows and older fixtures carry it. None for
    anything else, including the NaN a DataFrame gives a missing cell.
    """
    day = str(row.get("run_date") or "").strip()
    clock = str(row.get("run_time") or "").strip()
    for layout in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{day} {clock}",
                                     layout).replace(tzinfo=IST)
        except ValueError:
            continue
    return None


def in_scan_window(row) -> bool:
    """Whether a row's own scan instant is inside the scan window.

    FAILS CLOSED: a row whose instant cannot be read is outside, because
    it cannot be shown to be inside. Shared by remember(), log_row() and
    calibrate, so the feed and the report agree on what a valid row is.
    """
    when = scan_instant(row)
    return when is not None and scan_window(when)[0]


def _stamped_on(frame, day) -> bool:
    """Whether a per-symbol bar frame holds a bar on `day` (IST)."""
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        return False
    try:
        last = pd.Timestamp(frame.index.max())
    except (TypeError, ValueError):
        return False
    if last is pd.NaT:
        return False
    last = (last.tz_localize(IST) if last.tzinfo is None
            else last.tz_convert(IST))
    return last.date() == day


def session_symbols(symbols: list, today: dict, seed: "dict | None",
                    now: datetime) -> list:
    """The symbols holding at least one bar stamped with `now`'s date.

    THE SECOND HALF OF THE GUARD. scan_window reads the clock and not the
    exchange calendar, so on a trading holiday it is open all day while
    the feed receives nothing - and setups.measure, handed only prior
    sessions, measures the last one it finds. The same failure as the
    pre-open scans, reached through the calendar instead of the clock.
    It also covers a feed process that outlived its session: BarBuilder
    never clears, so its snapshot can hold only yesterday's bars.

    PER SYMBOL, NOT PER SESSION. The fallback is per frame, so one symbol
    with a bar today does not make another symbol's frame current. An
    earlier draft asked "does ANY frame have a bar today" and then scanned
    everything - so after a restart at 11:00, every symbol the seeder had
    not reached yet was measured on the previous session and stamped with
    today's date, and so was any name that simply had not traded yet.

    CHECKED ON THE SPLIT FRAMES, not the raw snapshot. frames_from drops
    partial bars, and a feed that joined mid-bucket has only partial bars
    for its first minutes - rows the raw snapshot holds and the scan never
    sees. The split is computed for the scan anyway, so this costs one
    index lookup per symbol.

    THE SEED COUNTS. A feed restarted at 11:00 closes its first bar at
    11:03, while seed_session has already fetched 09:15 onwards from Kite
    - that is today's session, and scanning it is what the seed is for.
    Each frame is dated by its own last bar rather than trusted for being
    non-empty, so a fetch that returned the prior session cannot pass.
    The seeder thread fills `seed` while this runs; a single get() per key
    is safe against that, where iterating the dict would not be.
    """
    day = now.astimezone(IST).date()
    seed = seed or {}
    return [symbol for symbol in symbols
            if _stamped_on(today.get(symbol), day)
            or _stamped_on(seed.get(symbol), day)]


def log_row(published: dict) -> dict:
    """One published scan row, reshaped into a scan_log row.

    BUILT FROM WHAT WAS PUBLISHED, not from the Setup object, so the
    logged row is by construction the row the UI was shown. The only
    fields not in the published table are derived rather than guessed:
    `risk_rupees` is quantity times the stop distance, which is how
    levels.build_levels computes lot_risk in the first place, and
    `blocked_by` is the first failing gate out of the audit trail.

    RAISES ValueError FOR A ROW SCANNED OUTSIDE THE SCAN WINDOW. There is
    no valid scan_log row to return for it: before the window its levels
    came from the previous session's bars, after it there is no session
    left to trade (see scan_window), and a None or an empty dict would
    travel on to whatever the caller does next. remember() filters first
    and never reaches this; it exists so a caller that skips the filter
    fails loudly instead of logging a stale setup.
    """
    if not in_scan_window(published):
        raise ValueError(
            f"refusing to log {published.get('symbol')!r} scanned at "
            f"{published.get('run_date')} {published.get('run_time')}: "
            f"outside the scan window")
    quantity = published.get("quantity")
    entry = published.get("entry")
    stop = published.get("stop")
    risk = None
    if not (_missing(quantity) or _missing(entry) or _missing(stop)):
        risk = round(abs(float(entry) - float(stop)) * int(quantity), 2)

    blocked = ""
    for reason in str(published.get("reasons") or "").split(" | "):
        if "[FAIL]" in reason:
            blocked = reason.split(" [")[0][:120]
            break

    row = dict(published)
    row.update({
        # The feed only ever scans now. A replay is a local, deliberate act.
        "replayed": False,
        "risk_rupees": risk,
        "taken": bool(published.get("actionable")),
        "blocked_by": blocked,
    })
    return {name: row.get(name, "") for name in _log_fields()}


def adopt_published_log(when: "datetime | None" = None) -> dict:
    """Read back the session's published log, keyed as _SESSION_LOG is.

    WHY. publish_log writes the whole in-memory set, and that memory dies
    with the process - so a restart mid-session republished only what the
    NEW task had seen, over the top of everything recorded before it.
    Observed 2026-09-21: a deploy at 13:21 replaced 3,450 setups spanning
    09:20 to 13:19 with 1,352 stamped 13:26, including every one that had
    cleared. The morning was not recoverable.

    Adopting first makes a restart additive instead of destructive. Rows
    already published win on first-seen by construction - they were
    recorded earlier - and the actionable-beats-blocked rule in remember()
    still applies on top of them.

    Never raises. A log that cannot be read back is a reason to start a
    fresh one, not a reason for the feed to fail.
    """
    rows: dict = {}
    try:
        payload = object_store.get(log_object_name(when))
    except Exception as exc:
        logger.warning("Could not read back the published scan log, so "
                       "this session starts from empty: %s", exc)
        return rows
    if not payload:
        return rows
    try:
        for row in csv.DictReader(io.StringIO(payload.decode("utf-8"))):
            key = (row.get("symbol"), row.get("direction"))
            if not key[0]:
                continue
            # The file stores booleans as text; remember() compares them.
            row["taken"] = str(row.get("taken", "")).lower() == "true"
            rows[key] = row
    except Exception as exc:
        logger.warning("Published scan log is unreadable, starting fresh: "
                       "%s", exc)
        return {}
    if rows:
        logger.info("Adopted %d setups already published today, so this "
                    "restart adds to the session rather than replacing it",
                    len(rows))
    return rows


def remember(table, when: "datetime | None" = None) -> int:
    """Accumulate this scan into the session's log. Returns rows held.

    ONE ROW PER SYMBOL AND DIRECTION PER SESSION, not one per scan. The
    feed scans every 30 seconds, so logging each pass would record the
    same setup four hundred times and every hit rate computed on it would
    be weighted by how long a symbol happened to stay on screen.

    WHICH observation is kept matters. A symbol blocked at 09:20 that
    clears at 11:00 must be recorded as it was when it CLEARED - that is
    the moment it would have been acted on. So an actionable observation
    replaces a blocked one, and among actionable ones the first wins.
    Among blocked ones the first also wins, which keeps the earliest
    evidence of why it never qualified.

    NOTHING SCANNED OUTSIDE THE SCAN WINDOW IS RECORDED, judged on each
    row's own run_date and run_time (see scan_instant), not on `when`: a
    table scanned at 15:20 and logged at 15:31 is valid, one scanned at
    06:37 is not whenever it is logged. loop() no longer scans outside the
    window; this is the second wall, so no caller can put a stale pre-open
    row into the log. The first-sighting rule is exactly what made those
    rows stick - a 06:37 row blocked every later, valid sighting of the
    same setup that day.

    ADOPTED ROWS ARE HELD TO THE SAME RULE. A log published before this
    guard existed can be full of them - all 2,485 rows of
    scan_log_20260924.csv are stamped before 06:49 - and adopting it on a
    restart would republish every one and hand them to outcomes.py again.
    """
    global _SESSION_LOG_DAY

    if table is None or getattr(table, "empty", True):
        return 0
    now = when or datetime.now(IST)
    refused = 0
    with _LOG_LOCK:
        if _SESSION_LOG_DAY != now.date():
            _SESSION_LOG.clear()
            # ADOPT WHAT IS ALREADY PUBLISHED before adding to it. This
            # fires on a genuine new day, where the object does not exist
            # and it costs one missing read - and on a RESTART mid-session,
            # where without it the next publish overwrites the whole
            # morning with whatever this process has seen so far.
            adopted = adopt_published_log(now)
            kept = {key: row for key, row in adopted.items()
                    if in_scan_window(row)}
            if len(kept) < len(adopted):
                logger.warning("Dropped %d adopted scan-log rows stamped "
                               "outside the scan window; a scan there "
                               "measures the wrong session, so they will "
                               "not be republished", len(adopted) - len(kept))
            _SESSION_LOG.update(kept)
            _SESSION_LOG_DAY = now.date()
        for published in table.to_dict("records"):
            # NaN, not just None - see _missing. A NO SETUP row reaches
            # here with entry NaN, which `is None` reads as present.
            if _missing(published.get("entry")):
                continue          # no levels, so nothing to resolve against
            if not in_scan_window(published):
                refused += 1
                continue
            key = (published.get("symbol"), published.get("direction"))
            row = log_row(published)
            held = _SESSION_LOG.get(key)
            if held is None or (row["taken"] and not held["taken"]):
                _SESSION_LOG[key] = row
        held_count = len(_SESSION_LOG)
    if refused:
        logger.warning("Refused %d scan-log rows stamped outside the scan "
                       "window; a scan there measures the wrong session",
                       refused)
    return held_count


def publish_log(when: "datetime | None" = None) -> int:
    """Write the session's accumulated log. Returns rows written.

    The whole day is rewritten each time because S3 has no append, and at
    a few hundred KB that is cheaper than any scheme which would let a
    crash lose the session.
    """
    with _LOG_LOCK:
        rows = list(_SESSION_LOG.values())
    if not rows:
        return 0
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_log_fields(),
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    object_store.put(log_object_name(when),
                     buffer.getvalue().encode("utf-8"))
    return len(rows)


def _iteration(builder, symbols: list, history: dict, tokens: dict,
               daily: dict, seed: "dict | None",
               now: datetime) -> "str | None":
    """One pass of the loop at `now`. Returns why it idled, or None.

    SPLIT OUT OF loop() so the window is testable with a frozen clock:
    loop() reads the wall clock and the stop event and nothing else, and
    everything that decides whether a scan happens is in here with `now`
    passed in. The same `now` stamps the table, the publish and the log,
    so they cannot straddle a midnight or a window edge between them.

    IDLING MEANS NOTHING AT ALL - no scan, no publish, no log. A table
    published from a closed window is exactly what the UI must never read
    as current, and a logged row from one is what outcomes.py scored
    against the wrong session. Two reasons to idle, checked cheapest
    first: the clock (scan_window), which costs no snapshot at all, then
    whether any symbol other than the benchmark has a bar stamped today.

    ONLY THOSE SYMBOLS ARE SCANNED (see session_symbols). A symbol with no
    bar today is left out of the table rather than measured on the last
    session it holds. The benchmark follows the same rule and loses
    nothing by it: benchmark_change_pct is pinned to the clock's day and
    returns None for a benchmark without a bar today either way.

    Raises only if scan_window or the live_bars import does; the scan and
    the log each keep their own handler, as before.
    """
    import live_bars

    is_open, reason = scan_window(now)
    if not is_open:
        return reason
    # EXCHANGE TIME FROM HERE ON. scan_window accepts any aware clock, but
    # row_for stamps run_date and run_time from now's own digits - a UTC
    # 04:00 would publish "04:00:00" and remember() would then refuse every
    # row as pre-open. loop() passes IST today; this keeps that true for
    # any other caller.
    now = now.astimezone(IST)
    started = time.monotonic()
    try:
        frame = builder.snapshot()
        today = live_bars.frames_from(frame, tokens)
        fresh = session_symbols(symbols, today, seed, now)
        if not any(symbol != config.SCAN_BENCHMARK for symbol in fresh):
            return NO_SESSION_BARS
        table, assembled = run_once(fresh, history, today, daily,
                                    seed=seed, now=now)
        # BEFORE the publish, so the table the UI reads carries the
        # stamps. It is also what remember() logs, so the scan log and
        # the published table agree on when a setup arrived.
        table = stamp_first_seen(table, now)
        rows = publish(table, now)
        logger.info("scan: %d setups from %d symbols in %.1fs; %d held "
                    "back with no bar stamped today", rows, assembled,
                    time.monotonic() - started, len(symbols) - len(fresh))
    except Exception as exc:
        logger.warning("scan failed (%.1fs): %s",
                       time.monotonic() - started, exc)
        return None
    # SEPARATE FROM THE SCAN, and after it. The published table is what
    # the UI reads and must not be delayed or lost because the log
    # failed; the log is a record, and a record is worth less than the
    # thing it records. Its own handler for the same reason.
    try:
        held = remember(table, now)
        written = publish_log(now)
        if written:
            logger.info("scan log: %d distinct setups held, %d written",
                        held, written)
    except Exception as exc:
        logger.warning("could not record the scan log: %s", exc)
    return None


def loop(builder, symbols: list, history: dict, tokens: dict, daily: dict,
         stop: threading.Event, every: int = DEFAULT_EVERY,
         seed: "dict | None" = None) -> None:
    """Scan on a timer until `stop`. Intended as a daemon thread target.

    NEVER RAISES OUT. This runs beside the tick stream, and a scan that
    fails must not take the feed down with it - the feed's job is the
    socket, and bars still being written is worth more than a table.

    ONLY INSIDE THE SCAN WINDOW, and only with bars stamped today. It used
    to scan on the timer alone, and a pre-open scan republished the prior
    session's bars under today's date - see scan_window. Outside, each
    tick does nothing (see _iteration).

    THE IDLE STATE IS LOGGED ONCE PER CHANGE, not once per tick. A feed
    started at 06:00 would otherwise print the same line 420 times
    before the window opens, and bury the one line that matters - the
    moment scanning actually starts or stops.
    """
    logger.info("Scan loop every %ds over %d symbols", every, len(symbols))
    idle = None                       # None means "scanning"
    while not stop.wait(every):
        try:
            reason = _iteration(builder, symbols, history, tokens, daily,
                                seed, datetime.now(IST))
        except Exception as exc:
            # Only scan_window sits outside _iteration's own handlers, and
            # it has no reason to raise - but see NEVER RAISES OUT.
            logger.warning("scan loop iteration failed: %s", exc)
            continue
        if reason == idle:
            continue
        if reason is not None:
            logger.info("scan loop idle - nothing scanned, published or "
                        "logged: %s", reason)
        else:
            logger.info("scan loop resuming: %s no longer applies", idle)
        idle = reason
