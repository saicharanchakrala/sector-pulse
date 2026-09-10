"""Intraday bars from the live tick stream, so a scan needs no download.

THE PROBLEM THIS SOLVES. Hitting "Run intraday scan" currently fetches
intraday bars for 216 symbols from Kite's historical endpoint, paced at its
documented 3 requests a second. That is minutes of waiting for data that was
already streaming past. Worse, the last completed candle can be a few
minutes stale by the time it arrives, so the levels are computed against a
price that has moved.

THE SHAPE OF THE FIX. A scan needs two different things:

  * PRIOR SESSIONS, for the relative-volume baseline and a gap-free ATR.
    These never change during the day, so they are fetched once against a
    window ending YESTERDAY and cached. A stable window means a cache hit
    all day and no network on any scan.
  * TODAY, which is what the stream supplies. Ticks are bucketed in
    memory at config.SCAN_BAR_INTERVAL - the same size the historical
    fetch above asks for, or the two halves of a session would not join -
    and flushed to parquet so another process can read them instantly.

VOLUME IS A DELTA, NOT A SUM. Kite's tick carries `volume` as the
CUMULATIVE volume traded so far today, not the size of that trade. Summing
it across a bar would multiply the true figure by the number of ticks, which
would send relative volume through the roof and make every quiet name look
like a breakout. Each bar's volume is therefore the difference between the
cumulative figure at its own close and at the previous bar's close.

PARTIAL BARS ARE MARKED, NOT HIDDEN. A feed starting part-way through a
bucket never saw that bucket's first minutes, and its volume is measured
from an unknown baseline. Such bars are flagged rather than silently served,
because a partial bar looks exactly like a quiet one.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent
STORE = ROOT / "live_bars"
# From config, so the feed and the scanner cannot disagree about the
# bucket size. A feed bucketing at 300 while scan_data asked Kite for
# 3-minute candles would have written bars nothing downstream could use.
BAR_SECONDS = config.SCAN_BAR_SECONDS
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def bucket_start(when: datetime) -> datetime:
    """The bucket a timestamp belongs to, floored, in IST.

    Bars are stamped at their START, matching Kite's historical candles, so
    at three-minute bars one labelled 10:00 covers 10:00 to 10:03. Every
    downstream guard in this project assumes that convention.

    Anchored on the hour rather than on the session open, which is what
    makes 10:00 a bucket boundary at every size this project offers.
    """
    local = when.astimezone(IST)
    floored = local.replace(second=0, microsecond=0)
    return floored - timedelta(minutes=floored.minute % (BAR_SECONDS // 60))


def in_session(stamp: datetime) -> bool:
    """Whether a bucket start falls inside NSE equity TRADING HOURS.

    Bounds come from config so the feed and the scanner cannot disagree
    about when the market is open. A bucket STARTING at the close would
    cover post-close time, so the last valid one starts strictly before it.

    Weekends are excluded; trading HOLIDAYS are not, because nothing in
    this project holds an exchange holiday calendar. That is tolerable
    here - the feed receives no ticks on a closed day, so there is nothing
    to admit - but it means this answers "within trading hours" and not
    "the market was open".
    """
    local = stamp.astimezone(IST)
    if local.weekday() >= 5:
        return False
    open_hour, open_minute = config.SCAN_SESSION_OPEN
    close_hour, close_minute = config.SCAN_SESSION_CLOSE
    start = local.replace(hour=open_hour, minute=open_minute, second=0,
                          microsecond=0)
    end = local.replace(hour=close_hour, minute=close_minute, second=0,
                        microsecond=0)
    return start <= local < end


def store_path(when: "date | None" = None) -> Path:
    """Where one session's live bars live."""
    day = when or datetime.now(IST).date()
    return STORE / f"live_{day:%Y%m%d}.parquet"


class BarBuilder:
    """Aggregates ticks into bars of BAR_SECONDS, one set per instrument.

    Thread-safe because the stream runs in an asyncio loop inside a
    background thread while flushes and snapshots come from elsewhere.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # token -> bucket_start -> partial bar dict
        self._open: dict[int, dict] = {}
        self._done: list[dict] = []
        # token -> cumulative volume at the close of its last finished bar
        self._volume_mark: dict[int, int] = {}
        self._seen_from: dict[int, datetime] = {}
        # Whether the currently OPEN bar has seen a tick that actually
        # carried a volume field. Index packets never do and an LTP-sized
        # packet does not either, so a bar can see hundreds of ticks and no
        # volume - which is not the same thing as no trading.
        self._saw_volume: dict[int, bool] = {}
        # token -> start of its last COMPLETED bar. Kept separately from
        # _open because close_open_bars empties that, and without it a
        # replayed tick could reopen a bucket already written out.
        self._closed_at: dict[int, datetime] = {}
        self._refused_late = 0
        self._refused_stale = 0
        self._refused_naive = 0
        # Completed bars already converted to a frame, and how many of
        # _done that covers. snapshot() used to rebuild the whole frame on
        # every call: measured at 1,805ms for 193,000 bars and 3,061ms for
        # 400,000. It runs outside the lock so it never blocked tick
        # ingestion, but seconds of CPU every 15-second flush competes for
        # the GIL with the tick handler. Only the new tail is converted.
        self._frame: "pd.DataFrame | None" = None
        self._framed = 0

    def add(self, ticks: list) -> None:
        """Fold a batch of ticks into their buckets."""
        with self._lock:
            for tick in ticks:
                self._add_one(tick)

    def _add_one(self, tick) -> None:
        stamp = tick.exchange_timestamp or tick.received_at
        if stamp is None or tick.last_price <= 0:
            return
        if stamp.tzinfo is None:
            # astimezone() on a naive datetime assumes the PROCESS
            # timezone, so on a UTC-configured host every stamp would land
            # outside the session gate below and the feed would write
            # nothing at all. Refused explicitly rather than guessed.
            self._refused_naive += 1
            return
        token = tick.instrument_token
        start = bucket_start(stamp)
        if not in_session(start):
            # Kite keeps sending after the close. The resulting buckets
            # carried a frozen price and no volume, and fed ATR, the
            # opening range and relative volume as though they were real.
            self._refused_late += 1
            return
        self._seen_from.setdefault(token, start)
        current = self._open.get(token)
        closed = self._closed_at.get(token)
        if current is None and closed is not None and start <= closed:
            # The bucket was already completed and written. Reopening it
            # would emit the same period twice.
            self._refused_stale += 1
            return
        if current is not None and start < current["start"]:
            # An OLDER bucket than the open one, which a reconnect replay
            # produces. Rolling back to it closed the newer bar early and
            # rewound the volume baseline, so it is refused instead.
            self._refused_stale += 1
            return
        if current is not None and current["start"] != start:
            self._finish(token, current)
            current = None
        if current is None:
            current = {"start": start, "open": tick.last_price,
                       "high": tick.last_price, "low": tick.last_price,
                       "close": tick.last_price,
                       "cum_volume": self._volume_mark.get(token),
                       "ticks": 0}
            self._open[token] = current
            self._saw_volume[token] = False
        current["high"] = max(current["high"], tick.last_price)
        current["low"] = min(current["low"], tick.last_price)
        current["close"] = tick.last_price
        # Latest cumulative reading wins; the delta is taken at close.
        # `is not None` rather than truthiness, because the old test could
        # not tell a genuine cumulative of 0 - no trade yet today - from a
        # packet that carries no volume field at all.
        reading = getattr(tick, "volume", None)
        if reading is not None:
            previous = current["cum_volume"]
            # Cumulative volume only rises. A lower reading is a replayed
            # or corrupt packet, and letting it through would rewind the
            # baseline and inflate the following bar.
            if previous is None or reading >= previous:
                current["cum_volume"] = reading
                self._saw_volume[token] = True
        current["ticks"] += 1

    def _finish(self, token: int, bar: dict,
                truncated: bool = False) -> None:
        """Close one bar, converting cumulative volume into this bar's own.

        `truncated` marks a bar closed before its bucket ended, which only
        close_open_bars can cause. Such a bar covers a fraction of its
        period, so it is flagged exactly as the forming-bar snapshot
        already flags it - otherwise a Ctrl-C 30 seconds into a bucket
        wrote a 30-second bar recorded as a finished full-length one.
        """
        mark = self._volume_mark.get(token)
        cumulative = bar["cum_volume"]
        saw = self._saw_volume.get(token, False)
        if mark is None or cumulative is None or not saw:
            # Unknowable rather than zero, by three routes: the feed joined
            # part-way through so there is no baseline; the bar saw no tick
            # carrying volume at all, which is permanent for an index; or
            # both. Reporting the whole day's volume here - which the old
            # code did on the bar AFTER this one, having reset the mark to
            # zero - is the spike this module exists to prevent.
            volume = float("nan")
        else:
            volume = max(0.0, float(cumulative - mark))
        # Advance the baseline only on a reading actually observed.
        if cumulative is not None and saw:
            self._volume_mark[token] = cumulative
        self._saw_volume[token] = False
        # `partial` means the bar did not cover its whole bucket - the feed
        # joined mid-way through it. It deliberately does NOT mean "volume
        # unknown": that is already said by Volume being NaN, and an index
        # carries no volume field at all, so folding the two together
        # flagged every index bar partial and load_today(drop_partial=True)
        # then dropped the entire benchmark the relative-strength gate
        # needs. The OHLC of a volume-less bar is perfectly good.
        partial = ((self._seen_from.get(token) == bar["start"]
                    and mark is None) or truncated)
        # Remember where this token's last completed bar sat, so a replay
        # arriving after _open was cleared cannot reopen it.
        self._closed_at[token] = bar["start"]
        self._done.append({
            "instrument_token": token, "Date": bar["start"],
            "Open": bar["open"], "High": bar["high"], "Low": bar["low"],
            "Close": bar["close"], "Volume": volume,
            "ticks": bar["ticks"], "partial": bool(partial),
        })

    EMPTY_COLUMNS = ["instrument_token", "Date"] + COLUMNS + ["ticks",
                                                              "partial"]

    def _forming_rows(self) -> list:
        """The bar still being built, per instrument. Caller holds the lock."""
        rows = []
        for token, bar in self._open.items():
            mark = self._volume_mark.get(token)
            cumulative = bar["cum_volume"]
            known = (mark is not None and cumulative is not None
                     and self._saw_volume.get(token, False))
            rows.append({
                "instrument_token": token, "Date": bar["start"],
                "Open": bar["open"], "High": bar["high"],
                "Low": bar["low"], "Close": bar["close"],
                "Volume": (max(0.0, float(cumulative - mark)) if known
                           else float("nan")),
                "ticks": bar["ticks"], "partial": True,
            })
        return rows

    def snapshot(self, include_forming: bool = False) -> pd.DataFrame:
        """Completed bars, optionally with the bar still being built.

        The forming bar is excluded by default: every indicator here treats
        a bar as a finished period, and half a bar reads as a real one.

        Built INCREMENTALLY. Completed bars never change once closed, so
        the frame for them is cached and only bars closed since the last
        call are converted. The forming bars are appended fresh each time
        and never cached, because they are still moving.
        """
        with self._lock:
            fresh = self._done[self._framed:]
            done_total = len(self._done)
            forming = self._forming_rows() if include_forming else []
        if fresh:
            block = pd.DataFrame(fresh)
            self._frame = (block if self._frame is None
                           else pd.concat([self._frame, block],
                                          ignore_index=True))
            self._framed = done_total
        completed = self._frame
        if completed is None and not forming:
            return pd.DataFrame(columns=self.EMPTY_COLUMNS)
        if not forming:
            return completed
        tail = pd.DataFrame(forming)
        if completed is None:
            return tail
        return pd.concat([completed, tail], ignore_index=True)

    def close_open_bars(self, now: "datetime | None" = None) -> int:
        """Finish every open bar. Without this the last one is lost.

        _finish runs only when a LATER tick arrives, so the final bucket of
        a session stays in _open until the process exits - and flush()
        writes completed bars only.

        A bar whose bucket is still the CURRENT one is closed early and is
        flagged partial, because it covers only part of its period. At
        15:30 nothing is current, so the session's last bar is closed
        complete, which is the case this exists for.
        """
        with self._lock:
            # `now` is injectable so the truncation rule can be tested
            # without depending on the wall clock being mid-session.
            current = bucket_start(now or datetime.now(IST))
            pending = list(self._open.items())
            for token, bar in pending:
                self._finish(token, bar, truncated=bar["start"] >= current)
            self._open.clear()
        return len(pending)

    def refused(self) -> dict:
        """Ticks turned away, so a caller can report rather than hide them."""
        with self._lock:
            return {"outside_session": self._refused_late,
                    "out_of_order": self._refused_stale,
                    "naive_timestamp": self._refused_naive}

    def flush(self, when: "date | None" = None) -> int:
        """Write completed bars to parquet, atomically. Returns rows written.

        Atomic because Streamlit may read this file at any moment, and a
        half-written parquet is not a smaller parquet - it is a crash.
        """
        frame = self.snapshot()
        if frame.empty:
            return 0
        STORE.mkdir(parents=True, exist_ok=True)
        target = store_path(when)
        temporary = target.with_suffix(".tmp")
        try:
            frame.to_parquet(temporary)
            os.replace(temporary, target)
        except Exception as exc:
            logger.warning("Could not flush live bars: %s", exc)
            return 0
        return len(frame)


def load_today(tokens: "dict[str, int] | None" = None,
               when: "date | None" = None,
               drop_partial: bool = True) -> dict:
    """Today's live bars per SYMBOL, or {} when the feed has written nothing.

    `tokens` maps symbol to instrument token; without it the frames come
    back keyed by token, which is only useful for diagnostics.
    """
    path = store_path(when)
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Unreadable live bar store %s: %s", path.name, exc)
        return {}
    if frame.empty:
        return {}
    if drop_partial and "partial" in frame.columns:
        frame = frame[~frame["partial"].astype(bool)]
    by_token = {token: group for token, group in frame.groupby("instrument_token")}
    lookup = {int(v): k for k, v in (tokens or {}).items()}
    out = {}
    for token, group in by_token.items():
        key = lookup.get(int(token), int(token))
        block = group.sort_values("Date").set_index("Date")
        block.index = pd.DatetimeIndex(block.index)
        if block.index.tz is None:
            block.index = block.index.tz_localize(IST)
        out[key] = block[COLUMNS]
    return out


def history_window(days: int = 12) -> tuple:
    """A prior-session window that ENDS YESTERDAY, so it caches all day.

    The point is stability. A window ending today changes every session and
    misses the parquet cache every morning; ending yesterday means one fetch
    and then no network for the rest of the day.
    """
    yesterday = datetime.now(IST).date() - timedelta(days=1)
    return yesterday - timedelta(days=max(1, days)), yesterday


def prewarm(symbols: list, days: int = 12,
            interval: "str | None" = None) -> dict:
    """Fetch and cache prior-session bars once. Returns what was obtained.

    The interval comes from config rather than a literal. It used to
    default to "5minute" while the stream bucketed at whatever
    SCAN_BAR_SECONDS said, so combined() spliced 5-minute prior sessions
    onto 3-minute live bars and every reading taken across that join -
    ATR, VWAP, relative volume, the opening range - was computed over two
    bar sizes at once.
    """
    import market_source
    interval = interval or market_source.kite_interval(config.SCAN_BAR_INTERVAL)
    start, end = history_window(days)
    try:
        return market_source.bars(symbols, start, end, interval=interval)
    except market_source.NoSession as exc:
        logger.warning("%s", exc)
        return {}


def backfill_today(symbols: list, interval: "str | None" = None) -> dict:
    """Today's session bars from Kite, for the part the feed missed.

    A feed started at 15:06 has no 09:15 bar, and the opening range is the
    session's FIRST fifteen minutes. Without it choose_direction returns no
    setup for every symbol - which reads as a quiet market when it is
    really a truncated session. Measured exactly that: 0 of 216 names got a
    direction from two bars of live data.

    One historical call per symbol, once, at feed startup. After that the
    stream carries the session forward and this is never needed again.
    """
    import market_source
    interval = interval or market_source.kite_interval(config.SCAN_BAR_INTERVAL)
    today = datetime.now(IST).date()
    try:
        return market_source.bars(symbols, today, today, interval=interval,
                                  refresh=True)
    except market_source.NoSession as exc:
        logger.warning("%s", exc)
        return {}
    except Exception as exc:
        logger.warning("Could not backfill today: %s", exc)
        return {}


def session_is_covered(frame, opening_minutes: int = 15) -> bool:
    """Whether today's bars reach back to the session open.

    The check is deliberately about the OPEN rather than bar count: fifty
    bars starting at noon still cannot form an opening range.
    """
    if frame is None or frame.empty:
        return False
    today = datetime.now(IST).date()
    session = frame[[ts.date() == today for ts in frame.index]]
    if session.empty:
        return False
    first = session.index[0].astimezone(IST)
    open_hour, open_minute = config.SCAN_SESSION_OPEN
    cutoff = first.replace(hour=open_hour, minute=open_minute,
                           second=0, microsecond=0) + timedelta(
                               minutes=opening_minutes)
    return first <= cutoff


def combined(symbols: list, tokens: "dict[str, int] | None" = None,
             days: int = 12) -> dict:
    """Prior sessions from cache plus today from the stream, per symbol.

    This is what makes a scan instant: neither half touches the network on
    a normal day. If the feed is not running, today is simply absent and
    the caller sees history only - which the scanner reports as a stale
    session rather than treating as a quiet market.
    """
    history = prewarm(symbols, days=days)
    live = load_today(tokens)
    # Today's bars from the seed file, written once at feed startup for the
    # part of the session that preceded the stream. Without it a mid-day
    # start silently loses the opening range.
    seed = load_seed()
    out = {}
    truncated = 0
    for symbol in symbols:
        parts = [f for f in (history.get(symbol), seed.get(symbol),
                             live.get(symbol))
                 if f is not None and not f.empty]
        if not parts:
            continue
        joined = pd.concat(parts)
        joined = joined[~joined.index.duplicated(keep="last")].sort_index()
        if not session_is_covered(joined):
            truncated += 1
        out[symbol] = joined
    if truncated:
        logger.warning(
            "Today's session is truncated for %d/%d symbols - the feed "
            "started after the open and no seed covers the gap, so the "
            "opening range cannot form and no direction will be chosen",
            truncated, len(out))
    return out


def seed_path(when: "date | None" = None) -> Path:
    """Where today's pre-stream backfill is kept."""
    day = when or datetime.now(IST).date()
    return STORE / f"seed_{day:%Y%m%d}.parquet"


def write_seed(frames: dict, when: "date | None" = None) -> int:
    """Persist the startup backfill so any reader sees the full session."""
    if not frames:
        return 0
    rows = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        block = frame[[c for c in COLUMNS if c in frame.columns]].copy()
        block["symbol"] = symbol
        block = block.reset_index()
        rows.append(block)
    if not rows:
        return 0
    STORE.mkdir(parents=True, exist_ok=True)
    joined = pd.concat(rows, ignore_index=True)
    target = seed_path(when)
    temporary = target.with_suffix(".tmp")
    try:
        joined.to_parquet(temporary)
        os.replace(temporary, target)
    except Exception as exc:
        logger.warning("Could not write today's seed: %s", exc)
        return 0
    return len(joined)


def load_seed(when: "date | None" = None) -> dict:
    """Today's pre-stream bars per symbol, or {} when there is no seed."""
    path = seed_path(when)
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Unreadable seed %s: %s", path.name, exc)
        return {}
    if frame.empty or "symbol" not in frame.columns:
        return {}
    stamp = "Date" if "Date" in frame.columns else frame.columns[0]
    out = {}
    for symbol, group in frame.groupby("symbol"):
        block = group.set_index(stamp).sort_index()
        block.index = pd.DatetimeIndex(block.index)
        if block.index.tz is None:
            block.index = block.index.tz_localize(IST)
        out[str(symbol)] = block[[c for c in COLUMNS if c in block.columns]]
    return out


def status(when: "date | None" = None) -> dict:
    """What the live store currently holds, for a UI line."""
    path = store_path(when)
    if not path.exists():
        return {"present": False}
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return {"present": False}
    if frame.empty:
        return {"present": True, "bars": 0, "instruments": 0}
    latest = pd.DatetimeIndex(frame["Date"]).max()
    return {
        "present": True,
        "bars": int(len(frame)),
        "instruments": int(frame["instrument_token"].nunique()),
        "latest_bar": latest,
        "partial": int(frame.get("partial", pd.Series(dtype=bool)).sum()),
        # float("nan") rather than None when the age is unknowable, so a
        # caller's NaN check behaves and a format string cannot blow up.
        "age_seconds": (max(0.0, (datetime.now(IST)
                                 - latest.tz_convert(IST)).total_seconds())
                        if latest.tzinfo else float("nan")),
    }
