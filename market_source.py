"""The single market-data source: Zerodha Kite. No yfinance anywhere.

Every price in this project now comes from here. That is a deliberate
narrowing with a real cost, stated plainly because it changes how the tools
behave rather than just what they import:

  * Kite needs a paid subscription and an interactive login with your
    password and 2FA. yfinance needed nothing. So the daily signal, the
    dashboard and the contribution planner no longer run out of the box -
    they need a session, and Kite expires it around 6am every morning with
    no non-interactive refresh for retail apps.
  * Kite serves Indian instruments only. The US profile's SPDR ETFs have no
    Kite equivalent, which is why that profile was removed rather than left
    permanently broken.

What is gained: prices are live rather than roughly fifteen minutes late,
and minute candles reach back years instead of eight days. Historical open
interest also becomes fetchable via `oi=True` - the field NSE publishes
nowhere - but nothing here passes it yet, so every past-date replay still
drops OI as lookahead and says so. The capability is available; it is not
wired up.

Symbols are Kite tradingsymbols throughout. Yahoo's conventions are gone
with the dependency: no `.NS` suffix, no `^` index prefix. The translation
table below exists only so an old profile string still resolves, and every
entry in it was verified against Kite's own public instrument master.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import config
import kite_client
import kite_instruments as ki

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

CACHE_DIR = config.PROJECT_ROOT / "bar_cache"

# The benchmark for relative strength, as Kite names it.
BENCHMARK = "NIFTY 50"

# Yahoo ticker to Kite tradingsymbol. Kept only so a profile string written
# for yfinance still resolves; new code should use the Kite name directly.
# Verified against Kite's public master: all thirteen resolve.
_LEGACY_ALIASES: dict[str, str] = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
    "^CNXAUTO": "NIFTY AUTO",
    "^CNXIT": "NIFTY IT",
    "^CNXFMCG": "NIFTY FMCG",
    "^CNXMETAL": "NIFTY METAL",
    "^CNXPHARMA": "NIFTY PHARMA",
    "^CNXPSUBANK": "NIFTY PSU BANK",
    "^CNXREALTY": "NIFTY REALTY",
    "^CNXENERGY": "NIFTY ENERGY",
    "^CNXINFRA": "NIFTY INFRA",
    "^CNXMEDIA": "NIFTY MEDIA",
    "NIFTY_FIN_SERVICE": "NIFTY FIN SERVICE",
    # F&O UNDERLYING names, which differ from the cash index tradingsymbol.
    # NSE's derivatives master calls the Bank Nifty underlying BANKNIFTY
    # while Kite's cash instrument is "NIFTY BANK", so the live feed was
    # refusing to subscribe to every index - including the benchmark that
    # relative strength is measured against. Verified against Kite's own
    # index list, which carries 236 names.
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "MIDSEL",
    "NIFTYNXT50": "NIFTY NEXT 50",
    "NIFTYFPI": "NIFTY FPI 150",
}

# Interval names differ between the two APIs. Kite's are the canonical ones
# now; the yfinance spellings map across so existing call sites keep working.
_INTERVALS: dict[str, str] = {
    "1m": "minute", "3m": "3minute", "5m": "5minute", "15m": "15minute",
    "30m": "30minute", "60m": "60minute", "1h": "60minute", "1d": "day",
}

_MASTER: "list | None" = None
_TOKENS: "dict[str, int] | None" = None
_EXCHANGES: "dict[str, str] | None" = None


class NoSession(RuntimeError):
    """No usable Kite session, so no price can be fetched.

    Covers both ways this happens, which matters because only the second is
    a daily event: no token file at all, and a token file Kite has expired.
    """


LOGIN_HINT = ("No usable Kite session, so no prices can be fetched. Run: "
              ".venv\\Scripts\\python -m kite_login")


def is_session_failure(exc: BaseException) -> bool:
    """Whether a Kite error means "your session is dead" rather than "that
    request failed".

    Matched on the message because kite_client.KiteError carries no status
    code. That is a coupling to a string built in this same repo
    (kite_client raises "Kite rejected the session (HTTP 403)"), so it is
    stable, but a `status` field on KiteError would be better and this
    should move to it. Getting this wrong in the safe direction only costs
    a log line; getting it wrong the other way means a whole scan reports
    two hundred symbols as individually unavailable when the real answer is
    one expired login.
    """
    return "403" in str(exc)


def _no_session(partial: "dict | None" = None, detail: str = "") -> None:
    """Raise NoSession - unless useful cached data was already gathered.

    A scan where 209 of 210 symbols are cached should not be thrown away
    because the last one needed the network. Every caller's degradation
    path handles a short dict; none of them handles losing the lot.
    """
    if partial:
        logger.warning("%s Continuing on %d cached symbol(s).",
                       detail or LOGIN_HINT, len(partial))
        return
    raise NoSession(f"{LOGIN_HINT}{(' ' + detail) if detail else ''}")


def canonical(symbol: str) -> str:
    """A Kite tradingsymbol from whatever spelling the caller had.

    Handles the three legacies at once: an explicit alias, a stray `.NS`
    suffix, and a `^`-prefixed Yahoo index. Anything else passes through
    unchanged, because the vast majority of symbols are already correct.
    """
    text = (symbol or "").strip().upper()
    if not text:
        return ""
    if text in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[text]
    if text.endswith(".NS"):
        text = text[:-3]
        if text in _LEGACY_ALIASES:
            return _LEGACY_ALIASES[text]
    if text.startswith("^"):
        stripped = text.lstrip("^")
        if text in _LEGACY_ALIASES:
            return _LEGACY_ALIASES[text]
        if stripped in _LEGACY_ALIASES:
            return _LEGACY_ALIASES[stripped]
        # Keep the caret on anything unrecognised. Stripping it turned
        # "^BSESN" into "BSESN", which can match an unrelated listed
        # instrument; no Kite tradingsymbol contains a caret, so leaving it
        # on guarantees a miss that gets logged instead of a wrong price.
        return text
    return text


def calendar_days(spec: str, default: int = 60) -> int:
    """Calendar days covering a span written in yfinance's units.

    The units changed underneath these strings and it is not cosmetic.
    yfinance's `period="60d"` meant sixty TRADING days: measured on this
    repo's own cache, `360ONE__5m_60d.parquet` holds 59 sessions spread over
    83 calendar days. Kite takes real dates, so passing 60 straight through
    as `end - 60 days` returns about 43 sessions - a 27% cut to the sample,
    silent, because nothing downstream gates on a minimum session count.

    So a bare "Nd" is read as N sessions and converted: five sessions per
    seven days, plus slack that grows with the span for public holidays and
    long weekends. "Nmo" and "Ny" are already calendar units and pass
    through. Erring long costs a few more paged requests; erring short
    quietly degrades every relative-volume median and ATR baseline built on
    top of it.
    """
    text = (spec or "").strip().lower()
    try:
        if text.endswith("mo"):
            return max(1, int(round(float(text[:-2]) * 30.44)))
        if text.endswith("y"):
            return max(1, int(round(float(text[:-1]) * 365)))
        if text.endswith("d"):
            sessions = int(text[:-1])
        else:
            sessions = int(text)
    except ValueError:
        return default
    if sessions < 1:
        return default
    return -(-sessions * 7 // 5) + max(3, sessions // 10)


def kite_interval(interval: str) -> str:
    """Kite's spelling of a candle interval."""
    text = (interval or "").strip().lower()
    return _INTERVALS.get(text, text)


def master(refresh: bool = False) -> list:
    """Kite's public instrument master, fetched once per process.

    Public and unauthenticated, so symbol resolution keeps working even
    without a session. Only prices need the login.
    """
    global _MASTER, _TOKENS, _EXCHANGES
    # `not _MASTER` rather than `is None`: fetch_master returns [] on any
    # network failure, and caching that meant one transient outage disabled
    # symbol resolution for the whole process - never retried, because []
    # is not None. An empty master is a failure to answer, not an answer.
    if not _MASTER or refresh:
        _MASTER = ki.fetch_master()
        _TOKENS = None
        _EXCHANGES = None
    return _MASTER


def _build_tables(refresh: bool = False) -> None:
    """Fill the token and exchange tables together, so they cannot disagree.

    The exchange has to be carried alongside the token. Futures live on NFO
    and some indices are not NSE, so a quote key built as "NSE:" + symbol
    for everything asks the wrong exchange and comes back as a silent miss.
    """
    global _TOKENS, _EXCHANGES
    rows = master(refresh=refresh)
    tokens_by_symbol: dict[str, int] = {}
    exchange_by_symbol: dict[str, str] = {}
    for group in (ki.indices(rows), ki.nse_futures(rows), ki.nse_equities(rows)):
        for contract in group:
            # Equities last so a cash symbol wins any collision with a
            # derivative or index of the same name.
            tokens_by_symbol[contract.tradingsymbol] = contract.instrument_token
            exchange_by_symbol[contract.tradingsymbol] = getattr(
                contract, "exchange", "") or "NSE"
    _TOKENS = tokens_by_symbol
    _EXCHANGES = exchange_by_symbol


def tokens(refresh: bool = False) -> dict[str, int]:
    """Tradingsymbol to instrument token, for cash, indices and futures."""
    if _TOKENS is None or refresh:
        _build_tables(refresh=refresh)
    assert _TOKENS is not None
    return _TOKENS


def exchange_for(symbol: str) -> str:
    """The exchange a symbol trades on, defaulting to NSE."""
    if _EXCHANGES is None:
        _build_tables()
    assert _EXCHANGES is not None
    return _EXCHANGES.get(canonical(symbol), "NSE")


def token_for(symbol: str) -> "int | None":
    """One symbol's instrument token, or None when it is not listed."""
    return tokens().get(canonical(symbol))


def _cache_path(symbol: str, interval: str, start: date, end: date,
                oi: bool = False):
    """Where one symbol's bars for one span and interval are cached.

    Byte-for-byte the same scheme as kite_bars._cache_path, deliberately:
    both modules write into the same bar_cache, so the two either agree or
    they corrupt each other. `oi` has to be part of the key. Without it a
    frame fetched without open interest is served verbatim to a later
    oi=True caller, silently dropping the one column that caller asked for -
    a bug kite_bars had already found and fixed, and which this module
    reintroduced by copying the filename but not the key.
    """
    safe = canonical(symbol).replace("/", "_").replace(":", "_").replace(" ", "_")
    suffix = "__oi" if oi else ""
    return (CACHE_DIR / f"kite__{safe}__{interval}__"
                        f"{start:%Y%m%d}_{end:%Y%m%d}{suffix}.parquet")


def _parse_cache_name(name: str):
    """(symbol, interval, start, end, oi) from a cache filename, or None.

    The inverse of _cache_path. Split rather than matched with a regex
    because a safe symbol may itself contain underscores - "NIFTY 50"
    becomes "NIFTY_50" and NIFTY_MEDIA is a real ETF - while a Kite
    interval never does.
    """
    if not name.startswith("kite__") or not name.endswith(".parquet"):
        return None
    body = name[len("kite__"):-len(".parquet")]
    oi = body.endswith("__oi")
    if oi:
        body = body[:-len("__oi")]
    try:
        symbol, interval, span = body.rsplit("__", 2)
        start_text, end_text = span.split("_")
        start = datetime.strptime(start_text, "%Y%m%d").date()
        end = datetime.strptime(end_text, "%Y%m%d").date()
    except ValueError:
        return None
    if not symbol:
        return None
    return symbol, interval, start, end, oi


_SPANS_LOCK = threading.Lock()
_SPANS_MEMO: dict = {}


def _cache_spans(interval: str, oi: bool) -> dict:
    """{safe symbol: [(start, end, path)]} already on disk for one interval.

    ONE directory listing for the whole call, not one glob per symbol.
    bar_cache holds over 12,000 files, and globbing per symbol across a
    210-name scan would walk all of them 210 times.

    MEMOISED ON THE DIRECTORY'S OWN mtime, because the listing costs 1.74
    seconds at that size and bars() is called several times per render.
    Adding or removing a file moves the directory's mtime, so a changed
    cache rebuilds. If mtime is unavailable the memo is skipped rather
    than trusted. A missed rebuild only costs a fetch that would have
    happened anyway; it can never serve the wrong bars, because the file
    a span points at is read and date-filtered on use.
    """
    try:
        stamp = CACHE_DIR.stat().st_mtime
    except OSError:
        stamp = None
    key = (interval, oi)
    if stamp is not None:
        with _SPANS_LOCK:
            remembered = _SPANS_MEMO.get(key)
        if remembered is not None and remembered[0] == stamp:
            return remembered[1]

    spans: dict = {}
    try:
        entries = list(CACHE_DIR.iterdir())
    except OSError:
        return spans
    for path in entries:
        parsed = _parse_cache_name(path.name)
        if parsed is None:
            continue
        symbol, found_interval, start, end, found_oi = parsed
        if found_interval != interval or found_oi != oi:
            continue
        spans.setdefault(symbol, []).append((start, end, path))
    if stamp is not None:
        with _SPANS_LOCK:
            _SPANS_MEMO[key] = (stamp, spans)
    return spans


def _covering_span(spans: dict, symbol: str, start: date, end: date):
    """A cached file whose span CONTAINS [start, end], or None.

    WHY THIS EXISTS. The cache key is the exact span, and every horizon
    the app offers is a window that rolls forward every day: the mid-term
    table asked for 2026-06-16..2026-09-15 yesterday and
    2026-06-17..2026-09-16 today. Same bars but for one session at each
    end, different key, total miss. So the first page load of each trading
    day refetched months of daily bars for the whole scope - measured
    2026-09-17, 222 spans in one morning, a burst of Kite calls during the
    session that showed up in the console as SSL drops.

    The narrowest containing span wins, so the slice stays small, with the
    most recently written breaking a tie - that is the one most likely to
    reach the end of the range.

    WHAT THIS DOES NOT PROVE, stated because it was raised in review and
    accepted rather than missed: the span comes from the FILENAME, which
    records what was asked for, not what Kite returned. A file named for a
    year can hold a fraction of one - Kite truncates intraday history,
    returns nothing before a listing date, and nothing across a
    suspension.

    Two things make that acceptable here. A capture cut short by being
    written mid-session is already refused, because bars() runs
    _cache_is_stale against the SOURCE file's mtime and a settled span is
    only trusted when the file was written after that date's close. What
    remains is a span Kite genuinely has less data for - and a fresh fetch
    would return exactly the same short answer, at the cost of the request.

    The tempting fix, requiring the slice to bracket [start, end], cannot
    be written without a trading calendar: holidays, listing dates and
    halts all make a legitimately short frame look truncated, and a
    threshold picked by eye would refetch the whole universe on the first
    holiday. The reason this cache exists at all is that refetching the
    whole universe is what took the machine down.
    """
    found = spans.get(canonical(symbol).replace("/", "_")
                      .replace(":", "_").replace(" ", "_"))
    if not found:
        return None
    covering = [(s, e, p) for s, e, p in found if s <= start and e >= end]
    if not covering:
        return None

    def rank(item):
        span_start, span_end, path = item
        try:
            written = path.stat().st_mtime
        except OSError:
            written = 0.0
        return ((span_end - span_start).days, -written)

    return min(covering, key=rank)[2]


def _slice_span(frame, start: date, end: date):
    """The rows of a wider cached frame that fall inside [start, end].

    Both ends inclusive, matching what Kite returns for the same request.
    A frame whose index is not timestamped cannot be sliced by date and is
    refused rather than served whole, which would hand the caller a wider
    range than it asked for.
    """
    if frame is None or len(frame) == 0:
        return None
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        return None
    try:
        days = index.date
    except (AttributeError, TypeError):
        return None
    keep = (days >= start) & (days <= end)
    return frame[keep]


def _interval_seconds(resolved: str) -> int:
    """Seconds in one candle of Kite's interval spelling."""
    text = (resolved or "").strip().lower()
    if text in ("day", "1day"):
        return 86_400
    if text == "minute":
        return 60
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) * 60 if digits else 300


def _cache_is_stale(path, frame, end: date, resolved: str,
                    now: "datetime | None" = None) -> bool:
    """Whether a cached span ending today has fallen behind the session.

    A span that ended BEFORE today is settled: prior sessions do not
    change, and refetching them would undo the one-fetch-a-day property
    the whole cache exists for. Only a span reaching into today can rot.

    Two conditions, because either alone misbehaves:

      * the data must be behind the reference clock - otherwise a file
        written seconds ago would be refetched;
      * the file must itself be older than the limit - otherwise a symbol
        that cannot catch up, on a holiday or after a halt, is refetched
        on every scan for ever, one Kite call per symbol per run.

    The reference clock is min(now, today's close): a file written at
    15:27 is not stale at 16:00, because the session is over and nothing
    further is coming. Comparing against a bare now() refetched the entire
    universe every evening.
    """
    now = now or datetime.now(IST)
    close_hour, close_minute = config.SCAN_SESSION_CLOSE
    try:
        written = datetime.fromtimestamp(path.stat().st_mtime, IST)
    except OSError:
        return False
    # "Today" is taken from the reference clock, not from date.today(): a
    # caller that supplies `now` supplies the whole clock, and mixing the
    # two made the tests for this function pass on the day they were
    # written and fail the next morning.
    if end < now.date():
        # A PAST span is settled only if the file was captured after that
        # date's close. One written DURING the day stops wherever the
        # fetch reached, and treating it as history freezes a truncated
        # session for ever: measured on 2026-09-13, the 25 Aug - 11 Sep
        # files held nothing after 10 Sep 15:12, because they were written
        # at 09:23 on the 11th. A study of the 11th read them and found no
        # bars for the day it was studying.
        end_close = datetime(end.year, end.month, end.day, close_hour,
                             close_minute, tzinfo=IST)
        return written < end_close
    limit = min(2 * _interval_seconds(resolved) + 60,
                config.CACHE_TODAY_MAX_AGE_SECONDS)
    close = now.replace(hour=close_hour, minute=close_minute, second=0,
                        microsecond=0)
    reference = min(now, close)
    if (reference - written).total_seconds() <= limit:
        return False
    newest = None
    if frame is not None and len(frame):
        try:
            newest = frame.index[-1].astimezone(IST)
        except (AttributeError, TypeError, IndexError):
            newest = None
    if newest is None:
        # Nothing to measure against, so the file's own age decides - and
        # it is already past the limit.
        return True
    return (reference - newest).total_seconds() > limit


def bars(symbols: list[str], start: date, end: date, interval: str = "5minute",
         oi: bool = False, refresh: bool = False,
         session=None) -> dict[str, pd.DataFrame]:
    """OHLCV frames per symbol, keyed by the symbol the CALLER passed in.

    Keyed on the caller's spelling rather than the canonical one so a module
    still holding yfinance-style tickers gets its own keys back and needs no
    translation of its own.

    Cached to parquet per symbol, span and interval, so a repeat costs
    nothing. Symbols that are unlisted or return nothing are omitted and
    logged; a sweep of two hundred names should not die on one delisting.
    """
    resolved = kite_interval(interval)
    live = session
    out: dict[str, pd.DataFrame] = {}
    needed: list[tuple[str, int]] = []

    # READ THE CACHE IN PARALLEL. A warm scan of 210 symbols is ~420
    # parquet reads across the intraday and daily spans, and they were
    # read one after another - measured at 21.3s of a 46.8s page render
    # with nothing being downloaded at all. Unlike the fetches below there
    # is no rate limit here: this is local IO and pyarrow releases the GIL
    # while decoding, so the pool is close to free.
    #
    # Only the READ is parallel. Staleness, the OpenInterest check and the
    # decision to refetch all stay in the sequential loop, so the logic
    # that decides what to serve is unchanged and reviewable in one place.
    # Listed once, outside the pool, and ONLY when some exact key is
    # missing. In the warm steady state every key hits and the directory
    # is never listed at all, so the fallback costs nothing on the path it
    # does not help.
    spans: dict = {}
    if not refresh and any(
            not _cache_path(s, resolved, start, end, oi=oi).exists()
            for s in symbols):
        spans = _cache_spans(resolved, oi)

    def read_cached(symbol: str):
        """(frame, the file it came from), or None when nothing serves.

        Returns the SOURCE path as well as the frame because the staleness
        rule is applied to the file's own mtime, and with a wider span
        standing in for the requested one those are different files.
        """
        path = _cache_path(symbol, resolved, start, end, oi=oi)
        if refresh:
            return None
        if path.exists():
            try:
                return pd.read_parquet(path), path
            except Exception as exc:
                logger.warning("Unreadable cache %s: %s", path.name, exc)
                return None
        wider = _covering_span(spans, symbol, start, end)
        if wider is None:
            return None
        try:
            frame = pd.read_parquet(wider)
        except Exception as exc:
            logger.warning("Unreadable cache %s: %s", wider.name, exc)
            return None
        sliced = _slice_span(frame, start, end)
        if sliced is None or sliced.empty:
            return None
        return sliced, wider

    readers = max(1, int(getattr(config, "CACHE_READ_WORKERS", 1)))
    preread: dict = {}
    if readers > 1 and len(symbols) > 1 and not refresh:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(readers, len(symbols))) as pool:
            for symbol, entry in zip(symbols,
                                     pool.map(read_cached, symbols)):
                if entry is not None:
                    preread[symbol] = entry

    for symbol in symbols:
        entry = preread.get(symbol)
        if entry is None and symbol not in preread and not refresh:
            entry = read_cached(symbol)
        if entry is not None:
            cached, source = entry
            # With oi=True the column must actually be there. A frame
            # missing it is refused rather than served, because the
            # absence is invisible downstream.
            if oi and "OpenInterest" not in cached.columns:
                logger.warning("Cache %s has no OpenInterest; refetching",
                               source.name)
            elif _cache_is_stale(source, cached, end, resolved):
                # Staleness is judged on the REQUESTED end against the
                # source file's mtime, so a wider span is held to exactly
                # the standard an exact-key file would have been.
                logger.info("Cached %s ends at %s, refetching today's "
                            "tail", source.name,
                            cached.index[-1] if len(cached) else "nothing")
            else:
                out[symbol] = cached
                continue
        token = token_for(symbol)
        if token is None:
            logger.warning("Not listed on Kite: %s", symbol)
            continue
        needed.append((symbol, token))
    if not needed:
        return out
    if live is None:
        live = kite_client.load_session()
    if live is None:
        _no_session(out)     # raises when nothing was cached
        return out
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Not fatal: without a cache directory every fetch is a live one.
        logger.warning("Cannot create %s, running uncached: %s",
                       CACHE_DIR, exc)
    # ONE SESSION PER THREAD, never one shared across them. A Kite session
    # wraps a requests.Session whose HTTPS connection pool is reused, and
    # concurrent workers recycling the same pooled connection produce
    # SSLEOFError("EOF occurred in violation of protocol") part-way through
    # a sweep. Observed live on 2026-09-15 the moment the feed's seed ran
    # through this path: APOLSINHOT, APOORVA, APTUS, AQYLON, ARCHIDPLY and
    # more, three of them failing inside the same millisecond.
    #
    # Sequential code never hit it because there was only ever one caller.
    # The fix is a session per worker, not fewer workers: the pacing gate
    # already limits the REQUEST RATE, and starving the pool would only
    # give back the speed without removing the race.
    _local = threading.local()

    def thread_session():
        existing = getattr(_local, "session", None)
        if existing is None:
            # Falls back to the caller's session only if this thread cannot
            # build its own - better a shared connection than no fetch.
            existing = kite_client.load_session() or live
            _local.session = existing
        return existing

    def fetch_one(symbol: str, token: int):
        """One symbol, returning (frame, exception) and never raising.

        Runs on a worker thread with its OWN Kite session. historical calls
        _pace itself, so the shared 3-a-second gate still meters every
        start - the pool only stops the pipeline draining while replies are
        in transit.
        """
        try:
            return kite_client.historical(token, start, end,
                                          interval=resolved, oi=oi,
                                          session=thread_session()), None
        # ValueError comes from kite_client on a bad span, OSError from the
        # filesystem. Neither should kill a sweep of two hundred names, and
        # market_data promises never to raise on this path.
        except (kite_client.KiteError, ValueError, OSError) as exc:
            return None, exc

    def keep(symbol: str, frame) -> None:
        if frame is None or frame.empty:
            logger.info("%s: no bars for %s..%s", symbol, start, end)
            return
        try:
            frame.to_parquet(_cache_path(symbol, resolved, start, end, oi=oi))
        except Exception as exc:
            logger.warning("Could not cache %s: %s", symbol, exc)
        out[symbol] = frame

    # CONCURRENT, because the limiter meters how often a request may START
    # and says nothing about how long Kite takes to ANSWER - measured at
    # about 1.65s. Fetched one at a time the quota sat idle waiting on
    # round trips, so 210 symbols cost ~7 minutes where 3 a second allows
    # ~70 seconds.
    workers = max(1, int(getattr(config, "KITE_FETCH_WORKERS", 1)))
    if workers == 1 or len(needed) == 1:
        for symbol, token in needed:
            frame, exc = fetch_one(symbol, token)
            if exc is not None:
                if is_session_failure(exc):
                    _no_session(out, detail=str(exc))
                    return out
                logger.warning("%s: %s", symbol, exc)
                continue
            keep(symbol, frame)
        return out

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(needed))) as pool:
        pending = [(symbol, pool.submit(fetch_one, symbol, token))
                   for symbol, token in needed]
        # Collected in the order `needed` lists them, so the result is the
        # same whatever order the replies arrive in.
        for symbol, future in pending:
            frame, exc = future.result()
            if exc is not None:
                if is_session_failure(exc):
                    # Every later symbol would fail the same way. Stop, and
                    # say what it is, rather than logging two hundred
                    # rejections. Cancel what has not started; the ones
                    # already in flight are left to finish and discarded.
                    for _, other in pending:
                        other.cancel()
                    _no_session(out, detail=str(exc))
                    return out
                logger.warning("%s: %s", symbol, exc)
                continue
            keep(symbol, frame)
    return out


def daily_bars(symbols: list[str], months: int = 6, oi: bool = False,
               session=None) -> dict[str, pd.DataFrame]:
    """Daily bars covering roughly the last `months` months."""
    end = date.today()
    start = end - timedelta(days=int(30.44 * max(1, months)))
    return bars(symbols, start, end, interval="day", oi=oi, session=session)


def intraday_bars(symbols: list[str], days: int = 10,
                  interval: str = "5minute",
                  session=None) -> dict[str, pd.DataFrame]:
    """Intraday bars covering roughly the last `days` calendar days."""
    end = date.today()
    start = end - timedelta(days=max(1, days))
    return bars(symbols, start, end, interval=interval, session=session)


def last_prices(symbols: list[str], session=None) -> dict[str, float]:
    """Last traded price per symbol, keyed by the caller's spelling.

    Live rather than delayed, which is the point of the migration. Kite
    caps one quote call at 500 instruments, so longer lists are batched.
    """
    wanted = [s for s in symbols if (s or "").strip()]
    if not wanted:
        return {}
    live = session or kite_client.load_session()
    if live is None:
        raise NoSession(LOGIN_HINT)
    by_key: dict[str, str] = {}
    for symbol in wanted:
        name = canonical(symbol)
        if token_for(symbol) is None:
            logger.warning("Not listed on Kite: %s", symbol)
            continue
        by_key[symbol] = f"{exchange_for(symbol)}:{name}"
    if not by_key:
        return {}
    out: dict[str, float] = {}
    requests_list = sorted(set(by_key.values()))
    batches = 0
    failures = 0
    for batch_start in range(0, len(requests_list), 500):
        batch = requests_list[batch_start:batch_start + 500]
        batches += 1
        try:
            quoted = kite_client.quote(batch, session=live)
        except kite_client.KiteError as exc:
            if is_session_failure(exc):
                raise NoSession(f"{LOGIN_HINT} {exc}") from exc
            logger.warning("Quote failed for %d instrument(s): %s",
                           len(batch), exc)
            failures += 1
            continue
        for symbol, key in by_key.items():
            row = quoted.get(key)
            if not row:
                continue
            price = row.get("last_price")
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                out[symbol] = value
    if batches and failures == batches:
        # Every batch errored - rate limiting, most likely. Returning {}
        # here is the one outcome this function exists to prevent: it is
        # indistinguishable from a book of unlisted symbols, and the
        # planner would price the lot at zero. A partial result is fine;
        # a total failure is not, so say so.
        raise NoSession(
            f"Every one of {batches} quote request(s) to Kite failed, so no "
            f"price is known for any of {len(by_key)} symbol(s). Treating "
            f"this as no session rather than as a book of zero-value "
            f"holdings - see the warnings above for the underlying errors.")
    missing = [s for s in by_key if s not in out]
    if missing:
        logger.warning("No price returned for %d symbol(s): %s",
                       len(missing), ", ".join(missing[:10]))
    return out


def session_available() -> bool:
    """Whether a Kite session exists, without proving it still works."""
    return kite_client.load_session() is not None
