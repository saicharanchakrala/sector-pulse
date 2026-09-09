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
from datetime import date, timedelta

import pandas as pd

import config
import kite_client
import kite_instruments as ki

logger = logging.getLogger(__name__)

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
    if _MASTER is None or refresh:
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
    for symbol in symbols:
        path = _cache_path(symbol, resolved, start, end, oi=oi)
        if path.exists() and not refresh:
            try:
                cached = pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Unreadable cache %s: %s", path.name, exc)
            else:
                # With oi=True the column must actually be there. A frame
                # missing it is refused rather than served, because the
                # absence is invisible downstream.
                if not oi or "OpenInterest" in cached.columns:
                    out[symbol] = cached
                    continue
                logger.warning("Cache %s has no OpenInterest; refetching",
                               path.name)
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
    for symbol, token in needed:
        try:
            frame = kite_client.historical(token, start, end,
                                           interval=resolved, oi=oi,
                                           session=live)
        # ValueError comes from kite_client on a bad span, OSError from the
        # filesystem. Neither should kill a sweep of two hundred names, and
        # market_data promises never to raise on this path.
        except (kite_client.KiteError, ValueError, OSError) as exc:
            if is_session_failure(exc):
                # Every later symbol would fail the same way. Stop, and say
                # what it is, rather than logging two hundred rejections.
                _no_session(out, detail=str(exc))
                return out
            logger.warning("%s: %s", symbol, exc)
            continue
        if frame is None or frame.empty:
            logger.info("%s: no bars for %s..%s", symbol, start, end)
            continue
        try:
            frame.to_parquet(_cache_path(symbol, resolved, start, end, oi=oi))
        except Exception as exc:
            logger.warning("Could not cache %s: %s", symbol, exc)
        out[symbol] = frame
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
