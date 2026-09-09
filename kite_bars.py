"""Bulk historical bars from Kite, cached to parquet for the edge harness.

The point of this module is sample size. yfinance serves eight days of
1-minute bars and about sixty of 5-minute, which left the edge search on 59
sessions whose signals are correlated within each day - enough to diagnose a
broken rule, not enough to confirm a working one. Kite serves years, so the
same harness can run on a sample that supports a conclusion.

Frames come back with the Open/High/Low/Close/Volume column names the rest
of this project already uses, so a Kite frame drops into edge_lab without
translation. With `oi=True` an OpenInterest column comes too, which is the
field NSE never publishes historically and which every past-date replay so
far had to discard as lookahead.

Requests are paced in kite_client at Kite's documented 3 per second, and
every symbol is cached after its first fetch, so a re-run costs nothing.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

import config
import kite_client
import kite_instruments as ki

logger = logging.getLogger(__name__)

CACHE_DIR = config.PROJECT_ROOT / "bar_cache"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(symbol: str, interval: str, start: date, end: date):
    """Where one symbol's bars for one span are cached."""
    safe = symbol.replace("/", "_").replace(":", "_").replace(" ", "_")
    return (CACHE_DIR /
            f"kite__{safe}__{interval}__{start:%Y%m%d}_{end:%Y%m%d}.parquet")


def token_map(contracts: "list | None" = None) -> dict[str, int]:
    """NSE cash symbol to Kite instrument token."""
    rows = contracts if contracts is not None else ki.fetch_master()
    return {c.tradingsymbol: c.instrument_token for c in ki.nse_equities(rows)}


def index_tokens(contracts: "list | None" = None) -> dict[str, int]:
    """Index name to instrument token, for benchmark series."""
    rows = contracts if contracts is not None else ki.fetch_master()
    return {c.tradingsymbol: c.instrument_token for c in ki.indices(rows)}


def fetch_symbol(symbol: str, instrument_token: int, start: date, end: date,
                 interval: str = "5minute", oi: bool = False,
                 refresh: bool = False,
                 session=None) -> "pd.DataFrame | None":
    """One symbol's bars, from cache when available."""
    path = _cache_path(symbol, interval, start, end)
    if path.exists() and not refresh:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Unreadable cache %s: %s", path.name, exc)
    try:
        frame = kite_client.historical(instrument_token, start, end,
                                       interval=interval, oi=oi,
                                       session=session)
    except kite_client.KiteError as exc:
        logger.warning("%s: %s", symbol, exc)
        return None
    if frame is None or frame.empty:
        logger.info("%s: no bars for %s..%s", symbol, start, end)
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path)
    except Exception as exc:
        logger.warning("Could not cache %s: %s", symbol, exc)
    return frame


def load_bars(symbols: list[str], start: date, end: date,
              interval: str = "5minute", oi: bool = False,
              refresh: bool = False, progress=None,
              contracts: "list | None" = None,
              workers: int = 6) -> dict[str, pd.DataFrame]:
    """Bars for many symbols, cached per symbol.

    `progress(done, total, symbol, cached)` is called per symbol so a long
    sweep can report where it is. Symbols with no token or no data are
    omitted and logged rather than raising: a sweep of two hundred names
    should not die on one delisting.
    """
    rows = contracts if contracts is not None else ki.fetch_master()
    tokens = token_map(rows)
    tokens.update(index_tokens(rows))
    session = kite_client.load_session()
    if session is None:
        raise kite_client.KiteError("No Kite session. Run kite_login first.")
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    total = len(symbols)
    # Fetches run concurrently while kite_client._pace still serialises
    # request STARTS at Kite's documented 3 per second. Sequentially each
    # symbol cost about 10 seconds for four chunked requests - almost all of
    # it round-trip latency rather than rate limiting - which projected to
    # 36 minutes for the F&O list. Overlapping the waiting fixes that
    # without ever exceeding the limit.
    wanted: list[tuple[str, int]] = []
    for symbol in symbols:
        token = tokens.get(symbol.strip().upper())
        if token:
            wanted.append((symbol, token))
        else:
            missing.append(symbol)
    done = 0
    lock = threading.Lock()

    def one(pair):
        nonlocal done
        symbol, token = pair
        was_cached = _cache_path(symbol, interval, start, end).exists()
        frame = fetch_symbol(symbol, token, start, end, interval=interval,
                             oi=oi, refresh=refresh, session=session)
        with lock:
            done += 1
            if progress:
                progress(done, total, symbol, was_cached)
        return symbol, frame

    if wanted:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for symbol, frame in pool.map(one, wanted):
                if frame is not None and not frame.empty:
                    out[symbol] = frame
    if missing:
        logger.warning("No Kite instrument token for %d symbol(s): %s",
                       len(missing), ", ".join(missing[:10]))
    return out


def benchmark_day_changes(index_symbol: str, start: date, end: date,
                          interval: str = "5minute",
                          contracts: "list | None" = None) -> dict:
    """Per-session percent change of an index, for relative strength.

    Uses each session's last close against the prior session's, which is
    what the live scanner compares against.
    """
    rows = contracts if contracts is not None else ki.fetch_master()
    token = index_tokens(rows).get(index_symbol.strip().upper())
    if not token:
        raise ValueError(f"No index token for {index_symbol!r}")
    frame = fetch_symbol(index_symbol, token, start, end, interval=interval)
    if frame is None or frame.empty:
        return {}
    closes = frame["Close"].dropna()
    by_day: dict = {}
    for stamp, value in closes.items():
        by_day[stamp.date()] = float(value)
    days = sorted(by_day)
    return {day: (by_day[day] / by_day[days[i - 1]] - 1.0) * 100.0
            for i, day in enumerate(days) if i > 0 and by_day[days[i - 1]] > 0}


def default_span(months: int = 12) -> tuple[date, date]:
    """A span ending today, `months` back, without needing a clock in tests."""
    end = date.today()
    return end - timedelta(days=int(30.44 * months)), end
