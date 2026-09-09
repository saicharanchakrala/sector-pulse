"""Batched bar downloads for the intraday scanner.

Scanning 210 symbols one request at a time would take minutes and get rate
limited, so tickers go out in chunks and each chunk is unpacked into
per-symbol OHLCV frames. Nothing here raises: a chunk that fails is logged
and omitted, and the report says how many symbols actually returned data so
a half-empty scan cannot be mistaken for a quiet market.

This module is the seam where a real broker feed would replace yfinance.
Everything downstream consumes plain OHLCV frames, so a Kite Connect
provider only has to produce the same shape - no indicator, gate or level
would change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

import config
import indicators
from intraday import field_series

logger = logging.getLogger(__name__)

_FIELDS = ("Open", "High", "Low", "Close", "Volume")


@dataclass
class BarSet:
    """Intraday and daily frames for every symbol that returned data."""

    intraday: dict[str, pd.DataFrame]
    daily: dict[str, pd.DataFrame]
    requested: int
    failed: list[str]

    @property
    def covered(self) -> int:
        """Symbols with usable intraday bars."""
        return len(self.intraday)


def chunks(items: list[str], size: int) -> list[list[str]]:
    """Split a list into consecutive chunks of at most `size`."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    return [items[i:i + size] for i in range(0, len(items), size)]


def _download(tickers: list[str], period: str, interval: str,
              start=None, end=None) -> "pd.DataFrame | None":
    """One batched yfinance call that returns None instead of raising.

    `start`/`end` take precedence over `period` when given, which is how a
    past-date replay pulls the window around that date instead of the most
    recent one.
    """
    window = ({"start": start, "end": end} if start is not None
              else {"period": period})
    try:
        data = yf.download(tickers, interval=interval, auto_adjust=True,
                           progress=False, group_by="column", **window)
    except Exception as exc:  # yfinance raises heterogeneous network errors
        logger.warning("Download failed for %d ticker(s) at %s (%s): %s",
                       len(tickers), interval, window, exc)
        return None
    if data is None or data.empty:
        logger.warning("No %s data returned for %d ticker(s)",
                       interval, len(tickers))
        return None
    return data


def _unpack(data: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split a batched frame into one OHLCV frame per ticker.

    A ticker missing Close is dropped outright; a ticker missing Volume is
    kept, because VWAP and relative volume already fail closed on absent
    volume and the remaining gates still say something useful.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        columns: dict[str, pd.Series] = {}
        for name in _FIELDS:
            series = field_series(data, name, ticker, tickers)
            if series is not None and not series.empty:
                columns[name] = series
        if "Close" not in columns:
            continue
        frame = pd.DataFrame(columns).sort_index()
        if not frame.empty:
            out[ticker] = frame
    return out


def replay_window(target: date,
                  lookback_days: int = config.SCAN_REPLAY_LOOKBACK_DAYS
                  ) -> tuple[str, str]:
    """The (start, end) yfinance needs to cover one past session plus context.

    `end` is exclusive in yfinance, so it sits the day after the target.
    """
    start = target - timedelta(days=max(1, lookback_days))
    return start.isoformat(), (target + timedelta(days=1)).isoformat()


def fetch_bars(tickers: list[str], batch_size: int = config.SCAN_BATCH_SIZE,
               target: "date | None" = None) -> BarSet:
    """Fetch intraday and daily bars for every ticker, in batches.

    With `target` set, the window is pulled around that date instead of the
    latest sessions, which is what makes a past-date replay possible at all.
    """
    unique = list(dict.fromkeys(t for t in tickers if t))
    if not unique:
        return BarSet({}, {}, 0, [])
    start = end = None
    if target is not None:
        start, end = replay_window(target)
    intraday: dict[str, pd.DataFrame] = {}
    daily: dict[str, pd.DataFrame] = {}
    for batch in chunks(unique, batch_size):
        fine = _download(batch, config.SCAN_BAR_LOOKBACK,
                         config.SCAN_BAR_INTERVAL, start=start, end=end)
        if fine is not None:
            intraday.update(_unpack(fine, batch))
        coarse = _download(batch, config.SCAN_DAILY_LOOKBACK, "1d",
                           start=start, end=end)
        if coarse is not None:
            daily.update(_unpack(coarse, batch))
        logger.info("Fetched %d/%d symbols so far", len(intraday), len(unique))
    failed = [t for t in unique if t not in intraday]
    if failed:
        logger.warning("No intraday bars for %d symbol(s): %s",
                       len(failed), ", ".join(failed[:10]))
    return BarSet(intraday=intraday, daily=daily,
                  requested=len(unique), failed=failed)


def truncate(bars: BarSet, cutoff) -> BarSet:
    """A point-in-time view of a BarSet: nothing after `cutoff` survives.

    This is what makes a replay honest. Passing an earlier clock to the gates
    while leaving the frames whole would compute the "10am signal" from the
    last bar of the day, a full-session VWAP and an ATR that has seen the
    afternoon - every reading contaminated by the future it is supposed to
    predict. Daily bars are cut on date so a replay of an earlier session
    cannot see later sessions' closes either.

    The intraday comparison is STRICTLY less than the cutoff, and that
    matters more than it looks. A bar is stamped at its START, so the bar
    labelled 10:00 covers 10:00 to 10:05 and its close is the 10:05 price.
    Keeping it made every "10:00" entry a 10:05 entry: an independent audit
    measured 16 actionable setups against 12 at a true 10:00 cutoff, with 8
    of the 16 existing only because of that single bar and 4 genuine ones
    deleted. Excluding it leaves the 09:55 bar last, whose close IS the
    price at 10:00.
    """
    fine: dict[str, pd.DataFrame] = {}
    for ticker, frame in bars.intraday.items():
        keep = frame.loc[[ts for ts in frame.index if ts < cutoff]]
        if not keep.empty:
            fine[ticker] = keep
    coarse: dict[str, pd.DataFrame] = {}
    for ticker, frame in bars.daily.items():
        # Strictly before the cutoff DATE. The replay date's own daily bar
        # is complete by the time a replay runs, so keeping it and relying
        # on each consumer to drop it again is how a leak got in.
        keep = frame.loc[[ts for ts in frame.index if ts.date() < cutoff.date()]]
        if not keep.empty:
            coarse[ticker] = keep
    return BarSet(intraday=fine, daily=coarse, requested=bars.requested,
                  failed=[t for t in bars.failed] +
                         [t for t in bars.intraday if t not in fine])


def benchmark_change_pct(bars: BarSet, ticker: str = config.SCAN_BENCHMARK
                         ) -> "float | None":
    """Today's percent change for the relative-strength benchmark.

    None when the benchmark did not download, which makes every
    relative-strength gate fail closed rather than silently comparing every
    symbol against zero and calling a falling market strength.
    """
    frame = bars.intraday.get(ticker)
    daily = bars.daily.get(ticker)
    if frame is None or daily is None:
        return None
    today = indicators.session_bars(frame)
    if today is None:
        return None
    closes = today["Close"].dropna()
    prior = daily.dropna(subset=["Close"])
    try:
        # Same clock anchor as setups.measure, for the same reason.
        cutoff_day = closes.index[-1].date()
        prior = prior.loc[[ts for ts in prior.index
                           if ts.date() < cutoff_day]]
    except AttributeError:
        logger.warning("Benchmark daily index is not timestamped; refusing to "
                       "compare against a partial bar")
        return None
    if closes.empty or prior.empty:
        return None
    return indicators.percent_change(float(closes.iloc[-1]),
                                     float(prior["Close"].iloc[-1]))
