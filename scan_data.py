"""Bar loading for the intraday scanner, on top of the Kite source.

This module was the seam where a real broker feed would replace yfinance,
and that has now happened: market_source fetches from Kite and hands back
one flat OHLCV frame per symbol. The claim that nothing downstream would
change held - no indicator, gate or level was touched by the swap.

Nothing here raises. A symbol that fails is logged and omitted, and the
report says how many actually returned data, so a half-empty scan cannot be
mistaken for a quiet market. The one thing that DOES now raise upstream is
the absence of a Kite session, and market_source raises rather than
returning empty precisely so it cannot be mistaken for a quiet market
either.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

import config
import indicators
import market_source
logger = logging.getLogger(__name__)


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


def _fetch(symbols: list[str], start: date, end: date,
           interval: str) -> dict[str, pd.DataFrame]:
    """Bars per symbol from Kite, or {} rather than raising.

    market_source returns one flat frame per symbol, so the MultiIndex
    unpacking the yfinance path needed is gone from this route entirely.
    """
    try:
        return market_source.bars(symbols, start, end, interval=interval)
    except market_source.NoSession as exc:
        logger.warning("%s", exc)
        return {}
    except Exception as exc:  # network and parse failures are heterogeneous
        logger.warning("Fetch failed for %d symbol(s) at %s: %s",
                       len(symbols), interval, exc)
        return {}


def replay_window(target: date,
                  lookback_days: int = config.SCAN_REPLAY_LOOKBACK_DAYS
                  ) -> tuple[date, date]:
    """The (start, end) span covering one past session plus its context.

    Both ends are inclusive, which differs from the yfinance route this
    replaced: Kite treats `to` as inclusive, so the end is the target date
    itself rather than the day after it.
    """
    start = target - timedelta(days=max(1, lookback_days))
    return start, target


def _lookback_days(spec: str, default: int) -> int:
    """Calendar days from a '10d' or '3mo' style span.

    Delegates so this and edge_lab cannot drift apart; see
    market_source.calendar_days for why "10d" is not ten days.
    """
    return market_source.calendar_days(spec, default)


def fetch_bars(tickers: list[str], batch_size: int = config.SCAN_BATCH_SIZE,
               target: "date | None" = None) -> BarSet:
    """Fetch intraday and daily bars for every symbol from Kite.

    With `target` set, the window ends at that date instead of today, which
    is what makes a past-date replay possible at all.

    `batch_size` is retained for callers that still pass it but no longer
    batches anything: Kite is queried per instrument and paced centrally at
    its documented rate, so grouping symbols buys nothing. It is kept rather
    than removed to avoid breaking a caller for no gain.
    """
    unique = list(dict.fromkeys(t for t in tickers if t))
    if not unique:
        return BarSet({}, {}, 0, [])
    end = target or date.today()
    fine_days = _lookback_days(config.SCAN_BAR_LOOKBACK, 10)
    coarse_days = _lookback_days(config.SCAN_DAILY_LOOKBACK, 92)
    if target is not None:
        # One definition of the replay window, not two. This used to compute
        # its own span inline while replay_window sat unused beside it under
        # a different rule.
        fine_start, end = replay_window(target, max(
            fine_days, config.SCAN_REPLAY_LOOKBACK_DAYS))
        fine_days = (end - fine_start).days

    intraday = _fetch(unique, end - timedelta(days=fine_days), end,
                      config.SCAN_BAR_INTERVAL)
    daily = _fetch(unique, end - timedelta(days=coarse_days), end, "day")
    logger.info("Fetched %d/%d symbols", len(intraday), len(unique))
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
