"""Intraday ETF snapshots from Kite: one daily fetch and one intraday fetch.

The daily bars provide previous-session closes and 20-session average
volumes; the intraday bars provide today's price action. Like market_data,
this module degrades gracefully - a symbol that fails is omitted and nothing
here raises, including the absence of a Kite session, because the caller
treats an empty result as "market closed today" and that is a survivable
reading.

Prices are now live rather than roughly fifteen minutes late, which matters
here more than anywhere else in the project: this module exists to confirm
that today's price action backs up what the news is claiming, and a stale
quote confirms nothing.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import market_source

logger = logging.getLogger(__name__)

_LAST_HOUR_BARS = 12  # ~12 five-minute bars = one trading hour


@dataclass(frozen=True)
class IntradaySnapshot:
    """Point-in-time intraday state for one tradeable ETF."""

    ticker: str
    last_price: float
    asof: datetime                  # tz-aware, IST as returned by Kite
    day_change_pct: float           # last vs previous session close
    last_hour_change_pct: float     # last vs ~12 5-minute bars earlier
    range_position: float           # (last - day_low) / (day_high - day_low), 0..1
    avg_volume_20d: float           # 20-session mean daily volume


def _fetch_daily(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Daily bars per symbol, or {} rather than raising."""
    try:
        return market_source.daily_bars(tickers, months=2)
    except market_source.NoSession as exc:
        logger.warning("%s", exc)
        return {}
    except Exception as exc:  # network and parse failures are heterogeneous
        logger.warning("Daily fetch failed for %d symbol(s): %s",
                       len(tickers), exc)
        return {}


def _fetch_intraday(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Recent 5-minute bars per symbol, or {} rather than raising.

    Five calendar days rather than one, so a Monday scan still has Friday's
    session behind it and a holiday does not empty the frame.
    """
    try:
        return market_source.intraday_bars(tickers, days=5,
                                           interval="5minute")
    except market_source.NoSession as exc:
        logger.warning("%s", exc)
        return {}
    except Exception as exc:
        logger.warning("Intraday fetch failed for %d symbol(s): %s",
                       len(tickers), exc)
        return {}


def _previous_close(closes: pd.Series, today) -> "float | None":
    """Previous session close: skip the last row if it is today's partial bar."""
    if closes.empty:
        return None
    try:
        last_date = closes.index[-1].date()
    except AttributeError:
        last_date = None
    if last_date == today and len(closes) >= 2:
        value = float(closes.iloc[-2])
    else:
        value = float(closes.iloc[-1])
    return None if math.isnan(value) or value == 0.0 else value


def _avg_volume_20d(volumes: "pd.Series | None") -> float:
    """Mean of the most recent 20 daily volume readings (0.0 if none)."""
    if volumes is None or volumes.empty:
        return 0.0
    mean = float(volumes.tail(20).mean())
    return 0.0 if math.isnan(mean) else mean


def _today_frame(frame: "pd.DataFrame | None") -> "pd.DataFrame | None":
    """The latest session's 5-minute Close/High/Low bars, or None.

    Takes one symbol's flat frame now that Kite returns per-symbol frames,
    so the MultiIndex-versus-flat defence the yfinance route needed does not
    apply on this path.
    """
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    closes = frame["Close"].dropna()
    if closes.empty:
        return None
    # The frame's OWN last session, not the wall clock. Asking for "today"
    # returned nothing on a holiday or before the first print, which read as
    # "market closed" when the real answer was "no bars yet".
    try:
        session_day = closes.index[-1].date()
    except AttributeError:
        return None
    keep = [ts for ts in closes.index if ts.date() == session_day]
    if not keep:
        return None
    wanted = [c for c in ("Close", "High", "Low", "Volume")
              if c in frame.columns]
    return frame.loc[keep, wanted]


def _snapshot_for(ticker: str, today_bars: pd.DataFrame, prev_close: float,
                  avg_volume: float) -> "IntradaySnapshot | None":
    """Assemble one snapshot from today's bars and daily context."""
    closes = today_bars["Close"].dropna()
    if closes.empty:
        return None
    last = float(closes.iloc[-1])
    if math.isnan(last) or last <= 0.0:
        return None
    base_idx = max(0, len(closes) - 1 - _LAST_HOUR_BARS)
    hour_base = float(closes.iloc[base_idx])
    last_hour = (last / hour_base - 1.0) * 100.0 if hour_base > 0.0 else 0.0
    highs = today_bars.get("High")
    lows = today_bars.get("Low")
    day_high = float(highs.max()) if highs is not None and highs.notna().any() else float(closes.max())
    day_low = float(lows.min()) if lows is not None and lows.notna().any() else float(closes.min())
    span = day_high - day_low
    range_position = (last - day_low) / span if span > 0.0 else 0.5
    return IntradaySnapshot(
        ticker=ticker,
        last_price=last,
        asof=closes.index[-1].to_pydatetime(),
        day_change_pct=(last / prev_close - 1.0) * 100.0,
        last_hour_change_pct=last_hour,
        range_position=max(0.0, min(1.0, range_position)),
        avg_volume_20d=avg_volume,
    )


def get_intraday_snapshots(tickers: list[str]) -> dict[str, IntradaySnapshot]:
    """Build intraday snapshots per ticker; {} when nothing traded."""
    if not tickers:
        return {}
    daily = _fetch_daily(tickers)
    intraday = _fetch_intraday(tickers)
    if not intraday:
        return {}
    snapshots: dict[str, IntradaySnapshot] = {}
    for ticker in tickers:
        try:
            today_bars = _today_frame(intraday.get(ticker))
            if today_bars is None:
                logger.warning("No intraday bars for %s; skipping", ticker)
                continue
            prev_close = None
            avg_volume = 0.0
            day_frame = daily.get(ticker)
            if day_frame is not None and not day_frame.empty:
                session_day = today_bars.index[-1].date()
                closes = (day_frame["Close"].dropna()
                          if "Close" in day_frame.columns else None)
                prev_close = (_previous_close(closes, session_day)
                              if closes is not None else None)
                avg_volume = _avg_volume_20d(
                    day_frame["Volume"].dropna()
                    if "Volume" in day_frame.columns else None)
            if prev_close is None:
                logger.warning("No previous close for %s; skipping", ticker)
                continue
            entry = _snapshot_for(ticker, today_bars, prev_close, avg_volume)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Intraday snapshot failed for %s: %s", ticker, exc)
            continue
        if entry is not None:
            snapshots[ticker] = entry
    return snapshots
