"""Intraday ETF snapshots via two batched yfinance downloads.

One daily download (period=2mo) provides previous session closes and
20-session average volumes; one intraday download (period=1d, interval=5m)
provides today's price action. Like market_data, this module degrades
gracefully: bad tickers are omitted and nothing here ever raises.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_LAST_HOUR_BARS = 12  # ~12 five-minute bars = one trading hour


@dataclass(frozen=True)
class IntradaySnapshot:
    """Point-in-time intraday state for one tradeable ETF."""

    ticker: str
    last_price: float
    asof: datetime                  # tz-aware, exchange tz as returned by yfinance
    day_change_pct: float           # last vs previous session close
    last_hour_change_pct: float     # last vs ~12 5-minute bars earlier
    range_position: float           # (last - day_low) / (day_high - day_low), 0..1
    avg_volume_20d: float           # 20-session mean daily volume


def field_series(data: pd.DataFrame, field: str, ticker: str,
                 requested_tickers: list[str]) -> "pd.Series | None":
    """Return one ticker's column for a field, handling MultiIndex vs flat frames.

    Public because the intraday scanner needs the same MultiIndex-versus-flat
    defence over High, Low and Volume. Duplicating it once produced a scanner
    that silently saw no volume on batched downloads.
    """
    columns = data.columns
    series = None
    if isinstance(columns, pd.MultiIndex):
        for key in ((field, ticker), (ticker, field)):
            if key in columns:
                series = data[key]
                break
    elif field in columns:
        if len(requested_tickers) == 1 and requested_tickers[0] == ticker:
            series = data[field]
    if series is None or not isinstance(series, pd.Series):
        return None
    return series.dropna()


def _download(tickers: list[str], **kwargs: object) -> "pd.DataFrame | None":
    """Batched yf.download that returns None instead of raising or going empty."""
    try:
        data = yf.download(tickers, auto_adjust=True, progress=False, **kwargs)
    except Exception as exc:  # yfinance raises heterogeneous network/parse errors
        logger.warning("yfinance download failed (%s): %s", kwargs, exc)
        return None
    if data is None or data.empty:
        logger.warning("yfinance returned no data (%s)", kwargs)
        return None
    return data


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


def _today_frame(intraday: pd.DataFrame, ticker: str,
                 tickers: list[str]) -> "pd.DataFrame | None":
    """Today's 5-minute Close/High/Low bars for one ticker, or None."""
    closes = field_series(intraday, "Close", ticker, tickers)
    if closes is None or closes.empty:
        return None
    tz = getattr(closes.index, "tz", None)
    today = datetime.now(tz).date() if tz is not None else datetime.now().date()
    mask = [ts.date() == today for ts in closes.index]
    if not any(mask):
        return None
    highs = field_series(intraday, "High", ticker, tickers)
    lows = field_series(intraday, "Low", ticker, tickers)
    frame = pd.DataFrame({"Close": closes})
    if highs is not None:
        frame["High"] = highs
    if lows is not None:
        frame["Low"] = lows
    return frame.loc[[ts for ts, keep in zip(closes.index, mask) if keep]]


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
    """Build intraday snapshots per ticker; {} when the market has not traded today."""
    if not tickers:
        return {}
    daily = _download(tickers, period="2mo", interval="1d")
    intraday = _download(tickers, period="1d", interval="5m")
    if intraday is None:
        return {}
    snapshots: dict[str, IntradaySnapshot] = {}
    for ticker in tickers:
        try:
            today_bars = _today_frame(intraday, ticker, tickers)
            if today_bars is None:
                logger.warning("No intraday bars for %s today; skipping", ticker)
                continue
            prev_close = None
            avg_volume = 0.0
            if daily is not None:
                closes = field_series(daily, "Close", ticker, tickers)
                today = today_bars.index[-1].date()
                prev_close = _previous_close(closes, today) if closes is not None else None
                avg_volume = _avg_volume_20d(field_series(daily, "Volume", ticker, tickers))
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
