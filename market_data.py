"""Sector ETF price momentum via a single batched yfinance download."""
from __future__ import annotations

import logging
import math

import pandas as pd
import yfinance as yf

import config
from models import SectorMomentum
from profiles import MarketProfile, get_profile

logger = logging.getLogger(__name__)


def close_series(data: pd.DataFrame, ticker: str,
                  requested_tickers: "list[str]") -> "pd.Series | None":
    """Return the Close series (leading NaNs trimmed) for one ticker, or None."""
    columns = data.columns
    series = None
    if isinstance(columns, pd.MultiIndex):
        # yf.download column layout depends on group_by: ("Close", ticker) by
        # default, (ticker, "Close") with group_by="ticker" — accept either.
        for key in (("Close", ticker), (ticker, "Close")):
            if key in columns:
                series = data[key]
                break
    elif "Close" in columns:
        # A flat (single-level) frame carries no ticker label, so it is only
        # trustworthy when the download was for exactly this one ticker.
        if len(requested_tickers) == 1 and requested_tickers[0] == ticker:
            series = data["Close"]
    if series is None or not isinstance(series, pd.Series):
        return None
    # Trim only leading NaNs; interior/trailing NaNs must stay in place so that
    # _window_return can skip windows per the spec instead of silently
    # compacting the series onto older closes.
    first_valid = series.first_valid_index()
    if first_valid is None:
        return None
    return series.loc[first_valid:]


def _window_return(closes: pd.Series, trading_days: int) -> "float | None":
    """Percent return over the trailing N trading days, or None if not computable."""
    if trading_days < 1 or len(closes) < trading_days + 1:
        return None
    last = float(closes.iloc[-1])
    base = float(closes.iloc[-(trading_days + 1)])
    if math.isnan(last) or math.isnan(base) or base == 0.0:
        return None
    return (last / base - 1.0) * 100.0


def _momentum_for(sector: str, etf: str, closes: pd.Series) -> "SectorMomentum | None":
    """Build a SectorMomentum from a Close series, or None if no window is usable."""
    returns: dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for label, (weight, scale_pct) in config.MOMENTUM_WINDOWS.items():
        try:
            trading_days = int(label.rstrip("dD"))
        except ValueError:
            logger.warning("Skipping unparseable momentum window label %s", label)
            continue
        pct_return = _window_return(closes, trading_days)
        if pct_return is None:
            continue
        returns[label] = round(pct_return, 2)
        weighted_sum += weight * math.tanh(pct_return / scale_pct)
        weight_sum += weight
    if not returns or weight_sum == 0.0:
        return None
    return SectorMomentum(sector=sector, etf=etf, returns=returns,
                          score=weighted_sum / weight_sum)


def get_sector_momentum(
    profile: MarketProfile | None = None,
) -> dict[str, SectorMomentum]:
    """Compute weighted tanh-squashed multi-window momentum per sector ticker."""
    resolved = get_profile() if profile is None else profile
    tickers = [sector_def.etf for sector_def in resolved.sectors.values()]
    try:
        data = yf.download(tickers, period="6mo", interval="1d",
                           auto_adjust=True, progress=False)
    except Exception as exc:  # yfinance raises heterogeneous network/parse errors
        logger.warning("yfinance download failed: %s", exc)
        return {}
    if data is None or data.empty:
        logger.warning("yfinance returned no price data for sector ETFs")
        return {}
    momentum: dict[str, SectorMomentum] = {}
    for name, sector_def in resolved.sectors.items():
        try:
            closes = close_series(data, sector_def.etf, tickers)
            if closes is None:
                logger.warning("No usable close prices for %s (%s)",
                               name, sector_def.etf)
                continue
            entry = _momentum_for(name, sector_def.etf, closes)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Momentum computation failed for %s (%s): %s",
                           name, sector_def.etf, exc)
            continue
        if entry is None:
            logger.warning("Insufficient price history for %s (%s)",
                           name, sector_def.etf)
            continue
        momentum[name] = entry
    return momentum
