"""Sector ETF price momentum from daily Kite bars."""
from __future__ import annotations

import logging
import math

import pandas as pd

import config
import market_source
from models import SectorMomentum
from profiles import MarketProfile, get_profile

logger = logging.getLogger(__name__)


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
    # Prefer the tradeable ETF over the sector index. You buy the ETF, so its
    # own history is the relevant one, and it carries its own premium and
    # tracking error. The original reason was a yfinance data gap - 34 bars
    # for the ETFs against one for most indices - which no longer applies on
    # Kite, but the first reason stands on its own.
    tickers = [resolved.trade_etfs.get(name) or sector_def.etf
               for name, sector_def in resolved.sectors.items()]
    try:
        frames = market_source.daily_bars(tickers, months=6)
    except market_source.NoSession as exc:
        logger.warning("%s", exc)
        return {}
    if not frames:
        logger.warning("No price data returned for %d sector ticker(s)",
                       len(tickers))
        return {}
    momentum: dict[str, SectorMomentum] = {}
    for name, sector_def in resolved.sectors.items():
        ticker = resolved.trade_etfs.get(name) or sector_def.etf
        try:
            frame = frames.get(ticker)
            if frame is None or "Close" not in frame.columns:
                logger.warning("No usable close prices for %s (%s)",
                               name, ticker)
                continue
            closes = frame["Close"]
            entry = _momentum_for(name, ticker, closes)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Momentum computation failed for %s (%s): %s",
                           name, ticker, exc)
            continue
        if entry is None:
            logger.warning("Insufficient price history for %s (%s)",
                           name, ticker)
            continue
        momentum[name] = entry
    return momentum
