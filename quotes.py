"""Last traded prices for bare NSE symbols via one batched yfinance download."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

import config
# Reused rather than duplicated: the same MultiIndex-vs-flat column defence is
# needed here as in the sector momentum path.
from market_data import _close_series

logger = logging.getLogger(__name__)


def to_ticker(symbol: str, suffix: str = config.QUOTE_SUFFIX) -> str:
    """Map a bare NSE symbol such as GOLDBEES to a yfinance ticker."""
    cleaned = symbol.strip().upper()
    return cleaned if "." in cleaned else f"{cleaned}{suffix}"


def fetch_last_prices(symbols: list[str],
                      suffix: str = config.QUOTE_SUFFIX) -> dict[str, float]:
    """Fetch the latest close per symbol. Returns {} on any failure."""
    wanted = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    if not wanted:
        return {}
    tickers = [to_ticker(symbol, suffix) for symbol in wanted]
    try:
        data = yf.download(tickers, period="5d", interval="1d",
                           auto_adjust=True, progress=False)
    except Exception as exc:  # yfinance raises heterogeneous network errors
        logger.warning("yfinance price download failed: %s", exc)
        return {}
    if data is None or data.empty:
        logger.warning("yfinance returned no price data for %d symbol(s)",
                       len(tickers))
        return {}
    prices: dict[str, float] = {}
    for symbol, ticker in zip(wanted, tickers):
        try:
            closes = _close_series(data, ticker, tickers)
            if closes is None:
                logger.warning("No usable close price for %s (%s)", symbol, ticker)
                continue
            valid = closes.dropna()
            if valid.empty:
                logger.warning("All close prices are NaN for %s (%s)",
                               symbol, ticker)
                continue
            price = float(valid.iloc[-1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Price lookup failed for %s (%s): %s",
                           symbol, ticker, exc)
            continue
        if price <= 0.0 or pd.isna(price):
            logger.warning("Ignoring non-positive price %s for %s", price, symbol)
            continue
        prices[symbol] = price
    return prices
