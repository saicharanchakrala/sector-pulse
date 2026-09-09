"""Last traded prices for NSE symbols, from Kite.

Previously one batched yfinance download of daily closes, which meant the
"last price" was really the most recent close and up to fifteen minutes
stale during a session. Kite's quote endpoint returns the actual last traded
price, which is what a planner sizing an order should be working from.

The trade: this needs a Kite session, so it raises rather than returning an
empty dict when there is none. An empty dict would be indistinguishable from
"every symbol is unlisted" and would let the planner silently price a book
at zero.
"""
from __future__ import annotations

import logging

import market_source

logger = logging.getLogger(__name__)

# Re-exported so callers that imported it from here keep working. Symbols are
# Kite tradingsymbols now, so this is close to an identity function.
canonical = market_source.canonical


def to_ticker(symbol: str) -> str:
    """Map a symbol to the identifier Kite uses.

    Kept for callers that still call it. Kite takes bare tradingsymbols, so
    the old `.NS` suffix is stripped rather than added.
    """
    return market_source.canonical(symbol)


def fetch_last_prices(symbols: list[str], session=None) -> dict[str, float]:
    """Last traded price per symbol, keyed by the symbol as passed in.

    Keyed on the caller's exact string, upper-casing and all: the symbols go
    out normalised, then the results are mapped back onto whatever came in,
    because a caller that asked for "reliance" should not have to know it
    will be answered about "RELIANCE".

    Symbols that are unlisted or return no price are omitted and logged, so
    a caller can tell the difference between "no price for GOLDBEES" and
    "no prices at all". Raises market_source.NoSession when there is no
    Kite session, because returning {} would look like a book of unlisted
    symbols and get priced at zero.
    """
    by_normalised: dict[str, str] = {}
    for symbol in symbols:
        if symbol and symbol.strip():
            by_normalised.setdefault(symbol.strip().upper(), symbol)
    if not by_normalised:
        return {}
    priced = market_source.last_prices(list(by_normalised), session=session)
    return {by_normalised[key]: value for key, value in priced.items()
            if key in by_normalised}
