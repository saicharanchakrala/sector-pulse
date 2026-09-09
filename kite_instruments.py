"""Zerodha's public instrument master: every tradeable contract, no auth.

This replaces a gap I had reported as unfillable. NSE's own endpoints serve
no bulk per-contract futures data - the derivatives bhavcopy 404s on every
published URL and /api/quote-derivative has been withdrawn - so futures
activity had to be inferred from per-underlying turnover. Zerodha publishes
the whole contract master as one CSV that needs no API key and no
subscription, and it contains what NSE would not give:

    647 futures contracts   216 underlyings across three expiries
 32,437 option contracts    every strike and expiry, both sides
 10,111 NSE cash equities
    236 indices, which carry instrument_type EQ and exchange NSE and are
        separated from equities by SEGMENT alone

Each row carries the instrument_token, which is the identifier Kite's
authenticated historical and WebSocket APIs key off. So fetching this now
also lays the groundwork for a live feed later without committing to one.

The boundary is worth stating precisely, because it is easy to over-read:
this is the contract UNIVERSE, not prices. What each contract last traded
at, its open interest, its historical candles - all of that needs a Kite
Connect subscription and the user's own login. Nothing here authenticates,
and nothing here should ever hold a credential.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime

import requests

logger = logging.getLogger(__name__)

MASTER_URL = "https://api.kite.trade/instruments"

EXCHANGE_NSE = "NSE"          # cash equities
SEGMENT_NSE_CASH = "NSE"      # segment, which unlike exchange excludes indices
SEGMENT_INDICES = "INDICES"
EXCHANGE_NFO = "NFO"          # NSE futures and options
TYPE_FUTURE = "FUT"
TYPE_CALL = "CE"
TYPE_PUT = "PE"
TYPE_EQUITY = "EQ"

_HEADERS = {"User-Agent": "sector-pulse/1.0 (educational market scanner)"}


@dataclass(frozen=True)
class Contract:
    """One row of the instrument master."""

    instrument_token: int
    tradingsymbol: str
    name: str                    # the underlying, for derivatives
    exchange: str
    segment: str
    instrument_type: str
    expiry: "date | None"
    strike: float
    lot_size: int
    tick_size: float

    @property
    def is_future(self) -> bool:
        """Whether this is a futures contract."""
        return self.instrument_type == TYPE_FUTURE

    @property
    def is_option(self) -> bool:
        """Whether this is a call or a put."""
        return self.instrument_type in (TYPE_CALL, TYPE_PUT)

    @property
    def days_to_expiry(self) -> "int | None":
        """Calendar days until expiry, or None for cash instruments.

        Needs a reference date from the caller in tests; uses today here
        because the value is only ever read for reporting.
        """
        if self.expiry is None:
            return None
        return (self.expiry - datetime.now().date()).days


def _to_int(value: str, default: int = 0) -> int:
    """Parse an integer field, defaulting rather than raising."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float = 0.0) -> float:
    """Parse a float field, defaulting rather than raising."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_date(value: str) -> "date | None":
    """Parse an ISO expiry, or None when the field is blank."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_master(text: str) -> list[Contract]:
    """Parse the instrument master CSV. Pure: no network.

    Rows that do not parse are skipped rather than raising: the file is
    third-party data and one malformed row should not lose the other
    hundred thousand.
    """
    rows: list[Contract] = []
    for row in csv.DictReader(io.StringIO(text)):
        token = _to_int(row.get("instrument_token"))
        symbol = str(row.get("tradingsymbol") or "").strip().upper()
        if not token or not symbol:
            continue
        rows.append(Contract(
            instrument_token=token,
            tradingsymbol=symbol,
            name=str(row.get("name") or "").strip().upper(),
            exchange=str(row.get("exchange") or "").strip().upper(),
            segment=str(row.get("segment") or "").strip().upper(),
            instrument_type=str(row.get("instrument_type") or "").strip().upper(),
            expiry=_to_date(row.get("expiry")),
            strike=_to_float(row.get("strike")),
            lot_size=_to_int(row.get("lot_size")),
            tick_size=_to_float(row.get("tick_size")),
        ))
    return rows


def fetch_master(timeout: int = 60) -> list[Contract]:
    """Download and parse the instrument master; [] on any failure."""
    try:
        response = requests.get(MASTER_URL, headers=_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch the Kite instrument master: %s", exc)
        return []
    contracts = parse_master(response.text)
    if not contracts:
        logger.warning("Instrument master parsed to nothing (%d bytes)",
                       len(response.content))
    return contracts


def nse_futures(contracts: list[Contract]) -> list[Contract]:
    """Every NSE futures contract."""
    return [c for c in contracts
            if c.exchange == EXCHANGE_NFO and c.is_future]


def nse_options(contracts: list[Contract],
                underlying: "str | None" = None) -> list[Contract]:
    """Every NSE option contract, optionally for one underlying."""
    wanted = (underlying or "").strip().upper()
    return [c for c in contracts
            if c.exchange == EXCHANGE_NFO and c.is_option
            and (not wanted or c.name == wanted)]


def nse_equities(contracts: list[Contract]) -> list[Contract]:
    """Every NSE cash equity instrument.

    Filtered on SEGMENT, not exchange. Indices carry instrument_type EQ and
    exchange NSE, differing only in segment, so filtering on exchange
    returned 136 of them as tradeable equities - NIFTY 50 and NIFTY BANK
    among them - and any caller sizing a position from that list would have
    been pricing an index it cannot buy.
    """
    return [c for c in contracts
            if c.segment == SEGMENT_NSE_CASH and c.instrument_type == TYPE_EQUITY]


def indices(contracts: list[Contract]) -> list[Contract]:
    """Every index instrument, which cannot be traded as cash equity."""
    return [c for c in contracts if c.segment == SEGMENT_INDICES]


def lot_sizes(contracts: list[Contract]) -> dict[str, int]:
    """Underlying to lot size, taken from the nearest futures expiry.

    Futures rather than options because both carry the same lot and the
    futures list is two orders of magnitude smaller to scan.
    """
    out: dict[str, int] = {}
    for contract in sorted(nse_futures(contracts),
                           key=lambda c: (c.expiry or date.max)):
        if contract.name and contract.name not in out and contract.lot_size > 0:
            out[contract.name] = contract.lot_size
    return out


def expiries(contracts: list[Contract], underlying: str) -> list[date]:
    """Sorted option expiries available for one underlying."""
    wanted = underlying.strip().upper()
    found = {c.expiry for c in contracts
             if c.name == wanted and c.is_option and c.expiry is not None}
    return sorted(found)


def strikes(contracts: list[Contract], underlying: str,
            expiry: "date | None" = None) -> list[float]:
    """Sorted strikes for one underlying, optionally one expiry."""
    wanted = underlying.strip().upper()
    found = {c.strike for c in contracts
             if c.name == wanted and c.is_option and c.strike > 0
             and (expiry is None or c.expiry == expiry)}
    return sorted(found)


def underlyings_with_derivatives(contracts: list[Contract]) -> set[str]:
    """Names that carry futures or options."""
    return {c.name for c in contracts
            if c.exchange == EXCHANGE_NFO and c.name}


def summary(contracts: list[Contract]) -> dict[str, int]:
    """Headline counts for a discovery report."""
    futures = nse_futures(contracts)
    options = nse_options(contracts)
    return {
        "total": len(contracts),
        "nse_equities": len(nse_equities(contracts)),
        "nse_futures": len(futures),
        "nse_options": len(options),
        "calls": sum(1 for c in options if c.instrument_type == TYPE_CALL),
        "puts": sum(1 for c in options if c.instrument_type == TYPE_PUT),
        "fo_underlyings": len(underlyings_with_derivatives(contracts)),
        "futures_expiries": len({c.expiry for c in futures if c.expiry}),
    }
