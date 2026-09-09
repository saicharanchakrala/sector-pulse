"""NSE option chain fetch and contract-quality assessment.

An honest boundary, stated up front: this module does not have a directional
view on options. Direction comes from the underlying's equity setup in
setups.py. What it does is answer a different question that decides whether
that view is tradeable at all - is there a contract with enough open
interest, enough volume and a tight enough spread that the round trip does
not eat the move?

That distinction matters because most option "signals" are really the
underlying's signal plus a contract choice, and the contract choice is where
retail loses money quietly: a 6-rupee premium with a 0.40 spread has given
up 6.7% before the underlying moves at all.

Chain metrics like put-call ratio and max pain are reported because traders
ask for them. Both are widely used and neither has a robust published edge;
they are descriptive context, labelled as such.
"""
from __future__ import annotations

import logging
import math
import urllib.parse
from dataclasses import dataclass

import requests

import config
import trade_costs
from levels import LONG, SHORT

logger = logging.getLogger(__name__)

CALL = "CE"
PUT = "PE"

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": config.NSE_BASE_URL + "/option-chain",
}


def kite_tradingsymbol(underlying: str, expiry: str, strike: float,
                       side: str) -> "str | None":
    """The exact Kite tradingsymbol for one contract, or None if unlisted.

    Resolved from the instrument master, not assembled from a naming rule.
    Kite spells monthly and weekly expiries differently, so a constructed
    symbol can look right and not exist - and a symbol that does not exist
    is exactly what must not reach a user about to place an order.

    `expiry` is NSE's spelling from the chain (for example 29-Sep-2026);
    it is parsed rather than compared as text.
    """
    import pandas as pd

    import kite_instruments as ki

    try:
        wanted_expiry = pd.to_datetime(expiry, dayfirst=True).date()
    except Exception:
        return None
    wanted_type = "CE" if str(side).upper().startswith("C") else "PE"
    try:
        rows = ki.nse_options(ki.fetch_master(), underlying)
    except Exception:
        return None
    for contract in rows:
        if (contract.instrument_type == wanted_type
                and contract.expiry == wanted_expiry
                and abs(contract.strike - float(strike)) < 0.01):
            return contract.tradingsymbol
    return None


@dataclass(frozen=True)
class Contract:
    """One strike and side of the chain, as NSE reports it."""

    symbol: str
    expiry: str
    strike: float
    side: str                       # CALL or PUT
    last_price: float
    bid: float
    ask: float
    open_interest: int
    oi_change: int
    volume: int
    implied_volatility: float

    @property
    def mid(self) -> float:
        """Mid price when both sides are quoted, else the last traded price."""
        if self.bid > 0.0 and self.ask > 0.0:
            return (self.bid + self.ask) / 2.0
        return self.last_price

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread; 0.0 when the book is one-sided."""
        if self.bid <= 0.0 or self.ask <= 0.0:
            return 0.0
        return max(0.0, self.ask - self.bid)

    @property
    def spread_pct(self) -> "float | None":
        """Spread as a percent of mid, or None when it cannot be measured.

        None rather than 0.0 for an unquoted book: a missing spread is not a
        tight spread, and treating it as zero would let the least liquid
        contracts through the gate that exists to catch them.
        """
        if self.bid <= 0.0 or self.ask <= 0.0 or self.mid <= 0.0:
            return None
        return self.spread / self.mid * 100.0


@dataclass(frozen=True)
class ChainMetrics:
    """Descriptive summary of one expiry's chain."""

    symbol: str
    expiry: str
    underlying: float
    atm_strike: float
    call_oi: int
    put_oi: int
    atm_call_iv: float
    atm_put_iv: float
    max_pain: "float | None"

    @property
    def put_call_ratio(self) -> "float | None":
        """Put OI over call OI across the expiry, or None without call OI."""
        if self.call_oi <= 0:
            return None
        return self.put_oi / self.call_oi


def open_session() -> requests.Session:
    """A session carrying the cookies NSE's API requires.

    NSE serves an empty body to callers that have not first loaded the
    option-chain page, so the bootstrap request is mandatory rather than
    polite.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    try:
        session.get(config.NSE_BASE_URL + "/option-chain",
                    timeout=config.NSE_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("NSE bootstrap failed: %s", exc)
    return session


def _get_json(session: requests.Session, path: str) -> "dict | None":
    """GET one NSE JSON endpoint, returning None on any failure or empty body."""
    try:
        response = session.get(config.NSE_BASE_URL + path,
                               timeout=config.NSE_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("NSE request failed (%s): %s", path, exc)
        return None
    if response.status_code != 200:
        logger.warning("NSE returned HTTP %d for %s", response.status_code, path)
        return None
    if len(response.content) < 50:
        logger.warning("NSE returned an empty body for %s (%d bytes); the "
                       "session cookie was probably rejected",
                       path, len(response.content))
        return None
    try:
        return response.json()
    except ValueError as exc:
        logger.warning("NSE returned non-JSON for %s: %s", path, exc)
        return None


def fetch_expiries(symbol: str,
                   session: "requests.Session | None" = None) -> list[str]:
    """Expiry dates for one underlying, nearest first; [] on failure.

    The contract-info endpoint does not distinguish indices from equities,
    so no instrument type is needed here.
    """
    live = session if session is not None else open_session()
    payload = _get_json(live, f"/api/option-chain-contract-info?symbol="
                              f"{urllib.parse.quote(symbol)}")
    if not payload:
        return []
    dates = payload.get("expiryDates") or []
    return [str(value) for value in dates if value]


def _to_contract(symbol: str, expiry: str, strike: float, side: str,
                 raw: dict) -> "Contract | None":
    """Build a Contract from one NSE chain leg, or None if it is unusable."""
    try:
        last = float(raw.get("lastPrice") or 0.0)
        return Contract(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            side=side,
            last_price=last,
            bid=float(raw.get("buyPrice1") or 0.0),
            ask=float(raw.get("sellPrice1") or 0.0),
            open_interest=int(float(raw.get("openInterest") or 0)),
            oi_change=int(float(raw.get("changeinOpenInterest") or 0)),
            volume=int(float(raw.get("totalTradedVolume") or 0)),
            implied_volatility=float(raw.get("impliedVolatility") or 0.0),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Unparseable %s leg at %s: %s", side, strike, exc)
        return None


def fetch_chain(symbol: str, is_index: bool, expiry: "str | None" = None,
                session: "requests.Session | None" = None
                ) -> tuple[list[Contract], "float | None", str]:
    """Contracts, underlying price and the expiry actually used.

    NSE requires the expiry parameter; without it the endpoint answers 200
    with an empty object, which is why the expiry is resolved first.
    """
    live = session if session is not None else open_session()
    chosen = expiry
    if not chosen:
        expiries = fetch_expiries(symbol, live)
        if not expiries:
            return [], None, ""
        chosen = expiries[0]
    kind = "Indices" if is_index else "Equity"
    path = (f"/api/option-chain-v3?type={kind}"
            f"&symbol={urllib.parse.quote(symbol)}"
            f"&expiry={urllib.parse.quote(chosen)}")
    payload = _get_json(live, path)
    if not payload:
        return [], None, chosen
    records = payload.get("records", payload)
    rows = records.get("data") or []
    underlying = records.get("underlyingValue")
    try:
        spot = float(underlying) if underlying is not None else None
    except (TypeError, ValueError):
        spot = None
    contracts: list[Contract] = []
    for row in rows:
        try:
            strike = float(row.get("strikePrice"))
        except (TypeError, ValueError):
            continue
        for key, side in ((CALL, CALL), (PUT, PUT)):
            leg = row.get(key)
            if isinstance(leg, dict):
                built = _to_contract(symbol, chosen, strike, side, leg)
                if built is not None:
                    contracts.append(built)
    return contracts, spot, chosen


def atm_strike(contracts: list[Contract], spot: float) -> "float | None":
    """Strike closest to spot."""
    strikes = sorted({c.strike for c in contracts})
    if not strikes or spot <= 0.0:
        return None
    return min(strikes, key=lambda k: abs(k - spot))


def max_pain(contracts: list[Contract]) -> "float | None":
    """Strike where total option-writer payout is smallest.

    Reported because it is asked for. It is a snapshot of current open
    interest, moves as OI moves, and has no established predictive record.
    """
    strikes = sorted({c.strike for c in contracts})
    if not strikes:
        return None
    calls = {c.strike: c.open_interest for c in contracts if c.side == CALL}
    puts = {c.strike: c.open_interest for c in contracts if c.side == PUT}
    best_strike, best_pain = None, math.inf
    for settle in strikes:
        pain = 0.0
        for strike in strikes:
            if settle > strike:
                pain += calls.get(strike, 0) * (settle - strike)
            if settle < strike:
                pain += puts.get(strike, 0) * (strike - settle)
        if pain < best_pain:
            best_strike, best_pain = settle, pain
    return best_strike


def summarise(symbol: str, contracts: list[Contract],
              spot: "float | None", expiry: str) -> "ChainMetrics | None":
    """Roll one expiry's chain into its descriptive metrics."""
    if not contracts or spot is None or spot <= 0.0:
        return None
    strike = atm_strike(contracts, spot)
    if strike is None:
        return None
    at_money = {c.side: c for c in contracts if c.strike == strike}
    return ChainMetrics(
        symbol=symbol,
        expiry=expiry,
        underlying=spot,
        atm_strike=strike,
        call_oi=sum(c.open_interest for c in contracts if c.side == CALL),
        put_oi=sum(c.open_interest for c in contracts if c.side == PUT),
        atm_call_iv=at_money[CALL].implied_volatility if CALL in at_money else 0.0,
        atm_put_iv=at_money[PUT].implied_volatility if PUT in at_money else 0.0,
        max_pain=max_pain(contracts),
    )


def quality_reasons(contract: Contract, lots: int,
                    lot_size: int) -> tuple[bool, list[str]]:
    """Whether a contract is tradeable, with one PASS/FAIL line per gate."""
    reasons: list[str] = []
    spread = contract.spread_pct
    if spread is None:
        spread_ok = False
        reasons.append("no two-sided quote, so the spread cannot be measured "
                       "[FAIL]")
    else:
        spread_ok = spread <= config.OPT_MAX_SPREAD_PCT
        reasons.append(f"spread {spread:.2f}% of mid vs ceiling "
                       f"{config.OPT_MAX_SPREAD_PCT:.2f}% "
                       f"[{'PASS' if spread_ok else 'FAIL'}]")
    oi_ok = contract.open_interest >= config.OPT_MIN_OPEN_INTEREST
    reasons.append(f"open interest {contract.open_interest:,} vs floor "
                   f"{config.OPT_MIN_OPEN_INTEREST:,} "
                   f"[{'PASS' if oi_ok else 'FAIL'}]")
    volume_ok = contract.volume >= config.OPT_MIN_VOLUME
    reasons.append(f"volume {contract.volume:,} vs floor "
                   f"{config.OPT_MIN_VOLUME:,} "
                   f"[{'PASS' if volume_ok else 'FAIL'}]")
    premium = contract.mid
    breakeven = trade_costs.options_breakeven_pct(premium, lots, lot_size)
    total_friction = breakeven + (spread if spread is not None else 0.0)
    reasons.append(f"round-trip cost {breakeven:.2f}% of premium, "
                   f"{total_friction:.2f}% including the spread [INFO]")
    return bool(spread_ok and oi_ok and volume_ok), reasons


def pick_contract(contracts: list[Contract], spot: float, direction: str,
                  lot_size: int, lots: int = 1
                  ) -> "tuple[Contract | None, list[str]]":
    """Cheapest tradeable near-the-money contract for a direction.

    A long view buys calls and a short view buys puts. Only buying is
    considered: selling naked options carries open-ended loss, which no
    scanner should quietly propose. Candidates run from at-the-money out to
    OPT_STRIKES_EITHER_SIDE, and the first that clears every quality gate
    wins, which biases toward at-the-money where the book is deepest.
    """
    if direction not in (LONG, SHORT):
        return None, [f"unknown direction {direction!r}"]
    side = CALL if direction == LONG else PUT
    strike = atm_strike(contracts, spot)
    if strike is None:
        return None, ["no strikes in the chain"]
    pool = sorted((c for c in contracts if c.side == side),
                  key=lambda c: abs(c.strike - strike))
    window = pool[:max(1, config.OPT_STRIKES_EITHER_SIDE * 2 + 1)]
    if not window:
        return None, [f"no {side} contracts in the chain"]
    rejected: list[str] = []
    for candidate in window:
        ok, reasons = quality_reasons(candidate, lots, lot_size)
        if ok:
            return candidate, reasons
        rejected.append(f"{candidate.strike:.0f} {side}: "
                        + "; ".join(r for r in reasons if "[FAIL]" in r))
    return None, ["no contract cleared the quality gates"] + rejected[:5]
