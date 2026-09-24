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

A picked contract is SIZED, not suggested one lot at a time: a bought
option can expire worthless, so its premium plus charges is the loss, and
the lot count comes from the same per-trade risk budget the equity sizer
uses (levels.risk_budget). When not one lot fits, nothing is suggested.
"""
from __future__ import annotations

import logging
import math
import urllib.parse
from dataclasses import dataclass, field

import requests

import config
import trade_costs
from levels import LONG, SHORT, risk_budget

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


# Same role as levels._SIZING_MAX_STEPS: a bound so an unforeseen input
# fails closed, not a knob. The per-lot loss has one fixed term (the flat
# 20-rupee brokerage per order), so the walk settles in a step or two.
_LOT_SIZING_MAX_STEPS = 64


def option_loss(premium: float, lots: int, lot_size: int) -> float:
    """Worst case of BUYING `lots` lots at `premium`: all of it, plus charges.

    A bought option can expire worthless, so the whole premium is the
    loss at the "stop" - there is no nearer one. The round-trip charges
    are priced at a flat exit, the conservative figure: selling near zero
    carries less STT and exchange charge than selling at the entry
    premium, so the real worst case sits a little below this.
    """
    quantity = lots * lot_size
    if quantity <= 0 or not math.isfinite(premium) or premium <= 0.0:
        return 0.0
    charges = trade_costs.options_cost(premium, premium, lots, lot_size).total
    return premium * quantity + charges


def lots_within_budget(premium: float, lot_size: int, budget: float) -> int:
    """Most lots whose worst-case loss, charges included, fits the budget.

    floor(budget / (premium * lot_size + per-lot charges)), except that
    the charges are not a fixed amount per lot - brokerage is a flat 20
    rupees an ORDER however many lots it carries - so it is solved the
    same way levels._quantity_inside_budget solves shares: start from the
    premium-only count, which charges can only reduce, and step down from
    the per-lot loss measured at each guess. 0 means not one lot fits.
    """
    if not (math.isfinite(premium) and premium > 0.0 and lot_size > 0
            and math.isfinite(budget) and budget > 0.0):
        return 0
    lots = int(budget // (premium * lot_size))
    for _ in range(_LOT_SIZING_MAX_STEPS):
        if lots <= 0:
            return 0
        loss = option_loss(premium, lots, lot_size)
        if loss <= budget:
            return lots
        lots = min(lots - 1, int(budget // (loss / lots)))
    logger.warning("Lot sizing did not settle (premium %s, lot %s, budget "
                   "%s); refusing rather than guessing", premium, lot_size,
                   budget)
    return 0


@dataclass(frozen=True)
class OptionPick:
    """The contract chosen for one setup, sized against the risk budget.

    `contract` is None when nothing is suggested; `reasons[0]` then says
    why in one sentence and `refused_by_budget` separates "the only liquid
    contract costs more than a trade may lose" from "nothing is liquid".
    """

    contract: "Contract | None"
    lots: int
    lot_size: int
    budget: float
    reasons: list[str] = field(default_factory=list)
    refused_by_budget: bool = False

    @property
    def quantity(self) -> int:
        """Options bought: lots times the lot size."""
        return self.lots * self.lot_size

    @property
    def premium_outlay(self) -> float:
        """Premium paid for the whole position, in rupees."""
        if self.contract is None:
            return 0.0
        return self.contract.mid * self.quantity

    @property
    def total_at_risk(self) -> float:
        """Premium plus round-trip charges: the loss if it expires worthless."""
        if self.contract is None:
            return 0.0
        return option_loss(self.contract.mid, self.lots, self.lot_size)

    @property
    def charges(self) -> float:
        """Round-trip charges on the whole position, at a flat exit."""
        return self.total_at_risk - self.premium_outlay


def pick_contract(contracts: list[Contract], spot: float, direction: str,
                  lot_size: int, capital: "float | None" = None,
                  risk_pct: "float | None" = None) -> OptionPick:
    """The nearest-the-money liquid contract, sized so its loss fits the budget.

    NOT the cheapest, which the docstring used to claim: candidates run
    from at-the-money out to OPT_STRIKES_EITHER_SIDE and the first that
    clears every quality gate wins, which biases toward at-the-money
    where the book is deepest.

    A long view buys calls and a short view buys puts. Only buying is
    considered: selling naked options carries open-ended loss, which no
    scanner should quietly propose.

    SIZED, WHERE IT USED TO SUGGEST ONE LOT. A bought option's worst case
    is the whole premium plus charges, so the position is
    lots_within_budget(mid, lot_size, budget) with the same per-trade
    budget the equity sizer uses (levels.risk_budget: capital * risk_pct
    / 100, refused above SCAN_MAX_RISK_PCT). One lot of an at-the-money
    single-stock option routinely costs several times a 1,000-rupee
    budget, and "Rs X per lot" printed beside it read as a suggestion to
    risk X.

    WHEN NOT ONE LOT FITS, NOTHING IS SUGGESTED - deliberately, rather
    than walking further out of the money until something is cheap
    enough. A far out-of-the-money option fits the budget BECAUSE it is
    unlikely to pay, so letting the budget choose the strike would turn a
    risk limit into a lottery-ticket selector. The refusal names one
    lot's worst case against the budget.

    `capital` and `risk_pct` default to config, read at call time.
    """
    capital = config.SCAN_CAPITAL if capital is None else capital
    risk_pct = config.SCAN_RISK_PCT_PER_TRADE if risk_pct is None else risk_pct
    budget = risk_budget(capital, risk_pct)

    def refuse(reasons: list[str], by_budget: bool = False) -> OptionPick:
        """An OptionPick that suggests nothing, carrying why."""
        return OptionPick(None, 0, lot_size, budget, reasons, by_budget)

    if budget <= 0.0:
        return refuse([f"no usable risk budget: {risk_pct}% of {capital} is "
                       f"missing, not positive, or above the "
                       f"{config.SCAN_MAX_RISK_PCT:g}% ceiling"],
                      by_budget=True)
    if lot_size <= 0:
        return refuse(["no lot size known, so neither the premium at risk "
                       "nor the charges can be computed"])
    if direction not in (LONG, SHORT):
        return refuse([f"unknown direction {direction!r}"])
    side = CALL if direction == LONG else PUT
    strike = atm_strike(contracts, spot)
    if strike is None:
        return refuse(["no strikes in the chain"])
    pool = sorted((c for c in contracts if c.side == side),
                  key=lambda c: abs(c.strike - strike))
    window = pool[:max(1, config.OPT_STRIKES_EITHER_SIDE * 2 + 1)]
    if not window:
        return refuse([f"no {side} contracts in the chain"])
    rejected: list[str] = []
    for candidate in window:
        ok, reasons = quality_reasons(candidate, 1, lot_size)
        if not ok:
            rejected.append(f"{candidate.strike:.0f} {side}: "
                            + "; ".join(r for r in reasons if "[FAIL]" in r))
            continue
        lots = lots_within_budget(candidate.mid, lot_size, budget)
        if lots <= 0:
            one_lot = option_loss(candidate.mid, 1, lot_size)
            premium = candidate.mid * lot_size
            return refuse([
                f"the risk budget refuses the nearest liquid contract: one "
                f"lot of the {candidate.strike:.0f} {side} is {lot_size:,} x "
                f"{candidate.mid:.2f} = {premium:,.0f} rupees of premium "
                f"plus {one_lot - premium:,.0f} of charges, all of it lost "
                f"if it expires worthless, against a {budget:,.0f} budget "
                f"({risk_pct:g}% of {capital:,.0f}) - {one_lot / budget:.1f}x "
                f"what one trade may lose. Cheaper strikes further out of "
                f"the money are not substituted.", *reasons],
                by_budget=True)
        # Re-assessed at the real size: the flat brokerage is a smaller
        # share of premium on two lots than on one, so the INFO cost line
        # must describe the position actually suggested.
        _, reasons = quality_reasons(candidate, lots, lot_size)
        at_risk = option_loss(candidate.mid, lots, lot_size)
        reasons.append(
            f"{lots} lot(s) = {lots * lot_size:,} options put "
            f"{at_risk:,.0f} rupees at risk (premium plus charges, lost in "
            f"full if it expires worthless) against the {budget:,.0f} "
            f"budget [PASS]")
        return OptionPick(candidate, lots, lot_size, budget, reasons)
    return refuse(["no contract cleared the quality gates", *rejected[:5]])
