"""Typeahead over every tradable NSE instrument - cash, futures and options.

WHY THIS EXISTS. The lookup used to search a pool of F&O UNDERLYINGS plus
cash equities, taken from the instrument snapshot. That pool holds the
symbol RELIANCE but not RELIANCE26SEPFUT, and nothing at all for the 32,807
option contracts, so a derivative was undiscoverable even when it was
perfectly resolvable - `market_source.token_for("RELIANCE26SEPFUT")` has
always returned a token.

WHERE THE DATA COMES FROM. Kite's public instrument master, which
market_source already fetches once per process and caches. That is 109,241
rows covering roughly 10,110 cash equities, 647 futures and 32,807 options.
No extra network call and no session needed: the master is unauthenticated,
which is why symbol resolution works even when prices do not.

THE ORDERING PROBLEM, which is most of the work here. A one-letter query
matches thousands of option contracts, and a list flooded with strikes of
one name is useless. So:

  * results are ordered by KIND first - cash, then index, then futures,
    then options - because the underlying is almost always what someone
    typing "RELI" wants;
  * options are withheld until the query is specific enough to narrow
    them (see MIN_CHARS_FOR_OPTIONS), rather than truncating the list and
    hiding the stock behind 20 strikes;
  * within a kind, the nearer expiry comes first, since that is the liquid
    one, and then the strike ascends.

WHAT THIS DOES NOT DO. Finding a contract is not the same as being able to
assess it. The daily store holds cash equity bars only, so selecting a
future or an option finds it and then has nothing to analyse. That is a
separate piece of work - it needs per-contract bars, a derivative cost
model, and horizons capped by expiry, since a September contract cannot be
held for 252 sessions.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

# Kinds, in the order they are offered. Cash first because someone typing
# a company name usually wants the company.
KIND_STOCK = "stock"
KIND_INDEX = "index"
KIND_FUTURE = "future"
KIND_OPTION = "option"
KIND_ORDER = (KIND_STOCK, KIND_INDEX, KIND_FUTURE, KIND_OPTION)

# Below this many characters, options are left out entirely.
#
# What this does NOT do, since an earlier comment claimed it: it does not
# "narrow" the option list. Four characters still matches every one of
# BANKNIFTY's contracts. What keeps the list usable is the kind-first tier
# ordering below, which puts the stock and its futures above every strike.
# This threshold does something smaller and more specific - it stops a
# one- or two-letter query, typed on the way to something else, from
# spending the whole result budget on strikes.
#
# It is not a performance guard either: a full scan of the 43,800-row
# catalogue costs about 7ms for "R" and 15ms for "RELI".
MIN_CHARS_FOR_OPTIONS = 4

_CATALOGUE: "list[Match] | None" = None
# Streamlit serves each session on its own thread, and two arriving together
# would both see an empty catalogue and both download the 109,000-row
# master. The rebind is atomic under the GIL so there was no corruption,
# only duplicated work - and, with an empty result, a duplicated failure.
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Match:
    """One searchable instrument, with what a person needs to tell it apart."""

    symbol: str                   # tradingsymbol - what gets analysed
    kind: str
    underlying: str               # the cash name, for derivatives
    expiry: "date | None" = None
    strike: float = 0.0
    right: str = ""               # CE or PE, empty for anything else
    lot_size: int = 0
    token: int = 0

    @property
    def is_derivative(self) -> bool:
        return self.kind in (KIND_FUTURE, KIND_OPTION)

    @property
    def days_to_expiry(self) -> "int | None":
        """Calendar days until expiry, or None for a cash instrument.

        What the horizon bound is computed from: a September contract
        cannot be held for 252 sessions, so a long-term verdict on one is
        not a cautious estimate but an answer to an impossible question.
        """
        if self.expiry is None:
            return None
        from datetime import date as _date
        return (self.expiry - _date.today()).days

    @property
    def label(self) -> str:
        """One line for a dropdown - the symbol plus what disambiguates it."""
        if self.kind == KIND_STOCK:
            return f"{self.symbol}  -  stock"
        if self.kind == KIND_INDEX:
            return f"{self.symbol}  -  index"
        when = f"{self.expiry:%d %b %Y}" if self.expiry else "unknown expiry"
        if self.kind == KIND_FUTURE:
            return (f"{self.symbol}  -  {self.underlying} future, expires "
                    f"{when}, lot {self.lot_size:,}")
        side = "call" if self.right == "CE" else "put"
        return (f"{self.symbol}  -  {self.underlying} {side} at "
                f"{self.strike:,.0f}, expires {when}, lot {self.lot_size:,}")


def build(rows: "list | None" = None) -> list:
    """The searchable catalogue, built from the instrument master.

    Deliberately tolerant: a group that cannot be read is logged and
    skipped rather than raised, because a typeahead that returns fewer
    kinds is usable and one that raises takes the page down.
    """
    import kite_instruments as ki
    import market_source

    if rows is None:
        rows = market_source.master()
    if not rows:
        logger.warning("Instrument master is empty; search has nothing to offer")
        return []
    out: list = []
    groups = (
        (KIND_STOCK, lambda: ki.nse_equities(rows)),
        (KIND_INDEX, lambda: ki.indices(rows)),
        (KIND_FUTURE, lambda: ki.nse_futures(rows)),
        (KIND_OPTION, lambda: ki.nse_options(rows)),
    )
    for kind, produce in groups:
        try:
            contracts = produce()
        except Exception as exc:
            logger.warning("Could not read %s instruments: %s", kind, exc)
            continue
        for contract in contracts:
            symbol = (getattr(contract, "tradingsymbol", "") or "").strip()
            if not symbol:
                continue
            underlying = (getattr(contract, "name", "") or "").strip().upper()
            out.append(Match(
                symbol=symbol.upper(), kind=kind,
                underlying=underlying or symbol.upper(),
                expiry=getattr(contract, "expiry", None),
                strike=float(getattr(contract, "strike", 0.0) or 0.0),
                right=(getattr(contract, "instrument_type", "") or "").strip().upper()
                      if kind == KIND_OPTION else "",
                lot_size=int(getattr(contract, "lot_size", 0) or 0),
                token=int(getattr(contract, "instrument_token", 0) or 0),
            ))
    return out


def catalogue(refresh: bool = False) -> list:
    """The catalogue, built once per process.

    The master itself is already cached by market_source, so this only
    avoids rebuilding ~43,000 Match objects on every keystroke.
    """
    global _CATALOGUE
    # `not _CATALOGUE`, so a build that failed because the master could not
    # be reached is retried on the next keystroke instead of being cached
    # as "there are no instruments".
    with _LOCK:
        if not _CATALOGUE or refresh:
            _CATALOGUE = build()
        return _CATALOGUE


def _score(match: Match, text: str) -> "tuple | None":
    """Sort key for one match against a query, or None when it does not hit.

    Lower sorts first. The tiers are what make the list useful rather than
    merely correct: an exact symbol beats a prefix, a prefix beats a
    substring, and the underlying's own row beats its derivatives.
    """
    symbol, underlying = match.symbol, match.underlying
    if symbol == text:
        tier = 0
    elif symbol.startswith(text):
        tier = 1
    elif underlying == text:
        tier = 2
    elif underlying.startswith(text):
        tier = 3
    elif text in symbol:
        tier = 4
    elif text in underlying:
        # A mid-word match on the underlying. Without this tier a query
        # that lands inside the underlying's name but not inside the
        # contract symbol matched nothing at all.
        tier = 5
    else:
        return None
    # Nearer expiry first - that is the liquid contract - with cash
    # instruments (no expiry) sorting ahead of everything dated.
    when = match.expiry.toordinal() if match.expiry else 0
    return (tier, KIND_ORDER.index(match.kind), when, match.strike, symbol)


def search(query: str, limit: int = 20, kinds: "tuple | None" = None,
           include_options: "bool | None" = None) -> list:
    """Ranked matches for a typeahead. [] for an empty or unmatched query.

    `include_options` overrides the MIN_CHARS_FOR_OPTIONS rule, for a
    caller that has its own way of narrowing - a chosen underlying, say.
    """
    text = (query or "").strip().upper()
    if not text:
        return []
    if include_options is None:
        include_options = len(text) >= MIN_CHARS_FOR_OPTIONS
    wanted = set(kinds) if kinds else set(KIND_ORDER)
    if not include_options:
        wanted.discard(KIND_OPTION)
    scored = []
    for match in catalogue():
        if match.kind not in wanted:
            continue
        key = _score(match, text)
        if key is not None:
            scored.append((key, match))
    scored.sort(key=lambda pair: pair[0])
    return [match for _, match in scored[:max(1, limit)]]


def find(symbol: str) -> "Match | None":
    """One instrument by exact tradingsymbol, or None."""
    text = (symbol or "").strip().upper()
    if not text:
        return None
    for match in catalogue():
        if match.symbol == text:
            return match
    return None


def expiries_for(underlying: str, kind: str = KIND_OPTION) -> list:
    """Sorted expiries available for one underlying, for a chained picker."""
    text = (underlying or "").strip().upper()
    seen = {m.expiry for m in catalogue()
            if m.underlying == text and m.kind == kind and m.expiry}
    return sorted(seen)


def strikes_for(underlying: str, expiry: date) -> list:
    """Sorted strikes for one underlying and expiry."""
    text = (underlying or "").strip().upper()
    seen = {m.strike for m in catalogue()
            if m.underlying == text and m.kind == KIND_OPTION
            and m.expiry == expiry and m.strike > 0}
    return sorted(seen)


def contract(underlying: str, expiry: date, strike: float,
             right: str) -> "Match | None":
    """The one option contract matching an underlying, expiry, strike, side."""
    text = (underlying or "").strip().upper()
    side = (right or "").strip().upper()
    for match in catalogue():
        if (match.kind == KIND_OPTION and match.underlying == text
                and match.expiry == expiry and match.right == side
                and abs(match.strike - strike) < 1e-6):
            return match
    return None


def main(argv=None) -> int:
    """CLI: python -m instrument_search RELIANCE"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Search every tradable NSE instrument")
    parser.add_argument("query", help="part of a symbol, e.g. RELIANCE")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--options", action="store_true",
                        help="include option contracts however short the query")
    args = parser.parse_args(argv)
    hits = search(args.query, limit=args.limit,
                  include_options=True if args.options else None)
    if not hits:
        print(f"nothing matched {args.query!r}")
        return 1
    for match in hits:
        print(match.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
