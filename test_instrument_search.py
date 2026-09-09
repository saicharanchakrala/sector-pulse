"""Tests for the instrument typeahead.

The ordering is the whole point of this module, so most of these are about
order rather than membership. A search that merely CONTAINS the right
instrument is not useful: one letter matches thousands of option strikes,
and a list that buries RELIANCE under twenty of them has failed even
though every row in it is a correct match.

No network: a synthetic master is injected, so these run without a Kite
session and without the 6-second instrument-master download.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

import instrument_search as isearch


@dataclass(frozen=True)
class FakeContract:
    """Shaped like kite_instruments.Contract, with only what build() reads."""

    tradingsymbol: str
    name: str
    instrument_type: str
    expiry: "date | None" = None
    strike: float = 0.0
    lot_size: int = 0
    instrument_token: int = 0

    @property
    def is_future(self) -> bool:
        return self.instrument_type == "FUT"

    @property
    def is_option(self) -> bool:
        return self.instrument_type in ("CE", "PE")


SEP = date(2026, 9, 29)
OCT = date(2026, 10, 27)


def universe():
    """A few stocks, an index, two futures expiries and a strike ladder."""
    rows = [
        FakeContract("RELIANCE", "RELIANCE", "EQ", lot_size=1, instrument_token=1),
        FakeContract("RELIABLE", "RELIABLE", "EQ", lot_size=1, instrument_token=2),
        FakeContract("RELIGARE", "RELIGARE", "EQ", lot_size=1, instrument_token=3),
        FakeContract("TCS", "TCS", "EQ", lot_size=1, instrument_token=4),
        FakeContract("NIFTY 50", "NIFTY 50", "EQ", instrument_token=5),
        FakeContract("RELIANCE26SEPFUT", "RELIANCE", "FUT", expiry=SEP,
                     lot_size=500, instrument_token=6),
        FakeContract("RELIANCE26OCTFUT", "RELIANCE", "FUT", expiry=OCT,
                     lot_size=500, instrument_token=7),
    ]
    for strike in (1400, 1300, 1500):
        for right in ("CE", "PE"):
            rows.append(FakeContract(
                f"RELIANCE26SEP{strike}{right}", "RELIANCE", right,
                expiry=SEP, strike=float(strike), lot_size=500))
    return rows


@pytest.fixture(autouse=True)
def catalogue(monkeypatch):
    """Install the synthetic catalogue, bypassing the network entirely."""
    import kite_instruments as ki

    rows = universe()
    monkeypatch.setattr(ki, "nse_equities",
                        lambda r: [c for c in r if c.instrument_type == "EQ"
                                   and not c.tradingsymbol.startswith("NIFTY")])
    monkeypatch.setattr(ki, "indices",
                        lambda r: [c for c in r
                                   if c.tradingsymbol.startswith("NIFTY")])
    monkeypatch.setattr(ki, "nse_futures", lambda r: [c for c in r if c.is_future])
    monkeypatch.setattr(ki, "nse_options", lambda r: [c for c in r if c.is_option])
    built = isearch.build(rows)
    monkeypatch.setattr(isearch, "_CATALOGUE", built)
    return built


# --- what the catalogue holds --------------------------------------------

def test_all_four_kinds_reach_the_catalogue(catalogue) -> None:
    kinds = {m.kind for m in catalogue}
    assert kinds == {"stock", "index", "future", "option"}


def test_a_derivative_carries_its_underlying_not_just_its_symbol(catalogue) -> None:
    # Without this the ordering cannot put a stock above its own options.
    future = isearch.find("RELIANCE26SEPFUT")
    assert future is not None
    assert future.underlying == "RELIANCE"
    assert future.lot_size == 500
    assert future.is_derivative


def test_a_stock_is_not_a_derivative(catalogue) -> None:
    stock = isearch.find("RELIANCE")
    assert stock is not None and not stock.is_derivative
    assert stock.expiry is None


# --- ordering, which is what makes the list usable -----------------------

def test_the_underlying_outranks_its_own_derivatives() -> None:
    hits = isearch.search("RELIANCE", limit=10)
    assert hits[0].symbol == "RELIANCE"


def test_futures_come_before_options() -> None:
    hits = isearch.search("RELIANCE", limit=10)
    kinds = [h.kind for h in hits]
    assert kinds.index("future") < kinds.index("option")


def test_the_nearer_expiry_comes_first() -> None:
    futures = [h for h in isearch.search("RELIANCE", limit=20)
               if h.kind == "future"]
    assert [f.symbol for f in futures] == ["RELIANCE26SEPFUT",
                                           "RELIANCE26OCTFUT"]


def test_strikes_ascend_rather_than_arriving_in_master_order() -> None:
    # The synthetic master lists 1400 before 1300 on purpose.
    calls = [h.strike for h in isearch.search("RELIANCE26SEP", limit=20)
             if h.right == "CE"]
    assert calls == sorted(calls)


def test_an_exact_symbol_wins_over_a_longer_prefix_match() -> None:
    hits = isearch.search("RELIABLE", limit=5)
    assert hits[0].symbol == "RELIABLE"


def test_a_short_query_does_not_flood_the_list_with_strikes() -> None:
    # Three characters is not enough to have narrowed the ladder, so the
    # stocks the user is probably after must not be pushed off the list.
    hits = isearch.search("REL", limit=5)
    assert all(h.kind != "option" for h in hits)
    assert "RELIANCE" in {h.symbol for h in hits}


def test_a_longer_query_does_admit_options() -> None:
    hits = isearch.search("RELIANCE26SEP1400", limit=5)
    assert {h.symbol for h in hits} == {"RELIANCE26SEP1400CE",
                                        "RELIANCE26SEP1400PE"}


def test_options_can_be_forced_on_for_a_short_query() -> None:
    hits = isearch.search("REL", limit=50, include_options=True)
    assert any(h.kind == "option" for h in hits)


def test_kinds_can_be_restricted() -> None:
    hits = isearch.search("RELIANCE", limit=20, kinds=("future",))
    assert hits and all(h.kind == "future" for h in hits)


# --- the empty and absent cases ------------------------------------------

def test_an_empty_query_returns_nothing_rather_than_everything() -> None:
    assert isearch.search("") == []
    assert isearch.search("   ") == []


def test_an_unmatched_query_returns_nothing() -> None:
    assert isearch.search("NOSUCHTHINGXYZ") == []
    assert isearch.find("NOSUCHTHINGXYZ") is None


def test_search_is_case_and_space_insensitive() -> None:
    assert isearch.search("  reliance ")[0].symbol == "RELIANCE"
    assert isearch.find(" reliance ").symbol == "RELIANCE"


def test_the_limit_is_respected_and_never_zero() -> None:
    assert len(isearch.search("RELIANCE", limit=3)) == 3
    assert len(isearch.search("RELIANCE", limit=0)) == 1


def test_an_empty_master_yields_an_empty_catalogue() -> None:
    assert isearch.build([]) == []


# --- the chained pickers -------------------------------------------------

def test_expiries_and_strikes_are_listed_for_a_chained_picker() -> None:
    assert isearch.expiries_for("RELIANCE", kind="future") == [SEP, OCT]
    assert isearch.expiries_for("RELIANCE") == [SEP]
    assert isearch.strikes_for("RELIANCE", SEP) == [1300.0, 1400.0, 1500.0]
    assert isearch.strikes_for("TCS", SEP) == []


def test_one_contract_can_be_addressed_by_its_four_parts() -> None:
    got = isearch.contract("RELIANCE", SEP, 1400.0, "PE")
    assert got is not None and got.symbol == "RELIANCE26SEP1400PE"
    assert isearch.contract("RELIANCE", SEP, 9999.0, "PE") is None


# --- the labels a person actually reads ----------------------------------

def test_each_kind_labels_what_distinguishes_it() -> None:
    assert isearch.find("RELIANCE").label == "RELIANCE  -  stock"
    assert isearch.find("NIFTY 50").label == "NIFTY 50  -  index"
    future = isearch.find("RELIANCE26SEPFUT").label
    assert "future" in future and "29 Sep 2026" in future and "500" in future
    call = isearch.find("RELIANCE26SEP1400CE").label
    assert "call" in call and "1,400" in call
    assert "put" in isearch.find("RELIANCE26SEP1400PE").label


# --- a failed download must not become a cached "there are nothing" ------

def test_an_empty_master_is_retried_rather_than_cached(monkeypatch) -> None:
    # THE REGRESSION. kite_instruments.fetch_master returns [] on any
    # network failure. market_source cached that - [] is not None, so it
    # was never retried - and the catalogue cached [] on top, so one
    # transient failure to reach Kite's CSV disabled the typeahead for the
    # life of the process. The lookup then treated "no matches" as a dead
    # end and refused to analyse anything at all.
    import kite_instruments as ki
    import market_source

    monkeypatch.setattr(isearch, "_CATALOGUE", None)
    monkeypatch.setattr(market_source, "_MASTER", None)
    monkeypatch.setattr(ki, "fetch_master", lambda timeout=60: [])
    assert isearch.catalogue() == []

    # The network recovers. The next call must actually try again.
    monkeypatch.setattr(ki, "fetch_master", lambda timeout=60: universe())
    monkeypatch.setattr(ki, "nse_equities",
                        lambda r: [c for c in r if c.instrument_type == "EQ"
                                   and not c.tradingsymbol.startswith("NIFTY")])
    monkeypatch.setattr(ki, "indices",
                        lambda r: [c for c in r
                                   if c.tradingsymbol.startswith("NIFTY")])
    monkeypatch.setattr(ki, "nse_futures", lambda r: [c for c in r if c.is_future])
    monkeypatch.setattr(ki, "nse_options", lambda r: [c for c in r if c.is_option])
    assert isearch.catalogue(), "an empty catalogue was cached and never retried"
    assert isearch.find("RELIANCE") is not None


def test_a_mid_word_match_on_the_underlying_is_found() -> None:
    # There was no "text inside the underlying" tier, so a query landing
    # inside the underlying's name but not inside the contract symbol
    # matched nothing. Ranked last, below every prefix match.
    hits = isearch.search("ELIANCE", limit=10, include_options=True)
    assert hits, "a mid-word underlying match found nothing"
    assert any(h.underlying == "RELIANCE" for h in hits)
