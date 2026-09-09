"""Tests for the single market-data source.

Every test here is a regression test for something an independent review of
the yfinance removal actually found, or for a claim market_source makes in
its own docstring. Nothing here touches the network: the instrument tables
and the Kite client are both substituted.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import kite_bars
import kite_client
import market_source


# --- symbol translation --------------------------------------------------

def test_every_legacy_alias_maps_to_a_kite_name() -> None:
    # The table exists only so a profile string written for yfinance still
    # resolves. A typo in it is silent: canonical() would pass the Yahoo
    # spelling straight through and the symbol would look unlisted.
    for yahoo, kite in market_source._LEGACY_ALIASES.items():
        assert market_source.canonical(yahoo) == kite
        assert not kite.startswith("^") and not kite.endswith(".NS")


def test_canonical_strips_the_ns_suffix_rather_than_adding_it() -> None:
    assert market_source.canonical("RELIANCE.NS") == "RELIANCE"
    assert market_source.canonical("reliance") == "RELIANCE"
    assert market_source.canonical("  M&M  ") == "M&M"
    assert market_source.canonical("") == ""
    assert market_source.canonical(None) == ""


def test_a_suffixed_index_still_reaches_its_alias() -> None:
    # NIFTY_FIN_SERVICE.NS was the profile's spelling; both halves of the
    # legacy have to come off in one call.
    assert market_source.canonical("NIFTY_FIN_SERVICE.NS") == "NIFTY FIN SERVICE"


def test_an_unknown_caret_symbol_keeps_its_caret_so_it_misses_loudly() -> None:
    # Stripping it turned "^BSESN" into "BSESN", which can match an
    # unrelated listed instrument and return a confidently wrong price. No
    # Kite tradingsymbol contains a caret, so keeping it guarantees a miss.
    assert market_source.canonical("^BSESN") == "^BSESN"
    assert market_source.canonical("^MADEUP") == "^MADEUP"


def test_interval_names_translate_and_pass_kite_spellings_through() -> None:
    assert market_source.kite_interval("5m") == "5minute"
    assert market_source.kite_interval("1d") == "day"
    assert market_source.kite_interval("60minute") == "60minute"


# --- B2: a span in trading days is not a span in calendar days ----------

def test_a_bare_day_span_is_read_as_trading_sessions() -> None:
    # yfinance's period="60d" meant sixty TRADING days - measured on this
    # repo's own cache, 59 sessions spread over 83 calendar days. Spending
    # 60 as calendar days returns about 43, a silent 27% cut to every
    # sample built on top of it.
    assert market_source.calendar_days("60d") >= 83
    assert market_source.calendar_days("10d") > 10


def test_month_and_year_spans_stay_calendar_units() -> None:
    assert market_source.calendar_days("3mo") == 91
    assert market_source.calendar_days("6mo") == 183
    assert market_source.calendar_days("1y") == 365


def test_an_unparseable_span_falls_back_instead_of_crashing() -> None:
    for bad in ("", "nonsense", "d", "-5d", "0d", None):
        assert market_source.calendar_days(bad, 42) == 42


# --- B1: the cache key must include oi, and agree with kite_bars --------

def test_the_cache_key_matches_kite_bars_for_both_oi_values() -> None:
    # Both modules write into the same bar_cache. They either agree or they
    # corrupt each other: an oi=True write under a no-OI filename poisons
    # what kite_bars.fetch_symbol(oi=False) reads back.
    start, end = date(2026, 1, 1), date(2026, 3, 1)
    for oi in (False, True):
        assert (market_source._cache_path("RELIANCE", "day", start, end, oi=oi)
                == kite_bars._cache_path("RELIANCE", "day", start, end, oi=oi))


def test_the_oi_variant_gets_its_own_filename() -> None:
    # Without this a frame fetched without open interest was served
    # verbatim to a later oi=True caller, dropping the one column that
    # caller asked for, with no warning.
    start, end = date(2026, 1, 1), date(2026, 3, 1)
    plain = market_source._cache_path("RELIANCE", "day", start, end, oi=False)
    with_oi = market_source._cache_path("RELIANCE", "day", start, end, oi=True)
    assert plain != with_oi


def test_a_space_in_an_index_name_does_not_become_a_path(tmp_path) -> None:
    path = market_source._cache_path("NIFTY 50", "5minute",
                                     date(2026, 1, 1), date(2026, 1, 2))
    assert " " not in path.name and path.parent.name == "bar_cache"


# --- C2: an expired session is the daily case, not a missing one ---------

def test_a_403_is_recognised_as_a_dead_session() -> None:
    # Kite expires the token around 6am, so this is the normal morning
    # failure, not an exotic one. Before this it surfaced as two hundred
    # individual per-symbol warnings.
    exc = kite_client.KiteError("Kite rejected the session (HTTP 403). The "
                                "access token is invalid or has expired.")
    assert market_source.is_session_failure(exc) is True


def test_an_ordinary_failure_is_not_mistaken_for_a_dead_session() -> None:
    for message in ("HTTP 500 upstream", "connection reset", "HTTP 429 too many"):
        assert market_source.is_session_failure(
            kite_client.KiteError(message)) is False


# --- C3: cached work must survive a session failure ---------------------

def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp("2026-09-08 09:15", tz="Asia/Kolkata")])
    return pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0],
                         "Close": [1.0], "Volume": [10.0]}, index=index)


def test_bars_raises_when_there_is_no_session_and_nothing_cached(monkeypatch) -> None:
    monkeypatch.setattr(market_source, "tokens", lambda refresh=False: {"AAA": 1})
    monkeypatch.setattr(kite_client, "load_session", lambda: None)
    monkeypatch.setattr(market_source, "_cache_path",
                        lambda *a, **k: pytest.importorskip("pathlib").Path(
                            "does-not-exist.parquet"))
    with pytest.raises(market_source.NoSession):
        market_source.bars(["AAA"], date(2026, 9, 1), date(2026, 9, 8))


def test_bars_keeps_the_cached_symbols_when_the_session_is_dead(monkeypatch,
                                                                tmp_path) -> None:
    # A scan where 209 of 210 symbols are cached must not be thrown away
    # because the last one needed the network. Every caller's degradation
    # path handles a short dict; none handles losing the lot.
    cached = tmp_path / "hit.parquet"
    _frame().to_parquet(cached)
    monkeypatch.setattr(market_source, "tokens",
                        lambda refresh=False: {"AAA": 1, "BBB": 2})
    monkeypatch.setattr(kite_client, "load_session", lambda: None)
    monkeypatch.setattr(market_source, "_cache_path",
                        lambda symbol, *a, **k: cached if symbol == "AAA"
                        else tmp_path / "miss.parquet")
    out = market_source.bars(["AAA", "BBB"], date(2026, 9, 1), date(2026, 9, 8))
    assert list(out) == ["AAA"]


def test_bars_keys_results_by_the_callers_own_spelling(monkeypatch) -> None:
    # A module still holding a yfinance-style ticker gets its own key back
    # and needs no translation of its own.
    monkeypatch.setattr(market_source, "tokens",
                        lambda refresh=False: {"RELIANCE": 1})
    monkeypatch.setattr(kite_client, "load_session", lambda: object())
    monkeypatch.setattr(kite_client, "historical",
                        lambda *a, **k: _frame())
    monkeypatch.setattr(market_source, "_cache_path",
                        lambda *a, **k: pytest.importorskip("pathlib").Path(
                            "nonexistent-dir-xyz") / "x.parquet")
    out = market_source.bars(["RELIANCE.NS"], date(2026, 9, 1), date(2026, 9, 8))
    assert list(out) == ["RELIANCE.NS"]


def test_one_bad_symbol_does_not_kill_the_sweep(monkeypatch) -> None:
    # market_data promises never to raise on this path, and a sweep of two
    # hundred names should not die on one delisting or one bad span.
    monkeypatch.setattr(market_source, "tokens",
                        lambda refresh=False: {"AAA": 1, "BBB": 2})
    monkeypatch.setattr(kite_client, "load_session", lambda: object())

    def flaky(token, *a, **k):
        if token == 1:
            raise ValueError("start is after end")
        return _frame()

    monkeypatch.setattr(kite_client, "historical", flaky)
    monkeypatch.setattr(market_source, "_cache_path",
                        lambda *a, **k: pytest.importorskip("pathlib").Path(
                            "nonexistent-dir-xyz") / "x.parquet")
    out = market_source.bars(["AAA", "BBB"], date(2026, 9, 1), date(2026, 9, 8))
    assert list(out) == ["BBB"]


# --- C4: quote keys carry the right exchange ---------------------------

def test_a_futures_symbol_is_quoted_on_nfo_not_nse(monkeypatch) -> None:
    # Folding futures into the token table made them pass the listed check
    # while still being asked for on NSE, which comes back as a silent miss
    # logged as "no price returned".
    monkeypatch.setattr(market_source, "_TOKENS", {"NIFTY26SEPFUT": 1})
    monkeypatch.setattr(market_source, "_EXCHANGES", {"NIFTY26SEPFUT": "NFO"})
    monkeypatch.setattr(kite_client, "load_session", lambda: object())
    seen: list[list[str]] = []

    def fake_quote(keys, session=None):
        seen.append(list(keys))
        return {keys[0]: {"last_price": 100.0}}

    monkeypatch.setattr(kite_client, "quote", fake_quote)
    market_source.last_prices(["NIFTY26SEPFUT"])
    assert seen == [["NFO:NIFTY26SEPFUT"]]


# --- the fail-open the planner must never see -------------------------

def test_a_total_quote_failure_raises_rather_than_pricing_a_book_at_zero(
        monkeypatch) -> None:
    # Returning {} is indistinguishable from "every symbol is unlisted",
    # and the planner would size against a zero-value book.
    monkeypatch.setattr(market_source, "_TOKENS", {"RELIANCE": 1})
    monkeypatch.setattr(market_source, "_EXCHANGES", {"RELIANCE": "NSE"})
    monkeypatch.setattr(kite_client, "load_session", lambda: object())

    def always_fails(keys, session=None):
        raise kite_client.KiteError("HTTP 429 too many requests")

    monkeypatch.setattr(kite_client, "quote", always_fails)
    with pytest.raises(market_source.NoSession):
        market_source.last_prices(["RELIANCE"])


def test_a_partial_quote_result_is_returned_not_refused(monkeypatch) -> None:
    monkeypatch.setattr(market_source, "_TOKENS", {"AAA": 1, "BBB": 2})
    monkeypatch.setattr(market_source, "_EXCHANGES", {"AAA": "NSE", "BBB": "NSE"})
    monkeypatch.setattr(kite_client, "load_session", lambda: object())
    monkeypatch.setattr(kite_client, "quote",
                        lambda keys, session=None: {"NSE:AAA": {"last_price": 7.5}})
    out = market_source.last_prices(["AAA", "BBB"])
    assert out == {"AAA": 7.5}


def test_a_nonpositive_price_is_dropped_rather_than_believed(monkeypatch) -> None:
    monkeypatch.setattr(market_source, "_TOKENS", {"AAA": 1})
    monkeypatch.setattr(market_source, "_EXCHANGES", {"AAA": "NSE"})
    monkeypatch.setattr(kite_client, "load_session", lambda: object())
    monkeypatch.setattr(kite_client, "quote",
                        lambda keys, session=None: {"NSE:AAA": {"last_price": 0.0}})
    assert market_source.last_prices(["AAA"]) == {}


def test_last_prices_raises_when_there_is_no_session_at_all(monkeypatch) -> None:
    monkeypatch.setattr(kite_client, "load_session", lambda: None)
    with pytest.raises(market_source.NoSession):
        market_source.last_prices(["RELIANCE"])


def test_an_empty_request_is_not_a_failure() -> None:
    assert market_source.last_prices([]) == {}
    assert market_source.last_prices(["", "   "]) == {}
