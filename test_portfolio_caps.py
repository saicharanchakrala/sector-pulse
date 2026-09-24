"""Tests for setups.apply_portfolio_caps: one scan's combined risk, bounded.

Run with: .venv\\Scripts\\python -m pytest test_portfolio_caps.py -q

WHY THIS EXISTS. Every setup is sized on its own against its own stop, so
the per-trade budget bounds each row and nothing bounded the total: on
2026-09-22 the scan log marked 569 setups taken, each risking 1,000 of
1,00,000. The cap walks the actionable setups in rank order and closes the
book once the combined loss-at-stop (charges included) or the combined
notional would pass its limit.

The setups here are built by hand, with levels whose risk and notional are
round numbers, so every expected count can be read straight off the
fixture rather than trusted to the sizer.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import ClassVar

import pytest

import config
import levels
import scan_intraday
import scan_publish
import setups
from levels import LONG


def _reading(symbol: str) -> setups.Readings:
    """The minimum Readings a Setup needs; no gate is re-run on it."""
    return setups.Readings(
        symbol=symbol, ticker=symbol, last=100.0, prev_close=99.0,
        day_change_pct=1.0, vwap=99.5, vwap_distance_pct=0.5,
        opening_range=None, cpr=None, atr_bar=1.0, rvol=2.0,
        relative_strength=1.0, turnover_20d=1e9, oi_change_pct=None,
        futures_share=None, derivatives_turnover=None, day_high=101.0,
        day_low=98.0, minutes_left=150, bars_left=25)


def _setup(symbol: str, score: float, lot_risk: float = 900.0,
           cost: float = 100.0, entry: float = 100.0, quantity: int = 100,
           passed: bool = True, with_levels: bool = True) -> setups.Setup:
    """A setup whose loss at the stop is lot_risk + cost, by construction."""
    trade = None
    if with_levels:
        stop = entry - lot_risk / quantity
        trade = levels.TradeLevels(
            direction=LONG, entry=entry, stop=stop,
            target=entry + 2.0 * (entry - stop), quantity=quantity,
            lot_risk=lot_risk, reward_risk=2.0,
            breakeven_pct=cost / (entry * quantity) * 100.0,
            cost_rupees=cost, expected_range=10.0,
            stop_source="volatility", capital_capped=False)
    return setups.Setup(_reading(symbol), LONG, trade, passed, score,
                        [f"{symbol} cleared every gate [PASS]"])


def _book(count: int, **kwargs) -> list:
    """`count` actionable setups, best first, each losing 1,000 at the stop."""
    return [_setup(f"S{i}", score=1.0 - i * 0.01, **kwargs)
            for i in range(count)]


def _kept(capped: list) -> list:
    return [s.symbol for s in capped if s.actionable]


# --- the risk cap ----------------------------------------------------------

def test_the_risk_cap_blocks_the_lowest_ranked_setups() -> None:
    """5% of 1,00,000 is 5,000: five 1,000-rupee setups fit, three do not."""
    book = _book(8)
    capped = setups.apply_portfolio_caps(book, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert _kept(capped) == ["S0", "S1", "S2", "S3", "S4"]
    blocked = [s for s in capped if not s.actionable]
    assert [s.symbol for s in blocked] == ["S5", "S6", "S7"]


def test_risk_is_counted_with_charges_not_price_alone() -> None:
    """800 of price risk + 200 of charges: price alone would fit six."""
    book = _book(8, lot_risk=800.0, cost=200.0)
    capped = setups.apply_portfolio_caps(book, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    # 6 * 800 = 4,800 would pass a price-only cap; 6 * 1,000 = 6,000 fails.
    assert len(_kept(capped)) == 5


def test_the_book_closes_rather_than_back_filling_a_smaller_setup() -> None:
    """Rank decides, not size: a later small setup is not squeezed in."""
    book = [*_book(4), _setup("BIG", 0.5, lot_risk=1400.0), _setup("SMALL", 0.4, lot_risk=400.0)]
    capped = setups.apply_portfolio_caps(book, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    # 4,000 used; BIG would make 5,500 and closes the book. SMALL alone
    # would have fitted (4,500) and is still turned away.
    assert _kept(capped) == ["S0", "S1", "S2", "S3"]
    small = next(s for s in capped if s.symbol == "SMALL")
    assert not small.actionable
    assert "not back-filled" in small.reasons[-1]


def test_a_setup_exactly_on_the_cap_is_kept() -> None:
    """Five setups of exactly one budget sum to exactly the cap."""
    capped = setups.apply_portfolio_caps(_book(5), capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert len(_kept(capped)) == 5


# --- the notional cap ------------------------------------------------------

def test_the_notional_cap_binds_on_its_own() -> None:
    """Tiny risk, big tickets: funding runs out long before risk does."""
    # 150 shares at 1,000 is 1,50,000 of notional each, against 1,00,000
    # at 5x = 5,00,000. Three fit (4,50,000); the fourth makes 6,00,000.
    # Risk used is 3 * 100 = 300 of a 5,000 cap, nowhere near binding.
    book = [_setup(f"N{i}", 1.0 - i * 0.01, lot_risk=60.0, cost=40.0,
                   entry=1_000.0, quantity=150) for i in range(5)]
    capped = setups.apply_portfolio_caps(book, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert _kept(capped) == ["N0", "N1", "N2"]
    fourth = capped[3]
    assert fourth.reasons[-1].startswith(setups.PORTFOLIO_NOTIONAL_CAP)


def test_the_risk_lead_in_is_used_when_risk_is_what_binds() -> None:
    capped = setups.apply_portfolio_caps(_book(6), capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert capped[5].reasons[-1].startswith(setups.PORTFOLIO_RISK_CAP)


# --- what passes through, and how a block is recorded ----------------------

def test_non_actionable_setups_pass_through_untouched() -> None:
    failed = _setup("FAILED", 0.0, passed=False)
    no_levels = _setup("NOLEVELS", 0.0, with_levels=False)
    book = [*_book(7), failed, no_levels]
    capped = setups.apply_portfolio_caps(book, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert capped[-2] is failed
    assert capped[-1] is no_levels
    assert capped[-2].reasons == failed.reasons


def test_a_blocked_setup_carries_the_reason_and_first_blocker_reads_it() -> None:
    book = _book(6)
    capped = setups.apply_portfolio_caps(book, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    blocked = capped[5]
    assert not blocked.passed and not blocked.actionable
    reason = blocked.reasons[-1]
    assert "[FAIL]" in reason
    # An ENTRY reason, like the room-to-move gate: it says not to OPEN
    # the trade, and the position watch must not read it as "the setup
    # stopped working" for someone already in.
    assert "[ENTRY]" in reason
    assert setups.capped_by_portfolio(blocked)
    assert scan_intraday.first_blocker(blocked).startswith(
        setups.PORTFOLIO_RISK_CAP)
    # The levels, the score and the earlier trail survive, so the row
    # still shows what it would have been and why it was not taken.
    assert blocked.levels is book[5].levels
    assert blocked.rank_score == pytest.approx(0.95)
    assert blocked.reasons[:-1] == book[5].reasons


def test_the_blocked_reason_survives_the_published_join() -> None:
    """The published table joins reasons with ' | ' and splits them back."""
    capped = setups.apply_portfolio_caps(_book(6), capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert " | " not in capped[5].reasons[-1]
    row = scan_publish.row_for(capped[5], datetime(2026, 9, 24, 10, 0))
    assert row["actionable"] is False
    logged = scan_publish.log_row(row)
    assert logged["taken"] is False
    assert logged["blocked_by"].startswith(setups.PORTFOLIO_RISK_CAP)


def test_the_cap_does_not_mutate_its_input() -> None:
    book = _book(8)
    before = [(s.passed, list(s.reasons)) for s in book]
    setups.apply_portfolio_caps(book, capital=100_000.0, max_risk_pct=5.0,
                                leverage=5.0)
    assert [(s.passed, list(s.reasons)) for s in book] == before


def test_applying_the_cap_twice_changes_nothing() -> None:
    once = setups.apply_portfolio_caps(_book(8), capital=100_000.0,
                                       max_risk_pct=5.0, leverage=5.0)
    twice = setups.apply_portfolio_caps(once, capital=100_000.0,
                                        max_risk_pct=5.0, leverage=5.0)
    assert _kept(twice) == _kept(once)
    assert [len(s.reasons) for s in twice] == [len(s.reasons) for s in once]


def test_an_unusable_cap_fails_closed() -> None:
    """NaN compares False with everything, so it must not mean 'no cap'."""
    capped = setups.apply_portfolio_caps(_book(3), capital=math.nan,
                                         max_risk_pct=5.0, leverage=5.0)
    assert _kept(capped) == []


def test_the_cap_reads_config_at_call_time(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_CAPITAL", 100_000.0)
    monkeypatch.setattr(config, "SCAN_MAX_OPEN_RISK_PCT", 2.0)
    assert len(_kept(setups.apply_portfolio_caps(_book(5)))) == 2


def test_the_configured_cap_is_five_percent() -> None:
    assert config.SCAN_MAX_OPEN_RISK_PCT == pytest.approx(5.0)


# --- the reported totals ---------------------------------------------------

def test_exposure_reports_what_the_cap_enforced() -> None:
    capped = setups.apply_portfolio_caps(_book(8), capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    exposure = setups.portfolio_exposure(capped, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert exposure.kept == 5
    assert exposure.blocked == 3
    assert exposure.risk_used == pytest.approx(5_000.0)
    assert exposure.risk_cap == pytest.approx(5_000.0)
    assert exposure.notional_used == pytest.approx(5 * 10_000.0)
    assert exposure.notional_cap == pytest.approx(500_000.0)
    assert exposure.risk_used_pct_of_cap == pytest.approx(100.0)


# --- wiring: every place a ranking is produced applies the cap -------------
#
# BEHAVIOURAL, not a search of the source. The previous versions asserted
# that the text "apply_portfolio_caps" appeared in each function, which a
# commented-out call satisfies. Each test below replaces the cap with a
# spy that records what it was handed and returns a DIFFERENT list, then
# checks that the ranked setups went in and the spy's list is what came
# out - so a call that is skipped, or whose result is dropped, fails.

class _CapSpy:
    """Stands in for setups.apply_portfolio_caps and records its calls."""

    def __init__(self, returns: list) -> None:
        self.returns = returns
        self.calls: list = []

    def __call__(self, ranked, **kwargs):
        self.calls.append((list(ranked), kwargs))
        return self.returns


def _from_the_cap() -> list:
    """What the spy hands back: nothing the ranking itself contains."""
    return [_setup("FROM_CAP", 0.9), _setup("ALSO_FROM_CAP", 0.8)]


def _stub_scoring(monkeypatch, book: list) -> None:
    """measure() returns the symbol and evaluate() looks its setup up."""
    by_symbol = {s.symbol: s for s in book}
    monkeypatch.setattr(setups, "measure",
                        lambda symbol, *args, **kwargs: symbol)
    monkeypatch.setattr(setups, "evaluate",
                        lambda reading, *args, **kwargs: by_symbol[reading])


def test_the_published_feed_publishes_what_the_cap_returns(
        monkeypatch) -> None:
    book = [_setup("LOW", 0.2), _setup("HIGH", 0.7)]
    frames = {"LOW": object(), "HIGH": object(),
              config.SCAN_BENCHMARK: object()}
    monkeypatch.setattr(scan_publish, "assemble",
                        lambda symbols, history, today, seed: frames)
    monkeypatch.setattr(scan_publish, "benchmark_change_pct",
                        lambda intraday, daily, now: 0.5)
    _stub_scoring(monkeypatch, book)
    spy = _CapSpy(_from_the_cap())
    monkeypatch.setattr(setups, "apply_portfolio_caps", spy)

    now = datetime(2026, 9, 24, 10, 0, tzinfo=scan_publish.IST)
    table, _ = scan_publish.run_once(list(frames), {}, {}, {}, now=now)

    assert len(spy.calls) == 1
    ranked, _ = spy.calls[0]
    assert [s.symbol for s in ranked] == ["HIGH", "LOW"], "rank order in"
    assert table["symbol"].tolist() == ["FROM_CAP", "ALSO_FROM_CAP"]


class _Universe:
    """The fields scan_intraday.run reads off an instrument snapshot."""

    gaps: ClassVar[list] = []
    fo_state: ClassVar[dict] = {}
    fo_underlyings: ClassVar[list] = []
    fo_indices: ClassVar[list] = []
    captured_at = datetime(2026, 1, 1, 9, 0, tzinfo=scan_intraday.IST)


def test_the_cli_reports_and_logs_what_the_cap_returns(monkeypatch) -> None:
    book = [_setup("LOW", 0.2), _setup("HIGH", 0.7),
            _setup("NOLEVELS", 0.1, with_levels=False)]
    returned = [*_from_the_cap(), _setup("NOLEVELS", 0.1, with_levels=False)]
    spy = _CapSpy(returned)
    seen: dict = {}
    bars = scan_intraday.scan_data.BarSet(
        intraday={"LOW.NS": object()}, daily={}, requested=3, failed=[])

    def report(ranked, *args, **kwargs):
        seen["reported"] = ranked
        return []

    def loggable(candidates, intraday, now):
        seen["logged"] = candidates
        return [], None

    monkeypatch.setattr(scan_intraday, "load_universe",
                        lambda refresh: _Universe())
    monkeypatch.setattr(scan_intraday.instruments, "to_ticker",
                        lambda symbol: symbol)
    monkeypatch.setattr(scan_intraday, "resolve_symbols",
                        lambda args, universe: (["LOW", "HIGH"], "test"))
    monkeypatch.setattr(scan_intraday.scan_data, "fetch_bars",
                        lambda *args, **kwargs: bars)
    monkeypatch.setattr(scan_intraday.scan_data, "benchmark_change_pct",
                        lambda bar_set: 0.5)
    monkeypatch.setattr(scan_intraday, "build_setups",
                        lambda *args, **kwargs: book)
    monkeypatch.setattr(setups, "apply_portfolio_caps", spy)
    monkeypatch.setattr(scan_intraday, "print_report", report)
    monkeypatch.setattr(scan_intraday, "loggable_now", loggable)
    # Never touch the real scan_log.csv, whatever the stubs above return.
    monkeypatch.setattr(scan_intraday, "append_log", lambda rows: None)

    args = scan_intraday._parse_args(["--capital", "250000"])
    assert scan_intraday.run(args) == 0

    assert len(spy.calls) == 1
    ranked, kwargs = spy.calls[0]
    assert ranked == setups.rank(book)
    # Capped against the capital the rows were sized on, not the default.
    assert kwargs.get("capital") == pytest.approx(250_000.0)
    assert seen["reported"] is returned
    assert [s.symbol for s in seen["logged"]] == ["FROM_CAP",
                                                  "ALSO_FROM_CAP"]


def test_the_dashboard_returns_what_the_cap_returns(monkeypatch) -> None:
    """app.run_scan runs without a Streamlit session once its I/O is stubbed."""
    import app

    book = [_setup("LOW", 0.2), _setup("HIGH", 0.7)]
    spy = _CapSpy(_from_the_cap())

    class _Inst:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

    class _Discovered:
        fo_stocks: ClassVar[list] = [_Inst("LOW"), _Inst("HIGH")]
        equities: ClassVar[list] = []
        fo_state: ClassVar[dict] = {}
        captured_at = datetime(2026, 1, 1, 9, 0, tzinfo=app.IST_ZONE)

    bars = app.scan_data.BarSet(
        intraday={"LOW": object(), "HIGH": object()}, daily={},
        requested=3, failed=[])
    monkeypatch.setattr(app.instruments, "load_latest",
                        lambda: _Discovered())
    monkeypatch.setattr(app.instruments, "to_ticker", lambda s: s)
    monkeypatch.setattr(app, "load_scan_bars",
                        lambda tickers, target=None, live_stamp="": bars)
    monkeypatch.setattr(app, "live_bar_stamp", lambda: "")
    monkeypatch.setattr(app.scan_data, "benchmark_change_pct",
                        lambda bar_set: 0.5)
    _stub_scoring(monkeypatch, book)
    monkeypatch.setattr(setups, "apply_portfolio_caps", spy)

    ranked, *_ = app.run_scan(app._DEFAULT_SCAN_SCOPE)

    assert len(spy.calls) == 1
    handed, kwargs = spy.calls[0]
    assert [s.symbol for s in handed] == ["HIGH", "LOW"], "rank order in"
    assert kwargs.get("capital") == pytest.approx(config.SCAN_CAPITAL)
    assert ranked is spy.returns


# --- a cap that cannot hold one trade is named, not called a quiet market --

def test_a_cap_below_the_per_trade_risk_is_a_configuration_problem() -> None:
    problem = setups.cap_config_problem(risk_pct=2.0, capital=100_000.0,
                                        max_risk_pct=1.5, leverage=5.0)
    assert "1.5%" in problem and "2%" in problem
    assert "SCAN_MAX_OPEN_RISK_PCT" in problem


def test_a_cap_equal_to_the_per_trade_risk_is_legal() -> None:
    """Tight - it holds exactly one full-size trade - but not a fault."""
    assert setups.cap_config_problem(risk_pct=2.0, capital=100_000.0,
                                     max_risk_pct=2.0, leverage=5.0) == ""


def test_the_shipped_caps_are_not_a_configuration_problem() -> None:
    assert setups.cap_config_problem() == ""


@pytest.mark.parametrize("override", [
    {"max_risk_pct": math.nan}, {"max_risk_pct": 0.0},
    {"max_risk_pct": -5.0}, {"leverage": 0.0}, {"leverage": math.nan},
])
def test_an_unusable_cap_is_a_configuration_problem(override) -> None:
    kwargs = {"risk_pct": 1.0, "capital": 100_000.0, "max_risk_pct": 5.0,
              "leverage": 5.0}
    kwargs.update(override)
    assert "unusable" in setups.cap_config_problem(**kwargs)


def _all_capped(max_risk_pct: float = 0.5) -> list:
    """A book a too-small cap turns away entirely: 1,000 each, 500 cap."""
    return setups.apply_portfolio_caps(_book(3), capital=100_000.0,
                                       max_risk_pct=max_risk_pct,
                                       leverage=5.0)


def test_the_all_blocked_message_names_the_cap_and_the_per_trade_risk(
        ) -> None:
    capped = _all_capped()
    assert _kept(capped) == []
    assert setups.closing_cap(capped) == setups.PORTFOLIO_RISK_CAP
    text = setups.cap_blocked_all_text(
        3, setups.closing_cap(capped), risk_pct=1.0, capital=100_000.0,
        max_risk_pct=0.5, leverage=5.0)
    assert "combined risk cap held back all 3" in text
    assert "0.5% of 100,000 = 500 rupees" in text
    assert "1% = 1,000 rupees" in text
    assert "configuration fault, not a quiet market" in text


def test_closing_cap_is_empty_when_nothing_was_capped() -> None:
    assert setups.closing_cap(_book(2)) == ""


def test_the_cli_names_the_cap_when_it_blocked_everything(
        monkeypatch, capsys) -> None:
    """The misconfiguration the reviewer found: cap 0.5% against 1% a trade."""
    monkeypatch.setattr(config, "SCAN_MAX_OPEN_RISK_PCT", 0.5)
    bars = scan_intraday.scan_data.BarSet(
        intraday={"S0.NS": object()}, daily={}, requested=3, failed=[])
    shown = scan_intraday.print_report(
        _all_capped(), bars, "test", 0.5,
        datetime(2026, 9, 24, 10, 0, tzinfo=scan_intraday.IST), top=5,
        capital=100_000.0, risk_pct=1.0)
    out = capsys.readouterr().out
    flat = " ".join(out.split())            # undo the report's line wrap
    assert shown == []
    assert "PORTFOLIO CAP BLOCKED IT ALL" in out
    assert "0.5% of 100,000 = 500 rupees" in flat
    assert "below the 1% one trade may lose" in flat
    assert "expected answer" not in out, "called a quiet market"


def test_the_cli_near_misses_do_not_print_the_entry_tag(capsys) -> None:
    scan_intraday._print_near_misses(_all_capped())
    out = capsys.readouterr().out
    assert setups.PORTFOLIO_RISK_CAP in out
    assert "[ENTRY]" not in out and "[FAIL]" not in out


def test_the_cli_refuses_a_cap_below_the_per_trade_risk(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MAX_OPEN_RISK_PCT", 1.5)
    with pytest.raises(SystemExit):
        scan_intraday._parse_args(["--risk-pct", "2"])


def test_the_cli_accepts_a_cap_equal_to_the_per_trade_risk(
        monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MAX_OPEN_RISK_PCT", 2.0)
    args = scan_intraday._parse_args(["--risk-pct", "2"])
    assert args.risk_pct == pytest.approx(2.0)


def test_the_cli_refuses_an_unusable_cap(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MAX_OPEN_RISK_PCT", math.nan)
    with pytest.raises(SystemExit):
        scan_intraday._parse_args([])


# --- the feed's table, checked against the cap -----------------------------

_NOW = datetime(2026, 9, 24, 10, 0)


def _published(capped: list) -> list:
    """The rows the feed would publish for this ranking."""
    return [scan_publish.row_for(s, _NOW) for s in capped]


def test_published_exposure_matches_the_enforced_totals() -> None:
    capped = setups.apply_portfolio_caps(_book(8), capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    enforced = setups.portfolio_exposure(capped, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    rebuilt = setups.published_exposure(_published(capped),
                                        capital=100_000.0,
                                        max_risk_pct=5.0, leverage=5.0)
    assert rebuilt.exposure.kept == enforced.kept == 5
    assert rebuilt.exposure.blocked == enforced.blocked == 3
    assert rebuilt.exposure.risk_used == pytest.approx(enforced.risk_used,
                                                       abs=5.0)
    assert rebuilt.exposure.notional_used == pytest.approx(
        enforced.notional_used, abs=5.0)
    assert rebuilt.held == ("S5", "S6", "S7")
    assert rebuilt.closed_by == setups.PORTFOLIO_RISK_CAP
    assert rebuilt.cap_seen and not rebuilt.over_cap


def test_a_table_exactly_on_the_cap_is_not_reported_as_over_it() -> None:
    """Rounding to two places must not turn a held cap into a breach."""
    # 999.997 lost at each stop, 4,999.985 for five: inside 5,000. The
    # published entry and stop round 2.7027 of distance up to 2.71, so the
    # rebuilt total comes to about 5,012 - past the cap on raw arithmetic
    # and inside it once the two-decimal rounding is allowed for.
    book = [_setup(f"R{i}", 1.0 - i * 0.01, lot_risk=899.997, cost=100.0,
                   entry=123.457, quantity=333) for i in range(5)]
    capped = setups.apply_portfolio_caps(book, capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert len(_kept(capped)) == 5
    rebuilt = setups.published_exposure(_published(capped),
                                        capital=100_000.0,
                                        max_risk_pct=5.0, leverage=5.0)
    assert rebuilt.exposure.risk_used > 5_000.0, "fixture lost its point"
    assert not rebuilt.over_cap


def test_a_table_from_a_feed_without_the_cap_is_reported_over_it() -> None:
    """Eight uncapped 1,000-rupee setups: 8,000 against a 5,000 cap."""
    rebuilt = setups.published_exposure(_published(_book(8)),
                                        capital=100_000.0,
                                        max_risk_pct=5.0, leverage=5.0)
    assert rebuilt.over_cap and not rebuilt.cap_seen
    assert rebuilt.exposure.kept == 8
    assert rebuilt.exposure.risk_used == pytest.approx(8_000.0, abs=5.0)


def test_published_exposure_survives_an_older_table() -> None:
    """No reasons, no cost column, a NaN actionable: none of it raises."""
    rows = [
        {"symbol": "A", "actionable": True, "entry": 100.0, "stop": 99.0,
         "quantity": 900},
        {"symbol": "B", "actionable": math.nan, "entry": 100.0,
         "stop": 99.0, "quantity": 900},
        {"symbol": "C", "actionable": True, "entry": None, "stop": None,
         "quantity": None},
        {"symbol": "D"},
    ]
    rebuilt = setups.published_exposure(rows, capital=100_000.0,
                                        max_risk_pct=5.0, leverage=5.0)
    assert rebuilt.exposure.kept == 2, "NaN is not actionable"
    assert rebuilt.exposure.risk_used == pytest.approx(900.0)
    assert rebuilt.uncharged == 1 and rebuilt.unpriced == 1
    assert rebuilt.held == () and not rebuilt.cap_seen
    assert setups.published_exposure([]).exposure.kept == 0


def test_the_dashboard_says_which_state_a_published_table_is_in() -> None:
    import app

    def state(rows):
        return app.published_cap_state(setups.published_exposure(
            rows, capital=100_000.0, max_risk_pct=5.0, leverage=5.0))

    capped = setups.apply_portfolio_caps(_book(8), capital=100_000.0,
                                         max_risk_pct=5.0, leverage=5.0)
    assert state(_published(capped)) == app.CAP_ENFORCED
    assert state(_published(_book(3))) == app.CAP_UNCONFIRMED
    assert state(_published(_book(8))) == app.CAP_EXCEEDED
