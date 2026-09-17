"""The UI must not offer a scope that can take the machine down.

WHAT HAPPENED. The widest intraday scope carried its cost in its own
label - "downloads the illiquid tail" - and nothing else. Picking it on
2026-09-17, during market hours, fetched bars per symbol for about 2,570
names inside the Streamlit process: bar_cache went from roughly 12,000
files to 31,537, a plain directory listing of it timed out at two
minutes, the browser tab stopped answering, and the machine ran out of
CPU and memory. The same requests competed with the live feed for Kite's
three-a-second budget.

A warning in a label is not a guard. These tests pin the guard.

The catalogue itself is deliberately NOT trimmed - run_scan still has to
resolve a scope that arrives from a Streamlit session which selected it
before the gate existed, and falling back is what it must do with one.
"""
from __future__ import annotations

import pytest

import app
import config


@pytest.fixture
def _may_download(monkeypatch):
    def set_to(value):
        monkeypatch.setattr(config, "SCAN_UI_MAY_DOWNLOAD", value)
    return set_to


def test_downloading_is_off_by_default() -> None:
    """The default is the guard. A gate defaulting open is not a gate."""
    assert config.SCAN_UI_MAY_DOWNLOAD is False


def test_the_unbounded_scope_is_not_offered(_may_download) -> None:
    _may_download(False)
    assert app._UNBOUNDED_SCAN_SCOPE not in app.offered_scopes()


def test_the_cheap_scopes_are_still_offered(_may_download) -> None:
    _may_download(False)
    offered = app.offered_scopes()
    assert app._DEFAULT_SCAN_SCOPE in offered
    assert "All listed equities, first 300" in offered


def test_turning_the_flag_on_restores_the_full_catalogue(
        _may_download) -> None:
    """Deliberate opt-in still works - this is a guard, not a removal."""
    _may_download(True)
    assert app.offered_scopes() == dict(app._SCAN_SCOPES)


def test_the_catalogue_itself_is_not_trimmed() -> None:
    """run_scan must still be able to look up a scope it will refuse."""
    assert app._UNBOUNDED_SCAN_SCOPE in app._SCAN_SCOPES


def test_a_stale_session_choice_falls_back_rather_than_downloading(
        _may_download) -> None:
    """Streamlit persists the selectbox under its key across restarts.

    A session that chose the unbounded scope before this gate existed
    arrives with it still selected, and honouring it would reproduce the
    outage the gate exists to prevent.
    """
    _may_download(False)
    scope, limit = app.resolve_scope(app._UNBOUNDED_SCAN_SCOPE)
    assert scope == app._DEFAULT_SCAN_SCOPE
    assert limit == app._SCAN_SCOPES[app._DEFAULT_SCAN_SCOPE]


def test_an_unknown_scope_falls_back_too(_may_download) -> None:
    _may_download(False)
    scope, _ = app.resolve_scope("something renamed last release")
    assert scope == app._DEFAULT_SCAN_SCOPE


def test_none_falls_back_rather_than_raising(_may_download) -> None:
    _may_download(False)
    assert app.resolve_scope(None)[0] == app._DEFAULT_SCAN_SCOPE


def test_an_allowed_scope_is_returned_unchanged(_may_download) -> None:
    _may_download(False)
    for name, limit in app.offered_scopes().items():
        assert app.resolve_scope(name) == (name, limit)


def test_the_unbounded_scope_is_honoured_once_opted_in(
        _may_download) -> None:
    _may_download(True)
    scope, limit = app.resolve_scope(app._UNBOUNDED_SCAN_SCOPE)
    assert scope == app._UNBOUNDED_SCAN_SCOPE
    assert limit == 0


def test_the_unbounded_scope_was_never_an_autoscan_scope() -> None:
    """It must not be reachable by the refresh loop either."""
    assert app._UNBOUNDED_SCAN_SCOPE not in app._AUTOSCAN_SCOPES


def test_every_autoscan_scope_survives_the_gate(_may_download) -> None:
    """Otherwise auto-refresh selects a scope the user cannot pick."""
    _may_download(False)
    offered = app.offered_scopes()
    for name in app._AUTOSCAN_SCOPES:
        assert name in offered, name


def test_run_scan_narrows_the_universe_it_actually_scans(
        _may_download, monkeypatch) -> None:
    """End to end, not just resolve_scope in isolation.

    The gate is worth nothing if run_scan resolves the label and then
    builds its ticker list from the refused one.
    """
    _may_download(False)
    seen = {}

    class _Inst:
        def __init__(self, symbol):
            self.symbol = symbol

    class _Discovered:
        fo_stocks = [_Inst(f"FO{i}") for i in range(7)]
        equities = [_Inst(f"EQ{i}") for i in range(900)]

    monkeypatch.setattr(app.instruments, "load_latest",
                        lambda: _Discovered())
    monkeypatch.setattr(app.instruments, "to_ticker", lambda s: s)

    def capture(tickers, target=None, live_stamp=""):
        seen["tickers"] = list(tickers)
        raise RuntimeError("stop here - the universe is what is under test")

    monkeypatch.setattr(app, "load_scan_bars", capture)
    with pytest.raises(RuntimeError):
        app.run_scan(app._UNBOUNDED_SCAN_SCOPE)

    picked = [t for t in seen["tickers"] if t != config.SCAN_BENCHMARK]
    assert picked == [f"FO{i}" for i in range(7)], (
        "scanned the refused scope's universe")
