"""The published scan panel has to update itself.

WHAT WAS WRONG. The only auto-refresh in the app re-RAN a local scan from
session state - refresh_scan_fragment, whose own message is "Run a scan
first; the timer refreshes an existing one". Nothing re-read the table the
FEED publishes, so the panel most of the page is given over to updated
only when something else happened to re-run the script. The feed
republishes about every half minute; the count and the "computed Ns ago"
caption sat frozen in between.

WHY POLLING IS AFFORDABLE HERE. It is a read, not a scan.
scan_publish.load HEADs the object and reuses the parsed table when the
ETag has not moved, so a tick with nothing new is one small request rather
than 244 KB and a parquet parse. That is the whole reason this can be on a
20-second timer while the local scan cannot.

The tests cover the poll-interval rule rather than Streamlit's fragment
machinery: when to arm a timer at all is the decision worth pinning, and
it is the part that can be wrong without anyone noticing - a timer left
armed overnight polls an object that cannot change until the next session.
"""
from __future__ import annotations

import app
import config
import object_store


def test_no_timer_when_there_is_no_object_store(monkeypatch) -> None:
    """On local disk there is no feed publishing anything to poll."""
    monkeypatch.setattr(object_store, "enabled", lambda: False)
    monkeypatch.setattr(app, "live_bars_in_hours", lambda: True)
    assert app.published_poll_interval() is None


def test_no_timer_outside_trading_hours(monkeypatch) -> None:
    """The feed is scaled to zero, so the object cannot change."""
    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(app, "live_bars_in_hours", lambda: False)
    assert app.published_poll_interval() is None


def test_a_timer_during_the_session(monkeypatch) -> None:
    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(app, "live_bars_in_hours", lambda: True)
    assert app.published_poll_interval() == \
        config.PUBLISHED_SCAN_POLL_SECONDS


def test_the_interval_has_a_floor(monkeypatch) -> None:
    """A misconfigured zero or negative would busy-loop the browser."""
    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(app, "live_bars_in_hours", lambda: True)
    monkeypatch.setattr(config, "PUBLISHED_SCAN_POLL_SECONDS", 0)
    assert app.published_poll_interval() >= 5
    monkeypatch.setattr(config, "PUBLISHED_SCAN_POLL_SECONDS", -30)
    assert app.published_poll_interval() >= 5


def test_the_poll_is_not_faster_than_the_feed_publishes() -> None:
    """Polling faster than the feed writes buys nothing but requests.

    The feed scans every scan_publish.DEFAULT_EVERY seconds and the scan
    itself takes tens of seconds, so new bytes appear at best once per
    interval.
    """
    import scan_publish
    assert config.PUBLISHED_SCAN_POLL_SECONDS >= \
        scan_publish.DEFAULT_EVERY / 2


def test_the_poll_is_faster_than_the_staleness_limit() -> None:
    """Otherwise the panel refuses its own table before refreshing it.

    render_published_scan warns and falls back past
    SCAN_PUBLISHED_MAX_AGE_SECONDS, so a poll slower than that would show
    the warning rather than fresher bars.
    """
    assert config.PUBLISHED_SCAN_POLL_SECONDS < \
        config.SCAN_PUBLISHED_MAX_AGE_SECONDS


def test_the_fragment_records_its_verdict_for_the_caller(
        monkeypatch) -> None:
    """A fragment cannot return to the script that declared it.

    render_scan_tab needs to know whether a published scan rendered, to
    decide whether to demote the local controls.
    """
    state: dict = {}
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "render_published_scan", lambda: True)
    app.published_scan_fragment()
    assert state["published_scan_shown"] is True

    monkeypatch.setattr(app, "render_published_scan", lambda: False)
    app.published_scan_fragment()
    assert state["published_scan_shown"] is False
