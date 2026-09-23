"""The short, mid and long horizon tables have to update themselves.

WHAT WAS WRONG. The refresh machinery was already here and unreachable.
horizon_anchor returns a time bucket that forms part of
load_horizon_picks' cache key, so the result self-invalidates every
HORIZON_ANCHOR_SECONDS, and the caption under the tables told the reader
how often the prices were refreshed. But render_horizon_tables was
called once at top level, wrapped in nothing, so only a full app run
re-evaluated the bucket. The tables sat on whatever they computed when the
page last happened to rerun, and the caption was a promise the page did
not keep.

WHY THE TICK IS AFFORDABLE. Out of hours horizon_anchor pins the bucket to
0, so the cache key does not move and the cached result is served without
recomputing or touching Kite. In hours it costs one batched quote sweep -
measured 2026-09-23 over the 210-symbol F&O universe at 4.1s, plus 3.3s to
assess, against an hourly tick. That sweep is on Kite's QUOTE budget,
batched 500 to a request, not the 3-a-second historical budget the feed
lives on.

These tests cover the anchor decision and the shape of the timer, not
Streamlit's fragment internals - the same split test_published_scan_refresh
makes, for the same reason.
"""
from __future__ import annotations

import inspect

import app


def _in_hours(monkeypatch, live=True, session=True):
    monkeypatch.setattr(app, "live_bars_in_hours", lambda: live)
    import market_source
    monkeypatch.setattr(market_source, "session_available", lambda: session)


# --- when the live anchor is used at all ---------------------------------

def test_out_of_hours_anchors_on_the_close(monkeypatch) -> None:
    """The last daily close is not a degraded anchor out of hours - it is
    where the instrument actually last traded."""
    _in_hours(monkeypatch, live=False)
    anchor_live, bucket, _ = app.horizon_anchor()
    assert anchor_live is False
    assert bucket == 0


def test_out_of_hours_the_bucket_is_frozen_so_the_tick_is_free(
        monkeypatch) -> None:
    """THE REASON AN UNCONDITIONAL TIMER IS SAFE.

    The bucket is part of load_horizon_picks' cache key. Pinned to 0, a
    tick re-reads the same key and Streamlit serves the cached result, so
    an overnight timer costs a rerun and no computation. If this ever
    returned a rotating bucket out of hours, every tick would reassess the
    whole universe against a daily store that cannot change until the
    nightly job runs.
    """
    _in_hours(monkeypatch, live=False)
    first = app.horizon_anchor()[1]
    second = app.horizon_anchor()[1]
    assert first == second == 0


def test_no_kite_session_falls_back_and_says_why(monkeypatch) -> None:
    _in_hours(monkeypatch, live=True, session=False)
    anchor_live, bucket, why = app.horizon_anchor()
    assert anchor_live is False
    assert bucket == 0
    assert "Kite session" in why, why


def test_in_hours_with_a_session_anchors_live(monkeypatch) -> None:
    _in_hours(monkeypatch, live=True, session=True)
    anchor_live, bucket, why = app.horizon_anchor()
    assert anchor_live is True
    assert bucket > 0
    assert why == ""


# --- the bucket is what makes the refresh happen -------------------------

def test_the_bucket_holds_still_inside_its_window(monkeypatch) -> None:
    """Otherwise every tick would recompute rather than every window."""
    _in_hours(monkeypatch)
    # horizon_anchor does `import time` inside the function, so the module
    # is what has to be patched. `app.time` is datetime.time - the name is
    # taken by the `from datetime import ... time` at the top of app.py.
    import time as _time
    # ALIGNED to a window boundary, whatever the window currently is. An
    # arbitrary epoch is not aligned, so "one second before the next
    # boundary" can already be past one - which fails against correct code.
    base = float(1_000_000 // app.HORIZON_ANCHOR_SECONDS
                 * app.HORIZON_ANCHOR_SECONDS)
    monkeypatch.setattr(_time, "time", lambda: base)
    first = app.horizon_anchor()[1]
    monkeypatch.setattr(_time, "time",
                        lambda: base + app.HORIZON_ANCHOR_SECONDS - 1)
    assert app.horizon_anchor()[1] == first


def test_the_bucket_moves_once_the_window_passes(monkeypatch) -> None:
    """THE REGRESSION TEST for a stale table. A bucket that never moved
    would make the cache key permanent and the tables frozen for the whole
    session, which is what the TTL alone would have allowed."""
    _in_hours(monkeypatch)
    import time as _time
    monkeypatch.setattr(_time, "time", lambda: 1_000_000.0)
    first = app.horizon_anchor()[1]
    monkeypatch.setattr(_time, "time",
                        lambda: 1_000_000.0 + app.HORIZON_ANCHOR_SECONDS + 1)
    assert app.horizon_anchor()[1] > first


# --- the timer's shape ---------------------------------------------------

def test_the_horizon_timer_is_unconditional() -> None:
    """THE BUG CLASS THIS GUARDS, and it has bitten this app before.

    positions_panel carries the note: "UNCONDITIONAL, and that is the fix
    for the worst bug review found. This used to pass run_every=None
    unless the market was open AND something was already watched - both
    evaluated on a full app run. Fragment ticks never cause a full run, so
    a page opened at 09:00 got no timer for the rest of the session."

    The horizons have no other fragment to wake them - published_scan gets
    away with a conditional interval only because autoscan_watch_fragment
    does. So the run_every here must be the bare constant, not a call that
    can return None.
    """
    source = inspect.getsource(app)
    marker = "lambda: render_horizon_tables("
    assert marker in source, "the horizon tables are not on a timer at all"
    tail = source.split(marker, 1)[1][:400]
    assert "run_every=HORIZON_ANCHOR_SECONDS" in tail, tail[:200]
    # A conditional interval is exactly the regression: any of these would
    # be evaluated once, on a full run, and frozen thereafter.
    for forbidden in ("run_every=None", "run_every=published_poll_interval",
                      "if live_bars_in_hours"):
        assert forbidden not in tail, forbidden


def test_the_interval_is_a_sane_cadence() -> None:
    """Bounded at both ends, for different reasons.

    The FLOOR is cost: one refresh is about 7.4s over the 210-symbol F&O
    universe, so a tick near that would overlap itself.

    The CEILING is meaning: a session runs 09:15 to 15:30, so beyond about
    two hours the tables would refresh two or three times a day and
    "refreshed about every ..." stops describing anything useful.

    RAISED TO AN HOUR on 2026-09-23. A shorter cadence bought nothing: the
    ranking columns come from completed daily bars over 10, 63 and 252
    sessions, and the daily store only advances overnight, so the ORDER
    cannot move during a session. Only the live price anchor does.
    """
    assert 30 <= app.HORIZON_ANCHOR_SECONDS <= 7200


def test_the_cache_ttl_matches_the_refresh_window() -> None:
    """THE REGRESSION TEST for a half-applied change.

    The refresh works by flooring the clock into a bucket that forms part
    of load_horizon_picks' cache key. If the TTL is shorter than the
    bucket, the entry expires mid-window and any full app run recomputes
    7.4s of work the bucket said was still current - so the cadence would
    describe the timer while the real work happened on the TTL. They have
    to be the same number.
    """
    import config
    assert config.HORIZON_REFRESH_SECONDS == app.HORIZON_ANCHOR_SECONDS
    source = inspect.getsource(app)
    marker = "def load_horizon_picks("
    head = source[:source.index(marker)]
    decorator = head[head.rindex("@st.cache_data"):]
    assert "ttl=config.HORIZON_REFRESH_SECONDS" in decorator, decorator
