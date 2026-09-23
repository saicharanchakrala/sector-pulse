"""Caches whose keys churn faster than their TTL must be capped.

WHAT HAPPENED. On 2026-09-17 the widest scan scope ran the machine out of
memory. The downloads filled it; these caches held onto it.

load_scan_bars is keyed partly on live_bar_stamp, which is
"{newest bar}|{total bar count}" - and the COUNT rises with every bar the
feed writes. Measured from the feed's own log that morning it moved every
20 seconds: 2257, 2322, 2357, 2380, 2397, 2412, 2423, 2432. Against a
900-second TTL and no cap that is about forty-five live BarSets, each
holding intraday and daily frames for the whole scope, and cache_resource
keeps the objects rather than sharing them. At 2,570 symbols one BarSet is
roughly 5,140 DataFrames.

load_horizon_picks HAD the same shape for the same reason - `bucket` is
the clock floored to a window, deliberately, so live-anchored prices
refresh - and cache_data PICKLES, so its entries are independent copies.
It no longer does: its TTL was set equal to that window on 2026-09-23, so
its key cannot turn over inside its own TTL. See the test at the bottom.

Neither is a leak in the strict sense: the TTL does evict. But a key that
turns over forty-five times inside one TTL window is unbounded in every
sense that matters, and it scales with the scope.

These tests read the decorators rather than the behaviour, because
Streamlit's cache internals are not a stable contract and the thing worth
pinning is the DECISION - that these two are capped at all.
"""
from __future__ import annotations

import inspect

import app
import config


def _decorator_args(func):
    """The kwargs a Streamlit cache decorator was applied with.

    Streamlit wraps the function and keeps the original on the wrapper,
    so the source of the enclosing module is the reliable place to look.
    """
    source = inspect.getsource(app)
    name = func.__name__ if hasattr(func, "__name__") else str(func)
    marker = f"def {name}("
    index = source.index(marker)
    # Walk back to the decorator line(s) immediately above the def.
    head = source[:index].rstrip().splitlines()
    collected = []
    for line in reversed(head):
        collected.append(line.strip())
        if line.lstrip().startswith("@st.cache"):
            break
    return " ".join(reversed(collected))


def test_the_scan_bar_cache_is_capped() -> None:
    decorator = _decorator_args(app.load_scan_bars)
    assert "max_entries" in decorator, decorator


def test_the_scan_bar_cache_cap_is_small() -> None:
    """Two: the current entry, and the one it replaces mid-rerun.

    A large cap would not help - the point is that each entry can be
    thousands of DataFrames.
    """
    decorator = _decorator_args(app.load_scan_bars)
    assert "max_entries=2" in decorator.replace(" ", ""), decorator


def test_the_horizon_pick_cache_is_capped() -> None:
    decorator = _decorator_args(app.load_horizon_picks)
    assert "max_entries" in decorator, decorator


def test_the_key_really_does_churn_faster_than_the_ttl() -> None:
    """The premise, asserted so the cap is not later read as cargo cult.

    If live_bar_stamp ever stops including the bar count, the cap becomes
    unnecessary rather than load-bearing, and this test says where to look.
    """
    source = inspect.getsource(app.live_bar_stamp)
    assert "bars" in source, (
        "live_bar_stamp no longer keys on the bar count - re-check whether "
        "load_scan_bars still needs max_entries")
    # One entry per bar-count change, against the TTL, is the multiplier.
    assert config.CACHE_TTL_SECONDS >= 300, config.CACHE_TTL_SECONDS


def test_the_horizon_bucket_cannot_outchurn_its_own_ttl() -> None:
    """THIS FILE'S PREMISE NO LONGER APPLIES TO THIS CACHE.

    It used to assert HORIZON_ANCHOR_SECONDS < CACHE_TTL_SECONDS: the
    bucket was one minute against the shared fifteen-minute TTL, so the
    key turned over fifteen times inside one window and max_entries was
    genuinely load-bearing.

    On 2026-09-23 load_horizon_picks was given its OWN ttl, equal to the
    bucket, because the two disagreeing was a bug in the other direction -
    a TTL shorter than the window expires the entry mid-window and any
    full app run recomputes 7.4s of work the bucket said was current. With
    them equal, a new key arrives exactly as the old one expires and at
    most one entry is live per window.

    The cap above stays. It costs nothing, it still guards the shape, and
    it is the right default for a pickling cache - but it is no longer
    what bounds this one, and a reader comparing against CACHE_TTL_SECONDS
    would be looking at a constant this cache does not use.
    """
    assert app.HORIZON_ANCHOR_SECONDS == config.HORIZON_REFRESH_SECONDS
    decorator = _decorator_args(app.load_horizon_picks)
    assert "ttl=config.HORIZON_REFRESH_SECONDS" in decorator, decorator
