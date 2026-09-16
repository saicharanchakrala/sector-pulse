"""Tests for running the feed somewhere other than this machine.

The defect guarded first here is the one an independent review caught, and
it is worth stating plainly because it was invisible to 688 tests, to ruff
and to a successful container build:

    by_turnover put its fallback in the `except` arm of
    `bar_store.load("day")`. That call does not raise for an absent store -
    it returns {}. So on the container, which has no daily store and never
    will, the except arm never fired, the published ranking was read by
    nothing, and the feed streamed the ~216 F&O underlyings that this whole
    change exists to widen. The runbook then documented that symptom with a
    remedy that could not work.

A single test would have caught it. This is that test, plus the ones for
everything else the split touches.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

import live_bars
import live_feed
import object_store
from test_object_store import FakeClientError, FakeS3


@pytest.fixture()
def published(monkeypatch):
    """Configure an object store holding a turnover ranking."""
    def build(symbols_and_values, bucket="a-bucket"):
        frame = pd.DataFrame({"symbol": [s for s, _ in symbols_and_values],
                              "turnover": [v for _, v in symbols_and_values]})
        buffer = io.BytesIO()
        frame.to_parquet(buffer)
        monkeypatch.setenv(object_store.BUCKET_ENV, bucket)
        monkeypatch.setenv(object_store.PREFIX_ENV, "zone-pulse")
        object_store.reset()
        fake = FakeS3({f"zone-pulse/{object_store.TURNOVER_OBJECT}":
                       buffer.getvalue()})
        monkeypatch.setattr(object_store, "_client", fake)
        return fake
    return build


def _daily(symbol_to_rows):
    """Daily frames in the shape bar_store.load returns."""
    return {symbol: pd.DataFrame({"Close": closes, "Volume": volumes})
            for symbol, (closes, volumes) in symbol_to_rows.items()}


# --- the regression -----------------------------------------------------

def test_an_absent_daily_store_falls_back_to_the_published_ranking(
        published, monkeypatch) -> None:
    # bar_store.load returns {} rather than raising, which is exactly how a
    # container arrives here. Keying the fallback off an exception meant it
    # never ran at all.
    import bar_store

    published([("AAA", 9e8), ("BBB", 5e8), ("CCC", 1e8)])
    monkeypatch.setattr(bar_store, "load", lambda *a, **k: {})
    assert live_feed.by_turnover(2) == ["AAA", "BBB"]


def test_the_fallback_also_runs_when_the_store_raises(
        published, monkeypatch) -> None:
    import bar_store

    published([("AAA", 9e8), ("BBB", 5e8)])

    def boom(*a, **k):
        raise FileNotFoundError("no such store")

    monkeypatch.setattr(bar_store, "load", boom)
    assert live_feed.by_turnover(5) == ["AAA", "BBB"]


def test_a_local_store_is_preferred_and_the_network_is_not_touched(
        published, monkeypatch) -> None:
    # The laptop has 559 MB of daily bars. It must rank on those rather
    # than on a file that may be a day behind.
    import bar_store

    fake = published([("PUBLISHED", 9e9)])
    monkeypatch.setattr(bar_store, "load", lambda *a, **k: _daily({
        "LOCAL_BIG": ([100.0] * 20, [10_000] * 20),
        "LOCAL_SMALL": ([10.0] * 20, [100] * 20),
    }))
    assert live_feed.by_turnover(2) == ["LOCAL_BIG", "LOCAL_SMALL"]
    assert fake.gets == [], "the published ranking must not have been read"


def test_nothing_published_and_no_store_reports_empty_rather_than_guessing(
        monkeypatch) -> None:
    import bar_store

    monkeypatch.setattr(bar_store, "load", lambda *a, **k: {})
    assert live_feed.by_turnover(5) == []


def test_an_unreachable_store_is_not_quietly_a_narrow_universe(
        monkeypatch) -> None:
    # Swallowing this would stream 216 names for the whole session because
    # a bucket was unreachable, with a warning nobody reads.
    import bar_store

    monkeypatch.setenv(object_store.BUCKET_ENV, "a-bucket")
    object_store.reset()
    monkeypatch.setattr(object_store, "_client",
                        FakeS3({}, raises=FakeClientError("AccessDenied")))
    monkeypatch.setattr(bar_store, "load", lambda *a, **k: {})
    with pytest.raises(object_store.StorageError):
        live_feed.by_turnover(5)


# --- the ranking itself -------------------------------------------------

def test_turnover_is_close_times_volume_over_twenty_sessions() -> None:
    frames = _daily({"A": ([10.0] * 25, [100] * 25)})
    # Only the last twenty count, and 10 * 100 = 1,000 each.
    assert live_feed.turnover_values(frames) == [(1000.0, "A")]


def test_the_ranking_is_descending_and_drops_what_cannot_be_ranked() -> None:
    frames = _daily({"BIG": ([10.0] * 20, [1000] * 20),
                     "SMALL": ([10.0] * 20, [10] * 20),
                     "ZERO": ([10.0] * 20, [0] * 20)})
    frames["NOCOLS"] = pd.DataFrame({"Close": [1.0]})
    frames["EMPTY"] = pd.DataFrame({"Close": [], "Volume": []})
    frames["NONE"] = None
    ranked = live_feed.turnover_values(frames)
    assert [s for _, s in ranked] == ["BIG", "SMALL"]


def test_a_nan_turnover_is_excluded_rather_than_sorted_as_zero() -> None:
    frames = _daily({"NAN": ([float("nan")] * 20, [100] * 20),
                     "REAL": ([10.0] * 20, [100] * 20)})
    assert [s for _, s in live_feed.turnover_values(frames)] == ["REAL"]


def test_the_published_file_is_trusted_to_be_ordered_but_reordered_anyway(
        published) -> None:
    # Written descending, but a hand-edited or partially rewritten file
    # must not silently invert the universe.
    published([("SMALL", 1.0), ("BIG", 9e9)])
    assert live_feed.turnover_from_store() == ["BIG", "SMALL"]


def test_an_unparseable_ranking_is_empty_rather_than_an_exception(
        monkeypatch) -> None:
    monkeypatch.setenv(object_store.BUCKET_ENV, "a-bucket")
    object_store.reset()
    monkeypatch.setattr(object_store, "_client", FakeS3(
        {f"zone-pulse/{object_store.TURNOVER_OBJECT}": b"not a parquet file"}))
    assert live_feed.turnover_from_store() == []


# --- the singleton guarantee --------------------------------------------

def test_the_file_lock_is_skipped_on_fargate(monkeypatch, tmp_path) -> None:
    # The lock guards a shared disk. A container's disk is neither shared
    # nor durable, so every task would take a fresh one and the guarantee
    # would be absent rather than merely different.
    monkeypatch.setattr(live_feed, "ON_FARGATE", True)
    monkeypatch.setattr(live_feed, "LOCK", tmp_path / "feed.lock")
    assert live_feed.take_lock() is True
    assert not (tmp_path / "feed.lock").exists(), "no lock file on Fargate"


def test_the_file_lock_still_works_locally(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(live_feed, "ON_FARGATE", False)
    monkeypatch.setattr(live_feed, "LOCK", tmp_path / "feed.lock")
    monkeypatch.setattr(live_bars, "STORE", tmp_path)
    assert live_feed.take_lock() is True
    assert (tmp_path / "feed.lock").exists()


def test_releasing_a_lock_that_was_never_taken_is_safe(
        monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(live_feed, "LOCK", tmp_path / "never-made.lock")
    live_feed.release_lock()


@pytest.mark.parametrize("pid", [0, -1, -12345])
def test_a_nonpositive_pid_is_not_alive(pid) -> None:
    # os.kill(0, 0) signals the whole process group, and a negative pid
    # signals another one. Neither is a question about a lock holder.
    assert live_feed._alive(pid) is False


# --- refusing to run somewhere it cannot publish ------------------------

def test_on_fargate_without_a_bucket_the_feed_refuses_to_start(
        monkeypatch) -> None:
    # Otherwise it runs all session, logs healthy flush lines, writes to a
    # disk that is discarded when the task stops, and reports nothing.
    monkeypatch.setattr(live_feed, "ON_FARGATE", True)
    assert object_store.enabled() is False
    assert live_feed.main([]) == 1


# --- live_bars through the object store ---------------------------------

@pytest.fixture()
def cloud_bars(monkeypatch):
    def build(objects=None):
        monkeypatch.setenv(object_store.BUCKET_ENV, "a-bucket")
        monkeypatch.setenv(object_store.PREFIX_ENV, "zone-pulse")
        object_store.reset()
        fake = FakeS3(objects or {})
        monkeypatch.setattr(object_store, "_client", fake)
        return fake
    return build


def _bar_frame():
    stamps = pd.date_range("2026-09-15 09:15", periods=3, freq="3min",
                           tz="Asia/Kolkata")
    return pd.DataFrame({
        "Date": stamps,
        "instrument_token": [256265] * 3,
        "Open": [100.0, 101.0, 102.0], "High": [101.0, 102.0, 103.0],
        "Low": [99.0, 100.0, 101.0], "Close": [100.5, 101.5, 102.5],
        "Volume": [10, 20, 30], "partial": [False, False, True],
    })


def test_bars_written_to_the_store_come_back_through_load_today(
        cloud_bars) -> None:
    from datetime import date

    when = date(2026, 9, 15)
    fake = cloud_bars()
    assert live_bars._write_frame(_bar_frame(), live_bars.store_name(when),
                                  live_bars.store_path(when), "live bars")
    assert fake.puts[0][1] == f"zone-pulse/{live_bars.store_name(when)}"
    back = live_bars.load_today({"NIFTY": 256265}, when=when)
    assert list(back) == ["NIFTY"]
    assert len(back["NIFTY"]) == 2, "the partial bar must be dropped"
    assert list(back["NIFTY"].columns) == live_bars.COLUMNS


def test_load_today_raises_on_a_fault_rather_than_reporting_no_feed(
        monkeypatch) -> None:
    from datetime import date

    monkeypatch.setenv(object_store.BUCKET_ENV, "a-bucket")
    object_store.reset()
    monkeypatch.setattr(object_store, "_client",
                        FakeS3({}, raises=FakeClientError("ExpiredToken")))
    with pytest.raises(object_store.StorageError):
        live_bars.load_today(when=date(2026, 9, 15))


def test_status_reports_a_fault_instead_of_raising(monkeypatch) -> None:
    # status() renders on every page load, so it degrades - but it must
    # carry the reason, because scan_data gates on this and would
    # otherwise read a broken store as a feed that had not started.
    from datetime import date

    monkeypatch.setenv(object_store.BUCKET_ENV, "a-bucket")
    object_store.reset()
    monkeypatch.setattr(object_store, "_client",
                        FakeS3({}, raises=FakeClientError("AccessDenied")))
    state = live_bars.status(when=date(2026, 9, 15))
    assert state["present"] is False
    assert state.get("error"), "the reason must survive for the caller to read"


def test_an_empty_store_is_simply_not_present(cloud_bars) -> None:
    from datetime import date

    cloud_bars()
    assert live_bars.status(when=date(2026, 9, 15)) == {"present": False}
