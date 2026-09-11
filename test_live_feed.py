"""Tests for the live feed's startup order and its background seeding.

WHY THIS FILE EXISTS. On 2026-09-11 the feed took its lock at 09:06 and
had still not subscribed by 09:28, because it fetched today's session so
far for all 1,006 symbols first - one historical call each at Kite's 3 a
second. The seed exists so a feed started mid-session still has the
opening range; at 1,006 symbols it became the reason the feed had no bars
for the first half hour. Nothing failed, nothing warned.

These tests pin the order and the chunking. Nothing here reaches the
network: backfill_today and write_seed are replaced.
"""
from __future__ import annotations

import inspect
import threading

import pytest

import live_bars
import live_feed


@pytest.fixture
def fake_seed(monkeypatch):
    """Record what backfill_today is asked for, and what gets written."""
    asked: list = []
    written: list = []

    def backfill(symbols, interval=None):
        asked.append(list(symbols))
        return {s: "frame" for s in symbols}

    def write(frames, when=None):
        written.append(sorted(frames))
        return len(frames)

    monkeypatch.setattr(live_bars, "backfill_today", backfill)
    monkeypatch.setattr(live_bars, "write_seed", write)
    monkeypatch.setattr(live_bars, "session_is_covered", lambda frame: True)
    return asked, written


def test_the_socket_opens_before_the_seed_runs() -> None:
    """The ordering that regressed, asserted on the source of main().

    Brittle, and deliberately so: the alternative is a test that needs a
    Kite session and a websocket, and the cost of not testing it was
    thirty-five minutes of a live session with no bars.
    """
    source = inspect.getsource(live_feed.main)
    seeder = source.index("seed_session")
    stream = source.index("asyncio.run(run())")
    assert seeder < stream, "the seed must be STARTED before the stream"
    # And it must be started on a thread rather than called, or starting it
    # earlier just moves the blockage.
    started = source.index("seeder.start()")
    assert "threading.Thread(target=seed_session" in source
    assert started < stream
    # Nothing may call the seed synchronously in main any more.
    assert "= seed_session(" not in source
    assert "backfill_today" not in source


def test_the_seed_is_written_in_chunks_as_it_arrives(fake_seed) -> None:
    # A seed file that appears only when all 1,006 symbols are done is
    # worth nothing to a scan running now.
    asked, written = fake_seed
    symbols = {f"S{i}": i for i in range(250)}
    rows = live_feed.seed_session(symbols)
    assert [len(chunk) for chunk in asked] == [100, 100, 50]
    assert len(written) == 3, "the seed was written once, not progressively"
    # Each write carries everything so far, so the file only ever grows.
    assert [len(names) for names in written] == [100, 200, 250]
    assert rows == 250


def test_one_bad_chunk_does_not_lose_the_rest(monkeypatch) -> None:
    calls = {"n": 0}

    def flaky(symbols, interval=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Kite said no")
        return {s: "frame" for s in symbols}

    monkeypatch.setattr(live_bars, "backfill_today", flaky)
    monkeypatch.setattr(live_bars, "write_seed",
                        lambda frames, when=None: len(frames))
    monkeypatch.setattr(live_bars, "session_is_covered", lambda frame: True)
    rows = live_feed.seed_session({f"S{i}": i for i in range(300)})
    # 300 symbols in three chunks, the middle one failed: 200 survive.
    assert rows == 200


def test_seeding_stops_when_the_feed_is_shutting_down(fake_seed) -> None:
    # Ctrl-C must not leave a thread pulling history for ten more minutes.
    asked, _ = fake_seed
    stop = threading.Event()
    stop.set()
    live_feed.seed_session({f"S{i}": i for i in range(300)}, stop)
    assert asked == [], "it kept fetching after the stop was set"


def test_a_seed_failure_never_reaches_the_stream(monkeypatch) -> None:
    # This runs in a thread beside the socket. An exception escaping here
    # would end the process and cost the whole session, not just the
    # opening range.
    def explode(symbols, interval=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(live_bars, "backfill_today", explode)
    assert live_feed.seed_session({"A": 1}) == 0
