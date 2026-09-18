"""The consolidated intraday store has to reach the container.

WHY. live_bars.prewarm reads the folded store first and refetches from
Kite whatever it cannot prove. A container has no forecast_cache and never
will, so it proved nothing and refetched everything: 17 days for ~2,500
symbols at Kite's three requests a second, measured 2026-09-18 at 836
seconds on every cold start. Three deploys that day spent about an hour of
a trading session in prewarm, and the 11:38 redeploy discarded a seed that
had already reached 795 of 2,486 symbols.

The file was always being built. It just never left the laptop.

WHAT MUST NOT BREAK:

  * A LOCAL store is authoritative. hydrate() must not overwrite the
    laptop's own 91 MB file with a published copy, ever.
  * The daily store must not be published by accident - 191 MB against
    the 1.4 MB daily_context.parquet the container actually reads.
  * A half-arrived download must not be readable, hence the rename.
  * Failure is the slow path, not a crash: every failure mode here has to
    leave the feed able to start.
"""
from __future__ import annotations

import pandas as pd
import pytest

import bar_store
import object_store


@pytest.fixture
def _store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bar_store, "STORE", tmp_path)
    return tmp_path


@pytest.fixture
def _bucket(monkeypatch):
    """An in-memory object store."""
    held: dict = {}
    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(object_store, "describe", lambda: "s3://test/prefix")
    monkeypatch.setattr(object_store, "put",
                        lambda name, payload: held.__setitem__(name, payload))
    monkeypatch.setattr(object_store, "get", lambda name: held.get(name))
    return held


def _write_local(path, rows=3):
    frame = pd.DataFrame({
        "symbol": ["AAA"] * rows,
        "stamp": pd.date_range("2026-09-18 09:15", periods=rows, freq="3min",
                               tz="Asia/Kolkata"),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
        "Volume": 1000.0,
    })
    frame.to_parquet(path, index=False)
    return frame


# --- the name ------------------------------------------------------------

def test_the_object_is_named_after_the_interval() -> None:
    assert bar_store.object_name("3minute") == "bars_3minute.parquet"
    assert bar_store.object_name("day") == "bars_day.parquet"


# --- publishing ----------------------------------------------------------

def test_publishing_uploads_the_folded_file(_store_dir, _bucket) -> None:
    _write_local(bar_store.store_path("3minute"))
    sent = bar_store.publish("3minute")
    assert sent > 0
    assert "bars_3minute.parquet" in _bucket


def test_publishing_without_a_store_is_a_no_op(_store_dir, _bucket) -> None:
    """Run before the fold, it must warn rather than upload nothing."""
    assert bar_store.publish("3minute") == 0
    assert _bucket == {}


def test_publishing_without_an_object_store_is_a_no_op(
        _store_dir, monkeypatch) -> None:
    _write_local(bar_store.store_path("3minute"))
    monkeypatch.setattr(object_store, "enabled", lambda: False)
    assert bar_store.publish("3minute") == 0


def test_the_daily_store_is_not_published_by_the_nightly_job() -> None:
    """191 MB, and no container reads it - daily_context.parquet is 1.4 MB.

    Asserted against the job itself, because the guard is the --publish
    argument rather than anything in the code.
    """
    from pathlib import Path
    job = Path(bar_store.ROOT / "run_nightly.cmd").read_text(
        encoding="utf-8", errors="replace")
    published = [line for line in job.splitlines()
                 if "--publish" in line and not line.strip().startswith("REM")]
    assert published, "the nightly job no longer publishes anything"
    for line in published:
        assert "--publish 3minute" in line, line
        assert "--publish day" not in line
        assert "--publish 3minute,day" not in line


# --- hydrating -----------------------------------------------------------

def test_hydrating_materialises_the_published_file(_store_dir,
                                                   _bucket) -> None:
    source = bar_store.store_path("3minute")
    original = _write_local(source, rows=5)
    _bucket["bars_3minute.parquet"] = source.read_bytes()
    source.unlink()

    assert bar_store.hydrate("3minute") is True
    assert source.exists()
    # And it is readable through the ordinary path, filters intact.
    got = bar_store.load("3minute")
    assert "AAA" in got
    assert len(got["AAA"]) == len(original)


def test_a_local_store_is_never_overwritten(_store_dir, _bucket) -> None:
    """THE ONE THAT MATTERS ON A LAPTOP.

    The local file is the authority - it is what the fold just built. A
    published copy is a convenience for machines that have none.
    """
    local = bar_store.store_path("3minute")
    _write_local(local, rows=9)
    before = local.read_bytes()
    _bucket["bars_3minute.parquet"] = b"a different, wrong payload"

    assert bar_store.hydrate("3minute") is False
    assert local.read_bytes() == before


def test_nothing_published_is_not_an_error(_store_dir, _bucket) -> None:
    assert bar_store.hydrate("3minute") is False
    assert not bar_store.store_path("3minute").exists()


def test_no_object_store_is_not_an_error(_store_dir, monkeypatch) -> None:
    monkeypatch.setattr(object_store, "enabled", lambda: False)
    assert bar_store.hydrate("3minute") is False


def test_a_failed_fetch_leaves_the_feed_able_to_start(
        _store_dir, monkeypatch) -> None:
    """Failure is the slow path, not a crash.

    The cost of failing here is refetching from Kite, which is exactly
    what happened before this existed - so it must never raise.
    """
    monkeypatch.setattr(object_store, "enabled", lambda: True)

    def boom(name):
        raise object_store.StorageError("denied")

    monkeypatch.setattr(object_store, "get", boom)
    assert bar_store.hydrate("3minute") is False


def test_no_temporary_file_is_left_behind(_store_dir, _bucket) -> None:
    """A half-arrived download must not be openable as the store."""
    source = bar_store.store_path("3minute")
    _write_local(source)
    _bucket["bars_3minute.parquet"] = source.read_bytes()
    source.unlink()

    bar_store.hydrate("3minute")
    leftovers = list(_store_dir.glob("*.tmp"))
    assert leftovers == [], leftovers


def test_a_round_trip_preserves_what_prewarm_reads(_store_dir,
                                                   _bucket) -> None:
    """publish then hydrate then load must give back the same bars."""
    local = bar_store.store_path("3minute")
    _write_local(local, rows=7)
    expected = bar_store.load("3minute")["AAA"]

    bar_store.publish("3minute")
    local.unlink()
    assert bar_store.hydrate("3minute") is True

    got = bar_store.load("3minute")["AAA"]
    pd.testing.assert_frame_equal(got, expected)
