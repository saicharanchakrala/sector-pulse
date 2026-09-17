"""Reading the published scan must not re-download what it already holds.

WHY. The feed republishes the scan roughly every half minute, while
Streamlit re-executes its whole script on every interaction - so most
reads are for bytes this process has already parsed. Measured 2026-09-17
from India against ap-southeast-2, the first read cost 7.65 seconds and
later ones 0.29, nearly all of it connection setup rather than transfer.

WHAT MUST NOT BREAK. Two things, and they are the reason the age is not
cached alongside the table:

  * The caller REFUSES a table past an age. Caching the age would freeze
    that judgement, so a stale scan would keep reading as fresh for as
    long as the object happened not to move.
  * A failed HEAD is a fault, not an absence. Downgrading it to a full GET
    would hide a denied read behind a second failure, and treating it as
    "no object" would throw away a good cached table.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import object_store
import scan_publish

IST = ZoneInfo("Asia/Kolkata")


def _table(scanned_at=None, rows=3):
    stamp = scanned_at or datetime.now(IST)
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(rows)],
        "direction": ["LONG"] * rows,
        "actionable": [False] * rows,
        "score": [0.0] * rows,
        "scanned_at": [stamp.isoformat(timespec="seconds")] * rows,
    })


def _payload(table):
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False)
    return buffer.getvalue()


@pytest.fixture
def _store(monkeypatch):
    """A stub object store counting HEADs and GETs."""
    state = {"etag": "v1", "body": _payload(_table()), "gets": 0, "heads": 0}

    def version(name):
        state["heads"] += 1
        return state["etag"]

    def get(name):
        state["gets"] += 1
        return state["body"]

    monkeypatch.setattr(object_store, "version", version)
    monkeypatch.setattr(object_store, "get", get)
    monkeypatch.setattr(scan_publish, "_PARSED", {})
    return state


def test_the_first_read_downloads(_store) -> None:
    found = scan_publish.load()
    assert found is not None
    assert _store["gets"] == 1


def test_an_unchanged_object_is_not_downloaded_again(_store) -> None:
    scan_publish.load()
    scan_publish.load()
    scan_publish.load()
    assert _store["gets"] == 1, "re-downloaded an object that had not moved"
    assert _store["heads"] == 3


def test_a_changed_object_is_downloaded_again(_store) -> None:
    scan_publish.load()
    _store["etag"] = "v2"
    _store["body"] = _payload(_table(rows=5))
    table, _ = scan_publish.load()
    assert _store["gets"] == 2
    assert len(table) == 5, "served the old table after a republish"


def test_the_age_is_recomputed_on_every_read_not_cached(_store,
                                                        monkeypatch) -> None:
    """The caller refuses a stale table, so a frozen age is a wrong answer.

    Asserted with a MOVED CLOCK and a strict inequality. `second >= first`
    passes when the age is cached, which is the implementation this test
    exists to forbid.
    """
    scanned = datetime(2026, 9, 17, 10, 0, tzinfo=IST)
    _store["body"] = _payload(_table(scanned_at=scanned))
    _store["etag"] = "aged"

    clock = {"now": scanned + timedelta(seconds=60)}

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"]

    monkeypatch.setattr(scan_publish, "datetime", _Clock)

    _, first = scan_publish.load()
    clock["now"] = scanned + timedelta(seconds=300)
    _, second = scan_publish.load()

    assert _store["gets"] == 1, "precondition: served from the memo"
    assert first == pytest.approx(60, abs=1)
    assert second == pytest.approx(300, abs=1), \
        "the age was cached alongside the table"


def test_callers_do_not_share_one_frame(_store) -> None:
    """Every Streamlit session in the process reads through here.

    Handing them all the same object means anything mutating it in place
    corrupts the others. Today's reader sorts, which copies, but that is a
    property of today's caller rather than a promise this makes.
    """
    first, _ = scan_publish.load()
    second, _ = scan_publish.load()
    assert first is not second
    first["symbol"] = "MUTATED"
    third, _ = scan_publish.load()
    assert list(third["symbol"]) != ["MUTATED"] * len(third)


def test_the_memo_keeps_one_day_not_every_day(_store) -> None:
    """The key is date-stamped, so keeping all of them leaks a table a day."""
    scan_publish.load()
    _store["etag"] = "v2"
    scan_publish.load(when=datetime(2026, 9, 18, 10, 0, tzinfo=IST))
    assert len(scan_publish._PARSED) == 1, scan_publish._PARSED.keys()


def test_a_denied_head_is_not_read_as_an_absent_object(monkeypatch) -> None:
    """object_store.version's own classification, not a stub of it.

    Reading a denied HEAD as "no object" would throw away a good cached
    table and refetch it; reading it as absent would report no scan at all.
    """
    class _Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403}}

    class _Client:
        def head_object(self, **kwargs):
            raise _Denied()

    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(object_store, "bucket", lambda: "b")
    monkeypatch.setattr(object_store, "client", lambda: _Client())
    with pytest.raises(object_store.StorageError):
        object_store.version("x.parquet")


def test_a_genuinely_absent_object_heads_as_none(monkeypatch) -> None:
    class _Missing(Exception):
        response = {"Error": {"Code": "404"},
                    "ResponseMetadata": {"HTTPStatusCode": 404}}

    class _Client:
        def head_object(self, **kwargs):
            raise _Missing()

    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(object_store, "bucket", lambda: "b")
    monkeypatch.setattr(object_store, "client", lambda: _Client())
    assert object_store.version("x.parquet") is None


def test_a_head_failure_surfaces_rather_than_downgrading(monkeypatch) -> None:
    def boom(name):
        raise object_store.StorageError("denied")

    monkeypatch.setattr(object_store, "version", boom)
    monkeypatch.setattr(scan_publish, "_PARSED", {})
    with pytest.raises(object_store.StorageError):
        scan_publish.load()


def test_a_store_without_versions_still_works(monkeypatch) -> None:
    """Local disk has no ETag. It must read every time, not never."""
    calls = {"gets": 0}

    def get(name):
        calls["gets"] += 1
        return _payload(_table())

    monkeypatch.setattr(object_store, "version", lambda name: None)
    monkeypatch.setattr(object_store, "get", get)
    monkeypatch.setattr(scan_publish, "_PARSED", {})
    assert scan_publish.load() is not None
    assert scan_publish.load() is not None
    assert calls["gets"] == 2, "cached on a missing version"


def test_an_absent_object_is_none_not_an_empty_table(monkeypatch) -> None:
    monkeypatch.setattr(object_store, "version", lambda name: None)
    monkeypatch.setattr(object_store, "get", lambda name: None)
    monkeypatch.setattr(scan_publish, "_PARSED", {})
    assert scan_publish.load() is None


def test_an_unreadable_body_is_none_and_is_not_remembered(
        monkeypatch) -> None:
    monkeypatch.setattr(object_store, "version", lambda name: "v1")
    monkeypatch.setattr(object_store, "get", lambda name: b"not parquet")
    monkeypatch.setattr(scan_publish, "_PARSED", {})
    assert scan_publish.load() is None
    assert scan_publish._PARSED == {}, "remembered a table it could not read"


def test_a_table_without_a_stamp_reads_as_an_unknown_age(
        monkeypatch) -> None:
    table = _table().drop(columns=["scanned_at"])
    monkeypatch.setattr(object_store, "version", lambda name: "v1")
    monkeypatch.setattr(object_store, "get", lambda name: _payload(table))
    monkeypatch.setattr(scan_publish, "_PARSED", {})
    got, age = scan_publish.load()
    assert len(got) == 3
    assert age != age, "an unknown age must be nan, not a number"


# --- the store's own helpers --------------------------------------------

def test_version_strips_the_quotes_s3_puts_round_an_etag(
        monkeypatch) -> None:
    """Quoting is transport, not identity - comparing it unstripped works
    but makes the stored key depend on a detail of the caller."""
    class _Client:
        def head_object(self, **kwargs):
            return {"ETag": '"abc123"'}

    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(object_store, "bucket", lambda: "b")
    monkeypatch.setattr(object_store, "client", lambda: _Client())
    assert object_store.version("x.parquet") == "abc123"


def test_version_is_none_when_the_store_is_off(monkeypatch) -> None:
    monkeypatch.setattr(object_store, "enabled", lambda: False)
    assert object_store.version("x.parquet") is None


def test_warming_never_raises(monkeypatch) -> None:
    """A failed warm-up must cost the warm-up and nothing else."""
    def boom():
        raise RuntimeError("no credentials")

    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(object_store, "client", boom)
    object_store.warm()
