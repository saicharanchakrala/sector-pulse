"""Tests for the object store that carries live bars between machines.

THE PROPERTY THIS FILE EXISTS FOR is the distinction in `_is_missing`: an
object that is not there yet, against a store that cannot be reached.

They must not be confused, because downstream they mean opposite things. A
missing object means the feed has not written yet, and the scan's correct
answer is to download. An unreachable store means something is broken, and
downloading is then exactly wrong - it re-fetches at three requests a
second what the feed has already published, while nothing reports a fault.

An independent review found that distinction was implemented correctly here
and then discarded one layer up, in the caller the scan actually gates on.
So this file pins the contract, and test_live_feed_cloud.py pins the
callers honouring it.
"""
from __future__ import annotations

import pytest

import object_store


class FakeClientError(Exception):
    """Shaped like botocore's ClientError, without needing botocore."""

    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


class FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class FakeS3:
    """Records calls so a test can assert on keys as well as content."""

    def __init__(self, objects=None, raises=None):
        self.objects = dict(objects or {})
        self.raises = raises
        self.puts = []
        self.gets = []

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3's own spelling
        self.gets.append((Bucket, Key))
        if self.raises:
            raise self.raises
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": FakeBody(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.puts.append((Bucket, Key, Body))
        if self.raises:
            raise self.raises
        self.objects[Key] = Body


@pytest.fixture()
def store(monkeypatch):
    """Configure the object store against a fake client."""
    def build(objects=None, raises=None, bucket="a-bucket", prefix="zone-pulse"):
        monkeypatch.setenv(object_store.BUCKET_ENV, bucket)
        monkeypatch.setenv(object_store.PREFIX_ENV, prefix)
        fake = FakeS3(objects, raises)
        object_store.reset()
        monkeypatch.setattr(object_store, "_client", fake)
        return fake
    return build


# --- off by default ------------------------------------------------------

def test_with_no_bucket_configured_the_store_is_inert() -> None:
    # Everything the app does locally must keep working with no AWS at all,
    # which is what lets the rest of the suite describe local behaviour.
    assert object_store.enabled() is False
    assert object_store.bucket() is None
    assert object_store.get("anything.parquet") is None
    assert object_store.put("anything.parquet", b"x") is None
    assert object_store.describe() == "local disk"


def test_a_blank_bucket_variable_counts_as_unset(monkeypatch) -> None:
    # An exported-but-empty variable is a normal way to turn something off
    # in a shell, and it must not read as a bucket named "".
    monkeypatch.setenv(object_store.BUCKET_ENV, "   ")
    assert object_store.enabled() is False


# --- the distinction that matters ---------------------------------------

def test_a_genuinely_absent_object_is_none(store) -> None:
    store({})
    assert object_store.get("live_20260915.parquet") is None


@pytest.mark.parametrize("code", [
    "AccessDenied",
    "403",
    "NoSuchBucket",
    "ExpiredToken",
    "InvalidAccessKeyId",
    "SlowDown",
    "RequestTimeTooSkewed",
])
def test_a_fault_raises_rather_than_reading_as_absent(store, code) -> None:
    # THE POINT OF THIS MODULE. Each of these, returned as None, would mean
    # "the feed has not started" - and the scan answers that by downloading
    # the whole universe while reporting nothing wrong.
    store({}, raises=FakeClientError(code))
    with pytest.raises(object_store.StorageError):
        object_store.get("live_20260915.parquet")


@pytest.mark.parametrize("code", ["NoSuchKey", "404", "NotFound"])
def test_the_three_absent_codes_are_the_only_absent_ones(store, code) -> None:
    store({}, raises=FakeClientError(code))
    assert object_store.get("live_20260915.parquet") is None


def test_an_exception_with_no_response_is_a_fault(store) -> None:
    # NoCredentialsError carries no .response at all. Treating a missing
    # attribute as "not found" would turn an unconfigured machine into a
    # silently empty feed.
    store({}, raises=RuntimeError("Unable to locate credentials"))
    with pytest.raises(object_store.StorageError):
        object_store.get("live_20260915.parquet")


def test_a_non_mapping_response_does_not_escape_as_attribute_error(store) -> None:
    # getattr(exc, "response", {}).get(...) raises AttributeError when
    # .response is a string, and that would leave get() rather than the
    # StorageError every caller is written against.
    boom = RuntimeError("odd")
    boom.response = "not a dict"
    store({}, raises=boom)
    with pytest.raises(object_store.StorageError):
        object_store.get("live_20260915.parquet")


# --- round trip ----------------------------------------------------------

def test_bytes_survive_a_round_trip_unchanged(store) -> None:
    fake = store({})
    payload = b"PAR1\x00\x01binary\xff"
    object_store.put("seed_20260915.parquet", payload)
    assert object_store.get("seed_20260915.parquet") == payload
    assert fake.puts[0][1] == "zone-pulse/seed_20260915.parquet"


def test_a_failed_write_raises_rather_than_reporting_success(store) -> None:
    store({}, raises=FakeClientError("AccessDenied"))
    with pytest.raises(object_store.StorageError):
        object_store.put("live_20260915.parquet", b"x")


# --- keys ----------------------------------------------------------------

def test_the_prefix_is_applied_and_slashes_do_not_double(monkeypatch) -> None:
    monkeypatch.setenv(object_store.BUCKET_ENV, "b")
    monkeypatch.setenv(object_store.PREFIX_ENV, "/zone-pulse/")
    assert object_store.key_for("live.parquet") == "zone-pulse/live.parquet"


def test_the_default_prefix_is_used_when_none_is_given(monkeypatch) -> None:
    monkeypatch.setenv(object_store.BUCKET_ENV, "b")
    assert object_store.prefix() == object_store.DEFAULT_PREFIX
    assert object_store.key_for("x") == f"{object_store.DEFAULT_PREFIX}/x"


def test_describe_names_the_bucket_and_prefix(monkeypatch) -> None:
    monkeypatch.setenv(object_store.BUCKET_ENV, "a-bucket")
    monkeypatch.setenv(object_store.PREFIX_ENV, "zone-pulse")
    assert object_store.describe() == "s3://a-bucket/zone-pulse"


def test_the_turnover_key_is_defined_once(monkeypatch) -> None:
    # live_feed reads it and publish_turnover writes it. Two literals would
    # let the feed rank on a file nothing was writing, with no error.
    import publish_turnover

    assert publish_turnover.OBJECT is object_store.TURNOVER_OBJECT
