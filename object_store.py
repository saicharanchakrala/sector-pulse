"""Shared object storage, so the feed and the app need not share a disk.

WHY THIS EXISTS. The feed and the scanner have always talked to each other
through one file: live_bars/live_YYYYMMDD.parquet, about 2 MB a session.
Everything else on disk - 690 MB of bar_cache, 559 MB of forecast_cache -
is history the app reads and the feed never touches. So moving the feed to
a container needs 2 MB a day to cross the network, not 1.25 GB.

This module is that crossing, and nothing more. It is deliberately not a
database: parquet round-trips byte-exact through an object store, which
means the partial-bar flags, the tz-aware index and the column contract
that live_bars guarantees all survive without a schema to keep in step.

OFF BY DEFAULT. With SECTOR_PULSE_S3_BUCKET unset every caller falls back
to the local filesystem and behaves exactly as it did before this module
existed. That is what lets the whole test suite go on describing local
behaviour, and what lets the app run with no AWS credentials at all.

A MISSING OBJECT AND AN UNREACHABLE BUCKET ARE NOT THE SAME THING. This is
the one subtlety worth the reader's attention. If a credentials failure
came back as "no data", load_today would return {} and the scan would
quietly decide the feed had not started yet - then download everything at
three requests a second while reporting nothing wrong. So get() returns
None only for a genuine 404, and raises StorageError for anything else.
Callers are expected to let that be loud.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

BUCKET_ENV = "SECTOR_PULSE_S3_BUCKET"
PREFIX_ENV = "SECTOR_PULSE_S3_PREFIX"
REGION_ENV = "SECTOR_PULSE_S3_REGION"
DEFAULT_PREFIX = "zone-pulse"
# The ranking the containerised feed picks its universe from.
# Named here rather than in either end so the writer and the
# reader cannot drift on to two different keys.
TURNOVER_OBJECT = "turnover.parquet"

# One client per process. boto3 clients are thread-safe for calls but
# creating one costs a credential resolution, and the feed flushes every
# twenty seconds for six and a half hours.
_client_lock = threading.Lock()
_client = None


class StorageError(RuntimeError):
    """The store was configured but could not be reached or read."""


def bucket() -> "str | None":
    """The configured bucket, or None when this is a local-disk run."""
    name = (os.environ.get(BUCKET_ENV) or "").strip()
    return name or None


def prefix() -> str:
    """Key prefix, without a trailing slash."""
    raw = (os.environ.get(PREFIX_ENV) or DEFAULT_PREFIX).strip()
    return raw.strip("/")


def enabled() -> bool:
    """Whether reads and writes should go to the object store."""
    return bucket() is not None


def key_for(name: str) -> str:
    """The object key a bare file name maps to."""
    stem = prefix()
    return f"{stem}/{name}" if stem else name


def client():
    """A cached S3 client. Imported lazily so local runs need no boto3."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - env dependent
                raise StorageError(
                    f"{BUCKET_ENV} is set but boto3 is not installed"
                ) from exc
            region = (os.environ.get(REGION_ENV) or "").strip() or None
            _client = boto3.client("s3", region_name=region)
    return _client


def reset() -> None:
    """Drop the cached client. For tests and for credential rotation."""
    global _client
    with _client_lock:
        _client = None


def get(name: str) -> "bytes | None":
    """Object bytes, or None when it genuinely does not exist.

    Raises StorageError for anything else - denied, throttled, no network.
    A caller that cannot tell those apart from an absent object will report
    an empty feed instead of a broken one.
    """
    if not enabled():
        return None
    try:
        answer = client().get_object(Bucket=bucket(), Key=key_for(name))
        return answer["Body"].read()
    except Exception as exc:
        if _is_missing(exc):
            return None
        raise StorageError(f"Could not read {key_for(name)}: {exc}") from exc


def version(name: str) -> "str | None":
    """The object's ETag, or None when it does not exist.

    A HEAD, so it moves a few hundred bytes rather than the body. Lets a
    caller skip re-downloading and re-parsing something it already holds:
    the published scan is republished every half minute or so, while a
    Streamlit page re-executes its whole script on every interaction, so
    most reads are of bytes the process already has.

    Same contract as get(): None means genuinely absent, and anything else
    raises. A denied HEAD read as "no object" would make a caller throw
    away a good cached table and go and fetch it again.
    """
    if not enabled():
        return None
    try:
        answer = client().head_object(Bucket=bucket(), Key=key_for(name))
    except Exception as exc:
        if _is_missing(exc):
            return None
        raise StorageError(f"Could not stat {key_for(name)}: {exc}") from exc
    tag = answer.get("ETag")
    if tag is None:
        return None
    # S3 quotes ETags; the quoting is not part of the identity.
    return str(tag).strip('"')


def warm() -> None:
    """Build the client and resolve credentials, off the critical path.

    Measured 2026-09-17 from India against ap-southeast-2: the first read
    of the published scan cost 7.65 seconds and the second 0.29, so nearly
    all of it is boto3 import, credential resolution and the TLS
    handshake. The client is already cached per process - this only moves
    who pays for building it, which matters when the alternative is the
    first page render.

    Never raises. A failure here costs the warm-up and nothing else; the
    real call will report it properly.
    """
    if not enabled():
        return
    try:
        client()
    except Exception as exc:
        # WARNING, not INFO. INFO is below the default level, so a genuine
        # credentials failure would be invisible here and then be
        # rediscovered - at the full cost - by the first render.
        logger.warning("Could not warm the object store client: %s", exc)


def put(name: str, payload: bytes) -> None:
    """Write bytes under `name`. Raises StorageError on failure.

    A single put_object of the whole body is atomic for readers: S3 serves
    the old object or the new one and never a half-written mixture, so the
    write-temporary-then-rename dance the local path needs has no analogue
    and no purpose here.
    """
    if not enabled():
        return
    try:
        client().put_object(Bucket=bucket(), Key=key_for(name), Body=payload)
    except Exception as exc:
        raise StorageError(f"Could not write {key_for(name)}: {exc}") from exc


def _is_missing(exc: Exception) -> bool:
    """Whether an exception means "no such object" rather than a fault."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        code = error.get("Code", "") if isinstance(error, dict) else ""
        if code in {"NoSuchKey", "404", "NotFound"}:
            return True
    return exc.__class__.__name__ == "NoSuchKey"


def describe() -> str:
    """One line for a status panel or a startup log."""
    if not enabled():
        return "local disk"
    return f"s3://{bucket()}/{prefix()}"
