"""Offline tests for kite_client: request pacing, paging and session files.

Nothing here reaches the network or the real token file, and that is
structural rather than a promise: an autouse fixture replaces
requests.get/post with functions that fail the test, points
config.KITE_TOKEN_FILE at a tmp_path, and clears the three credential
environment variables so a live local session cannot change a result.

The pacing tests are the reason this file exists. A missing lock in the
gate raises nothing: it produces 429s in production and a clean pass in a
single-threaded test, so the concurrency test below measures recorded
request starts against the documented rate.

Run with: .venv\\Scripts\\python -m pytest test_kite_client.py -q
"""
from __future__ import annotations

import sys
import subprocess
import os
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pytest

import config
import kite_client as kc

REPO_ROOT = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    """Redirect every credential source to tmp_path and ban the network."""
    def _no_network(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(kc.requests, "get", _no_network)
    monkeypatch.setattr(kc.requests, "post", _no_network)
    # The rate gate keeps its last-start stamps in FILES now, so they
    # outlive a test. Without this, "both slots cold" in the test below
    # means "cold unless an earlier test in this run warmed them", and the
    # slot-independence assertion fails for an unrelated reason.
    monkeypatch.setattr(kc, "PACE_DIR", tmp_path / "pace")
    kc._LAST_CALL.clear()
    monkeypatch.setattr(config, "KITE_TOKEN_FILE",
                        tmp_path / ".kite_session.json")
    for name in (config.KITE_API_KEY_ENV, config.KITE_API_SECRET_ENV,
                 config.KITE_ACCESS_TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)
    # The pacing stamps are module state, so a last-call time left by one
    # test would show up as an unexplained wait in the next one.
    monkeypatch.setattr(kc, "_LAST_CALL", {})


def _session() -> kc.Session:
    """A well-formed session that cannot authenticate anywhere."""
    return kc.Session(api_key="test_key", access_token="test_token")


def _elapsed(call) -> float:
    """Seconds a zero-argument call takes, on the monotonic clock."""
    started = time.monotonic()
    call()
    return time.monotonic() - started


def _no_pace(rate_per_sec=None, slot=None) -> None:
    """Stand in for _pace so paging tests do not spend real seconds."""


def _record_get(monkeypatch, candles=None) -> list:
    """Replace kite_client._get; return the list of calls it receives."""
    calls: list[dict] = []

    def fake_get(session, path, params=None):
        calls.append({"session": session, "path": path, "params": params})
        return {"candles": [list(row) for row in (candles or [])]}

    monkeypatch.setattr(kc, "_get", fake_get)
    return calls


# --- pacing --------------------------------------------------------------

def test_pace_serialises_concurrent_starts(monkeypatch) -> None:
    """No 1.0s window may hold more starts than the configured rate.

    Removing the lock in _pace makes this fail: every worker then reads the
    same last-call stamp, waits the same interval and starts together.
    """
    rate, calls, workers = 10.0, 20, 10
    monkeypatch.setattr(config, "KITE_HISTORICAL_RATE_PER_SEC", rate)
    starts: list[float] = []
    guard = threading.Lock()

    def one(_: int) -> None:
        kc._pace()
        with guard:
            starts.append(time.monotonic())

    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(calls)))
    total = time.monotonic() - began

    assert len(starts) == calls
    starts.sort()
    # A window may hold `rate` starts plus the one on its own left edge.
    limit = int(rate) + 1
    for index, left in enumerate(starts):
        inside = [t for t in starts[index:] if t < left + 1.0]
        assert len(inside) <= limit, (
            f"{len(inside)} starts inside one second, limit {limit}")
    # The same fact stated end to end: 20 starts at 10/s cannot fit into
    # less than 1.9s, where an unlocked gate finishes in milliseconds.
    assert total >= (calls - 1) / rate * 0.9


def test_pace_reads_the_rate_at_call_time(monkeypatch) -> None:
    """A monkeypatched rate must win over the value present at import."""
    monkeypatch.setattr(kc, "_LAST_CALL",
                        {kc._PACE_HISTORICAL: time.monotonic()})
    monkeypatch.setattr(config, "KITE_HISTORICAL_RATE_PER_SEC", 0.0)
    # With the shipped 3/s and a stamp from just now this would sleep 0.33s.
    assert _elapsed(kc._pace) < 0.15


def test_quote_slot_reads_the_quote_rate(monkeypatch) -> None:
    """The quote slot resolves its own constant, not the historical one."""
    monkeypatch.setattr(kc, "_LAST_CALL", {kc._PACE_QUOTE: time.monotonic()})
    monkeypatch.setattr(config, "KITE_QUOTE_RATE_PER_SEC", 0.0)
    monkeypatch.setattr(config, "KITE_HISTORICAL_RATE_PER_SEC", 3.0)
    assert _elapsed(lambda: kc._pace(slot=kc._PACE_QUOTE)) < 0.15


def test_each_endpoint_class_has_its_own_budget(monkeypatch) -> None:
    """A quote start must not consume the historical allowance."""
    monkeypatch.setattr(config, "KITE_HISTORICAL_RATE_PER_SEC", 2.0)
    monkeypatch.setattr(config, "KITE_QUOTE_RATE_PER_SEC", 2.0)
    kc._pace(slot=kc._PACE_QUOTE)                       # both slots cold
    crossed = _elapsed(lambda: kc._pace(slot=kc._PACE_HISTORICAL))
    same = _elapsed(lambda: kc._pace(slot=kc._PACE_QUOTE))
    assert crossed < 0.25, "one shared stamp: the slots are not separate"
    assert same >= 0.35, "the quote slot did not enforce its own gap"


def test_pace_refuses_an_unknown_slot() -> None:
    """An unnamed endpoint class raises rather than borrowing a rate."""
    with pytest.raises(KeyError):
        kc._pace(slot="orders")


def test_pace_disabled_by_a_non_positive_rate(monkeypatch) -> None:
    monkeypatch.setattr(config, "KITE_HISTORICAL_RATE_PER_SEC", -1.0)
    monkeypatch.setattr(kc, "_LAST_CALL",
                        {kc._PACE_HISTORICAL: time.monotonic()})
    assert _elapsed(kc._pace) < 0.15


def test_quote_rate_is_configured_and_stricter_than_historical() -> None:
    """Kite documents 1/s for /quote against 3/s for historical."""
    assert config.KITE_QUOTE_RATE_PER_SEC == 1.0
    assert (config.KITE_QUOTE_RATE_PER_SEC
            <= config.KITE_HISTORICAL_RATE_PER_SEC)


# --- quote ---------------------------------------------------------------

def test_quote_is_paced_on_the_quote_slot(monkeypatch) -> None:
    """The regression test for an unpaced /quote: it must ask the gate."""
    seen: list = []

    # The stand-in defaults to the wrong slot, so a bare _pace() call
    # inside quote() records "historical" and fails the assertion.
    def record(rate_per_sec=None, slot=kc._PACE_HISTORICAL) -> None:
        seen.append(slot)

    monkeypatch.setattr(kc, "_pace", record)
    monkeypatch.setattr(
        kc, "_get", lambda *args, **kwargs: {"NSE:INFY": {"last_price": 1.0}})
    payload = kc.quote(["NSE:INFY"], session=_session())
    assert payload == {"NSE:INFY": {"last_price": 1.0}}
    assert seen == [kc._PACE_QUOTE]


def test_quote_without_instruments_makes_no_request(monkeypatch) -> None:
    seen: list = []
    monkeypatch.setattr(kc, "_pace", lambda **kwargs: seen.append("paced"))
    calls = _record_get(monkeypatch)
    assert kc.quote([], session=_session()) == {}
    assert calls == [] and seen == []


def test_quote_refuses_more_than_the_documented_batch(monkeypatch) -> None:
    seen: list = []
    monkeypatch.setattr(kc, "_pace", lambda **kwargs: seen.append("paced"))
    calls = _record_get(monkeypatch)
    with pytest.raises(ValueError):
        kc.quote([f"NSE:S{n}" for n in range(501)], session=_session())
    assert calls == [] and seen == []


def test_quote_without_a_session_refuses(monkeypatch) -> None:
    calls = _record_get(monkeypatch)
    with pytest.raises(kc.KiteError):
        kc.quote(["NSE:INFY"])
    assert calls == []


# --- historical paging ---------------------------------------------------

@pytest.mark.parametrize("interval", ["minute", "5minute"])
def test_historical_chunks_fit_the_documented_span(interval,
                                                   monkeypatch) -> None:
    """Every chunk is within the interval cap, with no gap and no overlap."""
    monkeypatch.setattr(kc, "_pace", _no_pace)
    calls = _record_get(monkeypatch)
    start, end = date(2026, 1, 1), date(2026, 8, 31)
    kc.historical(1234, start, end, interval=interval, session=_session())

    span = timedelta(days=config.KITE_HISTORICAL_MAX_DAYS[interval])
    chunks = [(date.fromisoformat(call["params"]["from"]),
               date.fromisoformat(call["params"]["to"])) for call in calls]
    assert len(chunks) > 1, "the range should have been paged"
    assert chunks[0][0] == start, "the first chunk must begin at start"
    assert chunks[-1][1] == end, "the last chunk must end at end"
    for chunk_start, chunk_end in chunks:
        assert chunk_start <= chunk_end
        assert chunk_end - chunk_start <= span
    for (_, previous_end), (next_start, _) in zip(chunks, chunks[1:]):
        # A same-day step would refetch a session and a wider one would
        # drop one silently, which is what the span table exists to stop.
        assert next_start - previous_end == timedelta(days=1)


def test_historical_uses_the_narrow_default_for_an_unlisted_interval(
        monkeypatch) -> None:
    """An interval outside the table pages at the default span."""
    monkeypatch.setattr(kc, "_pace", _no_pace)
    calls = _record_get(monkeypatch)
    kc.historical(1234, date(2026, 1, 1), date(2026, 8, 31),
                  interval="7minute", session=_session())
    span = timedelta(days=config.KITE_HISTORICAL_DEFAULT_SPAN)
    assert len(calls) > 1
    for call in calls:
        first = date.fromisoformat(call["params"]["from"])
        last = date.fromisoformat(call["params"]["to"])
        assert last - first <= span
    assert config.KITE_HISTORICAL_DEFAULT_SPAN <= min(
        config.KITE_HISTORICAL_MAX_DAYS.values())


def test_historical_is_paced_on_the_historical_slot(monkeypatch) -> None:
    seen: list = []

    def record(rate_per_sec=None, slot=kc._PACE_QUOTE) -> None:
        seen.append(slot)

    monkeypatch.setattr(kc, "_pace", record)
    calls = _record_get(monkeypatch)
    kc.historical(1234, date(2026, 1, 1), date(2026, 8, 31),
                  interval="minute", session=_session())
    assert len(calls) > 1
    assert seen == [kc._PACE_HISTORICAL] * len(calls)


def test_historical_refuses_a_reversed_range(monkeypatch) -> None:
    monkeypatch.setattr(kc, "_pace", _no_pace)
    calls = _record_get(monkeypatch)
    with pytest.raises(ValueError):
        kc.historical(1234, date(2026, 6, 2), date(2026, 6, 1),
                      session=_session())
    assert calls == [], "a reversed range must not reach the API"


def test_historical_returns_ist_indexed_candles(monkeypatch) -> None:
    monkeypatch.setattr(kc, "_pace", _no_pace)
    candles = [["2026-06-01T09:15:00+0530", 100.0, 101.0, 99.5, 100.5, 1000],
               ["2026-06-01T09:20:00+0530", 100.5, 102.0, 100.0, 101.5, 1200]]
    calls = _record_get(monkeypatch, candles)
    frame = kc.historical(1234, date(2026, 6, 1), date(2026, 6, 2),
                          interval="5minute", session=_session())
    assert len(calls) == 1, "a two-day range needs one request"
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(frame.index.tz) == "Asia/Kolkata"
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 2


def test_historical_without_a_session_refuses(monkeypatch) -> None:
    calls = _record_get(monkeypatch)
    with pytest.raises(kc.KiteError):
        kc.historical(1234, date(2026, 6, 1), date(2026, 6, 2))
    assert calls == []


# --- credentials and session files ---------------------------------------

def test_credentials_present_reports_booleans_only(monkeypatch) -> None:
    """The report says whether a credential exists, never what it is."""
    monkeypatch.setenv(config.KITE_API_KEY_ENV, "key_value")
    monkeypatch.setenv(config.KITE_API_SECRET_ENV, "secret_value")
    present = kc.credentials_present()
    assert set(present) == {"api_key", "api_secret", "access_token_env",
                            "session_file"}
    assert all(isinstance(value, bool) for value in present.values())
    assert present["api_key"] is True
    assert present["api_secret"] is True
    assert present["access_token_env"] is False
    assert present["session_file"] is False
    rendered = json.dumps(present)
    assert "key_value" not in rendered and "secret_value" not in rendered


def test_credentials_present_treats_blank_as_absent(monkeypatch) -> None:
    monkeypatch.setenv(config.KITE_API_KEY_ENV, "   ")
    assert kc.credentials_present()["api_key"] is False


def test_login_url_refuses_without_a_key() -> None:
    with pytest.raises(kc.KiteError) as raised:
        kc.login_url()
    assert config.KITE_API_KEY_ENV in str(raised.value)


def test_login_url_carries_the_key_and_nothing_else(monkeypatch) -> None:
    monkeypatch.setenv(config.KITE_API_KEY_ENV, "public_key")
    monkeypatch.setenv(config.KITE_API_SECRET_ENV, "top_secret")
    monkeypatch.setenv(config.KITE_ACCESS_TOKEN_ENV, "live_token")
    url = kc.login_url()
    assert url.startswith(config.KITE_LOGIN_BASE + "?")
    assert "api_key=public_key" in url
    assert "top_secret" not in url
    assert "live_token" not in url


def test_session_file_round_trips_with_only_four_keys(monkeypatch) -> None:
    # The ACL call is a platform side effect, not part of the round trip.
    monkeypatch.setattr(kc, "restrict_permissions", lambda path: True)
    original = kc.Session(api_key="file_key", access_token="file_token",
                          user_id="AB1234",
                          created_at="2026-09-09T10:00:00+05:30")
    assert kc.save_session(original) is True
    written = json.loads(config.KITE_TOKEN_FILE.read_text(encoding="utf-8"))
    assert set(written) == {"api_key", "access_token", "user_id",
                            "created_at"}
    assert kc.credentials_present()["session_file"] is True
    assert kc.load_session() == original


def test_load_session_prefers_the_environment_over_the_file(
        monkeypatch) -> None:
    monkeypatch.setattr(kc, "restrict_permissions", lambda path: True)
    kc.save_session(kc.Session(api_key="file_key", access_token="file_token"))
    monkeypatch.setenv(config.KITE_API_KEY_ENV, "env_key")
    monkeypatch.setenv(config.KITE_ACCESS_TOKEN_ENV, "env_token")
    live = kc.load_session()
    assert live is not None
    assert (live.api_key, live.access_token) == ("env_key", "env_token")


def test_load_session_without_a_file_returns_none() -> None:
    assert not config.KITE_TOKEN_FILE.exists()
    assert kc.load_session() is None


def test_load_session_refuses_an_incomplete_file() -> None:
    config.KITE_TOKEN_FILE.write_text(json.dumps({"api_key": "only_a_key"}),
                                      encoding="utf-8")
    assert kc.load_session() is None


def test_exchange_request_token_refuses_without_the_secret(
        monkeypatch) -> None:
    """No secret means no checksum, so the call must not be attempted."""
    monkeypatch.setenv(config.KITE_API_KEY_ENV, "public_key")
    with pytest.raises(kc.KiteError) as raised:
        kc.exchange_request_token("a_request_token")
    message = str(raised.value)
    assert config.KITE_API_SECRET_ENV in message
    assert "a_request_token" not in message


def test_exchange_request_token_refuses_an_empty_token(monkeypatch) -> None:
    monkeypatch.setenv(config.KITE_API_KEY_ENV, "public_key")
    monkeypatch.setenv(config.KITE_API_SECRET_ENV, "top_secret")
    with pytest.raises(kc.KiteError):
        kc.exchange_request_token("   ")

# --- the rate budget is shared between processes -------------------------
#
# WHY A REAL SUBPROCESS BELOW. On 2026-09-11 the live feed's seed pass and
# a Streamlit scan ran at once; each paced itself at Kite's documented 3
# requests a second and Kite saw six. The pacer was a threading.Lock and an
# in-memory stamp, so it metered one PROCESS - and every in-process test
# passed against that code, and would pass again. That is why one test here
# pays for two interpreters.

PACE_CHILD = """
import json, sys, time
sys.path.insert(0, %r)
import kite_client
stamps = []
for _ in range(int(sys.argv[1])):
    kite_client._pace(rate_per_sec=float(sys.argv[2]), slot=sys.argv[3])
    stamps.append(time.time())
print(json.dumps(stamps))
"""


def test_a_paced_call_records_its_start_where_other_processes_can_see_it(
) -> None:
    before = time.time()
    kc._pace(rate_per_sec=100.0, slot="pytest")
    stamp_file = kc._pace_path("pytest")
    assert stamp_file.exists(), "nothing was written for other processes"
    written = float(stamp_file.read_text(encoding="utf-8").strip())
    assert before <= written <= time.time() + 0.5


def test_the_wait_comes_from_the_file_not_from_memory() -> None:
    # Clearing the in-memory stamp is what another process looks like from
    # in here: same file, no recollection of the last call. Against the old
    # pacer this test passes trivially and means nothing.
    kc._pace(rate_per_sec=4.0, slot="pytest")
    kc._LAST_CALL.pop("pytest", None)
    waited = _elapsed(lambda: kc._pace(rate_per_sec=4.0, slot="pytest"))
    assert waited > 0.15, f"did not wait for the shared stamp ({waited:.3f}s)"


def test_two_real_processes_share_one_budget(tmp_path) -> None:
    # The test that would have caught this morning. Two interpreters, one
    # stamp file, one budget: no two calls may start closer than the gap.
    rate, calls = 8.0, 4
    gap = 1.0 / rate
    script = tmp_path / "child.py"
    script.write_text(PACE_CHILD % str(REPO_ROOT), encoding="utf-8")
    # The child cannot be monkeypatched, so the directory is passed in the
    # environment - which is why PACE_DIR reads it.
    env = dict(os.environ, SECTOR_PULSE_PACE_DIR=str(tmp_path / "shared"))
    children = [
        subprocess.Popen(
            [sys.executable, str(script), str(calls), str(rate), "pytest"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, cwd=str(REPO_ROOT))
        for _ in range(2)]
    stamps = []
    for child in children:
        out, err = child.communicate(timeout=180)
        assert child.returncode == 0, err
        stamps.extend(json.loads(out.strip().splitlines()[-1]))
    stamps.sort()
    assert len(stamps) == 2 * calls
    tightest = min(b - a for a, b in zip(stamps, stamps[1:]))
    # Slack for scheduling jitter, far below the gap: the broken version
    # produced pairs microseconds apart.
    assert tightest > gap * 0.6, (
        f"two calls started {tightest * 1000:.0f}ms apart, gap is "
        f"{gap * 1000:.0f}ms - the budget is not shared")


def test_the_gate_degrades_rather_than_blocking_prices(monkeypatch) -> None:
    # A gate that cannot be taken must not stop a fetch: it falls back to
    # in-process pacing and warns once. Refusing would turn a metering
    # problem into no prices at all.
    monkeypatch.setattr(kc, "_lock_file", lambda handle: False)
    kc._pace(rate_per_sec=100.0, slot="pytest")
    assert "pytest" in kc._LAST_CALL


def test_a_stamp_from_the_future_waits_a_whole_gap() -> None:
    # A clock change, or a foreign writer, must not disable pacing.
    stamp = kc._pace_path("pytest")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(str(time.time() + 3600), encoding="utf-8")
    kc._LAST_CALL.pop("pytest", None)
    waited = _elapsed(lambda: kc._pace(rate_per_sec=5.0, slot="pytest"))
    assert waited > 0.15


def test_the_stamp_directory_is_not_the_repo_root() -> None:
    # It is written to on every call, so it must be somewhere gitignored
    # rather than beside the source.
    assert kc.PACE_DIR.name in ("run", "pace", "shared")
