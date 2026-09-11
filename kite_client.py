"""Authenticated Kite Connect access: quotes, historical candles, session.

Credentials are read from the environment and never written into this repo,
never logged, and never echoed. The login itself is yours to perform: Kite
issues a request_token only after an interactive login with your password
and 2FA, which is the correct design and means no tool can authenticate as
you unattended.

    setx KITE_API_KEY     your_key        (new shell afterwards, on Windows)
    setx KITE_API_SECRET  your_secret
    .venv\\Scripts\\python -m kite_login

kite_login prints a URL, you log in, Zerodha redirects to your app's
redirect URL carrying ?request_token=..., you paste that token back, and the
exchange happens locally. The resulting access token lands in a gitignored
file and expires around 6am the next day. Kite offers retail apps no
non-interactive refresh, so a daily login is unavoidable rather than an
oversight in this code.

Why this matters more than a live tick feed right now: the historical
endpoint serves minute candles going back years, where yfinance stops at
eight days and its 5-minute data at about sixty. The intraday edge search
is currently limited to 59 correlated sessions, which is why its results
are suggestive rather than conclusive. Depth fixes that. Speed does not.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)


class KiteError(RuntimeError):
    """A Kite request failed, or no session is available."""


@dataclass(frozen=True)
class Session:
    """A resolved Kite session. The token is a credential: do not log it."""

    api_key: str
    access_token: str
    user_id: str = ""
    created_at: str = ""

    @property
    def auth_header(self) -> dict[str, str]:
        """The Authorization header Kite expects."""
        return {"Authorization": f"token {self.api_key}:{self.access_token}",
                "X-Kite-Version": config.KITE_API_VERSION}


def _env(name: str) -> str:
    """One environment variable, stripped; empty string when unset."""
    return (os.environ.get(name) or "").strip()


def credentials_present() -> dict[str, bool]:
    """Which credentials are configured, without revealing any value."""
    return {
        "api_key": bool(_env(config.KITE_API_KEY_ENV)),
        "api_secret": bool(_env(config.KITE_API_SECRET_ENV)),
        "access_token_env": bool(_env(config.KITE_ACCESS_TOKEN_ENV)),
        "session_file": config.KITE_TOKEN_FILE.exists(),
    }


def login_url(api_key: "str | None" = None) -> str:
    """The URL to open in a browser to start a login.

    Contains only the public API key, which is not a secret: it identifies
    the app and appears in every redirect URL anyway.
    """
    key = api_key or _env(config.KITE_API_KEY_ENV)
    if not key:
        raise KiteError(
            f"{config.KITE_API_KEY_ENV} is not set. Set it in your "
            f"environment rather than passing it around.")
    return f"{config.KITE_LOGIN_BASE}?api_key={key}&v={config.KITE_API_VERSION}"


def exchange_request_token(request_token: str) -> Session:
    """Trade a request_token for an access token, locally.

    The checksum is SHA-256 of api_key + request_token + api_secret, which
    is why the secret must be present locally and must never travel through
    anything but this call. A request_token is single-use and expires within
    minutes, so it is not itself worth protecting the way the secret is.
    """
    key, secret = _env(config.KITE_API_KEY_ENV), _env(config.KITE_API_SECRET_ENV)
    token = (request_token or "").strip()
    missing = [name for name, value in
               ((config.KITE_API_KEY_ENV, key),
                (config.KITE_API_SECRET_ENV, secret)) if not value]
    if missing:
        raise KiteError(f"Not set in the environment: {', '.join(missing)}")
    if not token:
        raise KiteError("No request_token supplied")
    checksum = hashlib.sha256(f"{key}{token}{secret}".encode()).hexdigest()
    try:
        response = requests.post(
            f"{config.KITE_API_BASE}/session/token",
            data={"api_key": key, "request_token": token,
                  "checksum": checksum},
            headers={"X-Kite-Version": config.KITE_API_VERSION},
            timeout=config.KITE_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise KiteError(f"Session request failed: {exc}") from exc
    if response.status_code != 200:
        # Kite's error bodies do not contain the secret, but they can echo
        # the request_token, so only the message is surfaced.
        try:
            message = response.json().get("message", "")
        except ValueError:
            message = ""
        raise KiteError(f"Kite refused the session (HTTP "
                        f"{response.status_code}): {message}")
    payload = response.json().get("data") or {}
    access = payload.get("access_token")
    if not access:
        raise KiteError("Kite returned no access_token")
    return Session(api_key=key, access_token=access,
                   user_id=str(payload.get("user_id") or ""),
                   created_at=datetime.now().astimezone().isoformat())


def restrict_permissions(path) -> bool:
    """Restrict a file to the current user. True when it can be enforced.

    os.chmod only honours the write bit on Windows: read access and the
    NTFS ACL are untouched, so 0o600 leaves the mode at 0o666 and the file
    readable by anyone on the machine. icacls is what actually restricts it
    there, so the platform decides which mechanism is used - and the
    return value says whether anything was really enforced, because telling
    a user a token is protected when it is not is worse than saying nothing.
    """
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.warning("Could not chmod %s: %s", path, exc)
    if os.name != "nt":
        return True
    user = os.environ.get("USERNAME", "")
    if not user:
        return False
    try:
        import subprocess
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:R,W"],
            capture_output=True, text=True, timeout=20)
        return result.returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("Could not restrict the ACL on %s: %s", path, exc)
        return False


def save_session(session: Session) -> bool:
    """Persist a session to the gitignored token file.

    Returns whether the file's permissions could actually be restricted, so
    the caller can tell the truth about it rather than assuming.
    """
    payload = {"api_key": session.api_key, "access_token": session.access_token,
               "user_id": session.user_id, "created_at": session.created_at}
    path = config.KITE_TOKEN_FILE
    try:
        # Created with restrictive permissions rather than widened after the
        # fact, which leaves no window where the token sits world-readable.
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
    except OSError as exc:
        logger.warning("Could not save the Kite session: %s", exc)
        return False
    return restrict_permissions(path)


def load_session() -> "Session | None":
    """A session from the environment, else the token file, else None.

    The environment wins so a CI or scheduled run can inject a token
    without touching disk.
    """
    key = _env(config.KITE_API_KEY_ENV)
    token = _env(config.KITE_ACCESS_TOKEN_ENV)
    if key and token:
        return Session(api_key=key, access_token=token)
    try:
        with config.KITE_TOKEN_FILE.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not payload.get("api_key") or not payload.get("access_token"):
        return None
    return Session(api_key=payload["api_key"],
                   access_token=payload["access_token"],
                   user_id=payload.get("user_id", ""),
                   created_at=payload.get("created_at", ""))


def _get(session: Session, path: str, params: "dict | None" = None) -> dict:
    """One authenticated GET, raising KiteError on anything unexpected."""
    try:
        response = requests.get(f"{config.KITE_API_BASE}{path}",
                                headers=session.auth_header, params=params,
                                timeout=config.KITE_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise KiteError(f"Request failed ({path}): {exc}") from exc
    if response.status_code == 403:
        raise KiteError("Kite rejected the session (HTTP 403). The access "
                        "token has most likely expired - they last until "
                        "about 6am the next day. Run kite_login again.")
    if response.status_code != 200:
        try:
            message = response.json().get("message", "")
        except ValueError:
            message = response.text[:200]
        raise KiteError(f"HTTP {response.status_code} from {path}: {message}")
    try:
        return response.json().get("data") or {}
    except ValueError as exc:
        raise KiteError(f"Non-JSON reply from {path}") from exc


_PACE_HISTORICAL = "historical"
_PACE_QUOTE = "quote"
# Read through a callable so the rate is looked up when a request is about to
# go out rather than when this module is imported, which is what makes the
# limit configurable at run time.
_PACE_RATES = {
    _PACE_HISTORICAL: lambda: config.KITE_HISTORICAL_RATE_PER_SEC,
    _PACE_QUOTE: lambda: config.KITE_QUOTE_RATE_PER_SEC,
}
# One last-start timestamp per endpoint class, never one shared stamp: Kite
# meters /quote and historical separately, so sharing would have quote calls
# spending the historical budget and vice versa.
_LAST_CALL: dict[str, float] = {}
_PACE_LOCK = threading.Lock()
# Where the cross-process stamps live. Beside the code rather than in the
# system temp directory, so two processes started from the same checkout
# share one budget and two unrelated checkouts do not. The environment
# override exists so a test can use its own directory in a CHILD process,
# where monkeypatching cannot reach - and so a deployment can point two
# checkouts at one budget if they really do share an API key.
PACE_DIR = Path(os.environ.get("SECTOR_PULSE_PACE_DIR")
                or config.PROJECT_ROOT / "run")
_PACE_WARNED = False


def _pace_path(slot: str) -> Path:
    return PACE_DIR / f"kite_pace_{slot}.lock"


@contextmanager
def _file_gate(slot: str):
    """Hold an exclusive OS lock on this endpoint class's stamp file.

    Yields an open file positioned at 0, or None when the gate could not
    be taken - in which case the caller falls back to in-process pacing
    rather than refusing to fetch prices.

    The lock is held by the kernel, so it is released if this process
    dies. That is the whole reason it is not an exclusive-create file:
    feed.lock is that kind, and it needs a liveness check and a
    stale-adoption path to survive one badly timed Ctrl-C.
    """
    global _PACE_WARNED
    handle = None
    locked = False
    try:
        PACE_DIR.mkdir(parents=True, exist_ok=True)
        handle = open(_pace_path(slot), "a+", encoding="utf-8")
        locked = _lock_file(handle)
        yield handle if locked else None
    except Exception as exc:
        if not _PACE_WARNED:
            _PACE_WARNED = True
            logger.warning(
                "Cross-process rate pacing unavailable (%s), so each "
                "process meters itself. Two at once can exceed Kite's "
                "documented rate and the 429s look like missing bars.",
                exc)
        yield None
    finally:
        if handle is not None:
            try:
                if locked:
                    _unlock_file(handle)
            finally:
                handle.close()


def _lock_file(handle) -> bool:
    """Take an exclusive lock, blocking. True if it was taken."""
    if os.name == "nt":
        import msvcrt

        # msvcrt gives up after about ten seconds, so retry rather than
        # fail: a busy budget is normal, an unobtainable lock is not.
        for _ in range(30):
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                return True
            except OSError:
                continue
        return False
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return True


def _unlock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _pace(rate_per_sec: "float | None" = None,
          slot: str = _PACE_HISTORICAL) -> None:
    """Block until another request in this endpoint class may start.

    Kite documents 3 requests a second for historical data and 1 a second
    for /quote, and exceeding either returns 429s that look like missing
    data downstream. An unknown slot raises rather than defaulting to some
    rate, because guessing a limit here means guessing wrong quietly.

    THE BUDGET IS SHARED ACROSS PROCESSES, through a lock file per endpoint
    class holding the last start time. It used to be one in-memory stamp
    per process: the live feed seeding and Streamlit scanning each paced
    themselves at 3 a second and Kite saw six. Measured at 09:27 on
    2026-09-11, 322 historical calls in 180 seconds across the two.

    The sleep happens while both locks are held, and that is what makes
    the gate correct: two concurrent callers cannot both read the same
    last-start time and then both go. What overlaps between callers is the
    network round trip after this returns, not the wait. Wall-clock time
    rather than monotonic, because monotonic clocks are not comparable
    between processes.
    """
    if rate_per_sec is None:
        rate_per_sec = _PACE_RATES[slot]()
    if rate_per_sec <= 0:
        return
    gap = 1.0 / rate_per_sec
    with _PACE_LOCK:
        with _file_gate(slot) as shared:
            if shared is None:
                # In-process only. Better than nothing, and warned about.
                wait = gap - (time.time() - _LAST_CALL.get(slot, 0.0))
                if wait > 0:
                    time.sleep(wait)
                _LAST_CALL[slot] = time.time()
                return
            try:
                shared.seek(0)
                last = float((shared.read() or "0").strip() or 0.0)
            except ValueError:
                last = 0.0
            now = time.time()
            # A stamp from the future means the clock moved back or the
            # file was written by something else; waiting a whole gap is
            # the safe reading of it.
            wait = gap - (now - last) if last <= now else gap
            if wait > 0:
                time.sleep(wait)
            stamp = time.time()
            shared.seek(0)
            shared.truncate()
            shared.write(f"{stamp:.6f}")
            shared.flush()
            _LAST_CALL[slot] = stamp


def profile(session: "Session | None" = None) -> dict:
    """The logged-in user's profile: the cheapest way to test a session."""
    live = session or load_session()
    if live is None:
        raise KiteError("No Kite session. Run kite_login first.")
    return _get(live, "/user/profile")


def quote(instruments: list[str], session: "Session | None" = None) -> dict:
    """Full quotes for up to 500 instruments, e.g. ["NSE:RELIANCE"].

    Paced on the quote budget, which Kite meters at 1 request a second:
    a caller walking a universe in 500-name batches otherwise collects
    429s, and a 429 arrives here as a KiteError, not as a quote.
    """
    live = session or load_session()
    if live is None:
        raise KiteError("No Kite session. Run kite_login first.")
    if not instruments:
        return {}
    if len(instruments) > 500:
        raise ValueError(f"Kite caps a quote call at 500 instruments, "
                         f"got {len(instruments)}")
    _pace(slot=_PACE_QUOTE)
    return _get(live, "/quote", params=[("i", i) for i in instruments])


def historical(instrument_token: int, start: date, end: date,
               interval: str = "5minute", continuous: bool = False,
               oi: bool = False,
               session: "Session | None" = None) -> pd.DataFrame:
    """Candles for one instrument, paged over Kite's per-request day cap.

    The cap depends on the candle size, not on a single 60-day figure:
    KITE_HISTORICAL_MAX_DAYS holds the table and any interval missing from
    it falls back to the narrowest span rather than the widest.

    Returns a tz-aware DataFrame with the same Open/High/Low/Close/Volume
    column names the rest of this project uses, so a frame from here drops
    straight into edge_lab and the scanner without translation. With oi=True
    an OpenInterest column is included, which is the one thing NSE never
    serves historically.
    """
    live = session or load_session()
    if live is None:
        raise KiteError("No Kite session. Run kite_login first.")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    span = timedelta(days=config.KITE_HISTORICAL_MAX_DAYS.get(
        interval, config.KITE_HISTORICAL_DEFAULT_SPAN))
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + span, end)
        params = {"from": cursor.isoformat(), "to": chunk_end.isoformat()}
        if continuous:
            params["continuous"] = "1"
        if oi:
            params["oi"] = "1"
        _pace(slot=_PACE_HISTORICAL)
        data = _get(live, f"/instruments/historical/{instrument_token}/{interval}",
                    params=params)
        candles = data.get("candles") or []
        if candles:
            width = len(candles[0])
            names = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if width > 6:
                names.append("OpenInterest")
            frames.append(pd.DataFrame([row[:len(names)] for row in candles],
                                       columns=names))
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = pd.concat(frames, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True, format="mixed")
    frame = frame.set_index("Date").sort_index()
    frame.index = frame.index.tz_convert("Asia/Kolkata")
    return frame[~frame.index.duplicated(keep="last")]
