"""Persist option chains, because NSE keeps no history and nobody sells it.

Equity bars can be replayed: Kite serves minute candles going back years,
so a scan can be re-run honestly at almost any past minute. Option chains
cannot, and the move to Kite did not fix it. Kite will serve historical
candles for an option contract that still exists, but an expired contract
leaves the instrument master and takes its token with it - so the chain as
it stood on a past date, which strikes were listed and which were liquid,
is not reconstructible afterwards. NSE's own API serves one live snapshot
and no archive. Every option figure in a replay is therefore a
reconstruction from the underlying plus a delta estimate, not a
measurement.

The only fix is to start recording. This module writes a timestamped,
gzipped snapshot of whichever chains were asked for, so from the first
capture onward an option call becomes as replayable as an equity one, and
after a few weeks there is enough to ask whether the options layer adds
anything over the equity signal at all.

Storage is one file per capture: a gzipped JSON object holding every
contract for every requested underlying. About 210 underlyings compress to
a couple of megabytes, so a capture every session costs little.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import config
import options_chain

logger = logging.getLogger(__name__)


@dataclass
class ChainSnapshot:
    """Every chain captured in one pass, with what failed recorded."""

    captured_at: datetime
    chains: dict[str, dict] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)

    @property
    def contract_count(self) -> int:
        """Total option contracts captured."""
        return sum(len(entry.get("contracts") or [])
                   for entry in self.chains.values())

    def counts(self) -> dict[str, int]:
        """Headline counts for a sync report."""
        return {
            "underlyings": len(self.chains),
            "contracts": self.contract_count,
            "failed": len(self.failed),
        }


def snapshot_path(captured_at: datetime, root=None):
    """Where one capture is stored."""
    base = config.OPTION_SNAPSHOT_DIR if root is None else root
    return base / f"chains_{captured_at.strftime('%Y%m%d_%H%M%S')}.json.gz"


def capture(symbols: list[str], index_symbols: "set[str] | None" = None,
            session=None, pause: float = config.NSE_REQUEST_PAUSE_SECONDS,
            progress=None) -> ChainSnapshot:
    """Fetch the nearest-expiry chain for each symbol, paced.

    `progress` is called as progress(done, total, symbol) so a UI can show
    where it is. Paced deliberately: NSE throttles a burst of a couple of
    hundred requests, and a half-captured snapshot is worse than a slow one.
    """
    live = session if session is not None else options_chain.open_session()
    indices = index_symbols or set()
    snapshot = ChainSnapshot(captured_at=datetime.now().astimezone())
    total = len(symbols)
    for done, symbol in enumerate(symbols, start=1):
        if progress is not None:
            progress(done, total, symbol)
        contracts, spot, expiry = options_chain.fetch_chain(
            symbol, symbol in indices, session=live)
        if not contracts or spot is None:
            snapshot.failed.append(symbol)
        else:
            snapshot.chains[symbol] = {
                "spot": spot,
                "expiry": expiry,
                "contracts": [asdict(c) for c in contracts],
            }
        if pause > 0:
            time.sleep(pause)
    if snapshot.failed:
        logger.warning("No chain for %d underlying(s): %s",
                       len(snapshot.failed), ", ".join(snapshot.failed[:10]))
    return snapshot


def save(snapshot: ChainSnapshot, root=None):
    """Write one capture as gzipped JSON; returns the path or None."""
    base = config.OPTION_SNAPSHOT_DIR if root is None else root
    target = snapshot_path(snapshot.captured_at, base)
    payload = {
        "captured_at": snapshot.captured_at.isoformat(),
        "chains": snapshot.chains,
        "failed": snapshot.failed,
    }
    try:
        base.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError as exc:
        logger.warning("Could not write %s: %s", target, exc)
        return None
    return target


def list_snapshots(root=None) -> list:
    """Every stored capture, oldest first."""
    base = config.OPTION_SNAPSHOT_DIR if root is None else root
    try:
        return sorted(base.glob("chains_*.json.gz"))
    except OSError:
        return []


def load(path) -> "ChainSnapshot | None":
    """Read one stored capture, or None when it cannot be read."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    return ChainSnapshot(
        captured_at=datetime.fromisoformat(payload["captured_at"]),
        chains=payload.get("chains") or {},
        failed=payload.get("failed") or [],
    )


def load_latest(root=None) -> "ChainSnapshot | None":
    """The most recent stored capture, or None when there are none."""
    files = list_snapshots(root)
    return load(files[-1]) if files else None


def contracts_for(snapshot: ChainSnapshot, symbol: str
                  ) -> list[options_chain.Contract]:
    """Rebuild Contract objects for one underlying from a stored capture."""
    entry = snapshot.chains.get(symbol.strip().upper())
    if not entry:
        return []
    out: list[options_chain.Contract] = []
    for row in entry.get("contracts") or []:
        try:
            out.append(options_chain.Contract(**row))
        except TypeError as exc:
            logger.warning("Stored contract for %s does not match the current "
                           "Contract shape: %s", symbol, exc)
            break
    return out


def cache_summary(root=None) -> dict[str, object]:
    """What the local chain cache currently holds."""
    files = list_snapshots(root)
    total_bytes = 0
    for path in files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    first = last = None
    if files:
        head, tail = load(files[0]), load(files[-1])
        first = head.captured_at if head else None
        last = tail.captured_at if tail else None
    return {
        "snapshots": len(files),
        "megabytes": round(total_bytes / (1024 * 1024), 2),
        "first": first,
        "last": last,
    }
