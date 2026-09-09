"""Background tick recorder: python -m tick_recorder [--fo] [--minutes N]

Runs as its own long-lived process and appends every tick to a dated
newline-delimited file, with a small status file the dashboard can read to
show whether the feed is alive.

Deliberately NOT a thread inside Streamlit. Streamlit re-executes its script
on every interaction, so a thread started there has an unclear lifetime,
duplicates itself across reruns, and dies with the session. A separate
process that owns one socket and one file is far easier to reason about, and
it keeps recording while nobody is looking at the dashboard.

This is the only genuinely live source in the project. yfinance serves NSE
about fifteen minutes late, so no polling frequency ever made it current.

Kite caps one connection at 3,000 instruments and three connections per API
key, so the F&O universe fits comfortably in one socket.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from dataclasses import asdict
from datetime import datetime

import config
import instruments
import kite_client
import kite_instruments as ki
import kite_ticker as kt

logger = logging.getLogger("tick_recorder")

TICK_DIR = config.PROJECT_ROOT / "tick_store"
STATUS_FILE = TICK_DIR / "status.json"


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Record live Kite ticks to disk")
    parser.add_argument("--fo", action="store_true",
                        help="record every F&O underlying's cash instrument "
                             "(the default is a small watchlist)")
    parser.add_argument("--futures", action="store_true",
                        help="also record the nearest futures contract per "
                             "underlying, which is where open interest lives")
    parser.add_argument("--symbols", default="",
                        help="comma-separated NSE cash symbols to record")
    parser.add_argument("--mode", default=kt.MODE_FULL,
                        choices=[kt.MODE_LTP, kt.MODE_QUOTE, kt.MODE_FULL],
                        help="detail level; full includes open interest")
    parser.add_argument("--minutes", type=int, default=0,
                        help="stop after N minutes (0 runs until interrupted)")
    parser.add_argument("--status-every", type=int, default=15,
                        help="seconds between status-file writes")
    args = parser.parse_args(argv)
    if args.minutes < 0:
        parser.error(f"--minutes cannot be negative, got {args.minutes}")
    if args.status_every < 1:
        parser.error(f"--status-every must be at least 1, got {args.status_every}")
    return args


def resolve_tokens(args: argparse.Namespace) -> dict[int, str]:
    """Instrument tokens to record, mapped to a readable label."""
    master = ki.fetch_master()
    cash = {c.tradingsymbol: c for c in ki.nse_equities(master)}
    indices = {c.tradingsymbol: c for c in master if c.segment == "INDICES"}
    wanted: dict[int, str] = {}

    # The index is always worth recording: relative strength needs it.
    nifty = indices.get("NIFTY 50")
    if nifty:
        wanted[nifty.instrument_token] = "NIFTY 50"

    if args.symbols.strip():
        names = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.fo:
        universe = instruments.load_latest()
        if universe is None:
            raise SystemExit("No instrument snapshot. Run discover.py first.")
        names = [i.symbol for i in universe.fo_stocks]
    else:
        universe = instruments.load_latest()
        names = ([i.symbol for i in universe.fo_stocks][:25]
                 if universe else ["RELIANCE", "TCS", "HDFCBANK", "INFY"])

    for name in names:
        contract = cash.get(name)
        if contract:
            wanted[contract.instrument_token] = name
        else:
            logger.warning("No NSE cash instrument for %s", name)

    if args.futures:
        by_underlying: dict[str, object] = {}
        for contract in sorted(ki.nse_futures(master),
                               key=lambda c: (c.expiry or datetime.max.date())):
            if contract.name in names and contract.name not in by_underlying:
                by_underlying[contract.name] = contract
        for contract in by_underlying.values():
            wanted[contract.instrument_token] = contract.tradingsymbol

    if len(wanted) > kt.MAX_INSTRUMENTS_PER_CONNECTION:
        raise SystemExit(
            f"{len(wanted)} instruments exceeds the {kt.MAX_INSTRUMENTS_PER_CONNECTION} "
            f"one connection allows. Narrow the selection or split the run.")
    return wanted


def tick_path(when: datetime):
    """Where one session's ticks are appended."""
    return TICK_DIR / f"ticks_{when:%Y%m%d}.jsonl"


def write_status(stats: kt.TickerStats, labels: dict[int, str],
                 mode: str) -> None:
    """Overwrite the status file so a reader can see the feed is alive."""
    payload = dict(stats.summary())
    payload.update({"mode": mode, "instruments_named": len(labels),
                    "updated_at": datetime.now().astimezone().isoformat()})
    try:
        TICK_DIR.mkdir(parents=True, exist_ok=True)
        with STATUS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
    except OSError as exc:
        logger.warning("Could not write status: %s", exc)


def read_status() -> "dict | None":
    """The recorder's last status, for the dashboard; None when absent."""
    try:
        with STATUS_FILE.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# One failed write can be momentary. A run of them cannot be explained away.
MAX_CONSECUTIVE_WRITE_FAILURES = 3


class TickWriter:
    """Appends ticks as JSON lines, and stops the run if writing keeps failing.

    A write failure used to be indistinguishable from a network failure.
    The callback runs inside the stream loop, so an OSError from a full
    disk was caught by kite_ticker.stream, counted as a stream error and
    followed by a reconnect, forever, at WARNING level: the recorder looked
    alive and recorded nothing. Recording nothing is worse than stopping,
    so a run of failures sets the stop event and the run exits non-zero.

    `rows` counts ticks in batches that were written whole. A batch that
    fails part way through may still have put some lines on disk, so the
    count is a floor, not a guarantee.
    """

    def __init__(self, handle, labels: dict[int, str], stop: asyncio.Event,
                 max_failures: int = MAX_CONSECUTIVE_WRITE_FAILURES) -> None:
        self._handle = handle
        self._labels = labels
        self._stop = stop
        self._max_failures = max_failures
        self.rows = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error: "OSError | None" = None

    @property
    def gave_up(self) -> bool:
        """True once consecutive failures have stopped the recorder."""
        return self.consecutive_failures >= self._max_failures

    def __call__(self, ticks: "list[kt.Tick]") -> None:
        """Append one batch of ticks, one JSON line each."""
        try:
            for tick in ticks:
                self._handle.write(json.dumps(self._row(tick)) + "\n")
            self._handle.flush()
        except OSError as exc:
            self._on_failure(exc, len(ticks))
            return
        self.rows += len(ticks)
        self.consecutive_failures = 0

    def _row(self, tick: "kt.Tick") -> dict:
        """One tick as a JSON-safe row, labelled with its symbol."""
        row = asdict(tick)
        row["symbol"] = self._labels.get(tick.instrument_token, "")
        row["received_at"] = tick.received_at.isoformat()
        row["exchange_timestamp"] = (tick.exchange_timestamp.isoformat()
                                     if tick.exchange_timestamp else None)
        return row

    def _on_failure(self, exc: OSError, lost: int) -> None:
        """Count a failed batch, and stop the recorder once it is a pattern."""
        self.failures += 1
        self.consecutive_failures += 1
        self.last_error = exc
        if self.gave_up:
            logger.error("Stopping: %d consecutive tick-write failures, "
                         "the last %s: %s", self.consecutive_failures,
                         type(exc).__name__, exc)
            self._stop.set()
        else:
            logger.error("Tick write failed (%d of %d before stopping), "
                         "up to %d tick(s) lost: %s",
                         self.consecutive_failures, self._max_failures,
                         lost, exc)


async def _status_loop(stats: kt.TickerStats, labels: dict[int, str],
                       mode: str, every: int, stop: asyncio.Event) -> None:
    """Refresh the status file so staleness is visible to a reader."""
    while not stop.is_set():
        write_status(stats, labels, mode)
        try:
            await asyncio.wait_for(stop.wait(), timeout=every)
        except asyncio.TimeoutError:
            continue


def _install_stop_handlers(loop, stop: asyncio.Event) -> None:
    """Ask the loop to set `stop` on SIGINT and SIGTERM, where it can."""
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            # Windows ProactorEventLoop refuses signal handlers; KeyboardInterrupt
            # still unwinds the run, so this is a graceful-shutdown nicety only.
            pass


def _print_preamble(tokens: list[int], path, args: argparse.Namespace) -> None:
    """What the operator needs to know before any ticks arrive."""
    print(f"Recording {len(tokens)} instruments in {args.mode} mode")
    print(f"  -> {path}")
    if args.minutes:
        print(f"  stopping after {args.minutes} minute(s)")
    print("  Ctrl+C to stop. Ticks only flow while the market is open;")
    print("  on subscribe Kite sends one last-known snapshot per instrument.")


async def _wait_until_done(streamer: asyncio.Task, stop: asyncio.Event,
                           minutes: int) -> None:
    """Block until the time limit, a stop request, or the streamer ending.

    Waiting on the stop event alone would hang for the whole run if the
    stream task died: a rejected subscription or an expired token would
    look like a healthy recorder that simply never received a tick.
    """
    stopper = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({streamer, stopper},
                           timeout=minutes * 60 if minutes else None,
                           return_when=asyncio.FIRST_COMPLETED)
    finally:
        stopper.cancel()


async def _shutdown(tasks) -> "BaseException | None":
    """Cancel the background tasks and return the first real failure.

    CancelledError is expected here and means nothing, which is why it has
    to be named. Anything else is a bug or a fatal stream condition, and
    swallowing it is how a recorder ends up looking healthy while recording
    nothing.
    """
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    first = None
    for task, result in zip(tasks, results):
        if (not isinstance(result, BaseException)
                or isinstance(result, asyncio.CancelledError)):
            continue
        logger.error("Task %s ended in %s: %s", task.get_name(),
                     type(result).__name__, result)
        if first is None:
            first = result
    return first


def _report(stats: kt.TickerStats, writer: TickWriter,
            failure: "BaseException | None") -> int:
    """Print the closing summary and return the process exit code."""
    print()
    if writer.gave_up:
        print(f"Stopped: {writer.consecutive_failures} tick writes failed in "
              f"a row ({writer.last_error}). Nothing was reaching disk, so "
              f"the recorder gave up rather than pretend to record.")
    elif failure is not None:
        print(f"Stopped by a failure - {type(failure).__name__}: {failure}")
    elif writer.failures:
        print(f"Stopped, but {writer.failures} batch(es) failed to write; "
              f"the last error was {writer.last_error}")
    else:
        print("Stopped.")
    for key, value in stats.summary().items():
        print(f"  {key:<16}{value}")
    print(f"  {'rows_written':<16}{writer.rows}")
    return 1 if (writer.failures or failure is not None) else 0


async def run(args: argparse.Namespace) -> int:
    """Record until the time limit, an interrupt, or a fatal write failure."""
    session = kite_client.load_session()
    if session is None:
        print("No Kite session. Run: .venv\\Scripts\\python -m kite_login")
        return 1
    labels = resolve_tokens(args)
    tokens = list(labels)
    TICK_DIR.mkdir(parents=True, exist_ok=True)
    path = tick_path(datetime.now().astimezone())
    _print_preamble(tokens, path, args)

    stats = kt.TickerStats()
    stop = asyncio.Event()
    _install_stop_handlers(asyncio.get_running_loop(), stop)
    # The handle is a context manager so that anything raising between the
    # open and the close cannot leak it.
    with path.open("a", encoding="utf-8") as handle:
        writer = TickWriter(handle, labels, stop)
        tasks = (
            asyncio.create_task(
                kt.stream(tokens, writer, mode=args.mode, session=session,
                          stats=stats, stop_event=stop), name="stream"),
            asyncio.create_task(
                _status_loop(stats, labels, args.mode, args.status_every,
                             stop), name="status"),
        )
        try:
            await _wait_until_done(tasks[0], stop, args.minutes)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            failure = await _shutdown(tasks)
        write_status(stats, labels, args.mode)
    return _report(stats, writer, failure)


def main(argv: "list[str] | None" = None) -> int:
    """Configure logging and run the recorder."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(run(_parse_args(argv)))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
