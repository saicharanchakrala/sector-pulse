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


async def run(args: argparse.Namespace) -> int:
    """Record until the time limit or an interrupt."""
    session = kite_client.load_session()
    if session is None:
        print("No Kite session. Run: .venv\\Scripts\\python -m kite_login")
        return 1
    labels = resolve_tokens(args)
    tokens = list(labels)
    TICK_DIR.mkdir(parents=True, exist_ok=True)
    path = tick_path(datetime.now().astimezone())
    print(f"Recording {len(tokens)} instruments in {args.mode} mode")
    print(f"  -> {path}")
    if args.minutes:
        print(f"  stopping after {args.minutes} minute(s)")
    print("  Ctrl+C to stop. Ticks only flow while the market is open;")
    print("  on subscribe Kite sends one last-known snapshot per instrument.")

    stats = kt.TickerStats()
    stop = asyncio.Event()
    handle = path.open("a", encoding="utf-8")

    def on_ticks(ticks: list[kt.Tick]) -> None:
        """Append each tick as one JSON line."""
        for tick in ticks:
            row = asdict(tick)
            row["symbol"] = labels.get(tick.instrument_token, "")
            row["received_at"] = tick.received_at.isoformat()
            row["exchange_timestamp"] = (tick.exchange_timestamp.isoformat()
                                         if tick.exchange_timestamp else None)
            handle.write(json.dumps(row) + "\n")
        handle.flush()

    async def status_loop() -> None:
        """Refresh the status file so staleness is visible to a reader."""
        while not stop.is_set():
            write_status(stats, labels, args.mode)
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.status_every)
            except asyncio.TimeoutError:
                continue

    loop = asyncio.get_running_loop()
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

    watcher = asyncio.create_task(status_loop())
    streamer = asyncio.create_task(
        kt.stream(tokens, on_ticks, mode=args.mode, session=session,
                  stats=stats, stop_event=stop))
    try:
        if args.minutes:
            await asyncio.wait_for(stop.wait(), timeout=args.minutes * 60)
        else:
            await stop.wait()
    except asyncio.TimeoutError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for task in (streamer, watcher):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        write_status(stats, labels, args.mode)
        handle.close()
    print()
    print("Stopped.")
    for key, value in stats.summary().items():
        print(f"  {key:<16}{value}")
    return 0


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
