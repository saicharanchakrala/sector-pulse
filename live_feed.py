"""Background process: stream ticks for the whole F&O universe into bars.

    .venv\\Scripts\\python -m live_feed              # F&O underlyings
    .venv\\Scripts\\python -m live_feed --symbols RELIANCE,TATASTEEL
    .venv\\Scripts\\python -m live_feed --prewarm-only

A SEPARATE PROCESS, deliberately. Streamlit re-executes its whole script on
every interaction, so a thread started there duplicates across reruns and
dies with the browser session. This owns one socket and one file, and keeps
building bars while nobody is watching.

WHAT IT DOES AT STARTUP. Prewarms prior-session bars against a window
ending yesterday, so that fetch happens once rather than on every scan.
After that the scan touches no network: history comes from the parquet cache
and today comes from the file this process writes.

KITE'S LIMITS, respected rather than discovered. Three websocket
connections per API key and 3,000 instruments per connection, so one
connection carries the whole 216-name universe comfortably. Running two
copies of this is what will exhaust the connection budget, so it refuses to
start if another instance holds the lock.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import config
import instruments
import kite_client
import kite_ticker as kt
import live_bars

IST = ZoneInfo("Asia/Kolkata")
LOCK = live_bars.STORE / "feed.lock"
logger = logging.getLogger("live_feed")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream live ticks into 5-minute bars for the scanner")
    parser.add_argument("--symbols", default="",
                        help="comma-separated cash symbols; default is every "
                             "F&O underlying from the instrument snapshot")
    parser.add_argument("--minutes", type=int, default=0,
                        help="stop after N minutes (0 runs until interrupted)")
    parser.add_argument("--flush-every", type=int, default=20,
                        help="seconds between parquet flushes (default: 20)")
    parser.add_argument("--history-days", type=int, default=12,
                        help="prior calendar days to prewarm (default: 12)")
    parser.add_argument("--prewarm-only", action="store_true",
                        help="fetch and cache prior sessions, then exit")
    return parser.parse_args(argv)


def by_turnover(limit: int) -> list:
    """The `limit` most-traded symbols by 20-session turnover, or [].

    Turnover is close times volume averaged over the last twenty sessions,
    read from the consolidated daily store - the same measure the scan's
    own liquidity gate uses, so the feed covers what the gate can pass
    rather than an arbitrary slice.
    """
    try:
        import bar_store

        frames = bar_store.load("day")
    except Exception as exc:
        logger.warning("No daily store, so no turnover ranking: %s", exc)
        return []
    ranked = []
    for symbol, frame in frames.items():
        if frame is None or not {"Close", "Volume"} <= set(frame.columns):
            continue
        tail = frame.tail(20)
        if tail.empty:
            continue
        value = float((tail["Close"] * tail["Volume"]).mean())
        if value == value and value > 0:
            ranked.append((value, symbol))
    ranked.sort(reverse=True)
    return [symbol for _, symbol in ranked[:max(0, limit)]]


def universe(explicit: str, limit: "int | None" = None) -> list:
    """Symbols to stream: an explicit list, or the liquid universe.

    Previously this returned the F&O underlyings and nothing else, which
    left the scan's wider scopes downloading bars for ~2,350 symbols at
    three requests a second. Kite allows 3,000 instruments on the one
    connection this process uses, so there was never a reason for the
    feed to be that narrow.

    The F&O set is unioned in unconditionally rather than left to the
    turnover rank, so the intraday scanner's own default scope cannot
    lose a name to a ranking change.
    """
    if explicit.strip():
        return [s.strip().upper() for s in explicit.split(",") if s.strip()]
    snapshot = instruments.load_latest()
    if snapshot is None:
        logger.warning("No instrument snapshot; run the sync first")
        return []
    core = {inst.symbol for inst in snapshot.fo_underlyings}
    wanted = config.FEED_UNIVERSE_SIZE if limit is None else limit
    liquid = set(by_turnover(wanted))
    if not liquid:
        logger.warning("Turnover ranking unavailable; streaming the %d F&O "
                       "underlyings only", len(core))
        return sorted(core)
    combined = core | liquid
    cap = kt.MAX_INSTRUMENTS_PER_CONNECTION
    if len(combined) > cap:
        # The F&O set is never dropped; the ranking tail is trimmed.
        room = max(0, cap - len(core))
        combined = core | set(sorted(liquid - core)[:room])
        logger.warning("Trimmed the streaming universe to Kite's %d cap", cap)
    logger.info("Streaming %d symbols: %d F&O underlyings plus the top %d "
                "by 20-session turnover", len(combined), len(core), wanted)
    return sorted(combined)


def take_lock() -> bool:
    """Refuse to start a second feed. Kite allows only three sockets per key.

    A stale lock from a killed process is adopted rather than treated as
    fatal: the PID it names is checked, and a lock naming nothing alive is
    overwritten.
    """
    live_bars.STORE.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            held = int(LOCK.read_text(encoding="utf-8").strip())
        except Exception:
            held = -1
        if held > 0 and _alive(held):
            logger.error("Another live_feed is running as PID %d. Stop it "
                         "first - Kite allows three sockets per key and two "
                         "feeds double the tick load for no gain.", held)
            return False
        logger.warning("Adopting a stale lock from PID %s", held)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _alive(pid: int) -> bool:
    """Whether a PID is still running, without psutil."""
    try:
        out = os.popen(f'tasklist /FI "PID eq {pid}" /NH').read()
    except Exception:
        return False
    return str(pid) in out


def release_lock() -> None:
    try:
        if LOCK.exists():
            LOCK.unlink()
    except Exception:
        pass


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    symbols = universe(args.symbols)
    if not symbols:
        print("no symbols to stream")
        return 1
    print(f"universe: {len(symbols)} symbols")

    print(f"prewarming {args.history_days} prior days (once, cached)...")
    started = time.time()
    history = live_bars.prewarm(symbols, days=args.history_days)
    start, end = live_bars.history_window(args.history_days)
    print(f"  {len(history)}/{len(symbols)} symbols have prior bars "
          f"({start} .. {end}) in {time.time() - started:.0f}s")
    if args.prewarm_only:
        return 0

    import market_source
    tokens = {s: market_source.token_for(s) for s in symbols}
    tokens = {s: t for s, t in tokens.items() if t}
    missing = sorted(set(symbols) - set(tokens))
    if missing:
        print(f"  no instrument token for {len(missing)}: "
              f"{', '.join(missing[:8])}")
    if not tokens:
        print("nothing subscribable")
        return 1
    if not take_lock():
        return 1

    session = kite_client.load_session()
    if session is None:
        print("No Kite session. Run: .venv\\Scripts\\python -m kite_login")
        release_lock()
        return 1

    # Seed today's session BEFORE streaming. A feed started mid-day has no
    # 09:15 bar, and the opening range is the session's first fifteen
    # minutes - without it the scanner chooses no direction for any symbol,
    # which looks exactly like a quiet market. Measured: 0 of 216 names got
    # a direction from two bars of live data.
    print("seeding today's session so far...")
    seeded = live_bars.backfill_today(list(tokens))
    rows = live_bars.write_seed(seeded)
    covered = sum(1 for f in seeded.values()
                  if live_bars.session_is_covered(f))
    print(f"  {len(seeded)}/{len(tokens)} symbols seeded, {rows} bars, "
          f"{covered} reaching the open")

    builder = live_bars.BarBuilder()
    stop = threading.Event()

    def on_ticks(ticks: list) -> None:
        builder.add(ticks)

    def flusher() -> None:
        """Persist on a timer so a reader always sees a recent file."""
        while not stop.wait(args.flush_every):
            try:
                rows = builder.flush()
            except Exception as exc:
                # One bad flush must not end all future ones: this thread
                # is the only thing writing the file the scanner reads.
                logger.warning("flush failed: %s", exc)
                continue
            state = live_bars.status()
            age = state.get("age_seconds")
            # `age == age` alone was a NaN check, but None passes it too and
            # formatting None raised inside this thread - which killed the
            # flusher silently and stopped every write.
            fresh = (f", newest {age:.0f}s old"
                     if isinstance(age, (int, float)) and age == age else "")
            print(f"  {datetime.now(IST):%H:%M:%S}  bars {rows} across "
                  f"{state.get('instruments', 0)} instruments{fresh}",
                  flush=True)

    pump = threading.Thread(target=flusher, name="flusher", daemon=True)
    pump.start()

    def shutdown(*_) -> None:
        stop.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), shutdown)
            except Exception:
                pass

    print(f"streaming {len(tokens)} instruments; Ctrl-C to stop")
    deadline = (time.time() + args.minutes * 60) if args.minutes else None

    async def run() -> None:
        """Bridge the signal handler's threading.Event to kt.stream's.

        kt.stream takes an asyncio.Event, which only the event loop may
        set, while the signal handler and the --minutes deadline live
        outside it. A small watcher polls the outside world and sets the
        inside one; one second of latency on shutdown is not worth a more
        elaborate mechanism.
        """
        inner = asyncio.Event()

        async def watch() -> None:
            while not inner.is_set():
                if stop.is_set():
                    inner.set()
                    return
                if deadline is not None and time.time() >= deadline:
                    print("  reached --minutes deadline, stopping")
                    stop.set()
                    inner.set()
                    return
                await asyncio.sleep(1.0)

        watcher = asyncio.create_task(watch())
        try:
            await kt.stream(list(tokens.values()), on_ticks,
                            mode=kt.MODE_FULL, session=session,
                            stop_event=inner)
        finally:
            watcher.cancel()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("stream ended: %s", exc)
    finally:
        stop.set()
        # The bar still forming is a real bar for the part of the bucket it
        # covers, and at 15:30 it is the session's last. Closing it before
        # the final flush is the only way it reaches the file.
        closed = builder.close_open_bars()
        written = builder.flush()
        release_lock()
        refused = builder.refused()
        note = ""
        if any(refused.values()):
            note = (f", refused {refused['outside_session']} out-of-session "
                    f"and {refused['out_of_order']} out-of-order ticks")
        print(f"flushed {written} bars on exit "
              f"(closed {closed} still forming){note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
