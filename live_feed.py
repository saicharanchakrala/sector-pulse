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
import io
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import config
import instruments
import kite_client
import kite_ticker as kt
import live_bars
import object_store
import scan_publish

IST = ZoneInfo("Asia/Kolkata")
LOCK = live_bars.STORE / "feed.lock"
# Set by the ECS agent on every Fargate task, so it needs no config of our
# own and cannot drift out of step with where the process actually runs.
ON_FARGATE = bool(os.environ.get("ECS_CONTAINER_METADATA_URI_V4"))
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
    # No literal default: the window comes from config through
    # live_bars.history_days(), so the feed and a download agree on how
    # many prior sessions the relative-volume baseline is taken over.
    parser.add_argument("--history-days", type=int, default=0,
                        help="prior calendar days to prewarm "
                             "(default: from config.SCAN_BAR_LOOKBACK)")
    parser.add_argument("--scan-every", type=int, default=-1,
                        help="seconds between published scans; 0 disables, "
                             "-1 (default) means on when an object store is "
                             "configured and off otherwise")
    parser.add_argument("--prewarm-only", action="store_true",
                        help="fetch and cache prior sessions, then exit")
    return parser.parse_args(argv)


TURNOVER_SESSIONS = 20


def turnover_values(frames: dict) -> list:
    """[(turnover, symbol)] descending, from daily frames.

    Extracted so publish_turnover.py computes the ranking the SAME way the
    feed would have computed it from a local store. Two copies of this
    arithmetic would let the container stream a different universe from
    the one the laptop believed it had ranked, and nothing would report
    the divergence.
    """
    ranked = []
    for symbol, frame in (frames or {}).items():
        if frame is None or not {"Close", "Volume"} <= set(frame.columns):
            continue
        tail = frame.tail(TURNOVER_SESSIONS)
        if tail.empty:
            continue
        value = float((tail["Close"] * tail["Volume"]).mean())
        if value == value and value > 0:
            ranked.append((value, symbol))
    ranked.sort(reverse=True)
    return ranked


def turnover_from_store() -> list:
    """Symbols ranked by turnover, from the file publish_turnover writes.

    Returns [] when there is no object store or no published file, so the
    caller falls back to the F&O underlyings exactly as it did before.
    """
    if not object_store.enabled():
        return []
    # A StorageError is deliberately NOT caught. Swallowing it would narrow
    # the streaming universe to the F&O set for the whole session because a
    # bucket was unreachable, and log it as a warning nobody reads.
    payload = object_store.get(object_store.TURNOVER_OBJECT)
    if payload is None:
        logger.warning("No %s published yet, so the feed cannot rank beyond "
                       "the F&O underlyings - run publish_turnover.py",
                       object_store.TURNOVER_OBJECT)
        return []
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        logger.warning("Published turnover is unreadable: %s", exc)
        return []
    if frame.empty or "symbol" not in frame.columns:
        return []
    if "turnover" in frame.columns:
        frame = frame.sort_values("turnover", ascending=False)
    return [str(s) for s in frame["symbol"].tolist()]


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
        logger.info("No local daily store (%s)", exc)
        frames = {}
    ranked = [symbol for _, symbol in turnover_values(frames)]
    if not ranked:
        # KEYED ON AN EMPTY RANKING, NOT ON AN EXCEPTION. bar_store.load
        # returns {} for an absent store rather than raising, so a
        # container - which has no 559 MB forecast_cache and never will -
        # arrives here silently. An earlier version put this in an `except`
        # arm that therefore never fired at all: the feed streamed the ~216
        # F&O underlyings this change exists to widen, and the published
        # ranking was read by nothing.
        ranked = turnover_from_store()
        if ranked:
            logger.info("Ranking %d symbols from the published turnover "
                        "file rather than a local daily store", len(ranked))
    if not ranked:
        logger.warning("No turnover ranking from either the local daily "
                       "store or the object store")
    return ranked[:max(0, limit)]


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

    THE BENCHMARK IS NAMED CANONICALLY, and that is the whole point. Its
    ticks were always arriving: the F&O master calls the index NIFTY, it
    is in fo_underlyings, and token_for("NIFTY") is 256265 - the same
    token as "NIFTY 50". What was missing was the NAME. live_bars.frames_from
    inverts {symbol: token} into {token: symbol}, so an instrument reachable
    under two names keeps only ONE of them, decided by insertion order,
    while the scan and daily_context both key it "NIFTY 50". The bars
    landed under "NIFTY", the scan asked for "NIFTY 50", and the
    relative-strength gate therefore failed for EVERY symbol at every
    hour, silently: measured 2026-09-17, 2,498 of 2,498 published rows had
    a null relative_strength and nothing was actionable all morning.

    So the aliases are collapsed to the canonical name rather than the
    canonical name being added alongside them. Adding it would leave two
    names on one token and let a sorted() tie-break decide which survives
    - which happens to pick "NIFTY 50" today and would silently stop doing
    so if SCAN_BENCHMARK were spelled "^NSEI".

    Only the benchmark's own aliases are touched. BANKNIFTY and the rest
    canonicalise too (to "NIFTY BANK"), but nothing reads them by a
    canonical name, and renaming them here would change what is streamed
    for no stated reason.
    """
    # Lazily, like main() does: this module is imported by the scanner
    # thread and by tests that never touch the instrument master.
    import market_source

    wanted_name = market_source.canonical(config.SCAN_BENCHMARK)

    def is_benchmark(symbol: str) -> bool:
        return market_source.canonical(symbol) == wanted_name

    if explicit.strip():
        named = [s.strip().upper() for s in explicit.split(",") if s.strip()]
        # Compared canonically: "--symbols NIFTY" names the benchmark just
        # as surely as "--symbols NIFTY 50" does, and appending the other
        # spelling would put both on token 256265.
        named = [s for s in named if not is_benchmark(s)]
        named.append(wanted_name)
        return named
    snapshot = instruments.load_latest()
    if snapshot is None:
        logger.warning("No instrument snapshot; run the sync first")
        return []
    fo_names = {inst.symbol for inst in snapshot.fo_underlyings}
    fo_count = len(fo_names)
    core = {s for s in fo_names if not is_benchmark(s)}
    core.add(wanted_name)
    wanted = config.FEED_UNIVERSE_SIZE if limit is None else limit
    # Aliases stripped here too, so a ranking that ever carries the index
    # under its own spelling cannot put a second name on its token.
    liquid = {s for s in by_turnover(wanted) if not is_benchmark(s)}
    if not liquid:
        logger.warning("Turnover ranking unavailable; streaming %d F&O "
                       "underlyings and %s only", fo_count, wanted_name)
        return sorted(core)
    combined = core | liquid
    cap = kt.MAX_INSTRUMENTS_PER_CONNECTION
    if len(combined) > cap:
        # The F&O set is never dropped; the ranking tail is trimmed.
        room = max(0, cap - len(core))
        combined = core | set(sorted(liquid - core)[:room])
        logger.warning("Trimmed the streaming universe to Kite's %d cap", cap)
    logger.info("Streaming %d symbols: %d F&O underlyings (the index among "
                "them, as %s) plus the top %d by 20-session turnover",
                len(combined), fo_count, wanted_name, wanted)
    return sorted(combined)


def take_lock() -> bool:
    """Refuse to start a second feed. Kite allows only three sockets per key.

    A stale lock from a killed process is adopted rather than treated as
    fatal: the PID it names is checked, and a lock naming nothing alive is
    overwritten.
    """
    if ON_FARGATE:
        # The lock guards a SHARED disk, and a container's disk is neither
        # shared nor durable: every task would take a fresh one and the
        # guarantee would be silently absent rather than merely different.
        # ECS gives the real one instead - desiredCount 1 on the service
        # means the scheduler will not place a second task at all.
        logger.info("Running on ECS, so the singleton guarantee is the "
                    "service's desiredCount rather than a file lock")
        return True
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
    if pid <= 0:
        # os.kill(0, 0) signals the whole PROCESS GROUP and a negative pid
        # signals another one. Harmless with signal 0, but a lock file
        # holding a stray value should read as "nothing alive", not as a
        # question about someone else's processes.
        return False
    if os.name != "nt":
        # os.kill with signal 0 tests for existence without delivering
        # anything. tasklist does not exist outside Windows and returned
        # an empty string there, which read as "not alive" for every PID.
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
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


SEED_CHUNK = 100
# Three consecutive failures is one minute at the default
# twenty-second flush: long enough to ride out a blip, short
# enough that a broken policy does not cost the session.
MAX_PUBLISH_FAILURES = 3


def seed_session(tokens: dict, stop=None,
                 into: "dict | None" = None) -> int:
    """Fetch the part of today's session that preceded the stream.

    Runs in a thread while ticks are already arriving. Written in chunks
    rather than once at the end, because at 1,006 symbols "once at the
    end" is half an hour away and a seed file that does not exist yet is
    worth nothing to a scan running now.

    Returns the rows written. Never raises into the caller's thread: a
    failed seed costs the opening range, while an exception escaping here
    would take the process down and cost the whole session.
    """
    symbols = list(tokens)
    print(f"seeding today's session behind the stream "
          f"({len(symbols)} symbols, {SEED_CHUNK} at a time)...", flush=True)
    # SHARED, not re-read. The scan loop needs the same seed frames, and
    # reading them back from storage cost 2.3 seconds a pass - a third of
    # the scan's own budget - for bars this thread already has in hand.
    collected: dict = {} if into is None else into
    rows = 0
    for start in range(0, len(symbols), SEED_CHUNK):
        if stop is not None and stop.is_set():
            print("  seeding stopped", flush=True)
            break
        chunk = symbols[start:start + SEED_CHUNK]
        try:
            collected.update(live_bars.backfill_today(chunk))
            rows = live_bars.write_seed(collected)
        except Exception as exc:
            # One bad chunk must not end the rest: the opening range for
            # 900 symbols is worth more than a clean traceback for 100.
            logger.warning("seed chunk %d-%d failed: %s",
                           start, start + len(chunk), exc)
            continue
        covered = sum(1 for f in collected.values()
                      if live_bars.session_is_covered(f))
        print(f"  seeded {len(collected)}/{len(symbols)}, {rows} bars, "
              f"{covered} reaching the open", flush=True)
    return rows


# Beyond this the instrument snapshot is old enough that tokens may have
# been reissued and the F&O set may have changed at an expiry. Not fatal -
# a feed that will not start is worse than one ranking a slightly stale
# universe - so it warns rather than refusing.
SNAPSHOT_STALE_DAYS = 14


def _warn_if_snapshot_is_stale() -> None:
    """Say how old the baked-in instrument snapshot is.

    On a container this is frozen at image build time and nothing syncs it,
    so without this line a months-old universe.json would look exactly like
    a fresh one - and every instrument token the feed subscribes to comes
    out of it.
    """
    try:
        snapshot = instruments.load_latest()
        captured = getattr(snapshot, "captured_at", None) if snapshot else None
        if not captured:
            logger.warning("Instrument snapshot carries no capture date")
            return
        stamp = datetime.fromisoformat(str(captured).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=IST)
        days = (datetime.now(IST) - stamp).days
        message = "Instrument snapshot is %d day(s) old (captured %s)"
        if days >= SNAPSHOT_STALE_DAYS:
            logger.warning(message + " - rebuild the image to refresh it",
                           days, stamp.date())
        else:
            logger.info(message, days, stamp.date())
    except Exception as exc:
        logger.warning("Could not date the instrument snapshot: %s", exc)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if ON_FARGATE and not object_store.enabled():
        # Otherwise the task runs, logs healthy flush lines all session,
        # writes every bar to the container's own ephemeral disk and throws
        # the lot away when it stops - with nothing anywhere reporting it.
        logger.error("Running on ECS with no object store configured. Set "
                     "%s on the task definition, or this feed would publish "
                     "nothing.", object_store.BUCKET_ENV)
        return 1
    logger.info("Publishing to %s", object_store.describe())
    _warn_if_snapshot_is_stale()
    symbols = universe(args.symbols)
    if not symbols:
        print("no symbols to stream")
        return 1
    print(f"universe: {len(symbols)} symbols")

    # THE CONSOLIDATED STORE FIRST, if one has been published. Without it
    # prewarm refetches 17 days for every symbol from Kite at three
    # requests a second - measured 2026-09-18 at 836 seconds, on every
    # cold start, which made a mid-session redeploy cost twenty minutes of
    # the session. The nightly job already folds this file; it just never
    # left the laptop. A local store wins over the published copy, so this
    # is a no-op on a developer machine.
    try:
        # market_source is imported further down in this function, so it is
        # NOT in scope here - and without this line the NameError would be
        # swallowed by the handler below and the hydrate would silently
        # never run, which looks exactly like it working.
        import bar_store
        import market_source as ms

        interval = ms.kite_interval(config.SCAN_BAR_INTERVAL)
        if bar_store.hydrate(interval):
            print(f"hydrated the {interval} store from the object store")
    except Exception as exc:
        # Never fatal: the cost of failing here is the slow path, which is
        # exactly what happened before this existed.
        logger.warning("Could not hydrate the consolidated store, so "
                       "prewarm will refetch: %s", exc)

    history_days = args.history_days or live_bars.history_days()
    print(f"prewarming {history_days} prior days (once, cached)...")
    started = time.time()
    history = live_bars.prewarm(symbols, days=history_days)
    start, end = live_bars.history_window(history_days)
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
        # A container has no .kite_session.json and no way to run an
        # interactive login, so the local advice is unfollowable there.
        print("No KITE_API_KEY/KITE_ACCESS_TOKEN in the environment"
              if ON_FARGATE else
              "No Kite session. Run: .venv\\Scripts\\python -m kite_login")
        release_lock()
        return 1

    builder = live_bars.BarBuilder()
    stop = threading.Event()

    def on_ticks(ticks: list) -> None:
        builder.add(ticks)

    def flusher() -> None:
        """Persist on a timer so a reader always sees a recent file."""
        failures = 0
        while not stop.wait(args.flush_every):
            try:
                rows = builder.flush()
            except Exception as exc:
                # One bad flush must not end all future ones: this thread
                # is the only thing writing the file the scanner reads.
                logger.warning("flush failed: %s", exc)
                rows = 0
            # STATS FROM MEMORY, NOT FROM A FRESH READ. This used to call
            # live_bars.status(), which re-downloads the whole object it
            # has just written - about 1,170 extra GETs over a session,
            # each one larger than the last, purely to print a line.
            frame = builder.snapshot()
            if frame.empty:
                failures = 0
            elif rows:
                failures = 0
            else:
                # There were bars to write and none were written, so the
                # publish failed. Silently retrying forever means a bad
                # task-role policy costs the entire session with nothing
                # but a warning per flush to show for it.
                failures += 1
                logger.error("publish failed %d time(s) in a row - %d bars "
                             "held in memory and none persisted",
                             failures, len(frame))
                if failures >= MAX_PUBLISH_FAILURES:
                    logger.error("giving up after %d consecutive failures; "
                                 "a feed that cannot publish is doing no "
                                 "work, so stopping for ECS to restart it",
                                 failures)
                    stop.set()
                    return
            # Publish the numbers a reader needs, so nobody has to pull
            # the whole growing session object to learn them.
            live_bars.write_status(frame)
            instruments = (int(frame["instrument_token"].nunique())
                           if not frame.empty else 0)
            age = None
            if not frame.empty:
                latest = pd.DatetimeIndex(frame["Date"]).max()
                if latest is not None and latest.tzinfo:
                    age = (datetime.now(IST)
                           - latest.tz_convert(IST)).total_seconds()
            # `age == age` alone was a NaN check, but None passes it too and
            # formatting None raised inside this thread - which killed the
            # flusher silently and stopped every write.
            fresh = (f", newest {age:.0f}s old"
                     if isinstance(age, (int, float)) and age == age else "")
            print(f"  {datetime.now(IST):%H:%M:%S}  bars {rows} across "
                  f"{instruments} instruments{fresh}", flush=True)

    pump = threading.Thread(target=flusher, name="flusher", daemon=True)
    pump.start()

    # BEHIND THE SOCKET, not in front of it. This used to run to completion
    # before subscribing: at 1,006 symbols that is one historical call each
    # at 3 a second, so the feed stayed blind for the first half hour of
    # the session - the very part the seed exists to cover. Measured on
    # 2026-09-11: lock at 09:06, 397/1006 seeded by 09:28, socket not yet
    # open. Now the stream starts immediately and the seed fills in behind
    # it, which is safe because combined() keeps the LAST value for a
    # stamp and live bars come after the seed in that concatenation.
    # Shared with the scan loop below, and filled as the seeding proceeds,
    # so an early scan sees whatever is seeded so far rather than nothing.
    seeded: dict = {}
    seeder = threading.Thread(target=seed_session,
                              args=(dict(tokens), stop, seeded),
                              name="seeder", daemon=True)
    seeder.start()

    # THE SCAN, WHERE THE BARS ALREADY ARE. Measured 2026-09-16: on the
    # laptop a scan of these 216 names cost 48 seconds, of which only 5.8
    # was the scan - the rest was fetching bars this process built itself.
    # Here it costs the 5.8 and publishes a 56 KB table.
    #
    # Default -1 means "on when there is somewhere to publish to": the
    # container has an object store and wants this, a local feed has
    # neither and would burn CPU producing nothing.
    scan_every = args.scan_every
    if scan_every < 0:
        scan_every = (scan_publish.DEFAULT_EVERY
                      if object_store.enabled() else 0)
    if scan_every:
        import daily_context

        context = daily_context.load(symbols)
        if not context:
            logger.warning("No daily context published, so the scan would "
                           "have no previous close, pivot range or turnover "
                           "- run daily_context.py. Scanning anyway, with "
                           "those gates failing closed.")
        else:
            stamp = daily_context.age()
            if stamp:
                logger.info("Daily context covers %d symbols, newest %s "
                            "(%d day(s) ago)", len(context), stamp[0], stamp[1])
        scanner = threading.Thread(
            target=scan_publish.loop,
            args=(builder, symbols, history, tokens, context, stop,
                  scan_every),
            kwargs={"seed": seeded},
            name="scanner", daemon=True)
        scanner.start()
    else:
        logger.info("No scan loop: nowhere to publish to")

    def shutdown(*_) -> None:
        stop.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), shutdown)
            except Exception:
                pass

    print(f"streaming {len(tokens)} instruments now; the seed for the "
          f"earlier part of today fills in behind it. Ctrl-C to stop")
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
