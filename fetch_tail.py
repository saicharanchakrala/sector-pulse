"""Daily bars for listed equities the consolidated store has never seen.

WHY THIS EXISTS. The live feed subscribes to the symbols it can RANK, and
live_feed.by_turnover ranks on 20-session turnover read from the daily
store. A symbol with no rows there cannot be ranked, so it never reaches
the feed however much headroom the Kite socket has - measured 2026-09-15:

    listed equities        2,568
    streamed               2,485   (cap 3,000, so 515 spare)
    not streamed             283   ALL of them for want of daily bars

The chain is: no daily bars -> no turnover -> no rank -> not streamed ->
the scan downloads them one at a time while you wait. Fetching their
history once breaks it at the first link, and they then earn their way
onto the feed by ranking rather than being forced onto the socket.

WHAT IT DOES NOT DO. It never widens the feed directly and never edits
config. It only fills a hole in the daily store; live_feed picks up the
consequence on its next start.

RUN IT AFTER THE CLOSE. Every fetch competes with the live feed for the
same 3-a-second Kite budget, and during a session that budget is what
keeps the scan quick. Outside market hours nothing else is asking.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import bar_store
import config
import instruments
import market_source

# Enough history to rank on and to describe the daily horizons: the
# longest lookback is 252 sessions, and _assess_series wants lookback + 5.
YEARS = 2
SESSION_CLOSE = (15, 30)


def missing_symbols() -> list:
    """Listed equities with no rows in the consolidated daily store."""
    snapshot = instruments.load_latest()
    if snapshot is None:
        print("No instrument snapshot; run the sync first")
        return []
    listed = {i.symbol for i in snapshot.equities}
    try:
        have = set(bar_store.load("day"))
    except Exception as exc:
        print(f"No daily store to compare against ({exc}); treating all "
              f"{len(listed)} listed equities as missing")
        have = set()
    return sorted(listed - have)


def market_is_open(now=None) -> bool:
    """Whether a session is running, so the run can refuse to compete."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    open_at = config.SCAN_SESSION_OPEN[0] * 60 + config.SCAN_SESSION_OPEN[1] \
        if hasattr(config, "SCAN_SESSION_OPEN") else 9 * 60 + 15
    close_at = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    return open_at <= minutes <= close_at


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    force = "--now" in argv

    if market_is_open() and not force:
        print("The market is open, and every fetch here competes with the "
              "live feed for the same 3-a-second budget.\n"
              "Run it after 15:30, or pass --now to override.")
        return 2

    names = missing_symbols()
    if not names:
        # STILL REBUILD. On most days nothing is missing, and returning
        # here would mean the consolidated store never got folded - so a
        # nightly run would leave it drifting stale exactly as it was
        # found on 2026-09-15, three sessions behind, while reporting
        # success. The fold is local and costs no Kite requests.
        print("Nothing missing - every listed equity is in the daily store.")
        print("Folding the cache anyway, so the store stays current...")
        bar_store.main(["--intervals", "day"])
        return 0

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=int(YEARS * 365))
    workers = getattr(config, "KITE_FETCH_WORKERS", 1)
    print(f"{len(names):,} listed equities have no daily bars")
    print(f"span {start} .. {end}, {workers} in flight, "
          f"~{len(names) / max(1, workers * 1.8) / 60:.0f} min\n")

    started = time.time()
    # One call: market_source.bars fetches concurrently under the shared
    # rate gate and caches each symbol as it lands, so an interrupted run
    # costs nothing - the next one skips whatever already cached.
    got = market_source.bars(names, start, end, interval="day")
    print(f"\nfetched {len(got):,}/{len(names):,} in "
          f"{time.time() - started:.0f}s")
    empty = [s for s in names if s not in got]
    if empty:
        print(f"{len(empty):,} returned nothing - delisted, suspended or "
              f"never traded: {', '.join(empty[:8])}")
        print("  Almost all are ABSENT FROM KITE INSTRUMENT MASTER - "
              "token_for finds no token, so there is nothing to fetch "
              "and nothing to stream. Measured 2026-09-15: all 258 were "
              "of this kind and 0 had tokens. They sit in NSE listed "
              "equity data while the broker does not carry them, so no "
              "amount of fetching changes it.")

    if not got:
        # NOTHING WAS FETCHED, so the per-symbol cache is identical to
        # the one already folded and the store cannot have changed.
        # Rebuilding anyway burned 370 seconds on the first real run of
        # this script - and would have done so every night, for nothing.
        print("\nNothing new was fetched, so the store cannot have "
              "changed - skipping the fold.")
        return 0

    print("\nfolding the per-symbol cache into the consolidated store...")
    bar_store.main(["--intervals", "day"])

    after = set(bar_store.load("day"))
    print(f"\ndaily store now holds {len(after):,} symbols")
    print("The feed picks these up on its NEXT START, once they rank on "
          "20-session turnover. Nothing streams differently until then.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
