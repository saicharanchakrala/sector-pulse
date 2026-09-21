"""Fetch fresh intraday bars, so the consolidated store can advance.

WHY THIS EXISTS. Nothing on this machine was refreshing them. The
3-minute store is folded from the per-symbol cache, and that cache was
being filled as a SIDE EFFECT of the UI's own scans - which stopped on
2026-09-18 when SCAN_UI_MAY_DOWNLOAD went to False to keep the UI from
exhausting the machine. The only other caller of live_bars.prewarm is
live_feed, and that runs in a container whose disk is thrown away when it
stops.

So the store froze. Measured 2026-09-21:

    3-minute cache files ending 2026-09-17    217
                                 2026-09-16  1,590
                                 2026-09-15  2,502

and the consolidated store's newest session was 2026-09-17 - it did not
even hold Friday. live_bars.prewarm then refused it as stale and the feed
refetched everything from Kite on every cold start, which is the cost the
publishing of that store exists to remove.

This is the same failure the daily store had, in the second of the two
stores, and from the same cause. fetch_tail.extend_stale fixes the daily
side; this fixes the intraday side.

RUN IT AFTER THE CLOSE, before the fold. Every request competes with the
live feed for the same 3-a-second Kite budget.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import config

logger = logging.getLogger(__name__)


def universe() -> list:
    """The symbols the feed streams, which are the ones worth prewarming.

    Taken from live_feed so this cannot drift from what the container
    actually needs. Falls back to nothing rather than guessing: fetching
    the wrong universe would burn the Kite budget on symbols the scan
    never reads.
    """
    try:
        import live_feed

        return live_feed.universe("")
    except Exception as exc:
        logger.warning("Could not determine the streaming universe: %s", exc)
        return []


def refresh(symbols: "list | None" = None, days: "int | None" = None) -> int:
    """Prewarm the scan interval. Returns symbols with prior bars.

    Goes through live_bars.prewarm rather than fetching directly, so the
    store-first rule, the staleness refusal and the per-symbol coverage
    proof are the same ones the feed applies. A store that is already
    current therefore costs one read and no requests.
    """
    import live_bars
    import market_source

    names = symbols if symbols is not None else universe()
    if not names:
        print("no universe to refresh")
        return 0
    interval = market_source.kite_interval(config.SCAN_BAR_INTERVAL)
    days = days or live_bars.history_days()
    start, end = live_bars.history_window(days)
    print(f"refreshing {interval} bars for {len(names):,} symbols, "
          f"{start} .. {end}")

    started = time.time()
    try:
        held = live_bars.prewarm(names, days=days)
    except market_source.NoSession as exc:
        # The same daily condition fetch_tail reports rather than crashes
        # on: tokens die about 06:00 and this runs after the close.
        print(f"\nCANNOT REFRESH: {exc}")
        print("  The intraday store stays where it is, so the feed will "
              "refetch from Kite on its next start.")
        return 0
    print(f"  {len(held):,}/{len(names):,} symbols have prior bars in "
          f"{time.time() - started:.0f}s")
    return len(held)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch fresh intraday bars so the consolidated store "
                    "can advance")
    parser.add_argument("--days", type=int, default=None,
                        help="calendar days of history (default: the "
                             "feed's own window)")
    args = parser.parse_args(argv)

    try:
        import fetch_tail

        if fetch_tail.market_is_open():
            print("The market is open, and every fetch here competes with "
                  "the live feed for the same 3-a-second budget.\n"
                  "Run it after 15:30.")
            return 2
    except Exception:
        pass

    return 0 if refresh(days=args.days) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
