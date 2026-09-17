"""Delete per-symbol INTRADAY cache files nobody reads any more.

WHY. bar_cache accumulates one file per symbol per interval per window,
and every window the app uses rolls forward daily - so each trading day
writes a fresh file per symbol and never removes the previous one. It
reached 31,537 files on 2026-09-17, at which point a plain directory
listing of it took over two minutes and the app's own cache index over
that directory cost 1.74 seconds per build.

WHY NOT "DELETE 3-MINUTE BARS ONCE THE DAY IS OVER". That was the ask, and
taken literally it destroys data on the next nightly run. bar_store.rebuild
folds the CONSOLIDATED store from the per-symbol files ALONE - it never
reads the store it is replacing - so a per-symbol file deleted today is
gone from bars_3minute.parquet tomorrow night. And prior sessions are not
spare: relative_volume medians today's cumulative volume against the same
clock time across roughly a dozen PRIOR sessions, and the scan's own
window is 17 calendar days. Pruning to "yesterday" would leave the gate
with no baseline, and it would fail quietly, because a short history reads
as a low ratio rather than as an error.

So the rule here is "older than every reader", derived from config rather
than guessed:

    SCAN_BAR_LOOKBACK      -> 17 calendar days for the scan's own window
    live_bars.history_days -> the feed's prewarm, the same 17 by design
    SCAN_REPLAY_LOOKBACK_DAYS -> how far back a replay may be anchored

THE DAILY INTERVAL IS NEVER TOUCHED. bars_day.parquet carries history back
to about 2001 and is rebuilt from per-symbol day files by the same
full-rebuild path, so pruning those would delete twenty-five years of
daily bars. Intervals are allow-listed, not denied.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

import config
import market_source

logger = logging.getLogger(__name__)

# Slack on top of the longest reader's window. Covers a long weekend, a
# run of holidays, and a nightly job that did not run for a few days.
SLACK_DAYS = 10

# ONLY these. "day" is deliberately absent and must stay absent.
PRUNABLE = ("minute", "3minute", "5minute", "10minute", "15minute",
            "30minute", "60minute")


def retention_days() -> int:
    """Calendar days of intraday history every reader together needs."""
    wanted = [int(getattr(config, "SCAN_REPLAY_LOOKBACK_DAYS", 12))]
    try:
        wanted.append(int(market_source.calendar_days(
            config.SCAN_BAR_LOOKBACK, 12)))
    except Exception:
        wanted.append(17)
    try:
        import live_bars

        wanted.append(int(live_bars.history_days()))
    except Exception:
        pass
    return max(wanted) + SLACK_DAYS


def stale_files(today: "date | None" = None, keep_days: "int | None" = None):
    """[(path, end date)] for prunable files ending before the cutoff.

    Keyed on the window's END, because that is what decides whether any
    reader can still want it. A file is kept when its end falls on or
    after the cutoff, so a window that merely STARTS long ago - a wide
    span reaching up to yesterday - survives.
    """
    today = today or date.today()
    keep = retention_days() if keep_days is None else keep_days
    cutoff = today - timedelta(days=keep)
    found = []
    try:
        entries = list(market_source.CACHE_DIR.iterdir())
    except OSError as exc:
        logger.warning("Cannot read %s: %s", market_source.CACHE_DIR, exc)
        return found
    for path in entries:
        parsed = market_source._parse_cache_name(path.name)
        if parsed is None:
            continue
        _symbol, interval, _start, end, _oi = parsed
        if interval not in PRUNABLE:
            continue
        if end >= cutoff:
            continue
        found.append((path, end))
    return found


def redundant_files():
    """[(path, end)] for intraday spans another file already contains.

    THE ACTUAL WASTE, as it turns out. Profiled 2026-09-17: 19,515
    intraday files for about 2,500 symbols, roughly eight per symbol, and
    EVERY ONE ending within the last thirteen days. Nothing was old, so an
    age rule reclaimed nothing. The files pile up sideways instead - the
    app asks for a window ending today, and tomorrow asks for a window one
    day wider, so each day leaves another near-copy of the same bars:

        RELIANCE__3minute__20260829_20260915
        RELIANCE__3minute__20260830_20260916
        RELIANCE__3minute__20260831_20260917   <- contains both of those
        RELIANCE__3minute__20260915_20260915   <- contained in all three

    A file is redundant when another file for the same symbol, interval
    and oi flag spans at or beyond it at BOTH ends and was written no
    earlier. The freshness condition matters: the container is what the
    cache would serve for that range anyway - _covering_span picks exactly
    this file - so deleting the contained one removes a copy, not a source.
    """
    groups: dict = {}
    try:
        entries = list(market_source.CACHE_DIR.iterdir())
    except OSError as exc:
        logger.warning("Cannot read %s: %s", market_source.CACHE_DIR, exc)
        return []
    for path in entries:
        parsed = market_source._parse_cache_name(path.name)
        if parsed is None:
            continue
        symbol, interval, start, end, oi = parsed
        if interval not in PRUNABLE:
            continue
        try:
            written = path.stat().st_mtime
        except OSError:
            continue
        groups.setdefault((symbol, interval, oi), []).append(
            (start, end, written, path))

    victims = []
    for spans in groups.values():
        if len(spans) < 2:
            continue
        # Widest first, freshest breaking a tie, so the survivors are
        # considered before anything they might contain.
        spans.sort(key=lambda row: ((row[1] - row[0]).days, row[2]),
                   reverse=True)
        kept: list = []
        for start, end, written, path in spans:
            covered = any(k_start <= start and k_end >= end
                          and k_written >= written
                          for k_start, k_end, k_written, _ in kept)
            if covered:
                victims.append((path, end))
            else:
                kept.append((start, end, written, path))
    return victims


def prune(today: "date | None" = None, keep_days: "int | None" = None,
          apply: bool = False, subsumed: bool = True) -> tuple:
    """(files considered, bytes freed). Deletes only when `apply`."""
    victims = stale_files(today, keep_days)
    if subsumed:
        seen = {path for path, _ in victims}
        victims += [(path, end) for path, end in redundant_files()
                    if path not in seen]
    freed = 0
    removed = 0
    for path, _end in victims:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if not apply:
            freed += size
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path.name, exc)
            continue
        freed += size
        removed += 1
    return (len(victims) if not apply else removed), freed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete intraday bar cache files older than every "
                    "reader's window. Never touches daily bars.")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without it this only reports")
    parser.add_argument("--keep-days", type=int, default=None,
                        help=f"override the retention window "
                             f"(default {retention_days()})")
    parser.add_argument("--no-subsumed", action="store_true",
                        help="keep spans another file already contains")
    args = parser.parse_args(argv)

    keep = args.keep_days if args.keep_days is not None else retention_days()
    if keep < 1:
        print("refusing to prune with a retention window under one day")
        return 1
    count, freed = prune(keep_days=keep, apply=args.apply,
                         subsumed=not args.no_subsumed)
    verb = "deleted" if args.apply else "would delete"
    print(f"intraday cache: {verb} {count:,} file(s), "
          f"{freed / 1_048_576:.1f} MB, keeping {keep} days "
          f"(daily bars untouched)")
    if not args.apply and count:
        print("  re-run with --apply to remove them")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
