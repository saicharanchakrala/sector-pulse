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
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from zoneinfo import ZoneInfo

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


def settled_end(now: "datetime | None" = None) -> date:
    """The last date a completed daily bar can exist for.

    INCLUDES TODAY ONCE THE SESSION HAS CLOSED, and that is the whole
    point. An earlier version took yesterday unconditionally, which is
    right for a run during market hours and wrong for this job - which
    exists to run AFTER the close. On Monday 2026-09-21 at 16:02, half an
    hour after the 15:30 close, it returned Friday the 18th: every symbol
    already reached that date, so the extension fetched nothing and
    daily_context republished Friday's close as the previous close. The
    store would have sat exactly one session behind for ever, every night,
    while the coverage report said it was level.

    Then walked back off a weekend. Saturdays and Sundays are certain;
    holidays are not, and a symbol whose newest bar is a holiday behind
    gets one redundant request rather than a wrong answer.
    """
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    close = dt_time(*config.SCAN_SESSION_CLOSE)
    end = now.date()
    if end.weekday() >= 5 or now.time() < close:
        # Nothing settled today: the weekend, or the session is still on.
        end -= timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    return end


def newest_sessions() -> dict:
    """{symbol: its newest session} from the consolidated daily store."""
    try:
        frames = bar_store.load("day")
    except Exception as exc:
        print(f"No daily store to read ({exc})")
        return {}
    out = {}
    for symbol, frame in (frames or {}).items():
        if frame is None or frame.empty:
            continue
        try:
            out[symbol] = frame.index.max().date()
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def stale_groups(newest: dict, end: date) -> dict:
    """{newest session: [symbols]} for every symbol not reaching `end`.

    WHY THIS EXISTS AT ALL. This script's original job was only to backfill
    listed equities the store had NEVER seen - `listed - have`. Nothing
    extended a symbol that was already present. What kept the store
    current was an accident: the UI's longer-horizon tables re-downloaded
    three months of daily bars every morning, refreshing the per-symbol
    day files that the fold then consolidated. Switching that off on
    2026-09-18, to stop the UI hammering Kite, removed the only thing
    advancing the daily store - and four days later it looked like this:

        2026-09-09   2,518 symbols
        2026-09-11   2,315
        2026-09-15   1,055
        2026-09-16      12

    So prev_close was a DIFFERENT DATE per symbol, which is worse than a
    uniform lag: relative_strength compares a symbol's day change against
    the index across a universe whose members were measuring from
    different days.

    GROUPED BY THE DATE THEY REACH, because market_source.bars takes one
    span for a list of symbols. A handful of groups covers the universe
    and each symbol is requested exactly once, over the short span it
    actually needs rather than the two years a backfill would ask for.
    """
    groups: dict = {}
    for symbol, last in newest.items():
        if last >= end:
            continue
        groups.setdefault(last, []).append(symbol)
    return {day: sorted(names) for day, names in groups.items()}


def extend_stale(end: "date | None" = None) -> int:
    """Fetch every symbol forward to `end`. Returns symbols refreshed."""
    end = end or settled_end()
    newest = newest_sessions()
    if not newest:
        return 0
    groups = stale_groups(newest, end)
    if not groups:
        print(f"Every symbol in the daily store reaches {end}.")
        return 0

    behind = sum(len(names) for names in groups.values())
    workers = getattr(config, "KITE_FETCH_WORKERS", 1)
    print(f"{behind:,} of {len(newest):,} symbols are behind {end}:")
    for day in sorted(groups):
        print(f"  {day}  {len(groups[day]):,} symbols")
    print(f"~{behind / max(1, workers * 1.8) / 60:.0f} min at "
          f"{workers} in flight\n")

    refreshed = 0
    for day in sorted(groups):
        names = groups[day]
        # STARTS ON THE DAY IT ALREADY HAS, not the day after. The overlap
        # costs one bar and means a symbol whose last stored session was
        # itself partial gets it rewritten rather than built on.
        started = time.time()
        try:
            got = market_source.bars(names, day, end, interval="day")
        except market_source.NoSession as exc:
            # AN EXPIRED TOKEN IS A DAILY CONDITION, NOT A CRASH. Kite
            # tokens die about 06:00 and this job runs after the close, so
            # a session that has not been refreshed is the ordinary case
            # for a job run a day late. Letting it escape aborted the
            # whole of fetch_tail - including the fold - and printed a
            # traceback into the nightly log, where the one thing anybody
            # needs to read is which action fixes it.
            print(f"\n  CANNOT EXTEND: {exc}")
            print("  The daily store stays where it is. Every symbol's "
                  "prev_close remains older than one session until this "
                  "is fixed and the job is re-run.")
            return refreshed
        refreshed += len(got)
        print(f"  {day}: {len(got):,}/{len(names):,} in "
              f"{time.time() - started:.0f}s", flush=True)
    return refreshed


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

    # TWO JOBS, and the second one is why the store went stale. Backfill
    # only ever covered symbols the store had NEVER seen; nothing moved an
    # existing symbol forward, because the UI's daily downloads were doing
    # that by accident until they were switched off. Extending runs first
    # and unconditionally: it is the one that has to happen every night.
    print("Extending every symbol to the last settled session...")
    extended = extend_stale()

    names = missing_symbols()
    if not names:
        # STILL REBUILD. On most days nothing is missing, and returning
        # here would mean the consolidated store never got folded - so a
        # nightly run would leave it drifting stale exactly as it was
        # found on 2026-09-15, three sessions behind, while reporting
        # success. The fold is local and costs no Kite requests.
        print("\nNothing missing - every listed equity is in the daily store.")
        print("Folding the cache so the store picks up what was extended...")
        bar_store.main(["--intervals", "day"])
        report_coverage()
        return 0

    end = settled_end()
    start = end - timedelta(days=int(YEARS * 365))
    workers = getattr(config, "KITE_FETCH_WORKERS", 1)
    print(f"{len(names):,} listed equities have no daily bars")
    print(f"span {start} .. {end}, {workers} in flight, "
          f"~{len(names) / max(1, workers * 1.8) / 60:.0f} min\n")

    started = time.time()
    # One call: market_source.bars fetches concurrently under the shared
    # rate gate and caches each symbol as it lands, so an interrupted run
    # costs nothing - the next one skips whatever already cached.
    try:
        got = market_source.bars(names, start, end, interval="day")
    except market_source.NoSession as exc:
        # Same reasoning as the extension above: report the one action
        # that fixes it and let the fold run on whatever is already
        # cached, rather than aborting the step with a traceback.
        print(f"\nCANNOT BACKFILL: {exc}")
        got = {}
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

    if not got and not extended:
        # NOTHING WAS FETCHED AT ALL, so the per-symbol cache is identical
        # to the one already folded and the store cannot have changed.
        # Rebuilding anyway burned 370 seconds on the first real run of
        # this script - and would have done so every night, for nothing.
        #
        # `extended` is in the condition because skipping the fold after a
        # successful extension is exactly how the store stayed four days
        # behind while the job reported success.
        print("\nNothing new was fetched, so the store cannot have "
              "changed - skipping the fold.")
        return 0

    print("\nfolding the per-symbol cache into the consolidated store...")
    bar_store.main(["--intervals", "day"])
    report_coverage()
    print("The feed picks these up on its NEXT START, once they rank on "
          "20-session turnover. Nothing streams differently until then.")
    return 0


def report_coverage() -> None:
    """How far the store now reaches, per session, and whether that is level.

    Printed because a RAGGED store is the failure that hid for four days:
    a single "now holds 2,521 symbols" line was true the whole time and
    said nothing about prev_close being a different date for each of them.
    """
    newest = newest_sessions()
    if not newest:
        print("\ndaily store is empty")
        return
    end = settled_end()
    counts: dict = {}
    for day in newest.values():
        counts[day] = counts.get(day, 0) + 1
    reaching = counts.get(end, 0)
    print(f"\ndaily store holds {len(newest):,} symbols; "
          f"{reaching:,} reach {end}")
    for day in sorted(counts, reverse=True)[:5]:
        flag = "" if day >= end else "   BEHIND"
        print(f"  {day}  {counts[day]:,}{flag}")
    if reaching < len(newest):
        print(f"  {len(newest) - reaching:,} symbols did not reach {end} - "
              f"delisted, suspended, or absent from Kite's master. Their "
              f"prev_close is older than one session and every reading "
              f"derived from it is measured from a different day.")


if __name__ == "__main__":
    raise SystemExit(main())
