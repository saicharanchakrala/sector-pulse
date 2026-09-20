"""Bring the container's scan log down, so outcomes.py can resolve it.

WHY THIS EXISTS. scan_intraday.append_log is called by exactly one thing -
the command-line scanner. The Streamlit app evaluates setups and never
calls it, and the feed publishes a parquet table instead. So the log that
was supposed to become a track record sat at 8 rows from 9 September while
the feed scanned 2,485 symbols every 45 seconds for days, and
outcomes.csv held five setups from one instant, three of them replays.

The feed now accumulates its session and publishes scan_log_YYYYMMDD.csv.
This fetches those files to sit beside the local log, where
outcomes.log_files picks them up by glob. Downloaded rather than merged
into scan_log.csv on purpose: merging means matching headers, and a
header mismatch rotates the file - the exact failure that stranded four
rows once already. Separate files cannot misalign, and outcomes dedupes
by row_id across all of them anyway.

Run it before outcomes.py in the nightly job.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import config
import object_store

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# How many days back to look. A week covers a nightly job that did not run
# over a weekend or a holiday, which is not hypothetical: the job last ran
# on 2026-09-16 and the sessions of the 17th and 18th went unrecorded.
DEFAULT_LOOKBACK_DAYS = 7


def local_path(day: date):
    """Where one day's fetched log lives, beside the local scan log."""
    return config.SCAN_LOG_CSV.parent / f"scan_log_{day:%Y%m%d}.csv"


def fetch_day(day: date, force: bool = False) -> int:
    """Download one day's published log. Returns rows written, 0 if none.

    Skips a day already on disk unless `force`, because the published
    object is rewritten throughout the session and a completed local copy
    of a past day has nothing to gain from being refetched. TODAY is
    always refetched - the session is still being written to.
    """
    import scan_publish

    target = local_path(day)
    is_today = day == datetime.now(IST).date()
    if target.exists() and not force and not is_today:
        return 0
    try:
        payload = object_store.get(scan_publish.log_object_name(
            datetime.combine(day, datetime.min.time(), tzinfo=IST)))
    except object_store.StorageError as exc:
        logger.warning("Could not read the %s log: %s", day, exc)
        return 0
    if not payload:
        return 0
    text = payload.decode("utf-8", errors="replace")
    rows = max(0, text.count("\n") - 1)          # minus the header
    try:
        target.write_bytes(payload)
    except OSError as exc:
        logger.warning("Could not write %s: %s", target.name, exc)
        return 0
    logger.info("Fetched %s: %d rows", target.name, rows)
    return rows


def fetch(days: int = DEFAULT_LOOKBACK_DAYS, force: bool = False) -> dict:
    """{day: rows} for every published log found in the window."""
    if not object_store.enabled():
        logger.warning("No object store configured, so there is no "
                       "container log to fetch. Set %s.",
                       object_store.BUCKET_ENV)
        return {}
    today = datetime.now(IST).date()
    found = {}
    for back in range(max(1, days)):
        day = today - timedelta(days=back)
        rows = fetch_day(day, force=force)
        if rows:
            found[day] = rows
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the feed's published scan logs so outcomes.py "
                    "can resolve them")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"how far back to look "
                             f"(default {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--force", action="store_true",
                        help="refetch days already on disk")
    args = parser.parse_args(argv)

    found = fetch(days=args.days, force=args.force)
    if not found:
        print("no published scan logs found - has the feed run since the "
              "logging was deployed?")
        return 0
    total = sum(found.values())
    print(f"fetched {len(found)} day(s), {total:,} logged setups:")
    for day in sorted(found):
        print(f"  {day}  {found[day]:,}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
