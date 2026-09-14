"""Corporate announcements from NSE, timestamped by the exchange.

WHY THIS SOURCE AND NOT A NEWS FEED. news_fetcher reads RSS, which serves
only the most recent items and has no date parameter, so the archive can
only grow forwards - it holds 6 days against 228 sessions of bars. This
endpoint takes a date range and answers for at least two years back,
measured:

    last 7 days   3,224      6 months ago   2,383
    1 month ago   8,874      1 year ago     3,165
                             2 years ago    3,605

More importantly it solves the problem that ruins most backfilled-news
studies. A scraped article's date is often last-modified rather than
first-published, and outlets silently rewrite headlines as a story
develops, so a test can end up reading a headline written AFTER the move
it claims to predict. Each record here carries the exchange's own
dissemination time to the second:

    an_dt          13-Sep-2026 22:55:21    company filed
    exchdisstime   13-Sep-2026 22:55:22    NSE published
    difference     00:00:01

`exchdisstime` is when it became public, so an analysis may use a record
only against bars stamped after it, and the leak closes.

THE TWO ARE USUALLY A SECOND APART AND SOMETIMES NOT. Measured across all
358,791 rows: median 1s, but 184 over a minute, 84 over an hour, 31 over a
day, and one CROMPTON filing disseminated 2.3 DAYS after it was submitted.
So the fallback from one to the other is not cosmetic - on those rows it
would mark an announcement public several sessions early. That is why
`tidy` refuses a response missing the field rather than quietly falling
back for every row at once.

WHAT THIS IS NOT GOING TO FIX, stated here so the scale is honest.
Measured over 358,791 announcements across two years:

    after close (post 15:30)   64.3%
    DURING session             34.4%
    before open (pre 09:15)     1.3%

Two thirds arrive after the close and gap the stock at the next open,
which is a different trade from the intraday one. Of the third that land
during the session the largest single category is "Copy of Newspaper
Publication" - a regulatory formality carrying no information - and the
materially market-moving categories together are roughly a tenth of it.
Narrowed to the liquid F&O universe the scanner actually trades, that is
single digits a day. It cannot move a base rate measured over 685,000
trades. It can only carve out a small separate subset.

NOTHING IS FILTERED AT FETCH TIME. Every field the API returns is stored
and every category is kept; selection happens at analysis time, because a
filter applied here is irreversible without refetching.

RESUME STATE LIVES BESIDE THE DATA, NOT INSIDE IT. The completed windows
are recorded in a small JSON sidecar rather than inferred from a column,
so a genuinely empty week resumes correctly, and the grid is anchored to a
fixed epoch rather than to today - otherwise every tag changes overnight
and "resumable" means "refetches everything tomorrow".
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
import instruments

IST = "Asia/Kolkata"

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "forecast_cache"
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "announcements.parquet"
STATE = OUT_DIR / "announcements.windows.json"

API = "https://www.nseindia.com/api/corporate-announcements"
REFERER = ("https://www.nseindia.com/companies-listing/"
           "corporate-filings-announcements")
PAUSE = 0.9
DEFAULT_YEARS = 2

# Seven days per request. fetch_insider uses months and documents that the
# endpoint silently truncates a long window - and a truncated window looks
# exactly like a quiet week. Seven days measured at 3,200 records, well
# under any plausible cap, and SUSPECT_COUNT below catches it anyway.
WINDOW_DAYS = 7
# The grid is anchored HERE, not at today, so a window's identity does not
# move. Anchoring at `today - years` shifted every boundary by one day
# overnight: measured, 0 of 105 tags matched between one day's run and the
# next, and 0 of 53 between years=1 and years=2.
GRID_EPOCH = date(2015, 1, 1)
# A window returning this many is assumed truncated and is split in half.
# The busiest three-week span measured held 11,526, so a single week at
# 9,000 is already implausible.
SUSPECT_COUNT = 9_000
MIN_SPLIT_DAYS = 1

# Every field the endpoint returns. Listed rather than starred so a NEW
# field shows up as a diff here instead of appearing silently, but nothing
# the API sends is discarded.
FIELDS = ["symbol", "sm_name", "smIndustry", "sm_isin", "desc",
          "attchmntText", "attchmntFile", "attFileSize", "fileSize",
          "an_dt", "exchdisstime", "sort_date", "dt", "difference",
          "seq_id", "hasXbrl", "old_new", "bflag", "csvName", "orgid"]
# Without these the record cannot be placed in time or attributed to a
# stock, so a response lacking one is a schema change and not a quiet day.
REQUIRED = ("symbol", "an_dt", "exchdisstime", "seq_id")

SESSION_OPEN = (9, 15)
SESSION_CLOSE = (15, 30)


def window_tag(first: date) -> str:
    """Identity of one window. The SPAN is part of it deliberately.

    A tag of the start date alone collides when WINDOW_DAYS changes: a
    cached 7-day window and a new 14-day window claim the same name, and
    the days between them are never fetched by anything. Measured on a
    partial cache, changing 7 to 14 silently skipped 10 days.
    """
    return f"{first:%Y-%m-%d}+{WINDOW_DAYS}"


def week_windows(start: date, end: date) -> list:
    """Inclusive (first, last) pairs on a fixed grid covering start..end.

    Snapped back to GRID_EPOCH so the boundaries are a property of the
    calendar rather than of when the script happened to run. The first
    window may therefore begin a few days before `start`, which costs at
    most one extra request and buys a cache that survives to tomorrow.
    """
    offset = ((start - GRID_EPOCH).days // WINDOW_DAYS) * WINDOW_DAYS
    cursor = GRID_EPOCH + timedelta(days=offset)
    windows = []
    while cursor <= end:
        windows.append((cursor, cursor + timedelta(days=WINDOW_DAYS - 1)))
        cursor += timedelta(days=WINDOW_DAYS)
    return windows


def load_state() -> dict:
    """Completed windows from the sidecar, or {} if there is none."""
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  unreadable window state, refetching all: {exc}")
        return {}


def save_state(state: dict) -> None:
    """Write the sidecar atomically, so an interrupt cannot truncate it."""
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=1, sort_keys=True),
                    encoding="utf-8")
    os.replace(temp, STATE)


def fetch_window(session, first: date, last: date) -> "list | None":
    """Announcements disseminated in one window.

    Returns None on failure and [] on a genuinely empty window, because
    the two must not be recorded as the same thing: an empty week is data,
    a failed week is a hole that should be retried.
    """
    url = (f"{API}?index=equities"
           f"&from_date={first.strftime('%d-%m-%Y')}"
           f"&to_date={last.strftime('%d-%m-%Y')}")
    try:
        response = session.get(url, timeout=config.NSE_TIMEOUT_SECONDS,
                               headers={"Referer": REFERER})
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else None


def fetch_span(session, first: date, last: date) -> "list | None":
    """One window, split in half if the count looks capped rather than real.

    Splitting on suspicion rather than trusting the count is the whole
    point: silent truncation is indistinguishable from a quiet period at
    the call site, and every row lost that way is lost invisibly.

    If either half then fails this returns None rather than the parent's
    payload. Returning the parent would hand back the very response that
    tripped the suspicion, and `main` would store it, tag it complete and
    never look again - freezing an invisibly truncated week into the
    dataset, which is precisely what this function exists to prevent.
    """
    rows = fetch_window(session, first, last)
    if rows is None:
        return None
    span_days = (last - first).days + 1
    if len(rows) < SUSPECT_COUNT or span_days <= MIN_SPLIT_DAYS:
        return rows
    middle = first + timedelta(days=span_days // 2 - 1)
    print(f"      {len(rows):,} in {span_days}d looks capped - splitting",
          flush=True)
    time.sleep(PAUSE)
    left = fetch_span(session, first, middle)
    time.sleep(PAUSE)
    right = fetch_span(session, middle + timedelta(days=1), last)
    if left is None or right is None:
        return None
    return left + right


def _needs_fallback(frame: pd.DataFrame) -> pd.Series:
    """Rows whose exchdisstime is absent and must fall back to an_dt.

    astype(str) turns None and NaN into the literal "None"/"nan", which are
    not empty strings, so a bare .ne("") test would pass them through to a
    parse that fails and silently drops the fallback this exists to give.
    """
    blank = frame["exchdisstime"].astype(str).str.strip()
    return frame["exchdisstime"].isna() | blank.isin(
        ("", "None", "nan", "NaT", "-"))


def _public_at(frame: pd.DataFrame) -> pd.Series:
    """When each announcement became public, TIMEZONE-AWARE in IST.

    Prefers the exchange's dissemination time over the company's own filing
    time: the second is when it was submitted, the first is when anyone
    could act on it, and only the first is safe to test against a bar.

    AWARE, NOT NAIVE, and that is the whole point of this note. NSE prints
    these in IST without saying so, and pd.to_datetime hands back a naive
    column. Bars in this project are datetime64[us, Asia/Kolkata]. Comparing
    the two raises TypeError, which is the good outcome; the bad one is a
    caller who "fixes" it by stripping the bar's timezone and silently
    shifts every announcement by five and a half hours - far enough to move
    an after-close filing into the middle of the session and invent a
    signal. Localising here means the comparison a caller would naturally
    write is correct with no further handling.

    India has never observed daylight saving, so localisation has no
    ambiguous or non-existent times to resolve.
    """
    stamp = frame["exchdisstime"].where(~_needs_fallback(frame),
                                        frame["an_dt"])
    parsed = pd.to_datetime(stamp, format="%d-%b-%Y %H:%M:%S",
                            errors="coerce")
    # Bound while still naive, then localise: comparing a naive series
    # against an aware bound is the same TypeError in miniature. The
    # ceiling is NOW rather than midnight tomorrow, because a guard against
    # future-dating that admits the next 24 hours is not a guard.
    floor = pd.Timestamp("2015-01-01")
    ceiling = pd.Timestamp.now()
    bounded = parsed.where((parsed >= floor) & (parsed <= ceiling))
    return bounded.dt.tz_localize(IST)


def empty_frame() -> pd.DataFrame:
    """An empty frame with the full schema and stable dtypes.

    Concatenating a naively-built empty frame downgrades string columns to
    object and the timestamp to a different unit, so a resumed run that hit
    an empty week wrote a different schema than a clean one.
    """
    frame = pd.DataFrame({name: pd.Series(dtype="object") for name in FIELDS})
    frame["public_at"] = pd.Series(dtype=f"datetime64[us, {IST}]")
    frame["window"] = pd.Series(dtype="object")
    return frame


def tidy(rows: list) -> pd.DataFrame:
    """Keep every returned field and add the public-at timestamp.

    Raises on a response missing a REQUIRED field rather than manufacturing
    it as all-None. That distinction matters: a missing `exchdisstime`
    column would send every row down the an_dt fallback at once, and since
    public_at is recomputed on load, one drifted response would rewrite two
    years of timestamps to filing time with no error anywhere.
    """
    if not rows:
        return empty_frame()
    frame = pd.DataFrame(rows)
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise KeyError(
            f"NSE response is missing {missing} - the schema changed. "
            f"Refusing to store it: manufacturing these as empty would "
            f"silently move every timestamp to filing time. Got: "
            f"{sorted(frame.columns)}")
    for column in FIELDS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[FIELDS].copy()
    frame["public_at"] = _public_at(frame)
    return frame


def session_share(frame: pd.DataFrame) -> dict:
    """How the announcements fall against market hours.

    The number that decides whether this is an intraday input at all, so it
    is reported on every run rather than being left for the reader to work
    out from the raw file.
    """
    stamps = frame["public_at"].dropna()
    if stamps.empty:
        return {}
    minutes = stamps.dt.hour * 60 + stamps.dt.minute
    open_at = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    close_at = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    during = int(((minutes >= open_at) & (minutes < close_at)).sum())
    before = int((minutes < open_at).sum())
    return {"total": len(stamps), "during": during, "before": before,
            "after": len(stamps) - during - before}


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per seq_id, keeping the freshest, WITHOUT collapsing nulls.

    drop_duplicates treats NaN as equal to NaN, so a column of nulls
    reduces to a single row - verified, three null-id rows became one. That
    is latent today because seq_id is present on every one of the 359,297
    records, but it shares a trigger with the schema-drift guard in `tidy`:
    a rename upstream would make the column all-null and silently reduce
    the entire dataset to one row on save.

    `keep="last"` so a freshly fetched copy of a record beats the cached
    one. NOT VERIFIED: whether NSE actually reissues an amended
    announcement under the same seq_id. If it issues a new id instead, both
    copies are kept and the `old_new` field is what distinguishes them.
    """
    identified = frame[frame["seq_id"].notna()]
    anonymous = frame[frame["seq_id"].isna()]
    if len(anonymous):
        print(f"  {len(anonymous):,} rows have no seq_id and are kept "
              f"without deduplication")
    return pd.concat(
        [identified.drop_duplicates(subset=["seq_id"], keep="last"),
         anonymous], ignore_index=True)


def save(frame: pd.DataFrame) -> None:
    """Write the parquet atomically.

    37 MB straight onto the live path means an interrupt leaves a truncated
    file, the next run reports an unreadable cache and refetches two years.
    """
    temp = OUT.with_suffix(".tmp")
    frame.to_parquet(temp)
    os.replace(temp, OUT)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        years = float(argv[0]) if argv else DEFAULT_YEARS
    except ValueError:
        print(f"usage: fetch_announcements.py [years]   got {argv[0]!r}")
        return 2
    if years <= 0:
        print(f"years must be positive, got {years}")
        return 2

    today = date.today()
    start = today - timedelta(days=int(years * 365))
    have, state = None, load_state()
    if OUT.exists():
        try:
            have = pd.read_parquet(OUT)
            # public_at is DERIVED, never authoritative: recompute it from
            # the stored raw fields every load, so a change to the parsing
            # rule reaches two years of cached rows without refetching any
            # of them.
            have["public_at"] = _public_at(have)
            fell_back = int(_needs_fallback(have).sum())
            if fell_back:
                print(f"  {fell_back:,} cached rows have no exchdisstime "
                      f"and use the company's filing time instead")
        except Exception as exc:
            # Reset BOTH, or the recovery path lies: `have` and `state`
            # were already assigned, so a failure here would keep the
            # stale frame and keep suppressing every cached window.
            have, state = None, {}
            print(f"  unreadable cache, starting over: {exc}")

    windows = [w for w in week_windows(start, today)
               if not state.get(window_tag(w[0]), {}).get("complete")]
    print(f"span {start} .. {today}")
    print(f"windows complete {sum(1 for v in state.values() if v.get('complete'))}"
          f"; fetching {len(windows)} "
          f"(~{len(windows) * (PAUSE + 1.5) / 60:.0f} min)\n")

    session = instruments.open_session()
    fresh, failed, refetched = [], [], set()
    for first, last in windows:
        tag = window_tag(first)
        try:
            rows = fetch_span(session, first, last)
        except Exception as exc:
            rows, _ = None, print(f"  {tag}  ERROR {exc}")
        if rows is None:
            failed.append(tag)
            print(f"  {tag}  FAILED - will refetch on the next run",
                  flush=True)
            time.sleep(PAUSE)
            continue
        try:
            frame = tidy(rows)
        except KeyError as exc:
            failed.append(tag)
            print(f"  {tag}  SCHEMA CHANGE, not stored: {exc}", flush=True)
            time.sleep(PAUSE)
            continue
        frame["window"] = tag
        if len(frame):
            fresh.append(frame)
        refetched.add(tag)
        # A window still running is NOT complete. Caching the trailing
        # partial week as done meant a second run on the same day skipped
        # it and lost everything filed later that day.
        complete = last < today
        state[tag] = {"rows": len(frame), "complete": complete}
        print(f"  {tag}  {len(frame):>6,}{'' if complete else '  (partial)'}",
              flush=True)
        time.sleep(PAUSE)

    if have is None and not fresh:
        print("\nnothing fetched and no cache - nothing to save")
        return 1
    chunks = []
    if have is not None:
        # Drop the rows belonging to any window just refetched, so the new
        # copy replaces rather than races the old one on dedup order.
        chunks.append(have[~have["window"].isin(refetched)]
                      if "window" in have.columns else have)
    chunks.extend(fresh)
    data = deduplicate(pd.concat(chunks, ignore_index=True))
    save(data)
    save_state(state)

    print(f"\nrows {len(data):,} | symbols {data['symbol'].nunique():,} "
          f"| categories {data['desc'].nunique():,}")
    stamps = data["public_at"].dropna()
    if len(stamps):
        print(f"coverage {stamps.min():%Y-%m-%d} .. {stamps.max():%Y-%m-%d}")
        unparsed = len(data) - len(stamps)
        if unparsed:
            print(f"  {unparsed:,} rows have no usable timestamp and are "
                  f"UNUSABLE for any test that needs one")
    split = session_share(data)
    if split:
        print(f"\nagainst market hours (the number that decides whether "
              f"this is an intraday input):")
        for label, key in (("before open", "before"),
                           ("DURING session", "during"),
                           ("after close", "after")):
            count = split[key]
            print(f"  {label:<16} {count:>8,}  "
                  f"{count / split['total'] * 100:5.1f}%")
    print(f"\nsaved -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    if failed:
        print(f"\n{len(failed)} window(s) failed and were NOT stored, so "
              f"they are refetched rather than silently missing: "
              f"{', '.join(failed[:6])}")
        # A run that fetched nothing must not look like a clean run.
        # instruments.open_session returns a cookieless session on a failed
        # bootstrap rather than raising, so every window failing is exactly
        # what a dead session looks like from here.
        if not fresh:
            print("  nothing new was fetched at all - check the NSE session")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
