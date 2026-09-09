"""Insider and promoter share transactions, from NSE's PIT disclosures.

WHY THIS FIELD. SEBI's Prohibition of Insider Trading rules force directors,
promoters and designated employees to disclose their own dealings in the
company's shares. Unlike a headline, this is not an opinion about the
company - it is the people running it committing their own money. Insider
buying has genuine documented predictive power, and it is free.

A HARD LIMITATION, measured rather than assumed. NSE's bulk PIT endpoint
returns rows for 2020 through early April 2026 and then nothing:

    Feb 2026  1,279 rows      May 2026      3 rows
    Mar 2026  2,057 rows      Jun 2026      0 rows
    Apr 2026    392 rows      Jul-Sep 2026  0 rows

Per-symbol queries agree - the newest filing for INFY, out of 129 in 2026,
is 06-Apr-2026. Meanwhile the general corporate-announcements endpoint
returns 5,494 rows for September, so NSE's filings API is alive and it is
this dataset specifically that has stalled.

The consequence is not subtle: this feature is usable for BACKTESTING and
useless for LIVE PREDICTION, because it would be empty for exactly the
dates a forecast is made. It is therefore deliberately excluded from
forward tracking. If NSE backfills, re-run this and the gap closes on its
own; the fetcher is written to be resumable so that costs one command.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
import instruments

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "forecast_cache"
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "insider.parquet"

START = date(2020, 1, 1)
API = "https://www.nseindia.com/api/corporates-pit"
REFERER = ("https://www.nseindia.com/companies-listing/"
           "corporate-filings-insider-trading")
PAUSE = 0.9

# The disclosure carries far more than this; these are the fields that
# describe WHO dealt, WHICH WAY, HOW MUCH, and WHEN it became public.
FIELDS = ["symbol", "company", "acqName", "personCategory", "acqMode",
          "secType", "secAcq", "secVal", "tdpTransactionType",
          "befAcqSharesPer", "afterAcqSharesPer", "buyQuantity", "buyValue",
          "sellquantity", "sellValue", "acqfromDt", "acqtoDt", "intimDt",
          "date"]


def month_windows(start: date, end: date) -> list:
    """Inclusive (first, last) pairs, one per calendar month.

    Monthly rather than yearly because the endpoint silently truncates long
    windows, and a truncated window looks exactly like a quiet month.
    """
    windows = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            following = date(cursor.year + 1, 1, 1)
        else:
            following = date(cursor.year, cursor.month + 1, 1)
        windows.append((cursor, min(following - timedelta(days=1), end)))
        cursor = following
    return windows


def fetch_window(session, first: date, last: date) -> list:
    """Disclosures intimated in one window, or [] on any failure."""
    url = (f"{API}?index=equities"
           f"&from_date={first.strftime('%d-%m-%Y')}"
           f"&to_date={last.strftime('%d-%m-%Y')}")
    try:
        response = session.get(url, timeout=config.NSE_TIMEOUT_SECONDS,
                               headers={"Referer": REFERER})
    except Exception:
        return []
    if response.status_code != 200:
        return []
    try:
        payload = response.json()
    except Exception:
        return []
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


def tidy(rows: list) -> pd.DataFrame:
    """Keep the descriptive fields, coerce the numeric ones."""
    frame = pd.DataFrame(rows)
    for column in FIELDS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[FIELDS].copy()
    for column in ("secAcq", "secVal", "befAcqSharesPer",
                   "afterAcqSharesPer", "buyQuantity", "buyValue",
                   "sellquantity", "sellValue"):
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce")
    # intimDt is when the market learned, which is the only timestamp a
    # point-in-time feature may use. acqfromDt is when the insider dealt,
    # which is earlier and was not public then.
    parsed = pd.to_datetime(frame["intimDt"], errors="coerce",
                            dayfirst=True)
    # Bound the parse. Measured: 33 of 136,757 filings carry a date outside
    # any plausible range, including one in 2036. Most are genuine late
    # disclosures from 2016-2018, but a future-dated one is worse than
    # useless - it would mark a disclosure as public BEFORE it happened,
    # which is a lookahead leak of exactly the kind this project keeps
    # producing. Anything outside the window becomes NaT and is skipped.
    floor = pd.Timestamp("2010-01-01")
    ceiling = pd.Timestamp(date.today())
    frame["known_on"] = parsed.where(
        (parsed >= floor) & (parsed <= ceiling)).dt.date
    return frame


def main() -> int:
    end = date.today()
    have = None
    seen = set()
    if OUT.exists():
        try:
            have = pd.read_parquet(OUT)
            seen = set(have["window"].unique())
        except Exception as exc:
            print(f"  unreadable cache, starting over: {exc}")
    windows = [w for w in month_windows(START, end)
               if f"{w[0]:%Y-%m}" not in seen]
    print(f"span {START} .. {end}")
    print(f"months cached {len(seen)}; fetching {len(windows)} "
          f"(~{len(windows) * PAUSE:.0f}s)")

    session = instruments.open_session()
    chunks = [have] if have is not None else []
    empty_months = []
    for first, last in windows:
        rows = fetch_window(session, first, last)
        tag = f"{first:%Y-%m}"
        if rows:
            frame = tidy(rows)
            frame["window"] = tag
            chunks.append(frame)
            print(f"  {tag}  {len(rows):>5} disclosures", flush=True)
        else:
            empty_months.append(tag)
            print(f"  {tag}  {'0':>5} (none returned)", flush=True)
        time.sleep(PAUSE)

    if not chunks:
        print("nothing fetched")
        return 1
    data = pd.concat(chunks, ignore_index=True).drop_duplicates()
    data.to_parquet(OUT)
    print(f"\nrows {len(data):,} | symbols {data['symbol'].nunique():,}")
    known = data["known_on"].dropna()
    if len(known):
        print(f"coverage {known.min()} .. {known.max()}")
        gap = (end - known.max()).days
        print(f"NEWEST DISCLOSURE IS {gap} DAYS OLD")
        if gap > 45:
            print("  -> too stale to feed a live forecast. Backtest only.")
    if empty_months:
        print(f"months returning nothing: {', '.join(empty_months[-8:])}")
    print(f"saved -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
