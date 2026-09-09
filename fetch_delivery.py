"""Delivery percentage per stock per day, from NSE's full bhavcopy.

WHY THIS FIELD. Traded volume counts every share that changed hands,
including the same share bought and sold intraday by a day trader.
Delivery quantity counts only shares that were actually settled into
someone's demat account. The ratio is therefore a conviction proxy: a 3%
move on 20% delivery is churn, the same move on 70% delivery is somebody
taking ownership. It is per-stock and daily, which is what makes it usable
for choosing between stocks.

WHAT IT COSTS. NSE only publishes this file format from late 2019 -
measured: 2019-06-03 returns 404, 2019-12-02 returns 200. Price history in
bar_cache reaches back to 2016. So any model using delivery is restricted
to about 6.8 years instead of 10.7, which cuts the number of independent
one-year periods from roughly ten to roughly seven. The long horizon was
already the weakest and this makes it weaker. That trade is the honest
price of the feature and should be weighed rather than assumed away.

NOT INCLUDED: FII/DII. NSE publishes it as a MARKET-WIDE daily aggregate,
one number for the whole market rather than per stock. In a model that
ranks stocks within a date it is constant across the cross-section and
therefore contributes exactly nothing to the ranking - mathematically the
same as adding today's date as a feature. It could only inform market
timing, which is a different question. Per-stock foreign holdings exist
only quarterly, in shareholding filings.
"""
from __future__ import annotations

import io
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
OUT = OUT_DIR / "delivery.parquet"

# Measured boundary: the format does not exist before roughly this date.
START = date(2019, 6, 1)
BASE = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_"
PAUSE = 0.7            # NSE documents no rate, so pace conservatively
KEEP = ["SYMBOL", "DELIV_QTY", "DELIV_PER", "TTL_TRD_QNTY",
        "TURNOVER_LACS", "NO_OF_TRADES", "CLOSE_PRICE"]


def parse_day(text: str, when: date) -> "pd.DataFrame | None":
    """One day's cash-market rows, equities only.

    The file's headers carry leading spaces and the series column includes
    government securities, SME and bond rows, so both are cleaned here
    rather than left for every caller to rediscover.
    """
    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception:
        return None
    frame.columns = [c.strip() for c in frame.columns]
    if "SERIES" not in frame.columns or "DELIV_PER" not in frame.columns:
        return None
    frame["SERIES"] = frame["SERIES"].astype(str).str.strip()
    frame = frame[frame["SERIES"] == "EQ"]
    missing = [c for c in KEEP if c not in frame.columns]
    if missing or frame.empty:
        return None
    frame = frame[KEEP].copy()
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.strip()
    for column in KEEP[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # DELIV_PER is blank for symbols with no delivery that day; those rows
    # are real (the stock traded) so they are kept with NaN rather than
    # dropped, which would silently bias the sample toward active names.
    frame["date"] = when
    return frame


def existing() -> tuple:
    """Rows already fetched, and the dates they cover."""
    if not OUT.exists():
        return None, set()
    try:
        frame = pd.read_parquet(OUT)
    except Exception as exc:
        print(f"  unreadable {OUT.name}, starting over: {exc}")
        return None, set()
    return frame, set(frame["date"].unique())


def main() -> int:
    end = date.today()
    have, done = existing()
    session = instruments.open_session()
    headers = {"Referer": "https://www.nseindia.com/all-reports"}

    wanted = []
    cursor = START
    while cursor <= end:
        if cursor.weekday() < 5 and cursor not in done:
            wanted.append(cursor)
        cursor += timedelta(days=1)
    print(f"span {START} .. {end}")
    print(f"already have {len(done):,} dates; fetching {len(wanted):,} "
          f"(~{len(wanted) * PAUSE / 60:.0f} min at {PAUSE}s pacing)")

    chunks = [have] if have is not None else []
    ok = holidays = errors = 0
    for index, when in enumerate(wanted, start=1):
        url = f"{BASE}{when.strftime('%d%m%Y')}.csv"
        try:
            response = session.get(url, timeout=config.NSE_TIMEOUT_SECONDS,
                                   headers=headers)
        except Exception:
            errors += 1
            time.sleep(PAUSE)
            continue
        if response.status_code != 200 or len(response.content) < 20_000:
            # A holiday and a withdrawn file look identical from here, so
            # both are counted together rather than guessed apart.
            holidays += 1
        else:
            frame = parse_day(response.text, when)
            if frame is None:
                errors += 1
            else:
                chunks.append(frame)
                ok += 1
        time.sleep(PAUSE)
        if index % 100 == 0:
            print(f"  {index}/{len(wanted)}  ok={ok} no-file={holidays} "
                  f"err={errors}", flush=True)
            if chunks:
                pd.concat(chunks, ignore_index=True).to_parquet(OUT)

    if not chunks:
        print("nothing fetched")
        return 1
    data = pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset=["SYMBOL", "date"], keep="last")
    data.to_parquet(OUT)
    print(f"\nfetched {ok} new dates, {holidays} with no file, {errors} errors")
    print(f"rows {len(data):,} | dates {data['date'].nunique():,} | "
          f"symbols {data['SYMBOL'].nunique():,}")
    print(f"delivery % : median {data['DELIV_PER'].median():.1f}, "
          f"p10 {data['DELIV_PER'].quantile(0.1):.1f}, "
          f"p90 {data['DELIV_PER'].quantile(0.9):.1f}")
    print(f"saved -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
