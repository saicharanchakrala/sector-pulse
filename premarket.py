"""A pre-open watchlist, built entirely from bars that settled yesterday.

WHY IT EXISTS. The intraday scan can say nothing before about 09:30.
choose_direction needs an opening-range break, and an opening range that
has not closed yet has no break to compute - so the tab is honestly empty
from 08:00 until fifteen minutes after the bell. This fills that window
with the one thing that IS computable overnight.

WHAT IT RANKS ON, and why that and nothing else. Every ordering here is by
EXPECTED MOVEMENT, never by expected direction. That is not caution, it is
what this project measured: intraday direction scored 0.5121 AUC against a
0.4990 permutation null, material filings hit 29.6% against a 30.6%
control - worse than nothing - while the volatility effect replicated out
of sample at 2.03x in training and 1.22x in holdout. Volatility clusters;
direction, on this data, does not persist. So the list answers "which
names are likely to move enough to be worth watching, and are liquid
enough to get out of" and refuses the question it has no evidence for.

A NAME NEAR THE TOP IS NOT A BUY. It is a name whose ordinary daily range
is wide relative to its price. That cuts both ways by construction, and
anything here that read as a recommendation would be asserting the thing
the measurements rejected.

THE COST COLUMN IS THE USEFUL ONE. cost_in_atr is the round-trip
breakeven expressed as a fraction of a typical day's range. At 0.12% costs
and a 1.5% ATR the round trip eats 8% of an average day; at a 0.4% ATR it
eats 30%, and no intraday edge survives that. It is the cheapest way to
see which names are untradeable intraday before spending a session finding
out, and it comes straight from Varsity's rule that a trade whose
volatility-appropriate levels break your risk-reward appetite should be
dropped rather than sized down.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
import trade_costs

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Varsity's 14-period ATR, over DAILY bars.
#
# THE UNITS DIFFER FROM THE SCANNER'S, and the shared constant hides it.
# config.SCAN_ATR_BARS counts BARS: setups.py passes intraday bars to
# indicators.atr, so at a 3-minute interval its ATR spans 42 minutes.
# Here the same 14 is applied to daily bars, so it spans 14 sessions -
# about 130 times longer. An earlier comment claimed sharing the name
# meant the two "cannot describe different volatilities"; they always
# did, and the shared name disguised it rather than aligning it.
#
# Both are correct for their own purpose - an intraday stop should be
# sized to intraday movement, and a pre-open expected range to a daily
# one - but they are NOT the same measurement and must not be compared.
ATR_SESSIONS = config.SCAN_ATR_BARS          # 14, here meaning sessions
TURNOVER_SESSIONS = 20
VOLUME_BASELINE_SESSIONS = 10                # Varsity Ch 12.1's comparison
RANGE_SESSIONS = 20
YEAR_SESSIONS = 252

# Enough history for the yearly range plus slack; anything shorter and
# pos_52w silently describes a shorter window than its name claims.
MIN_SESSIONS = YEAR_SESSIONS + 10

SESSION_CLOSE = time(*config.SCAN_SESSION_CLOSE)


def store_written_at(path=None) -> "datetime | None":
    """When the consolidated daily store was last folded, or None."""
    import bar_store

    target = path or bar_store.store_path("day")
    try:
        return datetime.fromtimestamp(target.stat().st_mtime, IST)
    except (OSError, AttributeError):
        return None


def settled_through(now: "datetime | None" = None,
                    written_at: "datetime | None" = None):
    """The last date whose session is COMPLETE in the store.

    A PRE-OPEN LIST MAY ONLY SEE WHOLE SESSIONS. Two things can each make
    today's bar a partial one, and both have to be ruled out:

      * the session has not closed yet, and
      * the store was FOLDED before it closed.

    The second is the one that caught me. Measured 2026-09-15: the clock
    said 20:35, comfortably past the 15:30 close, so today looked settled -
    but forecast_cache/bars_day.parquet was written at 14:42, so the row
    it holds for today is a mid-session snapshot. RELIANCE carried 3.2M of
    volume in it against a 13M ten-session mean. Taken at face value that
    divides every relative volume by about four, reports an intraday price
    as a close, and truncates the most recent true range.

    Holidays need no special case: a day with no session has no bar.
    """
    now = now or datetime.now(IST)
    close_today = datetime.combine(now.date(), SESSION_CLOSE, tzinfo=IST)
    if now < close_today:
        return now.date() - timedelta(days=1)
    stamp = written_at if written_at is not None else store_written_at()
    if stamp is None or stamp < close_today:
        # The session is over but the store predates its close, so the row
        # it holds for today is partial. Tonight's fold fixes this.
        return now.date() - timedelta(days=1)
    return now.date()


def settled(frame: pd.DataFrame, through) -> pd.DataFrame:
    """`frame` truncated to sessions on or before `through`."""
    if frame is None or frame.empty or through is None:
        return frame
    stamps = pd.DatetimeIndex(frame.index)
    return frame[stamps.date <= through]


COLUMNS = [
    "symbol", "close", "atr", "atr_pct", "cost_in_atr", "expected_low",
    "expected_high", "turnover_20d", "spike_ratio", "rvol_10d", "pos_52w",
    "range_20d_pct", "change_pct",
]


def mean_true_range(frame: pd.DataFrame, sessions: int = ATR_SESSIONS) -> float:
    """Mean absolute true range over `sessions`, in rupees.

    A MEAN, not a standard deviation, and not Wilder's smoothed average.
    This project measured the distinction and recorded it: an ATR of this
    kind runs about 1.37 to 1.50 true sigmas, so treating it as one sigma
    would size every stop too tight. Kept the same here so a level quoted
    on the watchlist means what a level quoted by the scanner means.
    """
    if len(frame) < sessions + 1:
        return float("nan")
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    prior_close = frame["Close"].to_numpy(dtype=float)[:-1]
    ranges = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prior_close),
        np.abs(low[1:] - prior_close),
    ])
    window = ranges[-sessions:]
    return float(np.mean(window)) if len(window) else float("nan")


def describe(symbol: str, frame: pd.DataFrame,
             through=None) -> "dict | None":
    """One row of the watchlist, or None when the history is too short.

    `through` drops any session at or after the one being planned for, so
    a partially-folded current day cannot reach the numbers.
    """
    if frame is None or frame.empty:
        return None
    if not {"High", "Low", "Close", "Volume"} <= set(frame.columns):
        return None

    frame = settled(frame.sort_index(), through)
    if frame is None or len(frame) < MIN_SESSIONS:
        return None
    close = float(frame["Close"].iloc[-1])
    if not np.isfinite(close) or close <= 0:
        return None

    atr = mean_true_range(frame)
    if not np.isfinite(atr) or atr <= 0:
        return None
    atr_pct = atr / close * 100.0

    tail = frame.tail(TURNOVER_SESSIONS)
    daily_turnover = tail["Close"] * tail["Volume"]
    # MEDIAN, not mean, and the difference is not cosmetic. WEL traded
    # 129.5M shares on 2026-08-27 and 402K on 09-11; its 20-session MEAN
    # turnover is 110 crore and its median about 9. The gate asks "can I
    # get out of this tomorrow", and one enormous day three weeks ago
    # does not answer that question. live_feed.turnover_values still uses
    # the mean because it is choosing what to STREAM, where a spike is a
    # reason to include rather than a reason to doubt.
    turnover = float(daily_turnover.median())
    turnover_spike = (float(daily_turnover.mean()) / turnover
                      if turnover > 0 else float("nan"))

    baseline = frame["Volume"].tail(VOLUME_BASELINE_SESSIONS + 1)[:-1].mean()
    rvol = (float(frame["Volume"].iloc[-1]) / float(baseline)
            if baseline and np.isfinite(baseline) and baseline > 0
            else float("nan"))

    year = frame.tail(YEAR_SESSIONS)
    high52, low52 = float(year["High"].max()), float(year["Low"].min())
    span = high52 - low52
    pos_52w = (close - low52) / span * 100.0 if span > 0 else float("nan")

    recent = frame.tail(RANGE_SESSIONS)
    high20, low20 = float(recent["High"].max()), float(recent["Low"].min())
    midpoint = (high20 + low20) / 2.0
    # The Swing Trader's Bible compression measure: a 20-session range
    # narrow against its own midpoint. Reported, NOT ranked on - that a
    # coiled range predicts expansion is a claim this project has not
    # tested, and ranking on it would smuggle it in as though it had.
    range_pct = (high20 - low20) / midpoint * 100.0 if midpoint > 0 else float("nan")

    previous = float(frame["Close"].iloc[-2])
    change = (close / previous - 1.0) * 100.0 if previous > 0 else float("nan")

    # Costs as a share of a typical day's range. The breakeven is computed
    # on a real position rather than assumed, so brokerage caps apply.
    breakeven = trade_costs.equity_breakeven_pct(
        close, max(1, int(config.SCAN_CAPITAL // close)))
    cost_in_atr = breakeven / atr_pct if atr_pct > 0 else float("nan")

    return {
        "symbol": symbol,
        "close": round(close, 2),
        "atr": round(atr, 2),
        "atr_pct": round(atr_pct, 2),
        "cost_in_atr": round(cost_in_atr, 4),
        # Varsity's rule, made concrete: tomorrow's likely band. A stop
        # inside this is inside the noise and gets hit by ordinary trade.
        "expected_low": round(close - atr, 2),
        "expected_high": round(close + atr, 2),
        "turnover_20d": round(turnover, 0),
        "spike_ratio": round(turnover_spike, 1),
        "rvol_10d": round(rvol, 2),
        "pos_52w": round(pos_52w, 1),
        "range_20d_pct": round(range_pct, 2),
        "change_pct": round(change, 2),
    }


def build(limit: int = 40, min_turnover: "float | None" = None,
          min_price: "float | None" = None,
          frames: "dict | None" = None,
          now: "datetime | None" = None) -> pd.DataFrame:
    """The watchlist, ranked by expected movement. Never by direction.

    Gates on the same turnover and price floors the intraday scanner uses,
    so a name that reaches this list can actually be traded by the thing
    that would trade it. Ranking a name the scanner would reject on depth
    would be offering something that cannot be acted on.
    """
    floor_turnover = (config.SCAN_MIN_TURNOVER if min_turnover is None
                      else min_turnover)
    floor_price = config.SCAN_MIN_PRICE if min_price is None else min_price

    if frames is None:
        import bar_store

        try:
            frames = bar_store.load("day")
        except Exception as exc:
            logger.warning("No daily store, so no watchlist: %s", exc)
            return pd.DataFrame(columns=COLUMNS)

    through = settled_through(now)
    rows = []
    for symbol, frame in (frames or {}).items():
        described = describe(symbol, frame, through=through)
        if described is None:
            continue
        if described["close"] < floor_price:
            continue
        if described["turnover_20d"] < floor_turnover:
            continue
        rows.append(described)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    table = pd.DataFrame(rows)[COLUMNS]
    # THE ONE ORDERING, and the reason for it is in the module docstring:
    # volatility clusters and direction does not, so expected movement is
    # the only thing here with evidence behind it.
    table = table.sort_values("atr_pct", ascending=False, kind="stable")
    return table.head(limit).reset_index(drop=True)


def with_filings(table: pd.DataFrame, now: "datetime | None" = None,
                 hours: int = 18) -> pd.DataFrame:
    """Attach overnight filings as CONTEXT. Not a direction, not a reason.

    The default window reaches back past yesterday's close, which is where
    an announcement that has not yet traded would sit. Measured: a material
    filing preceded a 29.6% hit rate against a 30.6% control, so the only
    honest use of this column is to say the name may be lively.
    """
    if table.empty:
        return table
    import filings

    when = now or datetime.now(IST)
    notes = []
    for symbol in table["symbol"]:
        try:
            found = filings.recent_for(symbol, now=when, minutes=hours * 60)
        except Exception:
            found = []
        if not found:
            notes.append("")
            continue
        lead = str(found[0].get("desc", "")).strip()
        extra = f" +{len(found) - 1} more" if len(found) > 1 else ""
        notes.append(f"{lead}{extra}")
    out = table.copy()
    out["filing_18h"] = notes
    return out


def main(argv=None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    limit = 40
    for flag in argv:
        if flag.startswith("--limit="):
            limit = int(flag.split("=", 1)[1])

    through = settled_through()
    folded = store_written_at()
    table = build(limit=limit)
    print(f"Every figure below describes sessions up to and including "
          f"{through}.")
    if folded is not None:
        print(f"Daily store last folded {folded:%Y-%m-%d %H:%M}.")
    print()
    if table.empty:
        print("No watchlist: the daily store produced no qualifying names.")
        return 1
    table = with_filings(table)
    print(f"Pre-open watchlist, {len(table)} names, ranked by expected "
          f"movement (ATR as a share of price).")
    print("NOT a direction call and not a ranking of expected return - "
          "volatility is what replicated,")
    print("direction is not. A wide range cuts both ways by construction.\n")
    print(table.to_string(index=False))
    print("\ncost_in_atr is the round-trip breakeven as a share of a "
          "typical day's range.")
    print("Above ~0.15 the costs eat most of what an ordinary session "
          "offers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# The built list, so the page never pays to build it. It depends only on
# SETTLED daily bars, so it changes once a night and not once a render.
PUBLISHED = config.PROJECT_ROOT / "premarket.parquet"


def publish(limit: int = 200, path=None) -> int:
    """Build the watchlist once and write it, for the UI to just read.

    WHY. Measured 2026-09-16: build() takes 19.4 seconds, of which 19.9 is
    bar_store.load("day") pulling 3.98M rows across 2,520 symbols and 13.2
    is the per-symbol describe loop. Streamlit re-executes its whole
    script on every interaction, so putting that on a page makes the page
    cost twenty seconds whenever the cache is cold - which is every
    restart, and every first load of a trading morning.

    Nothing in it changes intraday. The inputs are sessions that have
    already settled, so building it once a night and reading ~50 KB is not
    an optimisation, it is the correct shape.

    A generous default limit: cheap to store, and it lets the UI's slider
    move without rebuilding anything.
    """
    table = build(limit=limit)
    if table.empty:
        logger.warning("Nothing qualified, so nothing published")
        return 0
    target = path or PUBLISHED
    table = with_filings(table)
    try:
        table.to_parquet(target, index=False)
    except Exception as exc:
        logger.warning("Could not write %s: %s", target, exc)
        return 0
    return len(table)


def load_published(path=None) -> "tuple | None":
    """(table, built_at) from the published file, or None when absent.

    The caller needs the build time as well as the rows: a watchlist from
    three days ago is not wrong so much as describing a different week,
    and only the timestamp distinguishes them.
    """
    target = path or PUBLISHED
    try:
        if not target.exists():
            return None
        built = datetime.fromtimestamp(target.stat().st_mtime, IST)
        return pd.read_parquet(target), built
    except Exception as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return None
