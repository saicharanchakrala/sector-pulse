"""What actually happened to the setups the scanner proposed.

WHY THIS EXISTS. scan_log.csv records 22 columns and every one of them is a
PREDICTION - entry, stop, target, score, the gates that passed. Not one
records the result. So on 2026-09-15 it was possible to see what the
scanner thought on 8 September and impossible to find out whether it was
right. Its own docstring says the log "is meant to become a track record";
without this module it is a track of intentions.

That gap blocks everything downstream. Calibration, "which gates actually
discriminate", whether the live scanner agrees with the offline study, and
any self-assessment loop at all - none of it is computable from
predictions alone.

THE TIE-BREAK IS BORROWED, NOT REINVENTED. A bar whose range spans both
the stop and the target carries no intra-bar ordering, so something must
decide which came first. forecast_intraday_data already decided: it counts
that bar as a stop, and `resolve` there resolves ties to the stop. This
module uses the same rule, and test_outcomes.py asserts the two agree on
random inputs rather than trusting that they do.

That matters more than it sounds. Kevin Davey's warning about back-tests
whose stop and target can both fill inside one bar is that the engine must
assume something about intra-bar price travel, and the results come out
"overly optimistic". The error is not random either: wider bars make the
ambiguous case more frequent, so the optimism grows with volatility -
which is precisely the thing this project measured its signals to predict.
Resolving pessimistically is what keeps a volatility effect from arriving
downstream disguised as a directional edge.

WHAT IT ASSUMES, stated so it can be argued with. The position is taken at
the logged entry price at the logged time, as a market order would be.
The scanner's entry is a level it believed tradeable at that moment, so
this measures the setup rather than the fill; slippage is already carried
in the cost model. A position still open at the session close is marked
out there rather than discarded, because dropping unresolved rows would
quietly keep only the ones that moved.
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent
STORE = ROOT / "outcomes.csv"

# From config, so this and market_source cannot drift on when a session
# ends. Two copies of 15:30 is two places to be wrong.
SESSION_CLOSE = time(*config.SCAN_SESSION_CLOSE)

# A run at 15:30:01 would resolve against a session whose final bar Kite
# may not have published yet, and write a permanent CLOSE against a
# truncated path. Nightly runs are hours later; this protects everyone else.
SETTLE_MINUTES = 15

TARGET = "TARGET"
STOP = "STOP"
CLOSE = "CLOSE"          # neither level reached; marked out at the close
NO_BARS = "NO_BARS"      # nothing to resolve against, not an outcome

FIELDS = [
    # `replayed` is NOT optional. scan_intraday records it because "a
    # replayed row stamped with a past time is otherwise indistinguishable
    # from a live scan made at that time" - and dropping it here produced
    # exactly that: a first track record that was 100% replayed and
    # presented as real. `score` is carried because calibration (does 0.68
    # beat 0.52?) is the main thing this file exists to make answerable,
    # and breakeven/stop let gross R be netted without a rejoin.
    "row_id", "run_date", "run_time", "symbol", "direction",
    "replayed", "control", "score", "entry", "stop", "target",
    "stop_pct", "breakeven_pct",
    "outcome", "exit_time", "exit_price", "r_multiple",
    "bars_held", "mfe_r", "mae_r", "resolved_at",
]


@dataclass(frozen=True)
class Outcome:
    """One resolved setup. r_multiple is GROSS - costs are applied later."""

    row_id: str
    outcome: str
    exit_time: str
    exit_price: float
    r_multiple: float
    bars_held: int
    mfe_r: float
    mae_r: float

    @property
    def resolved(self) -> bool:
        return self.outcome in {TARGET, STOP, CLOSE}


def row_id(run_date: str, run_time: str, symbol: str, direction: str) -> str:
    """A stable key joining a scan_log row to its outcome.

    Deliberately derived from the row's own fields rather than a counter or
    a uuid, so a log that is re-read, rotated or partially rebuilt still
    joins to the outcomes already computed for it.
    """
    return f"{run_date}T{run_time}|{symbol}|{direction}"


def resolve_one(entry: float, stop: float, target: float, long: bool,
                bars, row_key: str = "") -> Outcome:
    """Walk `bars` forward from entry and report what happened first.

    `bars` is a DataFrame indexed by timestamp with High and Low columns,
    covering the session FROM the entry moment onward. The caller is
    responsible for slicing it - passing bars from before the entry would
    resolve the trade against prices it could not have traded at.
    """
    if bars is None or len(bars) == 0:
        return Outcome(row_key, NO_BARS, "", 0.0, 0.0, 0, 0.0, 0.0)
    # SIGNS CHECKED, NOT REPAIRED. abs() on both distances would silently
    # mirror a SHORT whose target was logged above its entry and report a
    # clean TARGET for a trade that was never specified. A level on the
    # wrong side means the row is malformed, not that it needs fixing.
    if long:
        ok = stop < entry < target
    else:
        ok = target < entry < stop
    if not ok:
        logger.warning("%s has levels on the wrong side of entry "
                       "(entry %.2f stop %.2f target %.2f, long=%s)",
                       row_key, entry, stop, target, long)
        return Outcome(row_key, NO_BARS, "", 0.0, 0.0, 0, 0.0, 0.0)
    stop_distance = abs(entry - stop)
    target_distance = abs(target - entry)
    if stop_distance <= 0 or target_distance <= 0:
        # A zero-width stop cannot be resolved and must not read as a win.
        logger.warning("%s has a zero stop or target distance", row_key)
        return Outcome(row_key, NO_BARS, "", 0.0, 0.0, 0, 0.0, 0.0)

    highs = bars["High"].to_numpy(dtype=float)
    lows = bars["Low"].to_numpy(dtype=float)
    stamps = list(bars.index)

    # Excursions in R, tracked as the walk proceeds so they describe the
    # path actually taken rather than the whole session. MAE is what tells
    # you whether a stop was too tight; MFE whether a target was too far.
    mfe = 0.0
    mae = 0.0
    # strict=True: highs and lows come from the same frame and a
    # length mismatch would silently truncate the walk, resolving the
    # trade against a shorter path than it actually took.
    for position, (high, low) in enumerate(zip(highs, lows, strict=True)):
        favourable = (high - entry) if long else (entry - low)
        adverse = (entry - low) if long else (high - entry)
        mfe = max(mfe, favourable / stop_distance)
        mae = max(mae, adverse / stop_distance)

        if not (math.isfinite(high) and math.isfinite(low)):
            # Every comparison below is False on NaN, so the bar would be
            # skipped as "no touch" - understating hits on a gappy feed.
            # forecast_intraday_data's searchsorted orders NaN as larger
            # than everything and reports a spurious hit instead, so the
            # two resolvers disagree here. Refusing the row is the only
            # answer that is not quietly wrong in one direction or other.
            logger.warning("%s has a non-finite bar; refusing to resolve",
                           row_key)
            return Outcome(row_key, NO_BARS, "", 0.0, 0.0, 0, 0.0, 0.0)
        hit_target = (high >= entry + target_distance if long
                      else low <= entry - target_distance)
        hit_stop = (low <= entry - stop_distance if long
                    else high >= entry + stop_distance)

        # THE TIE GOES TO THE STOP. Both inside one bar means the bar tells
        # us nothing about ordering, and assuming the good one happened
        # first is how a back-test flatters itself. Same rule as
        # forecast_intraday_data.resolve, asserted equal in the tests.
        if hit_stop:
            price = entry - stop_distance if long else entry + stop_distance
            return Outcome(row_key, STOP, str(stamps[position]), float(price),
                           -1.0, position + 1, mfe, mae)
        if hit_target:
            price = entry + target_distance if long else entry - target_distance
            return Outcome(row_key, TARGET, str(stamps[position]), float(price),
                           target_distance / stop_distance,
                           position + 1, mfe, mae)

    # Neither level reached. Marked out at the last bar's close, which is
    # a real number rather than a discard - dropping these would keep only
    # the setups that moved and report a track record of those.
    last_close = float(bars["Close"].iloc[-1])
    if not math.isfinite(last_close):
        # round(nan, 4) is nan, which reaches the CSV as the string "nan"
        # and is then skipped by mean() - so the printed sample size and
        # the printed mean would describe different sets of rows.
        logger.warning("%s has a non-finite final close", row_key)
        return Outcome(row_key, NO_BARS, "", 0.0, 0.0, 0, 0.0, 0.0)
    move = (last_close - entry) if long else (entry - last_close)
    return Outcome(row_key, CLOSE, str(stamps[-1]), last_close,
                   move / stop_distance, len(stamps), mfe, mae)


def load_resolved(path: "Path | None" = None) -> set:
    """row_ids already resolved, so a re-run is cheap and idempotent."""
    target = path or STORE
    if not target.exists():
        return set()
    try:
        with target.open(newline="", encoding="utf-8") as handle:
            return {row["row_id"] for row in csv.DictReader(handle)
                    if row.get("row_id")}
    except (OSError, csv.Error) as exc:
        logger.warning("Could not read %s: %s", target.name, exc)
        return set()


def append(rows: list, path: "Path | None" = None) -> int:
    """Append resolved outcomes, writing the header on first use."""
    if not rows:
        return 0
    target = path or STORE
    # size check too: a crashed write leaves a zero-byte file, and
    # without a header DictReader reads the first outcome as one.
    fresh = not target.exists() or target.stat().st_size == 0
    try:
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS,
                                    extrasaction="ignore")
            if fresh:
                writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        logger.warning("Could not write %s: %s", target.name, exc)
        return 0
    return len(rows)


def to_row(logged: dict, result: Outcome) -> dict:
    """Join a scan_log row to its outcome, flat, for the outcomes file."""
    return {
        "row_id": result.row_id,
        "run_date": logged.get("run_date", ""),
        "run_time": logged.get("run_time", ""),
        "symbol": logged.get("symbol", ""),
        "direction": logged.get("direction", ""),
        "replayed": logged.get("replayed", ""),
        "control": logged.get("control", ""),
        "score": logged.get("score", ""),
        "entry": logged.get("entry", ""),
        "stop": logged.get("stop", ""),
        "target": logged.get("target", ""),
        "stop_pct": logged.get("stop_pct", ""),
        "breakeven_pct": logged.get("breakeven_pct", ""),
        "outcome": result.outcome,
        "exit_time": result.exit_time,
        "exit_price": round(result.exit_price, 2),
        "r_multiple": round(result.r_multiple, 4),
        "bars_held": result.bars_held,
        "mfe_r": round(result.mfe_r, 4),
        "mae_r": round(result.mae_r, 4),
        "resolved_at": datetime.now(IST).isoformat(timespec="seconds"),
    }


def session_bars(symbol: str, day):
    """Every intraday bar for one symbol's session, or None.

    Returns the WHOLE session rather than a slice, because the caller has
    to tell three cases apart: no bars at all (retry later), bars but none
    after the entry (a setup logged in the final bar, which will never
    have any and must not be retried forever), and the ordinary case.
    """
    import market_source

    try:
        interval = market_source.kite_interval(config.SCAN_BAR_INTERVAL)
        got = market_source.bars([symbol], day, day, interval=interval)
    except Exception as exc:
        logger.warning("No bars for %s on %s: %s", symbol, day, exc)
        return None
    frame = got.get(symbol)
    if frame is None or frame.empty:
        return None
    return frame


def _after(frame, taken):
    """Bars strictly after the entry moment, or an empty frame.

    STRICTLY after, deliberately. Kite stamps a bar at its OPEN, so the bar
    containing the scan instant has a high and low partly in the past and
    resolving against it would use prices from before the decision.

    The cost, recorded because it is a real measurement gap rather than a
    free win: `entry` is the previous bar's close, so up to two bars of
    genuine post-entry path are discarded. That biases outcomes toward
    CLOSE and understates fast stops, fast targets, MFE and MAE. The bias
    is conservative, which is the right direction, but it is not zero.
    """
    try:
        return frame[frame.index > taken]
    except TypeError as exc:
        logger.warning("Could not slice bars at %s: %s", taken, exc)
        return None


def log_files(log_path: "Path | None" = None) -> list:
    """The live scan log plus every rotated one beside it.

    scan_intraday rotates scan_log.csv to .csv.superseded whenever its
    field list changes, so rows unresolved at that moment become
    unreachable. One rotation has already happened and stranded four
    resolvable rows. row_id is derived from the row's own fields, so
    reading the rotated files is safe: anything already resolved is
    skipped by key.
    """
    live = log_path or config.SCAN_LOG_CSV
    found = [live] if live.exists() else []
    found.extend(sorted(live.parent.glob(f"{live.name}.superseded*")))
    return found


def resolve_log(log_path: "Path | None" = None,
                out_path: "Path | None" = None,
                now: "datetime | None" = None) -> dict:
    """Resolve every logged setup that has no outcome yet.

    Idempotent, and safe to run nightly: rows already in the outcomes file
    are skipped by key, and so are duplicates encountered within one pass.

    TODAY IS LEFT ALONE until the close has settled. A setup logged at
    10:00 has not finished happening at 14:00, and resolving it early
    records a CLOSE for a trade that went on to reach its target.
    """
    now = now or datetime.now(IST)
    done = load_resolved(out_path)
    stats = {"read": 0, "skipped": 0, "duplicate": 0, "resolved": 0,
             "unavailable": 0, "too_recent": 0, "malformed": 0}
    fresh = []

    for path in log_files(log_path):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                logged = list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            logger.warning("Could not read %s: %s", path.name, exc)
            continue

        for entry in logged:
            stats["read"] += 1
            key = row_id(entry.get("run_date", ""), entry.get("run_time", ""),
                         entry.get("symbol", ""), entry.get("direction", ""))
            if key in done:
                # Covers both "resolved on an earlier run" and "already seen
                # in THIS pass". The live log holds 8 rows and 3 distinct
                # keys - without this the first nightly run wrote a track
                # record counting two of three setups three times, and every
                # hit rate and mean R computed on it was weighted by how
                # often a row happened to be duplicated.
                stats["duplicate" if key in {r["row_id"] for r in fresh}
                      else "skipped"] += 1
                continue
            try:
                day = datetime.strptime(entry["run_date"], "%Y-%m-%d").date()
                clock = datetime.strptime(entry["run_time"], "%H:%M:%S").time()
                taken = datetime.combine(day, clock, tzinfo=IST)
                entry_price = float(entry["entry"])
                stop_price = float(entry["stop"])
                target_price = float(entry["target"])
            except (KeyError, ValueError) as exc:
                logger.warning("Unparseable row %s: %s", key, exc)
                stats["malformed"] += 1
                continue

            settled = (datetime.combine(day, SESSION_CLOSE, tzinfo=IST)
                       + timedelta(minutes=SETTLE_MINUTES))
            if now <= settled:
                stats["too_recent"] += 1
                continue

            frame = session_bars(entry["symbol"], day)
            if frame is None:
                # NOT an outcome. Leaving it unresolved means the next run
                # tries again; writing it now would be permanent.
                stats["unavailable"] += 1
                continue
            forward = _after(frame, taken)
            if forward is None:
                stats["unavailable"] += 1
                continue
            if len(forward) == 0:
                # The session traded but nothing after this row's stamp -
                # a setup logged in or after the final bar. No future bar
                # will ever arrive, so retrying nightly forever is wrong.
                # Mark it out at the session's own last close.
                forward = frame.iloc[-1:]

            result = resolve_one(entry_price, stop_price, target_price,
                                 long=str(entry.get("direction", "")).upper()
                                 == "LONG",
                                 bars=forward, row_key=key)
            if result.outcome == NO_BARS:
                stats["malformed"] += 1
                continue
            fresh.append(to_row(entry, result))
            done.add(key)
            stats["resolved"] += 1

    append(fresh, out_path)
    return stats


def summarise(path: "Path | None" = None) -> str:
    """Hit rate and R, split live from replayed, gross labelled as gross."""
    import pandas as pd

    target = path or STORE
    if not target.exists():
        return "no outcomes on file yet"
    frame = pd.read_csv(target)
    if frame.empty:
        return "no outcomes on file yet"

    lines = [f"{len(frame):,} outcomes on file"]
    replayed = frame["replayed"].astype(str).str.lower() == "true"
    for label, block in (("live", frame[~replayed]), ("replayed", frame[replayed])):
        if block.empty:
            continue
        settled = block[block["outcome"].isin([TARGET, STOP])]
        lines.append(f"\n{label}: {len(block):,} rows")
        lines.append("  " + block["outcome"].value_counts().to_dict().__str__())
        if len(settled):
            hits = int((settled["outcome"] == TARGET).sum())
            lines.append(f"  hit rate on settled: {hits}/{len(settled)} "
                         f"= {hits / len(settled):.1%}")
        gross = block["r_multiple"].mean()
        lines.append(f"  mean R (GROSS, before costs): {gross:+.3f}")
        # Costs in R, from the log's own figures: a breakeven of 0.12% on a
        # 1.86% stop is 0.066 R. Printing gross alone next to a hit rate
        # reads as a result when it is an input.
        try:
            drag = (block["breakeven_pct"].astype(float)
                    / block["stop_pct"].astype(float)).mean()
            lines.append(f"  mean cost drag: {drag:.3f} R")
            lines.append(f"  mean R (NET):   {gross - drag:+.3f}")
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            lines.append("  net R unavailable (no breakeven/stop columns)")
    if replayed.all():
        lines.append("\nEVERY row here is replayed. This is a simulation of "
                     "a track record, not one.")
    return "\n".join(lines)


def main(argv=None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stats = resolve_log()
    print(f"logged rows read        : {stats['read']:,}")
    print(f"  already resolved      : {stats['skipped']:,}")
    print(f"  duplicate in this pass : {stats['duplicate']:,}")
    print(f"  session not yet settled: {stats['too_recent']:,}")
    print(f"  bars unavailable      : {stats['unavailable']:,}")
    print(f"  malformed             : {stats['malformed']:,}")
    print(f"  resolved now          : {stats['resolved']:,}")
    if "--show" in argv:
        print()
        print(summarise())
    # Non-zero when a run did nothing but fail to find bars, so a nightly
    # log with a dead Kite token is actionable rather than silently fine.
    if stats["unavailable"] and not stats["resolved"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
