"""Do suggestions flip in and out of the table, and when?

WHY THIS EXISTS. The complaint was that "some stocks suggested for
intraday suddenly don't pass a gate in fraction of minutes, so the
suggestion table is not really great". It could not be settled from what
the system already keeps:

  - the published scan LOG is deduplicated to first sighting. 197 passes
    on 2026-09-22 produced 2,501 setups each, and the log held one row per
    symbol+direction stamped when first seen. By construction it cannot
    show a symbol entering, leaving and returning.
  - the published SNAPSHOT is the live table, but it is overwritten every
    pass, so yesterday is gone.

So the only sample available was two adjacent passes late in the session -
14:33:38 and 14:34:48 - which showed 71 actionable, 71 kept, zero
turnover. That is weak evidence against the complaint, not evidence for
it: late-session passes are exactly where things have settled. What was
missing is the OPEN, when relative volume, the opening range and the
benchmark are all still arriving.

WHAT IT RECORDS. One row per transition, plus one summary row per pass,
appended to a CSV that survives a restart. Transitions are what the
snapshot cannot keep and what the log discards. A row that LEAVES is
looked up in the pass that dropped it so the GATE IS NAMED - "it stopped
passing a gate" is only actionable when you know which one.

WHAT IT COSTS. One HEAD per poll and a GET only when the published object
has actually moved, which is what scan_publish.load already does. An
earlier version of this file claimed "one object read per pass" while
issuing an unconditional GET every 20 seconds - 3 to 5 full reads per
pass, about 1,100 over a session. It reads the object the UI already
polls and never touches Kite, so it competes with nothing.

RUN IT FROM BEFORE THE OPEN:

    . .\\deploy\\env.ps1
    .venv\\Scripts\\python churn_sample.py --watch

and read it afterwards with --report. It refuses to start without the
object store configured, because sourcing env.ps1 is easy to forget and
the failure is otherwise silent - the earlier version slept through the
whole session and exited 0.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config
import object_store
import scan_publish

IST = ZoneInfo("Asia/Kolkata")
SAMPLE_FILE = Path(__file__).with_name("churn_samples.csv")
FIELDS = ["sampled_at", "run_date", "pass_time", "kind", "symbol",
          "direction", "n_actionable", "detail"]

# How often to ask whether a NEW pass exists. The feed publishes every
# 70-90s, so this is a poll for a changed version, not a sampling rate.
POLL_SECONDS = 20


def snapshot():
    """The live table, or None when nothing new is published.

    Returns (frame, version). The version gate is what keeps this to one
    GET per published pass rather than one per poll.
    """
    try:
        payload = object_store.get(scan_publish.object_name())
    except Exception as exc:
        print(f"  read failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    if not payload:
        return None
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        print(f"  unreadable parquet: {exc}", flush=True)
        return None


def actionable_flags(frame):
    """The actionable column as real booleans.

    NOT a bare astype(bool), which is wrong twice over: NaN becomes True,
    so a row with a missing flag counts as being on the table, and the
    string "False" is truthy, so a parquet round-trip through an object
    column inverts the meaning entirely. Both manufacture phantom ENTER
    and LEAVE transitions - the exact measurement this script exists to
    produce.
    """
    column = frame["actionable"]
    # Tested on dtype SEMANTICS, not identity. `dtype == object` misses
    # pandas string[pyarrow], so a parquet round-trip that hands back
    # Arrow-backed strings fell through to astype(bool), where "False"
    # is truthy and every blocked row counted as being on the table.
    if pd.api.types.is_bool_dtype(column):
        return column.fillna(False).astype(bool)
    return (column.astype(str).str.strip().str.lower()
            .isin(("true", "1")))


def actionable_set(frame) -> dict:
    """{(symbol, direction): row} for everything currently on the table."""
    if frame is None or "actionable" not in frame.columns:
        return {}
    live = frame[actionable_flags(frame)]
    return {(r["symbol"], r["direction"]): r for _, r in live.iterrows()}


def first_failure(reasons) -> str:
    """The first FAIL in a row's reason chain - why it is not on the table.

    The reasons are one pipe-joined string and the first failure is the
    one that stopped it, so later entries describe gates never reached.
    Accepts None and NaN because the column can be absent.
    """
    if reasons is None or reasons != reasons:
        return "(no stated failure)"
    for part in str(reasons).split("|"):
        part = part.strip()
        if "[FAIL]" in part:
            return " ".join(part.split()[:10])
    return "(no stated failure)"


def _append(rows) -> None:
    new = not SAMPLE_FILE.exists()
    with SAMPLE_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def watch(until=None) -> int:
    """Sample every new pass until the close, appending transitions."""
    if not object_store.enabled():
        print("Object store not configured, so there is nothing to sample "
              "and this would sleep through the session reporting success.")
        print("Run:  . .\\deploy\\env.ps1   then start this again.")
        return 2

    close = tuple(until or config.SCAN_SESSION_CLOSE)
    previous: dict = {}
    seen_pass = None
    seen_version = None
    passes = 0

    print(f"watching {scan_publish.object_name()}; "
          f"writing {SAMPLE_FILE.name}; stopping at "
          f"{close[0]:02d}:{close[1]:02d}", flush=True)

    while True:
        now = datetime.now(IST)
        if (now.hour, now.minute) >= close:
            print(f"session closed; {passes} passes sampled", flush=True)
            return 0 if passes else 1

        # HEAD first. Only pay for the body when the object has moved.
        try:
            current_version = object_store.version(scan_publish.object_name())
        except Exception:
            current_version = None
        if current_version is not None and current_version == seen_version:
            time.sleep(POLL_SECONDS)
            continue

        frame = snapshot()
        if frame is None or frame.empty:
            time.sleep(POLL_SECONDS)
            continue
        seen_version = current_version

        pass_time = str(frame["run_time"].iloc[0])
        if pass_time == seen_pass:
            time.sleep(POLL_SECONDS)
            continue

        run_date = str(frame["run_date"].iloc[0]) \
            if "run_date" in frame.columns else now.date().isoformat()
        current = actionable_set(frame)
        stamp = now.isoformat(timespec="seconds")
        rows = [{"sampled_at": stamp, "run_date": run_date,
                 "pass_time": pass_time, "kind": "PASS", "symbol": "",
                 "direction": "", "n_actionable": len(current), "detail": ""}]

        if seen_pass is not None:
            by_key = {(r["symbol"], r["direction"]): r
                      for _, r in frame.iterrows()}
            for key in previous.keys() - current.keys():
                row = by_key.get(key)
                # row.get, not row["reasons"] - a missing column would
                # raise KeyError on the first LEAVE, which is precisely
                # the event this script exists to record, and would take
                # the rest of an unrepeatable session with it.
                detail = first_failure(row.get("reasons")) \
                    if row is not None else "(absent from this pass)"
                rows.append({"sampled_at": stamp, "run_date": run_date,
                             "pass_time": pass_time, "kind": "LEAVE",
                             "symbol": key[0], "direction": key[1],
                             "n_actionable": len(current), "detail": detail})
            for key in current.keys() - previous.keys():
                rows.append({"sampled_at": stamp, "run_date": run_date,
                             "pass_time": pass_time, "kind": "ENTER",
                             "symbol": key[0], "direction": key[1],
                             "n_actionable": len(current), "detail": ""})

        _append(rows)
        left = sum(1 for r in rows if r["kind"] == "LEAVE")
        entered = sum(1 for r in rows if r["kind"] == "ENTER")
        passes += 1
        print(f"  {pass_time}  actionable {len(current):>3}  "
              f"+{entered:<3} -{left:<3}", flush=True)
        previous, seen_pass = current, pass_time
        time.sleep(POLL_SECONDS)


def _bucket(row) -> "tuple | None":
    """(date, half hour) for a sampled row, or None if unparseable.

    BUCKETED BY DATE TOO. pass_time is a bare clock string, so bucketing
    on it alone merges every session in the file into one - and the file
    is designed to survive restarts and grow. A truncated final row from
    a Ctrl-C mid-write is also expected, hence the None.
    """
    stamp = str(row.get("pass_time", ""))
    parts = stamp.split(":")
    if len(parts) < 2 or not parts[0].strip().isdigit():
        return None
    try:
        minute = int(parts[1])
    except ValueError:
        return None
    return (str(row.get("run_date", "?")),
            f"{parts[0]}:{'00' if minute < 30 else '30'}")


def report(path=None) -> int:
    """Churn by half-hour, per day, and the gates that did the dropping."""
    source = Path(path) if path else SAMPLE_FILE
    if not source.exists():
        print(f"no {source.name} yet - run --watch during a session")
        return 1
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if _bucket(r)]
    passes = [r for r in rows if r["kind"] == "PASS"]
    if not passes:
        print(f"{source.name} holds no complete passes yet")
        return 1

    days = sorted({r.get("run_date", "?") for r in passes})
    print(f"{len(passes):,} passes sampled across {len(days)} day(s): "
          f"{', '.join(days)}")
    print()

    per = {}
    for r in rows:
        key = _bucket(r)
        b = per.setdefault(key, {"passes": 0, "enter": 0, "leave": 0,
                                 "act": []})
        if r["kind"] == "PASS":
            b["passes"] += 1
            try:
                b["act"].append(int(r["n_actionable"] or 0))
            except ValueError:
                pass
        elif r["kind"] == "ENTER":
            b["enter"] += 1
        elif r["kind"] == "LEAVE":
            b["leave"] += 1

    print(f"{'day':>11} {'half hour':>10} {'passes':>7} {'mean table':>11} "
          f"{'entered':>8} {'left':>6} {'churn/pass':>11}")
    print("-" * 70)
    for (day, half), v in sorted(per.items()):
        mean = (sum(v["act"]) / len(v["act"])) if v["act"] else 0.0
        # N passes give N-1 chances to change, and the first pass of a
        # watch run emits no transitions at all. Dividing by `passes`
        # overstates stability worst in small buckets - which is the
        # opening half hour, the one this exists to measure.
        diffs = max(0, v["passes"] - 1)
        per_pass = (v["leave"] / diffs) if diffs else 0.0
        rate = (per_pass / mean) if mean else 0.0
        print(f"{day:>11} {half:>10} {v['passes']:>7} {mean:>11.1f} "
              f"{v['enter']:>8} {v['leave']:>6} {rate:>10.1%}")
    print()
    print("`churn/pass` is the share of the table that dropped out between "
          "one pass and the next, over N-1 transitions rather than N "
          "passes. High early and low later means the gates are settling, "
          "not that the signal changed.")
    print()

    leaves = [r for r in rows if r["kind"] == "LEAVE"]
    if not leaves:
        print("nothing ever left the table.")
        return 0
    print(f"why {len(leaves):,} suggestions left the table:")
    for detail, n in Counter(r["detail"] for r in leaves).most_common(10):
        print(f"  {detail[:64]:<66} {n:>5}")
    print()
    repeats = Counter((r["symbol"], r["direction"]) for r in leaves)
    flapping = [(k, n) for k, n in repeats.most_common(10) if n > 1]
    if flapping:
        print("symbols that left more than once (flapping):")
        for (sym, side), n in flapping:
            print(f"  {sym:<14} {side:<6} left {n} times")
    else:
        print("no symbol left the table more than once.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure how much the suggestion table churns")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true",
                      help="sample every new pass until the close")
    mode.add_argument("--report", action="store_true",
                      help="summarise what has been sampled")
    parser.add_argument("--path", default=None,
                        help="sample file to report on (default: this one)")
    args = parser.parse_args(argv)
    if args.watch:
        return watch()
    return report(args.path)


if __name__ == "__main__":
    sys.exit(main())
