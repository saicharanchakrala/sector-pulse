"""Does a higher score actually resolve better? Read from the outcomes.

WHAT THIS IS FOR. setups.score_setup says its weights "are a judgement
call and have no backtest behind them", and the app's own caption records
that four horizons were measured and none showed predictive skill. So the
score is, today, an untested ordering. This turns it into a measured one:
for each band of score, what share of setups resolved in the target's
favour, and what did they average in R after costs.

WHAT IT REFUSES TO DO. It will not print a hit rate it cannot support.
Below MIN_SAMPLE resolved rows it says so and stops, because the whole
failure mode this exists to avoid is a number computed from five
correlated setups being read as a track record. It also reports the
REPLAYED and LIVE populations separately and never pools them - a replay
is a reconstruction and the `replayed` column exists because a replayed
row stamped with a past time is otherwise indistinguishable from a live
one.

WHAT A HIT RATE HAS TO BEAT. Each row carries required_win_rate, the
share of wins that merely covers the round trip. A band whose observed
hit rate sits below its own required rate lost money at that score, and
comparing the two is the only reading of this file that means anything.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

# Below this, no hit rate is reported. At n=5 the standard error of a
# proportion near 0.4 is about 0.22 - wider than any edge worth finding,
# so a number would be worse than silence.
MIN_SAMPLE = 30

# Score bands. Coarse on purpose: the score is bounded 0..1 and five bands
# over a few hundred rows keeps each one big enough to mean something.
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(path=None) -> list:
    """Every resolved outcome, across the live and rotated files."""
    import outcomes

    rows = []
    for source in outcomes.outcome_files(path):
        try:
            with source.open(newline="", encoding="utf-8") as handle:
                rows.extend(list(csv.DictReader(handle)))
        except OSError:
            continue
    seen, unique = set(), []
    for row in rows:
        key = row.get("row_id")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def band_for(score: float):
    for low, high in BANDS:
        if low <= score < high:
            return (low, high)
    return None


def summarise(rows: list) -> dict:
    """Per band: n, hit rate, mean net R, and the rate it had to beat."""
    bands = defaultdict(lambda: {"n": 0, "hits": 0, "net_r": 0.0,
                                 "required": 0.0})
    for row in rows:
        score = _float(row.get("score"))
        r_gross = _float(row.get("r_multiple"))
        if score is None or r_gross is None:
            continue
        band = band_for(score)
        if band is None:
            continue
        # NET of costs, in R. breakeven_pct is the round trip as a share of
        # price and stop_pct the stop distance, so their ratio is the cost
        # expressed in the same units as r_multiple.
        stop_pct = _float(row.get("stop_pct")) or 0.0
        breakeven = _float(row.get("breakeven_pct")) or 0.0
        cost_r = (breakeven / stop_pct) if stop_pct > 0 else 0.0
        entry = bands[band]
        entry["n"] += 1
        entry["hits"] += 1 if row.get("outcome") == "TARGET" else 0
        entry["net_r"] += r_gross - cost_r
        entry["required"] += _float(row.get("required_win_rate")) or 0.0
    return dict(bands)


def report(rows: list) -> str:
    """The whole readout, or an honest refusal."""
    live = [r for r in rows if str(r.get("replayed", "")).lower()
            not in ("true", "1")]
    replayed = [r for r in rows if str(r.get("replayed", "")).lower()
                in ("true", "1")]
    lines = [f"{len(rows)} resolved setups: {len(live)} live, "
             f"{len(replayed)} replayed."]

    if len(live) < MIN_SAMPLE:
        lines.append("")
        lines.append(f"NOT ENOUGH TO CALIBRATE. {len(live)} live outcomes "
                     f"against a floor of {MIN_SAMPLE}.")
        lines.append("")
        lines.append("At this size the error bar on a hit rate is wider "
                     "than any edge worth finding, so no rate is printed.")
        lines.append("A few hundred is where a score band starts to mean "
                     "something. The feed logs every directional setup "
                     "once per session; fetch_scan_log brings those down "
                     "and outcomes.py resolves them.")
        if replayed:
            lines.append("")
            lines.append(f"The {len(replayed)} replayed rows are excluded "
                         f"deliberately: a replay is a reconstruction, not "
                         f"a signal that was acted on.")
        return "\n".join(lines)

    bands = summarise(live)
    lines.append("")
    lines.append(f"{'score band':>12}  {'n':>5}  {'hit rate':>9}  "
                 f"{'needed':>7}  {'edge':>6}  {'mean net R':>10}")
    lines.append("-" * 62)
    for low, high in BANDS:
        entry = bands.get((low, high))
        if not entry or not entry["n"]:
            continue
        n = entry["n"]
        hit = entry["hits"] / n
        needed = entry["required"] / n
        lines.append(f"{low:.1f}-{high:.1f}".rjust(12) +
                     f"  {n:5d}  {hit:8.1%}  {needed:6.1%}  "
                     f"{hit - needed:+5.1%}  {entry['net_r'] / n:10.3f}")
    lines.append("")
    lines.append("`needed` is the mean required_win_rate - the share of "
                 "wins that merely covers the round trip.")
    lines.append("A band whose hit rate is below it lost money at that "
                 "score. `edge` is the difference and is the only column "
                 "worth reading on its own.")

    ordered = [(b, bands[b]["hits"] / bands[b]["n"])
               for b in sorted(bands) if bands[b]["n"] >= 10]
    if len(ordered) >= 3:
        rates = [rate for _, rate in ordered]
        rising = all(b <= a for a, b in zip(rates, rates[1:])) or \
            all(b >= a for a, b in zip(rates, rates[1:]))
        lines.append("")
        lines.append("Monotonic in score: " + ("yes" if rising else "NO") +
                     " - if the hit rate does not rise with the score, "
                     "the score is not ordering anything.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether a higher score resolved better")
    parser.add_argument("--path", default=None,
                        help="outcomes file (default: the project's)")
    args = parser.parse_args(argv)
    path = None
    if args.path:
        from pathlib import Path
        path = Path(args.path)
    rows = load(path)
    if not rows:
        print("no resolved outcomes at all - run outcomes.py first")
        return 0
    print(report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
