"""Does a higher score actually resolve better? Read from the outcomes.

WHAT THIS IS FOR. setups.score_setup says its weights "are a judgement
call and have no backtest behind them", and the app's own caption records
that four horizons were measured and none showed predictive skill. So the
score is, today, an untested ordering. This turns it into a measured one:
for each band of score, what share of setups resolved in the target's
favour, and what did they average in R after costs.

WHAT IT REFUSES TO DO. It will not print a hit rate it cannot support.
Below MIN_SAMPLE USABLE rows it says so and stops, because the whole
failure mode this exists to avoid is a number computed from five
correlated setups being read as a track record. The floor is counted on
rows that actually yield a number, not rows that exist - an earlier
version gated on the second and printed a per-trade rupee figure from
n=3. It also reports the REPLAYED and LIVE populations separately and
never pools them, because a replay is a reconstruction and the
`replayed` column exists because a replayed row stamped with a past time
is otherwise indistinguishable from a live one.

WHAT A HIT RATE HAS TO BEAT. The share of wins that merely covers the
round trip - and there are two honest answers, so both are printed.
`needed` assumes every non-winner loses the full stop, which is the
assumption levels.required_win_rate makes and the app displays. `real`
uses what the non-winners in that band actually did, which matters
because this scanner squares off at the close and most setups reach
neither level. `edge` is measured against `real`, so it agrees in sign
with the money column instead of contradicting it.

WHAT IT IS WORTH IN RUPEES. Expectancy is the average R per trade times
the rupee value of one R, and that value is a constant because the sizer
makes it one. Reported next to the R figure because the decision to keep
running a strategy is made in money, not in multiples of a stop - and
reported for the TAKEN rows separately, because the log is mostly setups
the gates refused and their expectancy is not the strategy's.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import config

# Below this, no rate and no expectancy is reported. At n=5 the standard
# error of a proportion near 0.4 is about 0.22 - wider than any edge worth
# finding, so a number would be worse than silence.
MIN_SAMPLE = 30

# Score bands. Coarse on purpose: the score is bounded 0..1 and five bands
# over a few hundred rows keeps each one big enough to mean something.
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001))

# The smallest price increment on NSE equities, and the reason the levels
# on a row cannot simply be differenced. See reward_risk.
TICK = 0.01

# A round trip costing more than the whole amount at risk is not a trade,
# it is a malformed row. They are real: the scan log stores stop_pct to
# three decimals, so a stop under a tenth of a percent rounds toward zero
# and the cost-to-risk ratio explodes. Measured on scan_log_20260921.csv -
# 2,750 rows, 74 with a ratio above 1R, worst 84.70R, and 2 with stop_pct
# rounded to 0.0 exactly. One such row moved a 40-row headline from
# +0.078R to -1.919R, so these are excluded and counted rather than
# averaged. All 74 were setups the gates had already refused.
MAX_COST_R = 1.0


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


def rupees_per_r() -> float:
    """What one R is worth, in rupees.

    A CONSTANT, and that is the whole point of the position sizing:
    quantity is derived so the rupee risk is the same whatever the stop
    distance, so every R multiple converts at one rate. With the current
    100,000 of capital and 1% per trade, one R is 1,000 rupees.

    TAKEN FROM CONFIG, not from the resolved rows, because outcomes.csv
    does not carry risk_rupees. The actual rupees at risk run slightly
    under the nominal because quantity has to be a whole number of
    shares, so this is the intended risk rather than the filled risk -
    the right basis for judging the strategy and the wrong one for
    reconciling against a broker statement.
    """
    return float(config.SCAN_CAPITAL) * float(
        config.SCAN_RISK_PCT_PER_TRADE) / 100.0


def cost_r(row) -> "float | None":
    """The round trip for this setup, expressed in R.

    breakeven_pct is the round trip as a share of price and stop_pct is
    the stop distance in the same units, so the ratio is the cost in
    units of the amount at risk and can be subtracted from an R multiple
    directly.

    RETURNS None RATHER THAN ZERO when it cannot be computed. A missing
    or zero stop_pct used to yield 0.0, which silently reported that
    row's GROSS R as net - in a file whose whole discipline is that
    costs are never omitted. See MAX_COST_R for the rounding that makes
    these rows real rather than hypothetical.
    """
    stop_pct = _float(row.get("stop_pct"))
    breakeven = _float(row.get("breakeven_pct"))
    if not stop_pct or stop_pct <= 0.0 or breakeven is None:
        return None
    ratio = breakeven / stop_pct
    if ratio < 0.0 or ratio > MAX_COST_R:
        return None
    return ratio


def reward_risk(row) -> "float | None":
    """The reward-to-risk ratio behind this row's levels.

    TAKEN FROM CONFIG, NOT FROM THE PRICES, and that is a measured
    choice rather than a shortcut. build_levels sets
    `target = entry +/- reward_risk * risk_per_share`, so the ratio is
    the configured constant by construction - but the row stores entry,
    stop and target rounded to two decimals, and differencing them
    recovers the ratio only as well as the rounding allows. For a
    low-priced name whose stop is a few ticks away, one tick is a large
    fraction of the distance.

    Measured on 2,739 real rows against the stored required_win_rate:

        differencing, no floor      97.6% kept   worst error 21.01%
        differencing, risk >= 0.10  88.5% kept   worst error  1.65%
        differencing, risk >= 1.00  56.6% kept   worst error  0.21%
        the configured constant     97.6% kept   worst error  0.19%

    The constant is more accurate than differencing with a floor that
    throws away two rows in five. The worst single row when differencing
    was MSCI360 at entry 9.9, stop 9.91, target 9.89 - recovered as
    1.0000 against a stored 0.7636.

    THE PRICES STILL GET A VOTE. If they disagree with the constant by
    more than rounding can explain, the row was built under a different
    configuration and the prices are the better source - and that check
    is only capable of firing when the stop is wide enough for the
    differenced ratio to be precise, which is exactly when it should be
    trusted.
    """
    entry = _float(row.get("entry"))
    stop = _float(row.get("stop"))
    target = _float(row.get("target"))
    if entry is None or stop is None or target is None:
        return None
    risk = abs(entry - stop)
    if risk <= 0.0:
        return None
    configured = float(config.SCAN_REWARD_RISK)
    derived = abs(target - entry) / risk
    # Each price carries up to half a tick, so the error in a differenced
    # ratio is about TICK * (1 + ratio) / risk. Doubled because the entry
    # is rounded on both sides of the fraction.
    slack = 2.0 * TICK * (1.0 + configured) / risk
    if abs(derived - configured) > slack:
        return derived
    return configured


def required_from_row(row) -> "float | None":
    """The win rate this setup had to beat, assuming a full stop loss.

    THE BUG THIS FIXES. This file read `required_win_rate` straight off
    each row, and outcomes.FIELDS does not carry that column -
    scan_publish and scan_intraday write it to the scan log and the
    resolver drops it. Verified: FIELDS holds 23 names and none is that
    one, the live outcomes.csv header has 21 columns, to_row never
    copies it, and no rotated outcomes file exists that would. So
    `_float(None) or 0.0` made `needed` 0.0% for every band and `edge`
    identical to the raw hit rate, in the one column the report used to
    call "the only column worth reading on its own". Every band looked
    profitable by the exact width of its own hit rate.

    DERIVED RATHER THAN ADDED as a column, because deriving it works on
    the rotated files that already exist, where a new field would be
    empty for every row resolved before today.

    levels.required_win_rate solves p*reward - (1-p)*risk = costs as
    (lot_risk + cost_rupees) / (reward + lot_risk). Dividing through by
    lot_risk puts it in R units, where cost_rupees/lot_risk is cost_r
    and reward/lot_risk is the reward-to-risk ratio:

        (1 + cost_r) / (rr + 1)

    capped at 1.0 exactly as the property caps it - costs exceeding the
    whole reward mean the trade breaks even at no win rate at all.
    Checked against the property over 4,000 built levels including
    structural stops, capital-capped sizes and quantities down to 1:
    worst disagreement 5.5e-15, so the algebra is the same number.

    CARRIES THE PROPERTY'S KNOWN FLAW, which is why `real` exists beside
    it. The formula counts every non-winner as a full loss of risk, and
    this scanner squares off at the close: measured over 93 sessions and
    2,436 signals, 1,374 - the majority - reached neither level and
    averaged +0.24R, not -1R. So this is an OVERSTATED bar.
    """
    ratio = reward_risk(row)
    cost = cost_r(row)
    if ratio is None or cost is None:
        return None
    return min(1.0, (1.0 + cost) / (ratio + 1.0))


def net_r(row) -> "float | None":
    """r_multiple after costs, in R, or None if it cannot be costed.

    `r_multiple` is GROSS - the Outcome docstring says so. Measured on
    the five rows resolved so far, a gross 614 rupee loss became 1,030
    net, costs taking 416. A strategy that reads positive gross and
    negative net is the exact trap worth surfacing, so nothing in this
    file reports gross R on its own, and a row whose costs cannot be
    established is dropped rather than reported gross.
    """
    gross = _float(row.get("r_multiple"))
    cost = cost_r(row)
    if gross is None or cost is None:
        return None
    return gross - cost


def is_live(row) -> bool:
    """A live scan, not a replay stamped with a past time."""
    return str(row.get("replayed", "")).lower() not in ("true", "1")


def disposition(row) -> str:
    """Whether the scanner would have ACTED on this setup.

    outcomes.FIELDS carries `taken` precisely so the rows the gates
    refused are kept as the counterfactual, and the resolver applies no
    filter - so most of the file is setups that were never endorsed. In
    scan_log_20260921.csv only 149 of 2,750 were taken, 5.4%.

    "unknown" IS A THIRD ANSWER AND NOT A SYNONYM FOR BLOCKED. The live
    outcomes.csv predates the column: all five of its rows have no
    `taken` at all, so treating absence as False would have silently
    reported an empty taken population as a measured zero.
    """
    value = str(row.get("taken", "")).strip().lower()
    if value in ("true", "1"):
        return "taken"
    if value in ("false", "0"):
        return "blocked"
    return "unknown"


def band_for(score: float):
    for low, high in BANDS:
        if low <= score < high:
            return (low, high)
    return None


def rateable(row) -> bool:
    """Whether every number this file reports can be computed for a row.

    ONE GATE FOR THE WHOLE FILE, because splitting it is what produced a
    smaller copy of the bug the derivation was meant to fix: `summarise`
    added `required_from_row(row) or 0.0` to the band total while
    counting the row in the denominator, so twenty rateable rows beside
    twenty with a blank stop_pct printed `needed` as 18.7% instead of
    37.4% - the same halving, one layer up. A row that cannot be costed
    is not data here, so it never enters a band and is counted as
    excluded instead.
    """
    return (_float(row.get("score")) is not None
            and net_r(row) is not None
            and required_from_row(row) is not None)


def summarise(rows: list) -> dict:
    """Per band: n, hits, net R, and the parts both break-even bars need."""
    bands = defaultdict(lambda: {"n": 0, "hits": 0, "net_r": 0.0,
                                 "required": 0.0, "cost_r": 0.0,
                                 "rr": 0.0, "loser_r": 0.0, "losers": 0})
    for row in rows:
        if not rateable(row):
            continue
        band = band_for(_float(row.get("score")))
        if band is None:
            continue
        entry = bands[band]
        entry["n"] += 1
        entry["net_r"] += net_r(row)
        entry["required"] += required_from_row(row)
        entry["cost_r"] += cost_r(row)
        entry["rr"] += reward_risk(row)
        if row.get("outcome") == "TARGET":
            entry["hits"] += 1
        else:
            # What the non-winners in this band actually returned, gross.
            # `real` is built from these rather than from an assumed -1R.
            entry["loser_r"] += _float(row.get("r_multiple")) or 0.0
            entry["losers"] += 1
    return dict(bands)


def realised_required(entry: dict) -> float:
    """The break-even hit rate given what this band's non-winners did.

    WHY THIS EXISTS. `needed` assumes every non-winner loses the whole
    stop, so a band can show a negative edge and a positive mean net R
    at the same time - observed at 0.2-0.4 reading edge -17.4% against
    +0.132 R. The old report called `edge` "the only column worth
    reading on its own" three lines above disclaiming it as overstated,
    and keeping both sentences was worse than either.

    Break-even solves p*rr + (1-p)*a = cost, where `a` is the mean gross
    R of the non-winners rather than -1:

        p = (cost - a) / (rr - a)

    which is sign-equivalent to the mean net R being positive, by the
    same algebra: mean net = p*rr + (1-p)*a - cost. So an `edge`
    measured against this cannot contradict the money column.

    NOT CLAMPED TO 0..100%, because clamping is what breaks the one
    property it exists for. A band whose non-winners on average returned
    more than the round trip made money with no target hits at all, and
    its honest bar is NEGATIVE - fewer than zero wins were required.
    Clamping that to 0.0 reproduced the contradiction one notch down:
    measured on a randomised sweep, a band with no hits and a mean net R
    of +0.125 read an edge of exactly 0.0. Above 100% is equally real and
    means the costs exceed the whole reward, so the trade breaks even at
    no hit rate. Both ends are printed as they come out.

    SOMEWHAT SELF-REFERENTIAL, and worth saying plainly: the bar is
    computed from the same rows whose hit rate is compared against it.
    It answers "did this band make money", which is the question, and
    not "will the next one", which needs the out-of-sample data this
    file is accumulating.
    """
    n = entry["n"]
    if not n:
        return 0.0
    cost = entry["cost_r"] / n
    ratio = entry["rr"] / n
    # An assumed full stop loss when the band has no non-winners to
    # measure - which is also the only case where the two bars agree.
    a = (entry["loser_r"] / entry["losers"]) if entry["losers"] else -1.0
    if ratio <= a:
        # Defensive only: a non-winner cannot have reached the target, so
        # its return is below the target multiple by construction.
        return 1.0
    return (cost - a) / (ratio - a)


def expectancy(rows: list) -> dict:
    """The one number the whole file exists for, and its parts.

    Expectancy is win rate times the average win minus loss rate times
    the average loss. Computed here from realised R rather than from
    that decomposition - the two agree, and the realised mean cannot
    drift out of step with the wins and losses it is made of - but the
    parts are reported alongside it, because a positive expectancy built
    on one outsized win is a different fact from one built on a steady
    edge. Net of costs throughout; see net_r.
    """
    wins, losses, net = [], [], []
    required = []
    hits = 0
    for row in rows:
        value = net_r(row)
        if value is None:
            continue
        net.append(value)
        (wins if value > 0 else losses).append(value)
        hits += 1 if row.get("outcome") == "TARGET" else 0
        rate = required_from_row(row)
        if rate is not None:
            required.append(rate)
    if not net:
        return {}
    count = len(net)
    mean = sum(net) / count
    return {
        "n": count,
        # TWO DIFFERENT RATES, deliberately, and both are printed because
        # they answer different questions. `win_rate` is the share that
        # made money after costs, which is what the decomposition is
        # built from. `hit_rate` is the share that reached the target,
        # which is what a break-even bar has to be compared against.
        # They diverge because most setups end at neither level.
        "win_rate": len(wins) / count,
        "hit_rate": hits / count,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "expectancy_r": mean,
        "expectancy_rupees": mean * rupees_per_r(),
        "required": (sum(required) / len(required)) if required else None,
    }


def _band_table(bands: dict) -> list:
    """The per-band rows."""
    lines = [f"{'score band':>12}  {'n':>5}  {'hit rate':>9}  "
             f"{'needed':>7}  {'real':>7}  {'edge':>7}  "
             f"{'mean net R':>10}",
             "-" * 72]
    for low, high in BANDS:
        entry = bands.get((low, high))
        if not entry or not entry["n"]:
            continue
        n = entry["n"]
        hit = entry["hits"] / n
        needed = entry["required"] / n
        real = realised_required(entry)
        lines.append(f"{low:.1f}-{high:.1f}".rjust(12) +
                     f"  {n:5d}  {hit:8.1%}  {needed:6.1%}  {real:6.1%}  "
                     f"{hit - real:+7.1%}  {entry['net_r'] / n:10.3f}")
    return lines


def _expectancy_lines(exp: dict, label: str) -> list:
    """The expectancy block, in R and in rupees."""
    rupees = rupees_per_r()
    lines = ["",
             f"EXPECTANCY over {exp['n']:,} {label}, net of costs, "
             f"where one R is Rs {rupees:,.0f}",
             f"  {'reached the target':<24}{exp['hit_rate']:>9.1%}",
             f"  {'made money after costs':<24}{exp['win_rate']:>9.1%}",
             f"  {'average winner':<24}{exp['avg_win']:>+8.3f} R"
             f"  {exp['avg_win'] * rupees:>+10,.0f}",
             f"  {'average loser':<24}{exp['avg_loss']:>+8.3f} R"
             f"  {exp['avg_loss'] * rupees:>+10,.0f}",
             f"  {'PER TRADE':<24}{exp['expectancy_r']:>+8.3f} R"
             f"  {exp['expectancy_rupees']:>+10,.0f}"]
    if exp.get("required") is not None:
        lines.append(f"  {'bar, full-loss basis':<24}"
                     f"{exp['required']:>9.1%}")
    if exp["expectancy_r"] <= 0:
        lines.append("")
        lines.append("NEGATIVE. At this expectancy the strategy loses "
                     "money per trade after costs, and trading it more "
                     "often loses it faster.")
    return lines


def _refusal(live: list, replayed: list, usable: int) -> list:
    """What is said instead of a number, and why."""
    lines = ["",
             f"NOT ENOUGH TO CALIBRATE. {usable} usable live outcomes "
             f"against a floor of {MIN_SAMPLE}.",
             "",
             "At this size the error bar on a hit rate is wider than any "
             "edge worth finding, so no rate is printed.",
             "A few hundred is where a score band starts to mean "
             "something. The feed logs every directional setup once per "
             "session; fetch_scan_log brings those down and outcomes.py "
             "resolves them."]
    if usable < len(live):
        lines.append(f"{len(live) - usable} of the {len(live)} live rows "
                     f"could not be costed or scored and are excluded - "
                     f"the floor counts rows that yield a number, not "
                     f"rows that exist.")
    # THE CONVERSION IS STATED EVEN THOUGH THE MEAN IS NOT. What one R is
    # worth is a config constant and knowable today; what the average
    # trade returns is a statistic and is not. Printing the first without
    # the second is the distinction this file exists to hold.
    lines.append("")
    lines.append(f"One R is Rs {rupees_per_r():,.0f} "
                 f"({config.SCAN_RISK_PCT_PER_TRADE:.2f}% of "
                 f"Rs {config.SCAN_CAPITAL:,.0f}), so expectancy in rupees "
                 f"is that times the mean net R - withheld here for the "
                 f"same reason the hit rate is.")
    if replayed:
        lines.append("")
        lines.append(f"The {len(replayed)} replayed rows are excluded "
                     f"deliberately: a replay is a reconstruction, not a "
                     f"signal that was acted on.")
    return lines


def _population_lines(live: list, usable: list, taken: list,
                      unknown: list) -> list:
    """Who the numbers below are about. See disposition."""
    lines = []
    if len(usable) < len(live):
        lines.append(f"{len(live) - len(usable)} live rows could not be "
                     f"costed or scored and are excluded; see MAX_COST_R.")
    blocked = len(usable) - len(taken) - len(unknown)
    tail = (f", {len(unknown)} predate the column"
            if unknown else "")
    lines.append(f"Of {len(usable)} usable live rows, {len(taken)} were "
                 f"TAKEN by the gates and {blocked} were blocked{tail}.")
    return lines


def report(rows: list) -> str:
    """The whole readout, or an honest refusal."""
    live = [r for r in rows if is_live(r)]
    replayed = [r for r in rows if not is_live(r)]
    usable = [r for r in live if rateable(r)]
    taken = [r for r in usable if disposition(r) == "taken"]
    unknown = [r for r in usable if disposition(r) == "unknown"]

    lines = [f"{len(rows)} resolved setups: {len(live)} live, "
             f"{len(replayed)} replayed."]
    lines.extend(_population_lines(live, usable, taken, unknown))

    # THE FLOOR COUNTS USABLE ROWS, not rows. Gating on len(live) while
    # computing the statistic over the rows that parsed let a per-trade
    # rupee figure out at n=3, in the file that says it will not print a
    # number it cannot support.
    if len(usable) < MIN_SAMPLE:
        return "\n".join(lines + _refusal(live, replayed, len(usable)))

    bands = summarise(usable)
    lines.append("")
    lines.extend(_band_table(bands))
    lines.append("")
    lines.append("`needed` is the break-even win rate assuming every "
                 "non-winner loses the whole stop - the bar the app shows "
                 "per setup, and an OVERSTATED one, because this scanner "
                 "squares off at the close.")
    lines.append("`real` is that bar recomputed from what this band's "
                 "non-winners actually returned, and `edge` is measured "
                 "against it - so edge and mean net R agree in sign "
                 "instead of contradicting each other.")
    lines.append("`real` is computed from the same rows it judges, so it "
                 "answers whether a band made money, not whether the next "
                 "one will.")

    exp = expectancy(usable)
    if exp:
        lines.extend(_expectancy_lines(exp, "live setups"))
        lines.append("")
        lines.append("Expectancy is the share that won times the average "
                     "win, less the share that lost times the average "
                     "loss - computed from realised R, which cannot drift "
                     "out of step with the wins and losses it is made of.")
        lines.append("The average winner and loser are printed beside it "
                     "because a positive expectancy carried by one "
                     "outsized win is a different fact from a steady one.")

    # THE POPULATION MATTERS MORE THAN THE NUMBER. The resolver keeps
    # blocked setups on purpose, as the counterfactual, so the block
    # above describes a population the gates mostly refused - 5.4% taken
    # in the current log. What the strategy would have earned is the
    # taken rows alone.
    taken_exp = expectancy(taken)
    if taken_exp and taken_exp["n"] >= MIN_SAMPLE:
        lines.extend(_expectancy_lines(taken_exp, "TAKEN live setups"))
        lines.append("")
        lines.append("This is the strategy's own expectancy. The block "
                     "above it includes setups the gates refused, which "
                     "is the counterfactual and not the track record.")
    else:
        lines.append("")
        lines.append(f"NO EXPECTANCY FOR THE TAKEN ROWS: "
                     f"{taken_exp.get('n', 0)} of them resolved, against "
                     f"a floor of {MIN_SAMPLE}. The figures above include "
                     f"setups the gates REFUSED - the counterfactual the "
                     f"log keeps on purpose, not what the strategy would "
                     f"have earned.")

    ordered = [(band, entry["hits"] / entry["n"])
               for band, entry in sorted(bands.items())
               if entry["n"] >= 10]
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
