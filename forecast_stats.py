"""Statistics for judging a trading rule, in one place.

These four functions decide whether a measured result is believed, and
during development each of them was reimplemented separately in four
scripts. Two of the errors that produced convincing false results lived
here, so they belong in one module with tests rather than copied per
experiment:

  * A permutation p computed as beats/shuffles instead of
    (1 + beats) / (1 + shuffles). With ten shuffles the first form reports
    0.0000 when none beat the observation, claiming a precision the test
    cannot deliver; the exact form floors at 1/11 = 0.091, which does not
    clear 0.05 and reverses the conclusion.
  * A confidence interval that resampled individual signals. Signals within
    one session share that session's market move, so the session is the
    unit of independence. Resampling rows shrinks the interval until noise
    looks significant.

Nothing here knows about markets. Given values, the group each value belongs
to, and a count of hypotheses tried, it reports how much of the result
survives.
"""
from __future__ import annotations

import numpy as np

DEFAULT_DRAWS = 2000
DEFAULT_SEED = 20260909


def block_bootstrap(values, groups, draws: int = DEFAULT_DRAWS,
                    seed: int = DEFAULT_SEED) -> tuple:
    """Mean and 95% interval, resampling whole GROUPS rather than rows.

    `groups` labels the unit of independence - a session-day for intraday
    signals, a calendar date for cross-sectional ones. Every value sharing a
    group is resampled together, because they share whatever moved that day.

    Returns (mean, low, high, p) where p is the one-sided bootstrap
    probability that the true mean is not above zero. With fewer than five
    groups the interval is not estimable and comes back as NaN rather than
    as a falsely narrow number.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if values.size == 0 or values.size != groups.size:
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(groups, kind="stable")
    values, groups = values[order], groups[order]
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    blocks = np.split(values, starts[1:])
    observed = float(values.mean())
    if len(blocks) < 5:
        return observed, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    count = len(blocks)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        pick = rng.integers(0, count, size=count)
        means[draw] = np.concatenate([blocks[p] for p in pick]).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return observed, float(low), float(high), float((means <= 0.0).mean())


def permutation_p(observed: float, null_values) -> float:
    """Exact one-sided permutation p: (1 + beats) / (1 + shuffles).

    The observed statistic counts as one of its own reference draws, which
    is what makes the test exact and what stops it ever reporting zero. The
    consequence worth remembering: N shuffles cannot produce a p below
    1 / (N + 1), so ten shuffles can never show significance at 0.05
    however large the effect. Use permutation_floor to check before
    trusting a p.
    """
    null_values = np.asarray(list(null_values), dtype=float)
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    beats = int(np.sum(null_values >= observed))
    return (1 + beats) / (1 + null_values.size)


def permutation_floor(shuffles: int) -> float:
    """Smallest p that `shuffles` permutations can possibly report."""
    return 1.0 / (1 + shuffles) if shuffles > 0 else float("nan")


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> list:
    """Which hypotheses survive at `alpha`, controlling false discovery.

    Every hypothesis actually tried must be passed in, including the ones
    that failed. Dropping the failures and correcting only the survivors
    defeats the purpose: with twelve rule variants tried, one clearing
    p < 0.05 alone is the expected outcome under a pure null.
    """
    pvalues = list(pvalues)
    total = len(pvalues)
    if total == 0:
        return []
    order = sorted(range(total), key=lambda i: pvalues[i])
    largest = 0
    for rank, index in enumerate(order, start=1):
        if pvalues[index] <= alpha * rank / total:
            largest = rank
    survive = [False] * total
    for rank, index in enumerate(order, start=1):
        if rank <= largest:
            survive[index] = True
    return survive


def required_hit_rate(reward: float, cost_in_r: float) -> float:
    """Hit rate a reward:risk geometry needs before costs are covered.

    Expected value in units of risk is p * reward - (1 - p) - cost, so
    breakeven sits at (1 + cost) / (1 + reward). Worth computing before
    modelling anything: if the base rate is twenty points below this, no
    realistic model closes the gap and the honest answer is to change the
    geometry or stop.
    """
    if reward <= -1.0:
        return float("nan")
    return (1.0 + cost_in_r) / (1.0 + reward)
