"""Losing streaks, drawdowns, and what it takes to climb back out of them.

The arithmetic every position-sizing rule rests on, computed rather than
asserted so the scan can print it beside the risk it is about to take:

  * A LOSS NEEDS A BIGGER GAIN TO RECOVER. Down 10% needs +11.1% to get
    back, down 30% needs +42.9%, down 50% needs +100%. recovery_needed.
  * STREAKS ARE NORMAL, NOT BAD LUCK. At a 33.3% win rate - the
    break-even rate of this scanner's 2:1 geometry - a run of five
    straight losses inside a hundred trades is all but certain (99.7%)
    and a run of ten turns up 43.5% of the time. prob_losing_streak
    computes those exactly.
  * THE RISK PER TRADE DECIDES WHETHER A STREAK IS SURVIVABLE. Ten losses
    at 1% is a 10% drawdown; at 5% it is half the account. Over a hundred
    no-edge trades a 50% fall from peak came out at 0.02% of paths at 1%
    risk, 8.7% at 2% and 59.7% at 5% (prob_drawdown, default seed).

THE SIZING MODEL, which every function here shares with levels.py: each
trade risks a FIXED RUPEE amount of a FIXED capital (capital * risk_pct /
100), and since the sizer now holds charges inside that budget, a full
loss costs the whole budget and no more. Losses therefore add, they do not
compound: k straight losses cost k * risk_pct percent of the starting
capital. A sizer that re-bases on current equity would lose less per trade
as it shrinks; this one does not, which makes these figures the harsher
and more honest ones for it.

Pure and deterministic: numpy only, no I/O, and the one Monte Carlo takes
an explicit seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config

# Shown beside the default win rate everywhere it is displayed. The rule
# has been measured to have no directional edge (see scan_intraday's
# docstring: 56,825 signals, indistinguishable from a coin flip), and the
# honest default for "how bad can a streak get" is the rate at which the
# geometry merely breaks even before costs.
NO_EDGE_LABEL = "a system with no edge, which is what has been measured"

# The drawdown that stands in for ruin in the summary line: from half the
# account gone, the other half has to double just to get back.
RUIN_DRAWDOWN = 0.5

# Default Monte Carlo size. At a probability near 0.5 the standard error
# is sqrt(0.25 / 20,000) = 0.0035, which is finer than the table prints.
DEFAULT_SIMS = 20_000
DEFAULT_SEED = 20260924

# Simulated paths are drawn in chunks of this many, so memory stays
# bounded (about 8 bytes x chunk x trades) however many are asked for.
_SIM_CHUNK = 5_000


def _check_rate(name: str, value: float) -> float:
    """A probability in [0, 1], or ValueError. NaN is refused, not passed."""
    rate = float(value)
    if not (math.isfinite(rate) and 0.0 <= rate <= 1.0):
        raise ValueError(f"{name} must be a probability in [0, 1], got "
                         f"{value!r}")
    return rate


def _check_risk_pct(risk_pct: float) -> float:
    """A finite, non-negative percent, or ValueError."""
    pct = float(risk_pct)
    if not (math.isfinite(pct) and pct >= 0.0):
        raise ValueError(f"risk_pct must be a finite percent >= 0, got "
                         f"{risk_pct!r}")
    return pct


def break_even_win_rate(reward_risk: "float | None" = None) -> float:
    """Win rate at which a reward:risk geometry breaks even, before costs.

    p * R = (1 - p) * 1  gives  p = 1 / (1 + R). At the configured 2:1 that
    is 33.3%. Used as the default win rate in streak_table because it is
    what a rule with no edge wins, and no edge is what has been measured.
    """
    ratio = float(config.SCAN_REWARD_RISK if reward_risk is None
                  else reward_risk)
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError(f"reward_risk must be positive, got {reward_risk!r}")
    return 1.0 / (1.0 + ratio)


def drawdown_after_losses(k: int, risk_pct: float) -> float:
    """Fraction of capital lost after k consecutive full losses.

    LINEAR, k * risk_pct / 100, because the sizer risks a fixed rupee
    amount of a fixed capital rather than a fraction of what is left, and
    because charges are now inside that amount, so a full loss is exactly
    one budget. Capped at 1.0: past that the account is gone and "more
    than all of it" is not a drawdown.
    """
    if int(k) != k or k < 0:
        raise ValueError(f"k must be a whole number >= 0, got {k!r}")
    return min(1.0, int(k) * _check_risk_pct(risk_pct) / 100.0)


def recovery_needed(drawdown: float) -> float:
    """Gain needed to get back to the peak after losing `drawdown` of it.

    d / (1 - d): losing 10% needs 11.1%, 30% needs 42.9%, 50% needs 100%.
    The asymmetry is the whole argument for small risk per trade. Returns
    inf at a total loss, where no gain recovers anything.
    """
    d = float(drawdown)
    if not (math.isfinite(d) and 0.0 <= d <= 1.0):
        raise ValueError(f"drawdown must be a fraction in [0, 1], got "
                         f"{drawdown!r}")
    if d >= 1.0:
        return math.inf
    return d / (1.0 - d)


def prob_losing_streak(k: int, n: int, win_rate: float) -> float:
    """Probability of at least one run of k consecutive losses within n trades.

    EXACT, by dynamic programming over the length of the current losing
    run, not by simulation. The state is the probability of having seen
    no k-run yet and currently sitting on a run of j losses, j = 0..k-1;
    a win sends every state to j = 0, a loss moves j to j + 1, and
    reaching k is absorbed as "the streak happened". O(n * k).

    Trades are treated as independent with a constant win rate. Real
    outcomes cluster - a bad regime produces several losers in a row -
    so if anything this UNDERSTATES how often long streaks arrive.
    """
    if int(k) != k or k < 1:
        raise ValueError(f"k must be a whole number >= 1, got {k!r}")
    if int(n) != n or n < 0:
        raise ValueError(f"n must be a whole number >= 0, got {n!r}")
    k, n = int(k), int(n)
    p = _check_rate("win_rate", win_rate)
    q = 1.0 - p
    if k > n:
        return 0.0
    state = np.zeros(k)
    state[0] = 1.0
    for _ in range(n):
        survived = state.sum()
        shifted = np.empty(k)
        shifted[0] = p * survived          # any win resets the run
        shifted[1:] = q * state[:-1]       # a loss extends it; state[k-1]
        state = shifted                    # extended reaches k: absorbed
    return float(min(1.0, max(0.0, 1.0 - state.sum())))


def _deepest_falls(paths: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Deepest fall from the running peak along each row of `paths`.

    Rows are equity paths AFTER each trade, without the starting value;
    `start` is the equity before the first trade and counts as a peak, so
    a path that only ever falls is measured from it rather than from its
    first (already lower) point. Returns one fraction per row: the largest
    (peak - equity) / peak seen along it, where peak is the highest value
    so far. A row with no trades has fallen nowhere and returns 0.0.

    The one place prob_drawdown's "fall" is defined, so the PEAK - not the
    start - is what a test can pin on a hand-built path via max_drawdown.
    """
    rows = np.atleast_2d(np.asarray(paths, dtype=float))
    if rows.shape[1] == 0:
        return np.zeros(rows.shape[0])
    peak = np.maximum(np.maximum.accumulate(rows, axis=1), float(start))
    return ((peak - rows) / peak).max(axis=1)


def max_drawdown(path, start: float = 1.0) -> float:
    """Deepest fall from the running peak along ONE equity path, as a fraction.

    `path` is the equity after each trade and `start` the equity before
    the first. Measured from the highest point reached so far, not from
    the start: 1.0 -> 1.2 -> 0.9 is a 25% drawdown (0.3 off a 1.2 peak),
    although it is only 10% below where it began. That is the definition
    prob_drawdown counts, and the one a trader lives through - a fall
    from a high is felt in full whether or not the account is still up.
    """
    start = float(start)
    if not (math.isfinite(start) and start > 0.0):
        raise ValueError(f"start must be a positive equity, got {start!r}")
    return float(_deepest_falls(np.asarray(path, dtype=float).reshape(1, -1),
                                start)[0])


def prob_drawdown(threshold: float, n: int, win_rate: float,
                  reward_risk: float, risk_pct: float,
                  sims: int = DEFAULT_SIMS,
                  seed: int = DEFAULT_SEED) -> float:
    """Probability equity falls `threshold` below its running peak within n trades.

    Seeded numpy Monte Carlo, so the same inputs give the same answer.
    Equity starts at 1.0 (the capital). Every trade risks the same
    risk_pct / 100 of that STARTING capital - the fixed-rupee sizing
    levels.py uses - winning reward_risk times it with probability
    win_rate and losing it otherwise. A path counts once it is ever
    `threshold` below its own peak, peak included the start.

    An idealisation in the system's favour: a winner here pays the full
    reward_risk multiple, while a real winner still pays its charges.
    """
    dd = float(threshold)
    if not (math.isfinite(dd) and 0.0 < dd <= 1.0):
        raise ValueError(f"threshold must be a fraction in (0, 1], got "
                         f"{threshold!r}")
    if int(n) != n or n < 0:
        raise ValueError(f"n must be a whole number >= 0, got {n!r}")
    if int(sims) != sims or sims < 1:
        raise ValueError(f"sims must be a whole number >= 1, got {sims!r}")
    ratio = float(reward_risk)
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError(f"reward_risk must be positive, got {reward_risk!r}")
    p = _check_rate("win_rate", win_rate)
    step = _check_risk_pct(risk_pct) / 100.0
    n, sims = int(n), int(sims)
    if n == 0 or step == 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    hits = 0
    remaining = sims
    while remaining > 0:
        batch = min(_SIM_CHUNK, remaining)
        remaining -= batch
        wins = rng.random((batch, n)) < p
        equity = 1.0 + np.cumsum(np.where(wins, step * ratio, -step), axis=1)
        # From the RUNNING PEAK, with the start counted as one, so a path
        # that only ever falls is measured from 1.0 - see _deepest_falls.
        falls = _deepest_falls(equity, start=1.0)
        hits += int(np.count_nonzero(falls >= dd - 1e-12))
    return hits / sims


@dataclass(frozen=True)
class StreakRow:
    """One line of the streak table."""

    streak: int                 # consecutive full losses
    drawdown: float             # fraction of capital they cost
    recovery: float             # gain needed to get back, as a fraction
    probability: float          # of at least one such run within `trades`


@dataclass(frozen=True)
class StreakTable:
    """Losing streaks at one risk level, with the ruin-style summary."""

    risk_pct: float
    win_rate: float
    win_rate_label: str
    reward_risk: float
    trades: int
    rows: tuple
    ruin_drawdown: float
    prob_ruin_drawdown: float


def streak_table(risk_pct: float, win_rate: "float | None" = None,
                 n: int = 100, streaks: tuple = (5, 10, 15, 20),
                 reward_risk: "float | None" = None,
                 win_rate_label: "str | None" = None,
                 sims: int = DEFAULT_SIMS,
                 seed: int = DEFAULT_SEED) -> StreakTable:
    """The losing-streak table for one risk level, ready to display.

    With no win_rate it uses break_even_win_rate(reward_risk) and labels it
    NO_EDGE_LABEL, because that is the honest default for a rule measured
    to have no edge. Pass a measured rate (and a label saying where it came
    from) to see the table for that instead.

    Each row: the streak length, the drawdown it costs at this risk, the
    gain needed to recover, and the exact probability of at least one such
    streak within n trades. The summary is the Monte Carlo probability of
    a RUIN_DRAWDOWN (50%) fall from peak within the same n trades.
    """
    ratio = float(config.SCAN_REWARD_RISK if reward_risk is None
                  else reward_risk)
    if win_rate is None:
        rate = break_even_win_rate(ratio)
        label = NO_EDGE_LABEL if win_rate_label is None else win_rate_label
    else:
        rate = _check_rate("win_rate", win_rate)
        label = "measured" if win_rate_label is None else win_rate_label
    pct = _check_risk_pct(risk_pct)
    rows = tuple(
        StreakRow(
            streak=int(k),
            drawdown=drawdown_after_losses(k, pct),
            recovery=recovery_needed(drawdown_after_losses(k, pct)),
            probability=prob_losing_streak(k, n, rate),
        )
        for k in streaks)
    return StreakTable(
        risk_pct=pct, win_rate=rate, win_rate_label=label,
        reward_risk=ratio, trades=int(n), rows=rows,
        ruin_drawdown=RUIN_DRAWDOWN,
        prob_ruin_drawdown=prob_drawdown(RUIN_DRAWDOWN, n, rate, ratio, pct,
                                         sims=sims, seed=seed),
    )


def format_percent(fraction: float) -> str:
    """A fraction as a percent for a table cell, with inf spelled out.

    A small but non-zero probability prints as "<0.1%", not "0.0%": two
    paths in ten thousand is rare, and it is not the same fact as never.
    """
    if math.isnan(fraction):
        return "n/a"
    if math.isinf(fraction):
        return "unrecoverable"
    if 0.0 < fraction < 0.0005:
        return "<0.1%"
    return f"{fraction * 100:.1f}%"


def text_lines(table: StreakTable) -> list[str]:
    """The table as plain-text lines, for the CLI report footer."""
    lines = [
        f"LOSING STREAKS at {table.risk_pct:g}% risk per trade, over "
        f"{table.trades} trades at a {table.win_rate * 100:.1f}% win rate "
        f"({table.win_rate_label}):",
        f"  {'Streak':>6}  {'Drawdown':>9}  {'To recover':>13}  "
        f"{'Chance in ' + str(table.trades):>14}",
    ]
    for row in table.rows:
        lines.append(
            f"  {row.streak:>6}  {format_percent(row.drawdown):>9}  "
            f"{format_percent(row.recovery):>13}  "
            f"{format_percent(row.probability):>14}")
    lines.append(
        f"  Chance of a {table.ruin_drawdown * 100:.0f}% fall from peak "
        f"within {table.trades} trades: "
        f"{format_percent(table.prob_ruin_drawdown)}.")
    return lines
