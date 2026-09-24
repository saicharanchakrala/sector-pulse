"""Tests for survival.py: streaks, drawdowns and the recovery asymmetry.

Run with: .venv\\Scripts\\python -m pytest test_survival.py -q

Where a number can be computed by hand it is asserted against the hand
figure - an enumeration of every sequence, or a closed form - so the
exact dynamic programme is checked against something other than itself.
"""
from __future__ import annotations

import itertools
import math

import pytest

import config
import survival


# --- recovery and drawdown arithmetic -------------------------------------

@pytest.mark.parametrize("drawdown,needed", [
    (0.10, 0.1111111), (0.30, 0.4285714), (0.50, 1.0), (0.0, 0.0),
])
def test_recovery_needed_is_d_over_one_minus_d(drawdown, needed) -> None:
    assert survival.recovery_needed(drawdown) == pytest.approx(needed,
                                                               abs=1e-6)


def test_a_total_loss_cannot_be_recovered() -> None:
    assert math.isinf(survival.recovery_needed(1.0))


def test_recovery_refuses_nonsense_rather_than_passing_nan() -> None:
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            survival.recovery_needed(bad)


def test_drawdown_is_linear_under_fixed_rupee_sizing() -> None:
    """k losses cost k budgets: they add, they do not compound."""
    one = survival.drawdown_after_losses(1, 1.0)
    assert one == pytest.approx(0.01)
    for k in (2, 5, 10, 20):
        assert survival.drawdown_after_losses(k, 1.0) == pytest.approx(
            k * one)
    assert survival.drawdown_after_losses(10, 2.0) == pytest.approx(0.20)
    assert survival.drawdown_after_losses(0, 1.0) == 0.0


def test_drawdown_saturates_at_the_whole_account() -> None:
    assert survival.drawdown_after_losses(30, 5.0) == 1.0


def test_the_break_even_rate_of_a_two_to_one_geometry_is_a_third() -> None:
    assert survival.break_even_win_rate(2.0) == pytest.approx(1.0 / 3.0)
    assert survival.break_even_win_rate() == pytest.approx(
        1.0 / (1.0 + config.SCAN_REWARD_RISK))


# --- exact streak probabilities --------------------------------------------

def _enumerated(k: int, n: int, win_rate: float) -> float:
    """P(a run of k losses within n trades) by listing every sequence."""
    total = 0.0
    for outcome in itertools.product((True, False), repeat=n):   # True = win
        run = longest = 0
        for won in outcome:
            run = 0 if won else run + 1
            longest = max(longest, run)
        if longest >= k:
            wins = sum(outcome)
            total += win_rate ** wins * (1.0 - win_rate) ** (n - wins)
    return total


def test_a_streak_of_one_is_any_loss_at_all() -> None:
    # P(at least one loss in n) = 1 - p^n.
    for p, n in ((0.4, 5), (1.0 / 3.0, 10), (0.9, 3)):
        assert survival.prob_losing_streak(1, n, p) == pytest.approx(
            1.0 - p ** n)


def test_a_streak_as_long_as_the_sample_is_every_trade_losing() -> None:
    # P(n losses in n) = q^n.
    for p, n in ((0.4, 5), (0.5, 8), (1.0 / 3.0, 4)):
        assert survival.prob_losing_streak(n, n, p) == pytest.approx(
            (1.0 - p) ** n)


def test_three_trades_two_losses_matches_the_hand_enumeration() -> None:
    # LLW, LLL and WLL: q^2 p + q^3 + p q^2 = q^2 (1 + p). At p = 0.5
    # that is 3 of the 8 equally likely sequences.
    assert survival.prob_losing_streak(2, 3, 0.5) == pytest.approx(3.0 / 8.0)
    p = 0.3
    q = 1.0 - p
    assert survival.prob_losing_streak(2, 3, p) == pytest.approx(
        q * q * (1.0 + p))


@pytest.mark.parametrize("k,n,p", [
    (2, 6, 0.4), (3, 8, 1.0 / 3.0), (4, 10, 0.5), (3, 12, 0.6), (5, 12, 0.2),
])
def test_the_dynamic_programme_matches_brute_force(k, n, p) -> None:
    assert survival.prob_losing_streak(k, n, p) == pytest.approx(
        _enumerated(k, n, p), abs=1e-12)


def test_a_streak_longer_than_the_sample_cannot_happen() -> None:
    assert survival.prob_losing_streak(11, 10, 0.3) == 0.0


def test_certain_wins_and_certain_losses_are_the_edges() -> None:
    assert survival.prob_losing_streak(3, 50, 1.0) == 0.0
    assert survival.prob_losing_streak(3, 50, 0.0) == 1.0


def test_more_trades_means_more_chance_of_a_streak() -> None:
    previous = 0.0
    for n in (10, 20, 50, 100, 200):
        current = survival.prob_losing_streak(8, n, 1.0 / 3.0)
        assert current >= previous
        previous = current
    assert previous > 0.5


def test_a_lower_win_rate_means_more_chance_of_a_streak() -> None:
    rates = (0.6, 0.5, 0.4, 1.0 / 3.0, 0.25)
    chances = [survival.prob_losing_streak(10, 100, p) for p in rates]
    assert chances == sorted(chances)
    assert chances[0] < chances[-1]


def test_the_streak_function_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError):
        survival.prob_losing_streak(0, 10, 0.5)
    with pytest.raises(ValueError):
        survival.prob_losing_streak(3, -1, 0.5)
    with pytest.raises(ValueError):
        survival.prob_losing_streak(3, 10, float("nan"))


# --- simulated drawdown ----------------------------------------------------

def test_the_drawdown_simulation_is_deterministic_under_a_seed() -> None:
    args = {"threshold": 0.2, "n": 100, "win_rate": 1.0 / 3.0,
            "reward_risk": 2.0, "risk_pct": 2.0, "sims": 4_000, "seed": 11}
    assert survival.prob_drawdown(**args) == survival.prob_drawdown(**args)


def test_more_risk_per_trade_means_more_chance_of_a_deep_drawdown() -> None:
    kwargs = {"n": 100, "win_rate": 1.0 / 3.0, "reward_risk": 2.0,
              "sims": 6_000, "seed": 3}
    at_one = survival.prob_drawdown(0.5, risk_pct=1.0, **kwargs)
    at_five = survival.prob_drawdown(0.5, risk_pct=5.0, **kwargs)
    assert at_five > at_one
    # Over 100 no-edge trades a 50% fall from peak came out near 60% of
    # paths at 5% risk and about 2 in 10,000 at 1%. The bounds are wide so
    # the assertion is about the ORDER of magnitude, not the sample.
    assert at_five > 0.3
    assert at_one < 0.01


def test_a_drawdown_that_needs_more_losses_than_trades_never_happens() -> None:
    # Ten trades at 1% can lose at most 10%, so 20% is out of reach.
    assert survival.prob_drawdown(0.2, 10, 0.0, 2.0, 1.0, sims=500) == 0.0


def test_certain_losses_always_reach_a_reachable_drawdown() -> None:
    # Twenty losses at 1% is exactly 20%, measured from the starting peak.
    assert survival.prob_drawdown(0.2, 20, 0.0, 2.0, 1.0, sims=500) == 1.0


# --- drawdown is measured from the running PEAK, not from the start --------
#
# The two tests above cannot tell the definitions apart: with every trade
# a loss the peak IS the start. These can.

def test_a_path_that_rises_then_falls_is_measured_from_its_high() -> None:
    # 1.0 -> 1.1 -> 1.2 -> 1.0 -> 0.9. From the 1.2 peak that is 0.3 off,
    # 25%; from the 1.0 start it is only 10%.
    assert survival.max_drawdown([1.1, 1.2, 1.0, 0.9]) == pytest.approx(0.25)


def test_a_path_that_only_falls_is_measured_from_the_start() -> None:
    # The start counts as a peak, so the first point's own level is not.
    assert survival.max_drawdown([0.9, 0.8]) == pytest.approx(0.2)


def test_a_fall_recovered_later_still_counts_in_full() -> None:
    # Down 1.5 -> 1.2 (20%) and then back to a new high: still 20%.
    assert survival.max_drawdown([1.5, 1.2, 1.6]) == pytest.approx(0.2)


def test_no_trades_is_no_drawdown_and_a_bad_start_is_refused() -> None:
    assert survival.max_drawdown([]) == 0.0
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            survival.max_drawdown([1.0], start=bad)


def test_the_simulation_counts_falls_from_the_peak() -> None:
    """Two trades risking 50% at 1:1, a fair coin, a 30% threshold.

    By hand over the four equally likely paths:
      WW  1.5, 2.0   never falls                      neither
      WL  1.5, 1.0   33% off the 1.5 peak, 0% down     peak only
      LW  0.5, 1.0   50% off the start                 both
      LL  0.5, 0.0   100% off the start                both
    From the peak that is 3 of 4 paths, 0.75; from the start 2 of 4, 0.5.
    At 20,000 paths the standard error is about 0.003, so 0.02 either side
    of 0.75 cannot be reached by a from-start count.
    """
    chance = survival.prob_drawdown(0.3, 2, 0.5, 1.0, 50.0, sims=20_000,
                                    seed=5)
    assert chance == pytest.approx(0.75, abs=0.02)


# --- the display table -----------------------------------------------------

def test_the_table_defaults_to_the_no_edge_win_rate_and_says_so() -> None:
    table = survival.streak_table(1.0, sims=2_000)
    assert table.win_rate == pytest.approx(
        survival.break_even_win_rate(config.SCAN_REWARD_RISK))
    assert table.win_rate_label == survival.NO_EDGE_LABEL
    assert [row.streak for row in table.rows] == [5, 10, 15, 20]


def test_the_table_rows_are_the_functions_they_summarise() -> None:
    table = survival.streak_table(2.0, win_rate=0.4, n=60, streaks=(3, 6),
                                  sims=2_000)
    for row in table.rows:
        assert row.drawdown == pytest.approx(row.streak * 0.02)
        assert row.recovery == pytest.approx(
            survival.recovery_needed(row.drawdown))
        assert row.probability == pytest.approx(
            survival.prob_losing_streak(row.streak, 60, 0.4))
    assert table.win_rate_label == "measured"
    assert 0.0 <= table.prob_ruin_drawdown <= 1.0


def test_the_text_footer_names_the_win_rate_it_assumed() -> None:
    lines = survival.text_lines(survival.streak_table(1.0, sims=1_000))
    assert survival.NO_EDGE_LABEL in lines[0]
    assert any("50% fall from peak" in line for line in lines)


def test_a_tiny_probability_is_not_printed_as_zero() -> None:
    assert survival.format_percent(0.0002) == "<0.1%"
    assert survival.format_percent(0.0) == "0.0%"
    assert survival.format_percent(math.inf) == "unrecoverable"
    assert survival.format_percent(math.nan) == "n/a"
