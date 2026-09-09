"""Tests for the statistics that decide whether a result is believed.

Every test here is a regression test for a mistake that actually produced a
convincing false result during development, or for a property the module's
docstrings claim.
"""
from __future__ import annotations

import numpy as np
import pytest

import forecast_stats as fs


# --- the permutation p-value ---------------------------------------------

def test_permutation_p_can_never_report_zero() -> None:
    # The bug: beats/shuffles returns 0.0000 when no shuffle beats the
    # observation, which reads as overwhelming significance. The exact form
    # floors at 1/(N+1) because the observation counts as its own draw.
    nulls = [-0.12] * 10
    assert fs.permutation_p(0.5, nulls) == pytest.approx(1 / 11)
    assert fs.permutation_p(0.5, nulls) > 0.05


def test_ten_shuffles_cannot_show_significance_at_five_percent() -> None:
    # This is why the intraday result reported p=0.0000 and was actually
    # p=0.091: the test was arithmetically incapable of the claim.
    assert fs.permutation_floor(10) == pytest.approx(1 / 11)
    assert fs.permutation_floor(10) > 0.05
    assert fs.permutation_floor(30) < 0.05
    # The exact boundary: 1/(N+1) < 0.05 needs N >= 20. Nineteen shuffles
    # floor at precisely 0.05, which does not CLEAR 0.05, so twenty is the
    # real minimum for a significant permutation result.
    assert fs.permutation_floor(19) == pytest.approx(0.05)
    assert fs.permutation_floor(20) < 0.05


def test_permutation_p_counts_ties_against_the_observation() -> None:
    # A shuffle equalling the observation is evidence against it, not for
    # it, so >= is the correct comparison.
    nulls = [1.0, 1.0, 0.0, 0.0]
    assert fs.permutation_p(1.0, nulls) == pytest.approx(3 / 5)


def test_permutation_p_rises_as_more_shuffles_beat_the_result() -> None:
    nulls = [0.0, 0.1, 0.2, 0.3, 0.4]
    weak = fs.permutation_p(0.05, nulls)
    strong = fs.permutation_p(0.5, nulls)
    assert strong < weak
    assert strong == pytest.approx(1 / 6)


def test_permutation_p_is_nan_without_shuffles() -> None:
    assert np.isnan(fs.permutation_p(1.0, []))
    assert np.isnan(fs.permutation_p(float("nan"), [1.0, 2.0]))


# --- the block bootstrap --------------------------------------------------

def test_grouped_resampling_gives_a_wider_interval_than_row_resampling() -> None:
    # The bug: resampling rows treats correlated same-day signals as
    # independent draws and shrinks the interval until noise looks real.
    # Construct data where every value in a day is identical, so the day
    # carries all the information and rows carry none beyond it.
    rng = np.random.default_rng(1)
    day_effects = rng.normal(0.0, 1.0, size=40)
    values = np.repeat(day_effects, 25)
    grouped = np.repeat(np.arange(40), 25)
    rows = np.arange(values.size)
    _, g_lo, g_hi, _ = fs.block_bootstrap(values, grouped, draws=800)
    _, r_lo, r_hi, _ = fs.block_bootstrap(values, rows, draws=800)
    assert (g_hi - g_lo) > 4 * (r_hi - r_lo)


def test_bootstrap_recovers_a_clear_positive_mean() -> None:
    values = np.full(500, 0.5)
    groups = np.repeat(np.arange(50), 10)
    mean, low, high, p = fs.block_bootstrap(values, groups, draws=500)
    assert mean == pytest.approx(0.5)
    assert low == pytest.approx(0.5) and high == pytest.approx(0.5)
    assert p == 0.0


def test_bootstrap_reports_a_high_p_for_a_negative_mean() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(-0.3, 0.1, size=400)
    groups = np.repeat(np.arange(40), 10)
    mean, _, _, p = fs.block_bootstrap(values, groups, draws=800)
    assert mean < 0
    assert p > 0.95


def test_bootstrap_refuses_to_invent_an_interval_from_few_groups() -> None:
    # Four groups cannot support a 95% interval. Returning NaN is honest;
    # returning a narrow number would be worse than returning nothing.
    values = np.array([1.0, 2.0, 3.0, 4.0])
    mean, low, high, p = fs.block_bootstrap(values, np.arange(4))
    assert mean == pytest.approx(2.5)
    assert np.isnan(low) and np.isnan(high) and np.isnan(p)


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=300)
    groups = np.repeat(np.arange(30), 10)
    first = fs.block_bootstrap(values, groups, draws=400)
    second = fs.block_bootstrap(values, groups, draws=400)
    assert first == second


def test_bootstrap_handles_empty_and_mismatched_input() -> None:
    assert all(np.isnan(x) for x in fs.block_bootstrap([], []))
    assert all(np.isnan(x) for x in fs.block_bootstrap([1.0, 2.0], [1]))


def test_bootstrap_does_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=200)
    groups = np.repeat(np.arange(20), 10)
    shuffle = rng.permutation(values.size)
    straight = fs.block_bootstrap(values, groups, draws=400)
    jumbled = fs.block_bootstrap(values[shuffle], groups[shuffle], draws=400)
    assert straight[0] == pytest.approx(jumbled[0])


# --- multiple testing -----------------------------------------------------

def test_benjamini_hochberg_rejects_a_lone_lucky_winner() -> None:
    # Twelve geometries were tried. One at p=0.04 is the expected outcome
    # under a pure null and must not survive.
    pvalues = [0.04] + [0.5] * 11
    assert fs.benjamini_hochberg(pvalues) == [False] * 12


def test_benjamini_hochberg_keeps_a_genuinely_strong_result() -> None:
    pvalues = [0.0001, 0.6, 0.7, 0.8]
    survive = fs.benjamini_hochberg(pvalues)
    assert survive[0] is True and not any(survive[1:])


def test_benjamini_hochberg_keeps_a_consistent_family() -> None:
    pvalues = [0.001, 0.002, 0.003, 0.004]
    assert all(fs.benjamini_hochberg(pvalues))


def test_benjamini_hochberg_handles_no_hypotheses() -> None:
    assert fs.benjamini_hochberg([]) == []


def test_dropping_failures_before_correcting_defeats_the_correction() -> None:
    # Documents the misuse the docstring warns about: correcting only the
    # survivor passes it, correcting the full family does not.
    full = [0.04] + [0.5] * 11
    assert fs.benjamini_hochberg(full)[0] is False
    assert fs.benjamini_hochberg([0.04])[0] is True


# --- the cost gate --------------------------------------------------------

def test_required_hit_rate_matches_the_measured_intraday_case() -> None:
    # Measured: a 2:1 target with cost 0.2537 R needed a 41.8% hit rate
    # against a base rate of 28.6%, a gap no realistic model closes.
    assert fs.required_hit_rate(2.0, 0.2537) == pytest.approx(0.4179, abs=1e-4)


def test_required_hit_rate_is_one_third_at_two_to_one_without_costs() -> None:
    assert fs.required_hit_rate(2.0, 0.0) == pytest.approx(1 / 3)


def test_a_wider_stop_lowers_the_required_hit_rate() -> None:
    # The whole cost argument in one assertion: the same fixed charge over
    # a bigger risk unit is a smaller fraction of it, so a wide stop needs
    # less skill for the same geometry.
    tight = fs.required_hit_rate(2.0, 0.2537)
    wide = fs.required_hit_rate(2.0, 0.0846)
    assert wide < tight
    assert tight - wide == pytest.approx(0.0564, abs=1e-3)
