"""Tests for the daily-horizon study.

Guards the defects an independent review found here: a top-decile basket
that was not a decile, an excess figure that was a row-weighted mean rather
than a portfolio return, and a confidence interval computed as though 477
heavily overlapping dates were independent draws.

All three reported a number more confident or more flattering than the data
supported, which is the failure mode this file is mostly about.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import forecast_horizons as fh


def _predictions(dates: int = 6, names: int = 20, ties: bool = False):
    """Out-of-fold rows in the shape `evaluate` expects."""
    rows = []
    for day in range(dates):
        for name in range(names):
            rows.append({
                "date": date(2026, 1, 1) + timedelta(days=day * 7),
                "symbol": f"SYM{name:02d}",
                "fold": day % 2,
                # Isotonic collapses predictions into few distinct values,
                # so ties are the normal case rather than an edge one.
                "p": 0.5 if ties else name / names,
                "label": int(name >= names // 2),
                "y_short": float(name) / names,
                "mdd_short": -float(name) / names,
            })
    return pd.DataFrame(rows)


# --- the basket that was not a decile -----------------------------------

def test_the_basket_is_the_requested_fraction_even_when_predictions_tie() -> None:
    # rank(pct=True) >= 0.90 does NOT select 10% once values tie: every
    # name sharing the cut-off takes the same percentile. Isotonic
    # collapsed 62,367 predictions into 81 distinct values, and the basket
    # held a median of 2.5% of names on some dates and 20% on others while
    # the code reported a flat 10%.
    frame = _predictions(dates=4, names=20, ties=True)
    picked = fh.top_per_date(frame, 0.10)
    per_date = picked.groupby("date").size()
    assert set(per_date) == {2}, f"expected 2 of 20 per date, got {dict(per_date)}"


def test_the_basket_rounds_up_so_a_thin_date_still_contributes() -> None:
    frame = _predictions(dates=2, names=5)
    picked = fh.top_per_date(frame, 0.10)
    assert set(picked.groupby("date").size()) == {1}


def test_ties_are_broken_reproducibly_and_never_by_the_outcome() -> None:
    # The fallback tie-break was frame.columns[0]. On a frame whose first
    # column happens to be an outcome that breaks ties by the ANSWER and
    # inflates the measured return. It must refuse instead.
    frame = _predictions(ties=True)
    first = fh.top_per_date(frame, 0.10)["symbol"].tolist()
    again = fh.top_per_date(frame.sample(frac=1.0, random_state=7), 0.10)
    assert first == again["symbol"].tolist(), "selection must not depend on row order"
    with pytest.raises(KeyError, match="symbol"):
        fh.top_per_date(frame.drop(columns=["symbol"]), 0.10)


# --- the interval that was too narrow to be honest ----------------------

def test_overlapping_labels_block_together_instead_of_resampling_dates() -> None:
    # A 252-session forward return sampled every 5 sessions repeats 98% of
    # its window on the next row, so treating dates as independent invents
    # roughly 25x more observations than exist.
    values = np.random.default_rng(0).normal(0.01, 1.0, 600)
    groups = np.repeat(np.arange(120), 5)
    _, lo_one, hi_one, _ = fh.date_bootstrap(values, groups, block_size=1)
    _, lo_five, hi_five, _ = fh.date_bootstrap(values, groups, block_size=5)
    assert (hi_five - lo_five) > 0 and (hi_one - lo_one) > 0
    # And once the block exceeds what the sample can support, it refuses.
    _, lo_big, hi_big, _ = fh.date_bootstrap(values, groups, block_size=25)
    assert np.isnan(lo_big) and np.isnan(hi_big), (
        "four independent blocks cannot support an interval")


def test_the_long_horizon_explains_its_missing_interval() -> None:
    # An unexplained NaN reads as a crash. It is not: 210 out-of-fold dates
    # blocked at 51 leave 4 independent observations, below the floor.
    frame = _predictions(dates=6, names=10)
    result = fh.evaluate(frame, "short", "SHORT")
    assert "note" in result
    if np.isnan(result["lo"]):
        assert result["note"], "a missing interval must say why"
        assert "below the floor" in result["note"]


# --- the excess that no portfolio earned --------------------------------

def test_excess_is_an_equal_weight_basket_return_not_a_row_pooled_mean() -> None:
    # Dates carry different numbers of names, so pooling rows weights a
    # crowded date above a thin one and reports something no portfolio
    # earns. Constructed: 100 names on a good date, 10 on a bad one.
    rows = []
    for name in range(100):
        rows.append({"date": date(2026, 1, 1), "symbol": f"A{name:03d}",
                     "fold": 0, "p": name / 100, "label": 1,
                     "y_short": 0.10, "mdd_short": -0.01})
    for name in range(10):
        rows.append({"date": date(2026, 2, 1), "symbol": f"B{name:03d}",
                     "fold": 1, "p": name / 10, "label": 0,
                     "y_short": -0.40, "mdd_short": -0.50})
    result = fh.evaluate(pd.DataFrame(rows), "short", "SHORT")
    # Equal weight across the two dates: (0.10 + -0.40) / 2 = -0.15
    assert result["excess"] == pytest.approx(-0.15, abs=1e-9)


def test_hit_and_drawdown_describe_the_same_basket_as_excess() -> None:
    # These were row-pooled while excess was date-averaged, so three
    # numbers printed under one heading described two different portfolios.
    rows = []
    for name in range(100):
        rows.append({"date": date(2026, 1, 1), "symbol": f"A{name:03d}",
                     "fold": 0, "p": name / 100, "label": 1,
                     "y_short": 0.10, "mdd_short": -0.01})
    for name in range(10):
        rows.append({"date": date(2026, 2, 1), "symbol": f"B{name:03d}",
                     "fold": 1, "p": name / 10, "label": 0,
                     "y_short": -0.40, "mdd_short": -0.50})
    result = fh.evaluate(pd.DataFrame(rows), "short", "SHORT")
    assert result["hit"] == pytest.approx(0.5), (
        "one good date and one bad date is a 50% hit rate, not 90%")
    assert result["mdd"] == pytest.approx(-0.255, abs=1e-6)


# --- AUC, same pooling defect as the intraday study ---------------------

def test_auc_is_scored_inside_each_fold() -> None:
    frame = _predictions()
    result = fh.evaluate(frame, "short", "SHORT")
    assert not np.isnan(result["auc"])
    # Removing fold identity must yield NaN rather than a pooled figure.
    stripped = fh.evaluate(frame.drop(columns=["fold"]), "short", "SHORT")
    assert np.isnan(stripped["auc"])


# --- folds ---------------------------------------------------------------

def test_the_embargo_separates_train_from_test() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=7 * i) for i in range(200)]
    folds = fh.folds_for(dates, embargo_sessions=63)
    assert folds
    embargo = fh.embargo_positions(dates, 63)
    for train_dates, test_dates in folds:
        assert not set(train_dates) & set(test_dates)
        gap = dates.index(min(test_dates)) - dates.index(max(train_dates))
        assert gap >= embargo, f"embargo too small: {gap} < {embargo}"


def test_the_embargo_is_converted_from_sessions_to_row_positions() -> None:
    # The dataset samples every 5th session, so a 252-session horizon is
    # about 51 rows, not 252. Getting this wrong either leaks or discards.
    dates = [date(2020, 1, 1) + timedelta(days=7 * i) for i in range(200)]
    assert 30 < fh.embargo_positions(dates, 252) < 80


def test_too_few_dates_produce_no_folds() -> None:
    assert fh.folds_for([date(2026, 1, 1)], embargo_sessions=63) == []


def test_the_sampling_stride_matches_the_builder() -> None:
    # A silent divergence would mis-size every bootstrap block while
    # everything still ran, so it is imported rather than repeated.
    import forecast_universe
    assert fh.SAMPLE_EVERY == forecast_universe.SAMPLE_EVERY
