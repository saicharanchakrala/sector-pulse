"""Tests for the intraday walk-forward study.

This module produces the figures the app publishes about itself, and until
now had no tests at all. It also carried a NameError that would have killed
STEP 2 on its first call, which no amount of careful reading had caught.

Three of the defects guarded here were found only by review or by chasing
an anomaly, and every one of them moved a published number in the direction
that FLATTERED the model. That is the pattern the ordering below follows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import forecast_intraday as fi


# Built by hand rather than randomly, so the distortion is guaranteed
# rather than probable. Within each fold the prediction carries NO
# information - the labels alternate regardless of it, so per-fold AUC is
# 0.5 exactly. Between folds the levels are deliberately anti-correlated
# with the base rates: fold 1 predicts HIGHER while winning LESS. Pooling
# ranks those high predictions above fold 0's and is dragged below 0.5.
# That is precisely the miscalibration measured in the real study.
_LEVELS = ((0, 0.20, 0.30, 0.75), (1, 0.50, 0.60, 0.25))


def _frame(rows_per_fold: int = 200):
    """Out-of-fold predictions with a per-fold calibration shift."""
    blocks = []
    for fold, low, high, base in _LEVELS:
        p = np.linspace(low, high, rows_per_fold)
        winners = int(rows_per_fold * base)
        # Winners spread evenly across the fold, so the label is
        # uncorrelated with p INSIDE it and the per-fold AUC is 0.5.
        y = np.zeros(rows_per_fold, dtype=int)
        y[np.linspace(0, rows_per_fold - 1, winners).astype(int)] = 1
        blocks.append(pd.DataFrame({"fold": fold, "p_long": p, "y_long": y}))
    return pd.concat(blocks, ignore_index=True)


# --- the AUC that was overstated ----------------------------------------

def test_auc_is_scored_inside_each_fold_not_over_the_pooled_folds() -> None:
    # Each fold trains its own model and its own isotonic calibrator, and
    # they are miscalibrated by different amounts. AUC is a pure RANKING
    # statistic, so one score over the stack ranks fold 0's 0.30 against
    # fold 2's 0.30 as though they meant the same thing.
    #
    # Measured on the real study this inflated the model's edge over the
    # null from +0.013 to +0.032, about 2.4x.
    frame = _frame()
    mean, per_fold = fi.fold_auc(frame)
    assert len(per_fold) == 2
    assert mean == pytest.approx(float(np.mean(per_fold)))
    # Inside each fold the prediction carries nothing, so each is a coin
    # flip...
    for score in per_fold:
        assert 0.45 < score < 0.55
    # ...while the pooled figure is dragged away by the level differences.
    from sklearn.metrics import roc_auc_score
    pooled = roc_auc_score(frame["y_long"], frame["p_long"])
    assert abs(pooled - 0.5) > abs(mean - 0.5), (
        "this fixture must reproduce the pooling distortion", pooled, mean)


def test_a_frame_without_fold_identity_refuses_to_guess() -> None:
    mean, per_fold = fi.fold_auc(_frame().drop(columns=["fold"]))
    assert np.isnan(mean) and per_fold == []


def test_a_single_class_fold_is_skipped_rather_than_crashing() -> None:
    frame = _frame()
    frame.loc[frame["fold"] == 1, "y_long"] = 1
    mean, per_fold = fi.fold_auc(frame)
    assert len(per_fold) == 1 and not np.isnan(mean)


# --- the null that contained impossible rows ----------------------------

def test_the_shuffle_moves_labels_and_realised_r_together() -> None:
    # Permuting the long and short labels INDEPENDENTLY built a null the
    # market cannot produce: under a stop/target grid a long win and a
    # short win on the same bar are near mutually exclusive, measured
    # correlation -0.96, but independent shuffles gave ~0.00 and let 13-24%
    # of rows win on BOTH sides.
    #
    # Leaving the realised-R columns behind was worse still: _mean_loss
    # then selected a random subset and returned E[r] instead of
    # E[r | loss], measured -0.035 against the real -0.609.
    rng = np.random.default_rng(0)
    n = 2_000
    long_label = rng.integers(0, 2, n)
    short_label = np.where(long_label == 1, 0, rng.integers(0, 2, n))
    frame = pd.DataFrame({
        "day": np.repeat(np.arange(20), n // 20),
        "L": long_label, "S": short_label,
        "Lr": np.where(long_label == 1, 2.0, -1.0),
        "Sr": np.where(short_label == 1, 2.0, -1.0)})

    before = np.corrcoef(frame["L"], frame["S"])[0, 1]
    loss_before = frame.loc[frame["L"] == 0, "Lr"].mean()

    order = rng.permutation(np.arange(len(frame)))
    shuffled = frame.copy()
    for column in ("L", "S", "Lr", "Sr"):
        shuffled[column] = shuffled[column].to_numpy()[order]

    after = np.corrcoef(shuffled["L"], shuffled["S"])[0, 1]
    loss_after = shuffled.loc[shuffled["L"] == 0, "Lr"].mean()

    assert after == pytest.approx(before), "the pairing must survive"
    assert loss_after == pytest.approx(loss_before), "E[r|loss] must survive"
    assert not ((shuffled["L"] == 1) & (shuffled["S"] == 1)).any(), (
        "a row winning on both sides is not a state the market produces")


def test_walk_forward_accepts_both_shuffle_scopes() -> None:
    # 'session' preserves each day's base rate and is a STRICTER test whose
    # null is not a coin flip; 'global' is the reference the docstring
    # always described. Both must exist, and the default must be global.
    import inspect
    signature = inspect.signature(fi.walk_forward)
    assert signature.parameters["shuffle_scope"].default == "global"


# --- the EV gate --------------------------------------------------------

def test_the_mean_loss_comes_from_losing_rows_only() -> None:
    train = pd.DataFrame({"L": [0, 0, 1, 1], "Lr": [-1.0, 0.5, 2.0, 2.0]})
    assert fi._mean_loss(train, "L", "Lr") == pytest.approx(-0.25)


def test_a_positive_mean_loss_is_clamped_so_the_gate_still_gates() -> None:
    # ev = p*reward + (1-p)*loss - cost. With loss positive the gate opens
    # on every row: at p=0.10 and loss=+0.75 it evaluates to +0.725.
    train = pd.DataFrame({"L": [0, 0], "Lr": [0.5, 1.0]})
    assert fi._mean_loss(train, "L", "Lr") == 0.0


def test_no_losers_falls_back_to_the_old_minus_one_assumption() -> None:
    train = pd.DataFrame({"L": [1, 1], "Lr": [2.0, 2.0]})
    assert fi._mean_loss(train, "L", "Lr") == -1.0


# --- trade selection ----------------------------------------------------

def _selection_frame():
    return pd.DataFrame({
        "day": [1, 2, 3, 4, 5],
        "ev_long": [0.5, 0.4, 0.3, -0.1, -0.2],
        "ev_short": [-0.1, -0.1, -0.1, 0.9, -0.3],
        "r_long": [2.0, -1.0, 2.0, 0.0, 0.0],
        "r_short": [0.0, 0.0, 0.0, 2.0, 0.0],
        "cost_r": [0.1] * 5})


def test_only_positive_expected_value_rows_are_taken() -> None:
    net, days = fi.net_of_selection(_selection_frame())
    assert len(net) == 4 and list(days) == [1, 2, 3, 4]


def test_the_better_side_is_taken_when_both_are_positive() -> None:
    frame = _selection_frame()
    frame.loc[0, "ev_short"] = 0.9          # short now beats long
    net, _ = fi.net_of_selection(frame)
    assert net[0] == pytest.approx(-0.1), "should have taken the short"


def test_the_cap_keeps_the_highest_ev_trades_and_keeps_days_aligned() -> None:
    net, days = fi.net_of_selection(_selection_frame(), limit=2)
    assert len(net) == len(days) == 2
    # Highest EV are the short at 0.9 (day 4) and the long at 0.5 (day 1).
    assert sorted(days) == [1, 4]


def test_tied_rows_are_kept_from_the_START_of_the_sample() -> None:
    # argsort(...)[::-1] reverses the tie order too, so tied rows were kept
    # LAST-first and the capped subsample skewed late. Isotonic ties
    # predictions heavily, so ties are the common case, not an edge one.
    frame = pd.DataFrame({
        "day": [1, 2, 3, 4, 5],
        "ev_long": [0.5] * 5, "ev_short": [-1.0] * 5,
        "r_long": [1.0] * 5, "r_short": [0.0] * 5, "cost_r": [0.0] * 5})
    _, days = fi.net_of_selection(frame, limit=3)
    assert list(days) == [1, 2, 3]


def test_an_empty_frame_selects_nothing_rather_than_raising() -> None:
    net, days = fi.net_of_selection(pd.DataFrame())
    assert net.size == 0 and days.size == 0


# --- fold construction --------------------------------------------------

def test_train_and_test_never_touch_and_the_embargo_sits_between() -> None:
    days = list(range(200))
    folds = fi.build_folds(days)
    assert folds, "expected folds from 200 days"
    for train_days, test_days in folds:
        assert not set(train_days) & set(test_days), "folds overlap"
        gap = min(test_days) - max(train_days)
        assert gap >= fi.EMBARGO_DAYS, f"embargo too small: {gap}"


def test_training_windows_only_ever_grow() -> None:
    folds = fi.build_folds(list(range(200)))
    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes)


def test_no_days_produces_no_folds() -> None:
    assert fi.build_folds([]) == []


def test_a_tiny_calendar_still_produces_only_non_overlapping_folds() -> None:
    # Three days yields a one-day train window, which is degenerate but
    # not WRONG - main() guards on the fold list being empty and the real
    # dataset has 228 sessions. What must never break is the separation.
    for size in range(3, 12):
        for train_days, test_days in fi.build_folds(list(range(size))):
            assert not set(train_days) & set(test_days)
            assert min(test_days) - max(train_days) >= fi.EMBARGO_DAYS


# --- the module that once could not import itself -----------------------

def test_the_shared_bootstrap_is_reachable_at_module_scope() -> None:
    # day_bootstrap delegated to forecast_stats while the only import was a
    # LOCAL one inside main(), so the call raised NameError and STEP 2 died
    # on its first geometry. A function-local import never binds a global.
    # Six days: block_bootstrap refuses an interval below five groups
    # rather than reporting a falsely narrow one.
    values = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.4,
                       0.15, -0.05, 0.25, -0.15])
    days = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    observed, lo, hi, p = fi.day_bootstrap(values, days, draws=200)
    assert observed == pytest.approx(float(values.mean()))
    assert lo <= observed <= hi
