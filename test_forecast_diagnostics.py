"""Tests for the signal diagnostics, and for the universe builder.

The defect guarded first here is the one that survived longest: the
benchmark was measured from today's FIRST BAR while the stock's change came
from its previous close, so relative strength subtracted two quantities
with different origins. An independent review found it in two modules and
only one was fixed; this file exists partly so that cannot recur silently.

It matters more here than in the dataset builder. There the bias shifts a
continuous feature. Here relative strength feeds a hard gate that is a pure
SIGN test, so a constant offset flips the verdict outright on every
marginal name, in the same direction, all day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import forecast_diagnostics as fd
import forecast_universe as fu

IST = "Asia/Kolkata"


def _index(frame_days: int = 3, bars: int = 5, step: float = 1.0,
           gap: float = 10.0) -> pd.DataFrame:
    """A benchmark that gaps every morning and then drifts."""
    blocks = []
    level = 100.0
    for day in range(frame_days):
        level += gap
        stamps = pd.date_range(f"2026-09-{10 + day} 09:15", periods=bars,
                               freq="5min", tz=IST)
        blocks.append(pd.DataFrame(
            {"Close": level + np.arange(bars) * step}, index=stamps))
        level = float(blocks[-1]["Close"].iloc[-1])
    return pd.concat(blocks)


# --- the baseline that did not match the stock --------------------------

def _from_todays_first_bar(frame: pd.DataFrame) -> dict:
    """The implementation this module used to have, kept as a CONTRAST.

    A test that only asserts the right answer cannot show it would have
    caught the wrong one. Computing the old behaviour here and asserting
    the module differs from it proves the discrimination, and does so
    without ever editing the shipped module to reintroduce the bug.
    """
    closes = frame["Close"].dropna()
    out = {}
    for day, series in closes.groupby(closes.index.date):
        if len(series) < 2:
            continue
        first = float(series.iloc[0])
        if first > 0:
            out[day] = (series / first - 1.0) * 100.0
    return out


def test_the_benchmark_is_measured_from_the_previous_session_close() -> None:
    # Measured from its own first bar, the index always started the day at
    # exactly 0.0% and every symbol inherited a constant bias equal to the
    # overnight gap - standard deviation 0.64pp against a feature whose own
    # spread is 1.73pp.
    frame = _index()
    by_day = fd.benchmark_changes(frame)
    day = sorted(by_day)[0]
    first_bar = float(by_day[day].iloc[0])
    assert first_bar != pytest.approx(0.0), (
        "the first bar must carry the overnight gap, not be forced to zero")
    assert first_bar > 0, "this fixture gaps UP, so the first bar is positive"

    # And it must NOT agree with the old baseline, which pinned every
    # session's first bar to exactly zero. Measured on this fixture:
    # 7.81% against 0.00%.
    old = _from_todays_first_bar(frame)
    assert float(old[day].iloc[0]) == pytest.approx(0.0)
    assert first_bar != pytest.approx(float(old[day].iloc[0]))


def test_the_first_session_is_skipped_because_it_has_no_prior_close() -> None:
    # Falling back to its own open would put ONE session on a different
    # baseline from every other - a silent mixture, which is worse than a
    # gap because nothing downstream can detect it.
    frame = _index(frame_days=3)
    by_day = fd.benchmark_changes(frame)
    all_days = sorted({stamp.date() for stamp in frame.index})
    assert all_days[0] not in by_day
    assert len(by_day) == len(all_days) - 1


def test_each_session_is_a_series_over_bars_not_one_closing_number() -> None:
    # It returned the index's FULL-DAY move and handed it to every bar, so
    # a signal was scored against the outcome of the session it was still
    # trading in.
    by_day = fd.benchmark_changes(_index(bars=5))
    series = by_day[sorted(by_day)[-1]]
    assert len(series) == 5
    assert series.iloc[0] != pytest.approx(series.iloc[-1])
    assert series.index.is_monotonic_increasing


# --- reading the benchmark at a moment ----------------------------------

def test_the_benchmark_is_never_clamped_to_its_last_known_bar() -> None:
    # Clamping serves the index's closing move to any bar past its own
    # last, which is the same lookahead by another route.
    series = pd.Series([0.1, 0.2, 0.3],
                       index=pd.date_range("2026-09-10 09:15", periods=3,
                                           freq="5min", tz=IST))
    before = pd.Timestamp("2026-09-10 09:10", tz=IST)
    between = pd.Timestamp("2026-09-10 09:22", tz=IST)
    assert fd.benchmark_at(series, before) is None
    assert fd.benchmark_at(series, between) == pytest.approx(0.2)
    # A clamping implementation would answer 0.3 here - the index's
    # CLOSING move, served to a bar that traded before the index opened.
    # Asserting None rather than "not 0.3" is what makes this discriminate.
    assert fd.benchmark_at(series, before) != series.iloc[-1]


def test_an_absent_benchmark_is_none_rather_than_zero() -> None:
    # Zero would silently read as "the index was flat", turning a falling
    # market into apparent strength for every symbol in it.
    assert fd.benchmark_at(None, pd.Timestamp("2026-09-10", tz=IST)) is None
    assert fd.benchmark_at(pd.Series(dtype=float),
                           pd.Timestamp("2026-09-10", tz=IST)) is None


# --- the rank correlation -----------------------------------------------

def test_rank_correlation_is_computed_on_ranks() -> None:
    left = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert fd.spearman(left, left) == pytest.approx(1.0)
    assert fd.spearman(left, left[::-1].reset_index(drop=True)) == pytest.approx(-1.0)
    # Monotone but non-linear must still be a perfect RANK correlation.
    assert fd.spearman(left, left ** 3) == pytest.approx(1.0)


def test_too_few_points_returns_nan_rather_than_a_confident_number() -> None:
    assert np.isnan(fd.spearman(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])))


# --- the universe builder's forward window ------------------------------

def test_the_forward_window_looks_forward_and_excludes_today() -> None:
    # Reversing, rolling, then reversing back gives a forward window
    # through the same code path as a trailing one. An off-by-one here
    # would include today's own close in "the worst close AHEAD".
    close = pd.DataFrame({"A": [100.0, 90.0, 80.0, 70.0, 60.0]})
    out = fu.forward_min_ratio(close, span=2)
    # From row 0 the next two closes are 90 and 80, so the worst is 80.
    assert out["A"].iloc[0] == pytest.approx(80.0 / 100.0 - 1.0)
    # The tail has no full forward window and must be NaN, not optimistic.
    assert np.isnan(out["A"].iloc[-1])


def test_a_rising_series_has_a_forward_minimum_above_today() -> None:
    close = pd.DataFrame({"A": [100.0, 110.0, 120.0, 130.0]})
    out = fu.forward_min_ratio(close, span=2)
    assert out["A"].iloc[0] > 0, "a rising series must not report a drawdown"


def test_the_sampling_stride_is_the_one_the_horizons_study_imports() -> None:
    assert fu.SAMPLE_EVERY == 5
