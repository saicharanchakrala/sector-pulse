"""The fast relative_volume must agree with the readable one.

WHY BOTH EXIST. cumulative_volume_by_time builds a full cumulative curve
per session - about 125 points each - and relative_volume then reads a
SINGLE point out of each. Profiled 2026-09-16 that was 1.64 seconds of a
4.37 second evaluation, 38% of the whole thing, to produce twelve numbers.

So relative_volume now computes those numbers directly. The curve version
survives as `_relative_volume_from_curves` for one reason: it is the
readable statement of what the arithmetic means, and it is what these
tests hold the fast path to. It is NOT a fallback - it raises on an
unsorted frame, which is why the fast path sorts instead of delegating.

Two implementations that agree today and drift in a month would be worse
than the slow one alone, which is why the agreement is asserted on random
data rather than on a couple of hand-built cases.
"""
from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import pytest

import indicators

IST = "Asia/Kolkata"


def _frame(sessions=12, bars=75, seed=0, truncate=None, start_day=None):
    """Intraday bars across several sessions, with random volume."""
    rng = np.random.default_rng(seed)
    first = start_day or date(2026, 9, 1)
    blocks = []
    for offset in range(sessions):
        day = first + timedelta(days=offset)
        count = bars
        if truncate is not None and offset == truncate[0]:
            count = truncate[1]
        stamps = pd.date_range(f"{day} 09:15", periods=count, freq="3min",
                               tz=IST)
        price = 100.0 + rng.normal(0, 1, count).cumsum()
        blocks.append(pd.DataFrame({
            "Open": price, "High": price + 0.5, "Low": price - 0.5,
            "Close": price,
            "Volume": rng.integers(1_000, 100_000, count).astype(float),
        }, index=stamps))
    return pd.concat(blocks)


# --- THE AGREEMENT -------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_the_fast_path_matches_the_curve_version(seed) -> None:
    frame = _frame(seed=seed)
    fast = indicators.relative_volume(frame)
    slow = indicators._relative_volume_from_curves(frame)
    assert (fast is None) == (slow is None), f"seed {seed}"
    if fast is not None:
        assert fast == pytest.approx(slow, rel=1e-9), f"seed {seed}"


@pytest.mark.parametrize("clock", [time(9, 30), time(10, 0), time(11, 15),
                                   time(13, 45), time(15, 0)])
def test_they_agree_at_any_point_in_the_session(clock) -> None:
    frame = _frame(seed=7)
    fast = indicators.relative_volume(frame, asof=clock)
    slow = indicators._relative_volume_from_curves(frame, asof=clock)
    assert (fast is None) == (slow is None)
    if fast is not None:
        assert fast == pytest.approx(slow, rel=1e-9)


def test_they_agree_when_a_prior_session_is_truncated() -> None:
    # The case the original comment records: a session cut to 20 of 75
    # bars reads as "quiet at this hour" and inflates the ratio, so both
    # implementations must skip it rather than use its final total.
    frame = _frame(seed=3, truncate=(4, 20))
    clock = time(14, 0)
    fast = indicators.relative_volume(frame, asof=clock)
    slow = indicators._relative_volume_from_curves(frame, asof=clock)
    assert (fast is None) == (slow is None)
    if fast is not None:
        assert fast == pytest.approx(slow, rel=1e-9)


def test_they_agree_when_volume_has_gaps() -> None:
    frame = _frame(seed=11)
    frame.loc[frame.index[5:9], "Volume"] = np.nan
    fast = indicators.relative_volume(frame)
    slow = indicators._relative_volume_from_curves(frame)
    assert (fast is None) == (slow is None)
    if fast is not None:
        assert fast == pytest.approx(slow, rel=1e-9)


def test_an_unsorted_frame_is_sorted_rather_than_summed_across_days() -> None:
    """A cumulative sum over shuffled rows would add volume across sessions.

    Note this is now BETTER than the reference: _relative_volume_from_curves
    raises "asof requires a sorted index" on this input, so unsorted frames
    never worked at all. The fast path sorts and answers, and the answer
    must match the reference fed the sorted frame.
    """
    frame = _frame(seed=5)
    shuffled = frame.sample(frac=1.0, random_state=2)

    with pytest.raises(ValueError, match="sorted"):
        indicators._relative_volume_from_curves(shuffled)

    fast = indicators.relative_volume(shuffled)
    reference = indicators._relative_volume_from_curves(shuffled.sort_index())
    assert fast == pytest.approx(reference, rel=1e-9)


# --- the guards ----------------------------------------------------------

def test_one_session_is_not_enough_for_a_baseline() -> None:
    assert indicators.relative_volume(_frame(sessions=1)) is None


def test_an_empty_frame_is_none_not_an_error() -> None:
    assert indicators.relative_volume(pd.DataFrame()) is None
    assert indicators.relative_volume(None) is None


def test_a_frame_without_volume_is_none() -> None:
    frame = _frame()[["Open", "High", "Low", "Close"]]
    assert indicators.relative_volume(frame) is None


def test_a_non_timestamped_index_is_none_rather_than_wrong() -> None:
    frame = _frame(sessions=2)
    frame.index = range(len(frame))
    assert indicators.relative_volume(frame) is None


def test_zero_volume_throughout_gives_none_not_a_division() -> None:
    frame = _frame(sessions=3)
    frame["Volume"] = 0.0
    assert indicators.relative_volume(frame) is None


def test_a_typical_session_reads_near_one() -> None:
    # Sanity on the meaning, not just the agreement: identical volume
    # every session must give a ratio of exactly 1.
    blocks = []
    for offset in range(5):
        day = date(2026, 9, 1) + timedelta(days=offset)
        stamps = pd.date_range(f"{day} 09:15", periods=30, freq="3min", tz=IST)
        blocks.append(pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
            "Volume": 5_000.0}, index=stamps))
    frame = pd.concat(blocks)
    assert indicators.relative_volume(frame) == pytest.approx(1.0)


def test_double_the_usual_volume_reads_as_two() -> None:
    blocks = []
    for offset in range(5):
        day = date(2026, 9, 1) + timedelta(days=offset)
        stamps = pd.date_range(f"{day} 09:15", periods=30, freq="3min", tz=IST)
        volume = 10_000.0 if offset == 4 else 5_000.0
        blocks.append(pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
            "Volume": volume}, index=stamps))
    frame = pd.concat(blocks)
    assert indicators.relative_volume(frame) == pytest.approx(2.0)
