"""Tests for the outcome resolver that turns predictions into a record.

THE TEST THIS FILE EXISTS FOR is the last one: that resolve_one agrees
with forecast_intraday_data.resolve on random paths. Two resolvers with
the same job and different tie-break rules would let the live track record
and the offline study disagree about the same trade, and nothing would
report the divergence - you would simply have two numbers and no way to
tell which was wrong.

The tie-break is not a detail. A bar spanning both stop and target carries
no intra-bar ordering, and assuming the target came first is how a
back-test flatters itself. Worse, the ambiguous case gets MORE common as
bars widen, so the flattery grows with volatility - which is the one thing
this project's signals were measured to predict. Both resolvers therefore
count that bar as a stop, and this file holds them to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import outcomes


def _bars(rows, start="2026-09-15 09:15"):
    """Bars from (high, low) or (high, low, close) tuples."""
    stamps = pd.date_range(start, periods=len(rows), freq="3min",
                           tz="Asia/Kolkata")
    highs = [r[0] for r in rows]
    lows = [r[1] for r in rows]
    closes = [r[2] if len(r) > 2 else (r[0] + r[1]) / 2 for r in rows]
    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes},
                        index=stamps)


# --- the tie-break -------------------------------------------------------

def test_a_bar_spanning_both_levels_counts_as_a_stop() -> None:
    # The defect this guards: a 3-minute bar that reaches both 102 and 98
    # says nothing about which came first. Counting it as a target would
    # turn a coin flip into a win, and would do so more often on volatile
    # names - exactly where the signal is strongest.
    bars = _bars([(102.0, 98.0)])
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=bars)
    assert got.outcome == outcomes.STOP
    assert got.r_multiple == -1.0


def test_the_same_tie_resolves_to_the_stop_when_short() -> None:
    bars = _bars([(102.0, 98.0)])
    got = outcomes.resolve_one(100.0, 102.0, 98.0, long=False, bars=bars)
    assert got.outcome == outcomes.STOP


# --- ordinary resolution -------------------------------------------------

def test_a_target_reached_first_is_a_win() -> None:
    bars = _bars([(100.5, 99.8), (102.5, 100.0), (99.0, 97.0)])
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=bars)
    assert got.outcome == outcomes.TARGET
    assert got.r_multiple == pytest.approx(1.0)
    assert got.bars_held == 2, "it resolved on the second bar, not the third"


def test_a_stop_reached_first_is_a_loss_of_exactly_one_r() -> None:
    bars = _bars([(100.5, 99.8), (100.2, 97.5), (103.0, 102.0)])
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=bars)
    assert got.outcome == outcomes.STOP
    assert got.r_multiple == -1.0
    assert got.bars_held == 2


def test_a_short_resolves_on_the_mirror_levels() -> None:
    bars = _bars([(100.2, 99.0), (100.1, 97.9)])
    got = outcomes.resolve_one(100.0, 102.0, 98.0, long=False, bars=bars)
    assert got.outcome == outcomes.TARGET
    assert got.r_multiple == pytest.approx(1.0)


def test_neither_level_reached_is_marked_out_not_discarded() -> None:
    # Discarding these keeps only the setups that moved, which is a track
    # record of the exciting ones rather than of the strategy.
    bars = _bars([(100.5, 99.5), (100.8, 99.2, 100.6)])
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=bars)
    assert got.outcome == outcomes.CLOSE
    assert got.r_multiple == pytest.approx(0.3), "0.6 of a 2.0 stop"


# --- excursions ----------------------------------------------------------

def test_excursions_describe_the_path_up_to_the_exit() -> None:
    # MAE says whether the stop was nearly hit on the way to the target,
    # which is what tells you a tighter stop would have lost this trade.
    bars = _bars([(100.4, 98.6), (102.5, 100.0)])
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=bars)
    assert got.outcome == outcomes.TARGET
    assert got.mae_r == pytest.approx(0.7), "reached 98.6 against a 2.0 stop"
    assert got.mfe_r >= 1.0


def test_an_entirely_favourable_bar_has_no_adverse_excursion() -> None:
    # Asserting ">= 0" was a tautology: both start at 0.0 and only ever go
    # through max(), so it could not fail for any non-NaN input. The value
    # is what discriminates - this bar never traded below the entry, so
    # MAE is exactly zero and MFE is the 0.1 move over a 2.0 stop.
    bars = _bars([(100.1, 100.05, 100.08)])
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=bars)
    assert got.mae_r == pytest.approx(0.0)
    assert got.mfe_r == pytest.approx(0.05)


# --- degradation ---------------------------------------------------------

def test_no_bars_is_not_an_outcome() -> None:
    # It must stay resolvable later. Recording it as a non-event would be
    # permanent, and a briefly unavailable symbol would enter the track
    # record as a trade that did nothing.
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=None)
    assert got.outcome == outcomes.NO_BARS
    assert not got.resolved


@pytest.mark.parametrize("long,entry,stop,target", [
    (True, 100.0, 102.0, 105.0),    # long stop ABOVE entry
    (True, 100.0, 98.0, 97.0),      # long target BELOW entry
    (False, 100.0, 98.0, 95.0),     # short stop BELOW entry
    (False, 100.0, 102.0, 105.0),   # short target ABOVE entry
])
def test_levels_on_the_wrong_side_are_refused_not_mirrored(
        long, entry, stop, target) -> None:
    # abs() on both distances silently mirrors a malformed row and reports
    # a clean TARGET or STOP for a trade that was never specified. A level
    # on the wrong side means the row is broken, not that it needs fixing.
    bars = _bars([(110.0, 90.0, 100.0)])
    got = outcomes.resolve_one(entry, stop, target, long=long, bars=bars)
    assert got.outcome == outcomes.NO_BARS


def test_a_non_finite_bar_is_refused_rather_than_skipped() -> None:
    # Every comparison is False on NaN, so the bar would read as "no
    # touch" - understating hits. forecast_intraday_data's searchsorted
    # orders NaN above everything and reports a spurious hit instead, so
    # the two resolvers genuinely disagree here. Refusing is the only
    # answer that is not quietly wrong in one direction or the other.
    bars = _bars([(float("nan"), 99.0, 100.0), (103.0, 99.0, 102.0)])
    got = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=bars)
    assert got.outcome == outcomes.NO_BARS


def test_a_zero_width_stop_cannot_read_as_a_win() -> None:
    bars = _bars([(103.0, 99.0)])
    got = outcomes.resolve_one(100.0, 100.0, 102.0, long=True, bars=bars)
    assert got.outcome == outcomes.NO_BARS


# --- the join key --------------------------------------------------------

def test_the_row_id_is_stable_and_distinguishes_direction() -> None:
    first = outcomes.row_id("2026-09-15", "10:00:00", "RELIANCE", "LONG")
    assert first == outcomes.row_id("2026-09-15", "10:00:00", "RELIANCE", "LONG")
    assert first != outcomes.row_id("2026-09-15", "10:00:00", "RELIANCE", "SHORT")
    assert first != outcomes.row_id("2026-09-16", "10:00:00", "RELIANCE", "LONG")


def test_resolved_ids_round_trip_through_the_file(tmp_path) -> None:
    path = tmp_path / "outcomes.csv"
    result = outcomes.Outcome("k1", outcomes.TARGET, "t", 102.0, 1.0, 2, 1.0, 0.2)
    outcomes.append([outcomes.to_row({"symbol": "X"}, result)], path)
    assert outcomes.load_resolved(path) == {"k1"}
    # And appending again does not rewrite the header into the data.
    second = outcomes.Outcome("k2", outcomes.STOP, "t", 98.0, -1.0, 1, 0.1, 1.0)
    outcomes.append([outcomes.to_row({"symbol": "Y"}, second)], path)
    assert outcomes.load_resolved(path) == {"k1", "k2"}


def test_an_absent_outcomes_file_is_an_empty_set(tmp_path) -> None:
    assert outcomes.load_resolved(tmp_path / "missing.csv") == set()


# --- THE ONE THAT MATTERS ------------------------------------------------

def _research_verdict(bars, entry, stop_distance, target_distance, long):
    """What forecast_intraday_data.resolve makes of the same path."""
    import forecast_intraday_data as fid

    highs = bars["High"].to_numpy(dtype=float)
    lows = bars["Low"].to_numpy(dtype=float)
    cmax = np.maximum.accumulate(highs)
    neg_cmin = np.maximum.accumulate(-lows)
    close_price = float(bars["Close"].iloc[-1])
    return fid.resolve(cmax, neg_cmin, close_price, long, entry,
                       stop_distance, target_distance)


def test_the_live_resolver_agrees_with_the_research_one() -> None:
    """The property this file exists for, on paths that actually test it.

    An earlier version parametrised 12 seeds x 2 directions over a 2.0/3.0
    geometry and had ZERO power: across those 24 walks the ambiguous
    both-levels-in-one-bar case arose 0 times, and mutating resolve_one to
    the optimistic tie-break was caught in 0 of 24. The walks produced only
    TARGET and STOP, never a tie and never a CLOSE.

    The geometry here is deliberately tight against the step size so ties
    are common, and the test ASSERTS it saw both a tie and a close-out
    rather than trusting that it did. A coverage claim that is not checked
    is how the previous version passed while testing nothing.
    """
    import forecast_intraday_data as fid

    entry = 100.0
    ambiguous = closed = 0

    # TWO GEOMETRIES ON PURPOSE. A tight bracket against this step size
    # produces the both-levels-in-one-bar tie constantly but always
    # resolves, so it never exercises the mark-out-at-close branch. A wide
    # one is the reverse. Running only the tight geometry passed the
    # ambiguity assertion and then failed the close-out one, which is how
    # this comment came to exist.
    for seed in range(60):
        stop_distance = target_distance = 0.5 if seed % 2 else 9.0
        for long in (True, False):
            rng = np.random.default_rng(seed)
            steps = rng.normal(0.0, 1.2, 30).cumsum() + entry
            spread = np.abs(rng.normal(0.0, 0.9, 30))
            frame = pd.DataFrame(
                {"High": steps + spread, "Low": steps - spread,
                 "Close": steps},
                index=pd.date_range("2026-09-15 09:15", periods=30,
                                    freq="3min", tz="Asia/Kolkata"))

            stop = entry - stop_distance if long else entry + stop_distance
            target = entry + target_distance if long else entry - target_distance

            # Did any single bar span both levels before either resolved?
            for high, low in zip(frame["High"], frame["Low"], strict=True):
                spans_target = (high >= entry + target_distance if long
                                else low <= entry - target_distance)
                spans_stop = (low <= entry - stop_distance if long
                              else high >= entry + stop_distance)
                if spans_target and spans_stop:
                    ambiguous += 1
                    break
                if spans_target or spans_stop:
                    break

            mine = outcomes.resolve_one(entry, stop, target, long=long,
                                        bars=frame)
            if mine.outcome == outcomes.CLOSE:
                closed += 1

            highs = frame["High"].to_numpy(dtype=float)
            lows = frame["Low"].to_numpy(dtype=float)
            label, gross_r = fid.resolve(
                np.maximum.accumulate(highs), np.maximum.accumulate(-lows),
                float(frame["Close"].iloc[-1]), long, entry,
                stop_distance, target_distance)

            assert (mine.outcome == outcomes.TARGET) == bool(label), (
                f"seed {seed} long={long}: {mine.outcome} vs label {label}")
            assert mine.r_multiple == pytest.approx(gross_r, abs=1e-9), (
                f"seed {seed} long={long}: {mine.r_multiple} vs {gross_r}")

    assert ambiguous > 0, (
        "no walk produced a bar spanning both levels, so this test has no "
        "power against the tie-break it exists to guard")
    assert closed > 0, "the CLOSE branch of the agreement was never exercised"


def test_they_agree_on_the_constructed_ambiguous_bar() -> None:
    # Belt and braces: the loop above asserts it saw ties, this pins the
    # exact case so a failure names it directly.
    import forecast_intraday_data as fid

    frame = _bars([(103.5, 97.5, 100.0)])
    mine = outcomes.resolve_one(100.0, 98.0, 102.0, long=True, bars=frame)
    highs = frame["High"].to_numpy(dtype=float)
    lows = frame["Low"].to_numpy(dtype=float)
    label, gross_r = fid.resolve(np.maximum.accumulate(highs),
                                 np.maximum.accumulate(-lows),
                                 100.0, True, 100.0, 2.0, 2.0)
    assert mine.outcome == outcomes.STOP
    assert label == 0
    assert mine.r_multiple == pytest.approx(gross_r)
