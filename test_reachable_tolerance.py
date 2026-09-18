"""The reach gate must not be decided by floating-point rounding.

WHAT WAS WRONG. TradeLevels.reachable compared target_distance to
expected_range with a bare `<=`. With the default volatility stop at half
a sigma and a 2:1 target those two quantities are the SAME NUMBER by
construction:

    target_distance = 2 * (0.5 * sigma) = sigma = expected_range

setups._reach_reason already says so - it "is an identity, not a test" and
"can only fail when there is no remaining move at all". But an identity
compared with `<=` on floats is a coin flip. Measured over 4,000
randomised volatility-stopped trades: 100% landed mathematically on the
line and 48.8% were refused, per symbol, at random.

That silently suppressed about half of every scan's actionable setups -
and it is the sort of defect that looks like a quiet market rather than a
bug, because the trail printed a plausible sentence about the target
sitting exactly on the plausible move.

A relative tolerance fixes it. The tests below pin BOTH halves: the
identity must pass, and a target genuinely beyond the range must still be
refused, because a tolerance wide enough to hide a real overshoot would
be worse than the rounding.
"""
from __future__ import annotations

import random

import levels


def _volatility_trade(entry, atr, bars):
    """A trade whose stop comes from volatility, so the identity applies."""
    return levels.build_levels(
        direction="LONG", entry=entry, atr_per_bar=atr, bars_left=bars,
        opening_low=None, opening_high=None, day_low=None, day_high=None)


def test_the_identity_is_reachable_across_the_whole_input_space() -> None:
    """The regression test. A bare `<=` fails about half of these."""
    random.seed(1)
    checked = refused = 0
    for _ in range(2000):
        trade = _volatility_trade(entry=random.uniform(50, 5000),
                                  atr=random.uniform(0.05, 8.0),
                                  bars=random.randint(10, 110))
        if trade is None or trade.stop_source != "volatility":
            continue
        checked += 1
        if not trade.reachable:
            refused += 1
    assert checked > 1000, f"only {checked} volatility-stopped trades built"
    assert refused == 0, (
        f"{refused}/{checked} volatility-stopped trades refused by "
        f"rounding alone")


def test_the_two_quantities_really_are_the_same_number() -> None:
    """Pins the PREMISE, so the tolerance is not later read as arbitrary.

    If the stop or the reward:risk ever changes so that the target no
    longer lands on sigma, this stops being an identity and the tolerance
    stops being justified - and this test says where to look.
    """
    trade = _volatility_trade(entry=1000.0, atr=2.0, bars=90)
    assert trade is not None and trade.stop_source == "volatility"
    # Equal to within one part in a billion, which is what the tolerance
    # is sized for - and NOT equal bit for bit, which is the whole point.
    diff = abs(trade.target_distance - trade.expected_range)
    assert diff <= 1e-9 * trade.expected_range, diff


def test_a_target_genuinely_beyond_the_range_is_still_refused() -> None:
    """The tolerance must not hide a real overshoot."""
    class _Stub:
        expected_range = 10.0
        target_distance = 12.0
        _REACH_TOLERANCE = levels.TradeLevels._REACH_TOLERANCE
        reachable = levels.TradeLevels.reachable

    assert _Stub().reachable is False


def test_a_hair_beyond_the_range_is_refused() -> None:
    """One part in a thousand over is an overshoot, not rounding."""
    class _Stub:
        expected_range = 10.0
        target_distance = 10.01
        _REACH_TOLERANCE = levels.TradeLevels._REACH_TOLERANCE
        reachable = levels.TradeLevels.reachable

    assert _Stub().reachable is False


def test_no_remaining_move_is_refused() -> None:
    """The one case _reach_reason says can legitimately fail."""
    class _Stub:
        expected_range = 0.0
        target_distance = 0.0
        _REACH_TOLERANCE = levels.TradeLevels._REACH_TOLERANCE
        reachable = levels.TradeLevels.reachable

    assert _Stub().reachable is False


def test_the_tolerance_is_relative_not_absolute() -> None:
    """A fixed epsilon would be meaningless across a 50x price range.

    An absolute 1e-9 is enormous for a 0.01 range and nothing for a
    10,000 one.
    """
    assert 0.0 < levels.TradeLevels._REACH_TOLERANCE < 1e-6

    class _Big:
        expected_range = 1e6
        target_distance = 1e6 + 1.0          # 1 part per million over
        _REACH_TOLERANCE = levels.TradeLevels._REACH_TOLERANCE
        reachable = levels.TradeLevels.reachable

    assert _Big().reachable is False, "a relative tolerance must still bind"
