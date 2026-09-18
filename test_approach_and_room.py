"""Two answers to "it only shows me what already moved".

THE COMPLAINT, measured. choose_direction requires the opening range to
be BROKEN before a symbol has a direction at all, and the
relative-strength gate requires today's outperformance - so a long cannot
clear until it is already up more than the index. On 2026-09-18 the
cleared setups had moved 3.16% at the median and asked for 3.10% more,
and 46% of them needed the rest of the session to travel further than the
whole morning had.

TWO SEPARATE FIXES, deliberately not one:

  approach()      reads the scan one step EARLIER - the level a name has
                  not broken, how far away it is in its own volatility,
                  and which gates would still block it on arrival. It adds
                  names; it predicts nothing.

  _room_reason()  refuses a setup with less plausible move left than the
                  day has already made. It removes names; it finds none.

Neither has a backtest behind it. The room threshold is a config value
with its measurement recorded beside it precisely because it is a
judgement, and the tests below pin the REFUSALS - the cases where these
must stay quiet - rather than only the happy paths.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import config
import indicators
import setups

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 18, 11, 0, tzinfo=IST)


def _reading(**over):
    """A Readings sitting quietly inside a closed opening range."""
    base = dict(
        symbol="AAA", ticker="AAA", last=100.0, prev_close=99.0,
        day_change_pct=1.01, vwap=99.5, vwap_distance_pct=0.5,
        opening_range=indicators.OpeningRange(high=101.0, low=98.0, bars=5,
                                              minutes=15),
        cpr=None, atr_bar=2.0, rvol=1.5, relative_strength=1.0,
        turnover_20d=5e8, oi_change_pct=None, futures_share=None,
        derivatives_turnover=None, day_high=100.5, day_low=98.5,
        minutes_left=270, bars_left=90, range_closed=True,
        session_bar_count=40,
    )
    # Passed straight through, so a mistyped override raises TypeError
    # instead of silently testing the default. An earlier version filtered
    # to the dataclass's own field names, which meant _reading(atr=0.0)
    # tested atr_bar=2.0 and passed.
    base.update(over)
    return setups.Readings(**base)


# --- approach: when it must stay silent ---------------------------------

def test_a_name_that_already_broke_is_not_approaching() -> None:
    """evaluate() owns it from that point; two owners would double-report."""
    assert setups.approach(_reading(last=101.5)) is None


def test_a_name_that_broke_downward_is_not_approaching() -> None:
    assert setups.approach(_reading(last=97.5, vwap=99.5)) is None


def test_an_unclosed_opening_range_is_not_approaching() -> None:
    """The comparison is not defined yet - the latest bar defines the range."""
    assert setups.approach(_reading(range_closed=False)) is None


def test_no_opening_range_at_all_is_none() -> None:
    assert setups.approach(_reading(opening_range=None)) is None


def test_no_vwap_is_none() -> None:
    """Without VWAP there is no side, so no trigger to approach."""
    assert setups.approach(_reading(vwap=None)) is None


@pytest.mark.parametrize("atr", [None, 0.0])
def test_without_volatility_distance_has_no_units(atr) -> None:
    assert setups.approach(_reading(atr_bar=atr)) is None


def test_a_name_too_far_from_its_trigger_is_not_reported() -> None:
    """A whole ATR-plus away is not approaching anything."""
    far = _reading(last=99.6, atr_bar=0.2)      # 1.4 / 0.2 = 7 ATR
    assert setups.approach(far) is None


# --- approach: the side comes from VWAP ---------------------------------

def test_above_vwap_approaches_the_range_high_as_a_long() -> None:
    near = setups.approach(_reading(last=100.0, vwap=99.5))
    assert near is not None
    assert near.side == setups.LONG
    assert near.trigger == 101.0
    assert near.distance == pytest.approx(1.0)
    assert near.distance_atr == pytest.approx(0.5)


def test_below_vwap_approaches_the_range_low_as_a_short() -> None:
    near = setups.approach(_reading(last=99.0, vwap=99.5,
                                    relative_strength=-1.0))
    assert near is not None
    assert near.side == setups.SHORT
    assert near.trigger == 98.0
    assert near.distance == pytest.approx(1.0)


def test_the_side_is_not_whichever_edge_is_closer() -> None:
    """A name below VWAP but nearer the HIGH would break into a direction
    choose_direction refuses, so the side must come from VWAP.

    max_atr is lifted here on purpose: at the default ceiling this
    geometry is 1.45 ATR from the low and is rejected on DISTANCE before
    the side is ever in question, which would make the test pass for the
    wrong reason.
    """
    near = setups.approach(_reading(last=100.9, vwap=101.5,
                                    relative_strength=-1.0), max_atr=2.0)
    assert near is not None
    assert near.side == setups.SHORT, "reported a break VWAP would veto"
    assert near.trigger == 98.0


# --- approach: readiness uses the real gates ----------------------------

def test_vwap_between_price_and_the_high_still_reports_the_long() -> None:
    """THE REGRESSION TEST for the side rule.

    Breaking the high CROSSES VWAP, so choose_direction accepts the long.
    An earlier version read the current VWAP side, called it SHORT, and
    pointed at the far edge - naming the wrong trigger and hiding the
    right one on a geometry that is common mid-session.
    """
    reading = _reading(last=100.0, vwap=100.5)
    near = setups.approach(reading)
    assert near is not None
    assert near.side == setups.LONG, "reported the edge VWAP does not agree with"
    assert near.trigger == 101.0

    # And the direction it names is the one choose_direction would give
    # once the break happened.
    broken = _reading(last=101.2, vwap=100.5)
    assert setups.choose_direction(broken)[0] == setups.LONG


def test_a_ready_name_would_not_immediately_fail_the_room_gate() -> None:
    """THE CONTRADICTION the panel must not present as a promise.

    Reaching the trigger is itself more move spent, so a name can look
    clear at 0.6 ATR away and fail the room gate the instant it breaks.
    Readiness is therefore judged on levels built AT THE TRIGGER, not at
    the current price.
    """
    reading = _reading(last=103.0, prev_close=100.0, day_change_pct=3.0,
                       vwap=102.0, atr_bar=0.25,
                       opening_range=indicators.OpeningRange(
                           high=103.15, low=101.0, bars=5, minutes=15),
                       day_high=103.1, day_low=101.2,
                       minutes_left=150, bars_left=50)
    near = setups.approach(reading)
    assert near is not None
    assert near.side == setups.LONG

    # Whatever it reported, breaking must not contradict it.
    broke = setups.evaluate(_reading(
        last=near.trigger + 0.05, prev_close=100.0, day_change_pct=3.05,
        vwap=102.0, atr_bar=0.25,
        opening_range=indicators.OpeningRange(high=103.15, low=101.0,
                                              bars=5, minutes=15),
        day_high=103.25, day_low=101.2, minutes_left=150, bars_left=50))
    if near.ready:
        assert broke.actionable, (
            "reported ready, then refused on breaking: "
            f"{[r for r in broke.reasons if '[FAIL]' in r]}")
    else:
        assert near.blockers, "not ready but named no blocker"


def test_ready_when_only_the_break_is_missing() -> None:
    near = setups.approach(_reading())
    assert near is not None
    assert near.ready is True
    assert near.blockers == []


def test_thin_volume_is_reported_as_the_blocker() -> None:
    near = setups.approach(_reading(rvol=0.4))
    assert near is not None
    assert near.ready is False
    assert any("relative volume" in b for b in near.blockers)


def test_the_wrong_side_of_the_index_blocks_a_long() -> None:
    near = setups.approach(_reading(relative_strength=-2.0))
    assert near is not None
    assert near.ready is False
    assert any("relative strength" in b for b in near.blockers)


def test_an_illiquid_name_is_not_ready() -> None:
    near = setups.approach(_reading(turnover_20d=1_000.0))
    assert near is not None
    assert near.ready is False


def test_too_late_in_the_session_is_not_ready() -> None:
    near = setups.approach(_reading(minutes_left=5))
    assert near is not None
    assert near.ready is False


def test_readiness_is_not_a_claim_the_break_happens() -> None:
    """Documented intent, pinned so it is not quietly reinterpreted."""
    assert "not a prediction" in setups.Approach.__doc__.lower() or \
        "not a forecast" in setups.Approach.__doc__.lower()


# --- room to move -------------------------------------------------------

class _Trade:
    """Enough of TradeLevels for the room gate: the range and the SIDE.

    The side is not optional - the gate measures move spent in the
    trade's own direction, because an unsigned denominator refused longs
    that had fallen and passed shorts that had risen.
    """

    def __init__(self, expected_range, direction=setups.LONG):
        self.expected_range = expected_range
        self.direction = direction


def test_a_setup_with_less_left_than_it_has_moved_is_refused(
        monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.5)
    # Moved 5.00, only 1.00 plausibly left: ratio 0.20.
    ok, line = setups._room_reason(_reading(last=104.0, prev_close=99.0),
                                   _Trade(1.0))
    assert ok is False
    assert "[FAIL]" in line
    assert "0.20" in line


def test_a_setup_with_room_left_passes(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.5)
    ok, line = setups._room_reason(_reading(last=100.0, prev_close=99.0),
                                   _Trade(3.0))
    assert ok is True
    assert "[PASS]" in line


def test_an_unmoved_name_has_all_its_room(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.5)
    ok, _ = setups._room_reason(_reading(last=99.0, prev_close=99.0),
                                _Trade(0.1))
    assert ok is True


def test_a_missing_previous_close_fails_OPEN(monkeypatch) -> None:
    """A data gap must not become a verdict.

    Every other gate fails closed on missing data because absence is
    weaker evidence. This one is not about evidence - refusing for want of
    a daily bar would block setups for a reason unrelated to chasing.
    """
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.5)
    ok, line = setups._room_reason(_reading(prev_close=None), _Trade(0.01))
    assert ok is True
    assert "[INFO]" in line


def test_a_zero_threshold_reports_without_binding(monkeypatch) -> None:
    """The escape hatch, matching SCAN_REQUIRE_OI_CONFIRMATION's pattern."""
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.0)
    ok, line = setups._room_reason(_reading(last=110.0, prev_close=99.0),
                                   _Trade(0.01))
    assert ok is True
    assert "[INFO]" in line


def test_the_default_threshold_is_not_a_veto_on_most_output() -> None:
    """Measured 2026-09-18: 1.0 cut 56% of cleared setups, 0.5 cut 20%.

    Promoting an unvalidated rule to the larger veto is what
    SCAN_REQUIRE_OI_CONFIRMATION's comment exists to warn against.
    """
    assert 0.0 < config.SCAN_MIN_ROOM_RATIO <= 0.75


def _broke_upward():
    """A real break that clears every OTHER gate, so the room gate decides.

    The ATR and bars are scaled to a plausible intraday setup on purpose.
    An earlier fixture used atr_bar=2.0 with bars_left=90, which put
    target_distance one ulp ABOVE expected_range - so _reach_reason
    already failed and `actionable is False` held whether the room gate
    was wired in or not. The test named for proving the wiring was the one
    test that could not fail.
    """
    return _reading(last=101.5, prev_close=99.0, atr_bar=1.0, bars_left=25,
                    minutes_left=150, day_high=101.6, day_low=98.5)


def test_the_gate_is_wired_into_the_verdict(monkeypatch) -> None:
    """Otherwise it reports a FAIL that does not bind on anything.

    Asserted as a DIFFERENCE between two thresholds on one fixture: with
    the gate slack the setup is actionable, and with it strict the same
    setup is not. Nothing else about the fixture changes, so only the gate
    can account for the flip.
    """
    broke = _broke_upward()

    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.01)
    slack = setups.evaluate(broke)
    assert slack.direction == setups.LONG
    assert slack.actionable is True, slack.reasons
    assert slack.rank_score > 0.0

    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 5.0)
    strict = setups.evaluate(broke)
    assert strict.direction == setups.LONG
    assert strict.actionable is False, "the gate does not bind on `passed`"
    assert strict.rank_score == 0.0
    assert any("plausible move left" in r and "[FAIL]" in r
               for r in strict.reasons)


def test_the_reason_appears_in_the_trail_even_when_it_passes(
        monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.01)
    verdict = setups.evaluate(_broke_upward())
    assert any("plausible move left" in r and "[PASS]" in r
               for r in verdict.reasons)


def test_the_room_failure_is_tagged_as_an_entry_gate(monkeypatch) -> None:
    """A held position must not be told its setup "no longer clears".

    The ratio falls monotonically as a trade works - further from the
    previous close, fewer bars left - so without the tag the watch tab
    headlines a failure on exactly the positions going well.
    """
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 5.0)
    verdict = setups.evaluate(_broke_upward())
    room = [r for r in verdict.reasons if "plausible move left" in r]
    assert room and "[ENTRY]" in room[0], room


def test_a_short_is_measured_against_its_own_direction(
        monkeypatch) -> None:
    """The denominator must be signed.

    Unsigned, a long sitting BELOW its previous close - the whole gap
    above it as room - was refused as a chase, and a short carrying
    adverse move was let through.
    """
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.5)

    class _Long:
        expected_range = 1.5
        direction = setups.LONG

    class _Short:
        expected_range = 1.5
        direction = setups.SHORT

    down = _reading(last=96.0, prev_close=100.0)
    ok, line = setups._room_reason(down, _Long())
    assert ok is True, line
    assert "against this direction" in line

    ok, _ = setups._room_reason(down, _Short())
    assert ok is False, "4.00 spent downward with 1.50 left should refuse"
