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
    base.update(over)
    fields = {f.name for f in __import__("dataclasses").fields(setups.Readings)}
    return setups.Readings(**{k: v for k, v in base.items() if k in fields})


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
    def __init__(self, expected_range):
        self.expected_range = expected_range


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


def test_the_gate_is_wired_into_the_verdict(monkeypatch) -> None:
    """Otherwise it reports a FAIL that does not bind on anything."""
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 5.0)
    broke = _reading(last=101.5, prev_close=99.0)
    verdict = setups.evaluate(broke)
    assert verdict.direction == setups.LONG
    assert verdict.actionable is False
    assert any("plausible move left" in r for r in verdict.reasons)
    assert verdict.rank_score == 0.0


def test_the_reason_appears_in_the_trail_even_when_it_passes(
        monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_MIN_ROOM_RATIO", 0.01)
    verdict = setups.evaluate(_reading(last=101.5, prev_close=99.0))
    assert any("plausible move left" in r or "room to move" in r
               for r in verdict.reasons)
