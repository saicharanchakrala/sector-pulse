"""When a setup arrived, which the table alone cannot say.

WHY. Every scan rebuilds the published table from scratch, so `run_time`
is when THAT pass ran. A setup that has held since 09:45 was
indistinguishable from one that printed four seconds ago - and those are
very different facts about the same row.

Two stamps, answering different questions:

  first_seen_at  when the symbol first showed this direction today
  cleared_at     when it first passed every gate, blank if it never did

FIRST WINS FOR BOTH, and cleared_at is never overwritten by a later
block: a setup that cleared at 10:42 and failed at 11:30 was still
suggested at 10:42, and that is the moment it would have been acted on.

WHAT IS DELIBERATELY ABSENT is an expected time for the target. The
target is placed at the plausible remaining move - volatility per bar
times the square root of the bars remaining - so inverting it returns the
bars remaining. Measured on the live 2026-09-21 scan: 49.0 bars left
against 48.8 implied by the target. The horizon is the session close for
every row by construction, which is an assumption rather than a forecast,
and outcomes.csv contains no TARGET resolution to build an empirical one
from.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import scan_publish

IST = ZoneInfo("Asia/Kolkata")
OPEN = datetime(2026, 9, 21, 9, 45, tzinfo=IST)


def _row(symbol="AAA", direction="LONG", actionable=False, entry=100.0,
         stop=98.0, quantity=10):
    return {"symbol": symbol, "direction": direction, "actionable": actionable,
            "entry": entry, "stop": stop, "target": 104.0,
            "quantity": quantity, "score": 0.5, "reasons": ""}


def _table(*rows):
    return pd.DataFrame(list(rows))


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(scan_publish, "_FIRST_DIRECTION", {})
    monkeypatch.setattr(scan_publish, "_FIRST_CLEARED", {})
    monkeypatch.setattr(scan_publish, "_SEEN_DAY", None)


def _stamp(table, when):
    return scan_publish.stamp_first_seen(table, when)


# --- first_seen_at -------------------------------------------------------

def test_the_first_appearance_is_recorded() -> None:
    out = _stamp(_table(_row()), OPEN)
    assert out["first_seen_at"].iloc[0] == "09:45:00"


def test_a_later_scan_keeps_the_first_time() -> None:
    """THE POINT. Without this every row reads as if it just appeared."""
    _stamp(_table(_row()), OPEN)
    out = _stamp(_table(_row()), OPEN + timedelta(hours=1))
    assert out["first_seen_at"].iloc[0] == "09:45:00"


def test_the_two_directions_of_one_symbol_are_tracked_apart() -> None:
    _stamp(_table(_row(direction="LONG")), OPEN)
    out = _stamp(_table(_row(direction="LONG"), _row(direction="SHORT")),
                 OPEN + timedelta(hours=1))
    by_side = dict(zip(out["direction"], out["first_seen_at"]))
    assert by_side["LONG"] == "09:45:00"
    assert by_side["SHORT"] == "10:45:00"


def test_a_row_with_no_levels_gets_no_stamp() -> None:
    """Nothing was suggested, so there is no time it was suggested at."""
    out = _stamp(_table(_row(direction="NO SETUP", entry=None, stop=None,
                             quantity=None)), OPEN)
    assert out["first_seen_at"].iloc[0] is None
    assert out["cleared_at"].iloc[0] is None


def test_a_nan_entry_from_a_mixed_table_gets_no_stamp() -> None:
    """A DataFrame renders a missing number as NaN, not None - and NaN is
    truthy, which is how the scan log broke on 2026-09-21."""
    out = _stamp(_table(
        _row(symbol="HASLEVELS", entry=100.0),
        _row(symbol="NOSETUP", direction="NO SETUP", entry=None, stop=None,
             quantity=None),
    ), OPEN)
    stamps = dict(zip(out["symbol"], out["first_seen_at"]))
    assert stamps["HASLEVELS"] == "09:45:00"
    assert stamps["NOSETUP"] is None


# --- cleared_at ----------------------------------------------------------

def test_a_blocked_setup_has_no_cleared_time() -> None:
    out = _stamp(_table(_row(actionable=False)), OPEN)
    assert out["first_seen_at"].iloc[0] == "09:45:00"
    assert out["cleared_at"].iloc[0] is None


def test_clearing_later_records_when_it_cleared_not_when_it_appeared(
) -> None:
    """The two stamps have to be able to differ - that gap is the answer
    to "how long has this been building?"."""
    _stamp(_table(_row(actionable=False)), OPEN)
    out = _stamp(_table(_row(actionable=True)), OPEN + timedelta(minutes=57))
    assert out["first_seen_at"].iloc[0] == "09:45:00"
    assert out["cleared_at"].iloc[0] == "10:42:00"


def test_a_later_block_does_not_erase_the_cleared_time() -> None:
    """It cleared at 10:42. That it later failed does not unmake that."""
    _stamp(_table(_row(actionable=True)), OPEN)
    out = _stamp(_table(_row(actionable=False)), OPEN + timedelta(hours=2))
    assert out["cleared_at"].iloc[0] == "09:45:00"


def test_the_first_clearing_wins_not_the_latest() -> None:
    _stamp(_table(_row(actionable=True)), OPEN)
    out = _stamp(_table(_row(actionable=True)), OPEN + timedelta(hours=1))
    assert out["cleared_at"].iloc[0] == "09:45:00"


# --- hygiene -------------------------------------------------------------

def test_a_new_session_starts_fresh() -> None:
    """Yesterday's 09:45 must not be reported as today's."""
    _stamp(_table(_row()), OPEN)
    out = _stamp(_table(_row()), OPEN + timedelta(days=1, minutes=30))
    assert out["first_seen_at"].iloc[0] == "10:15:00"


def test_the_original_table_is_not_mutated() -> None:
    """The caller publishes and logs from this; a surprise column in the
    input would reach both by a route nobody declared."""
    table = _table(_row())
    out = _stamp(table, OPEN)
    assert "first_seen_at" not in table.columns
    assert "first_seen_at" in out.columns


def test_an_empty_table_is_returned_unchanged() -> None:
    empty = pd.DataFrame()
    assert scan_publish.stamp_first_seen(empty, OPEN) is empty
    assert scan_publish.stamp_first_seen(None, OPEN) is None


def test_every_row_gets_a_value_so_the_column_is_not_ragged() -> None:
    out = _stamp(_table(_row(symbol="A"), _row(symbol="B", entry=None,
                                               stop=None, quantity=None),
                        _row(symbol="C", actionable=True)), OPEN)
    assert len(out["first_seen_at"]) == 3
    assert len(out["cleared_at"]) == 3


# --- the planned exit ----------------------------------------------------

def test_the_exit_is_a_clock_time_not_a_countdown() -> None:
    """minutes_left answers "when does the market shut", which read as
    "when am I out of this" is the wrong number."""
    import config

    out = _stamp(_table(_row()), OPEN)
    hour, minute = config.SCAN_EXIT_BY
    assert out["exit_by"].iloc[0] == f"{hour:02d}:{minute:02d}"


def test_the_exit_defaults_to_the_session_close() -> None:
    """Because that is what the scanner has always assumed. Changing it
    changes which setups clear and where targets sit, so it is a decision
    rather than a default worth guessing at."""
    import config

    assert config.SCAN_EXIT_BY == config.SCAN_SESSION_CLOSE


def test_an_earlier_exit_is_honoured(monkeypatch) -> None:
    """The point of making it configurable at all."""
    import config

    monkeypatch.setattr(config, "SCAN_EXIT_BY", (15, 15))
    out = _stamp(_table(_row()), OPEN)
    assert out["exit_by"].iloc[0] == "15:15"


def test_every_row_carries_the_exit_including_no_setup_rows() -> None:
    """It is a property of the session, not of the setup."""
    out = _stamp(_table(_row(symbol="A"),
                        _row(symbol="B", direction="NO SETUP", entry=None,
                             stop=None, quantity=None)), OPEN)
    assert out["exit_by"].nunique() == 1
    assert out["exit_by"].notna().all()


# --- the absent feature, pinned ------------------------------------------

def test_the_target_horizon_is_the_bars_left_by_construction() -> None:
    """So an "expected time to target" would restate minutes_left.

    target_distance = atr * sqrt(bars_left) for the default volatility
    stop, so (target_distance / atr)^2 = bars_left exactly. Pinned because
    it is the reason no ETA column exists, and if the target geometry ever
    stops being an identity this test says where to look.
    """
    import levels

    trade = levels.build_levels(
        direction="LONG", entry=1000.0, atr_per_bar=2.0, bars_left=49,
        opening_low=None, opening_high=None, day_low=None, day_high=None)
    assert trade is not None and trade.stop_source == "volatility"
    implied = (trade.target_distance / 2.0) ** 2
    assert implied == pytest.approx(49, rel=1e-6)
