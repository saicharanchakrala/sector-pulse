"""The feed has to record its own scans, once per setup, not once per pass.

WHY IT DID NOT BEFORE. scan_intraday.append_log has exactly one caller -
the command-line scanner. The Streamlit app evaluates setups at
app.py:1429 and never calls it; the feed publishes a parquet table
instead. So the log that was meant to become a track record held 8 rows
from 9 September while the feed scanned 2,485 symbols every 45 seconds
for days, and outcomes.csv held 5 setups from a single instant, three of
them replays.

THE TWO RULES THAT MATTER:

  ONE ROW PER SYMBOL AND DIRECTION PER SESSION. The feed scans every 30
  seconds, so logging each pass would record the same setup four hundred
  times and weight every hit rate by how long a symbol stayed on screen.

  THE ACTIONABLE OBSERVATION WINS. A symbol blocked at 09:20 that clears
  at 11:00 has to be recorded as it was when it CLEARED - that is the
  moment it would have been acted on. Keeping the first observation
  regardless would label it blocked and record levels nobody would have
  traded.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import object_store
import scan_publish

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 21, 10, 30, tzinfo=IST)


def _published(symbol="AAA", direction="LONG", actionable=False, score=0.0,
               entry=100.0, stop=98.0, quantity=10, reasons=""):
    return {
        "run_date": "2026-09-21", "run_time": "10:30:00", "symbol": symbol,
        "direction": direction, "passed": actionable,
        "actionable": actionable, "score": score, "rvol": 1.5,
        "relative_strength": 1.0, "oi_change_pct": None,
        "futures_share": None, "vwap": 99.5, "turnover_20d": 5e8,
        "reasons": reasons, "entry": entry, "stop": stop, "target": 104.0,
        "quantity": quantity, "stop_pct": 2.0, "target_pct": 4.0,
        "breakeven_pct": 0.12, "cost_rupees": 12.0,
        "required_win_rate": 0.34,
    }


def _table(*rows):
    return pd.DataFrame(list(rows))


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(scan_publish, "_SESSION_LOG", {})
    monkeypatch.setattr(scan_publish, "_SESSION_LOG_DAY", None)


@pytest.fixture
def _bucket(monkeypatch):
    held: dict = {}
    monkeypatch.setattr(object_store, "enabled", lambda: True)
    monkeypatch.setattr(object_store, "put",
                        lambda name, payload: held.__setitem__(name, payload))
    return held


# --- the row shape -------------------------------------------------------

def test_the_row_carries_every_column_the_log_defines() -> None:
    """A missing column makes the file malformed to outcomes.py."""
    import scan_intraday

    row = scan_publish.log_row(_published())
    assert set(row) == set(scan_intraday._CSV_FIELDS)


def test_risk_is_derived_from_quantity_and_the_stop_distance() -> None:
    """Not published, but needed to turn an R multiple into rupees.

    Checked against a real logged row: GVT&D, quantity 11, entry 4704.0,
    stop 4616.3 recorded risk_rupees 964.7.
    """
    row = scan_publish.log_row(
        _published(entry=4704.0, stop=4616.3, quantity=11))
    assert row["risk_rupees"] == pytest.approx(964.7, abs=0.05)


def test_a_row_without_levels_has_no_risk() -> None:
    row = scan_publish.log_row(
        _published(entry=None, stop=None, quantity=None))
    assert row["risk_rupees"] is None


def test_the_feed_never_marks_a_row_replayed() -> None:
    """The feed only scans now; a replay is a local, deliberate act."""
    assert scan_publish.log_row(_published())["replayed"] is False


def test_taken_follows_actionable() -> None:
    assert scan_publish.log_row(_published(actionable=True))["taken"] is True
    assert scan_publish.log_row(_published(actionable=False))["taken"] is False


def test_the_first_failing_gate_is_recorded() -> None:
    reasons = ("price 100.00 vs floor 20.00 [PASS] | relative volume 0.53x "
               "vs floor 1.20x [FAIL] | cost 0.30 [FAIL]")
    row = scan_publish.log_row(_published(reasons=reasons))
    assert row["blocked_by"] == "relative volume 0.53x vs floor 1.20x"


def test_a_cleared_setup_names_no_blocker() -> None:
    row = scan_publish.log_row(
        _published(actionable=True, reasons="everything [PASS]"))
    assert row["blocked_by"] == ""


# --- deduplication -------------------------------------------------------

def test_the_same_setup_seen_four_hundred_times_is_one_row() -> None:
    table = _table(_published())
    for _ in range(400):
        scan_publish.remember(table, when=NOW)
    assert len(scan_publish._SESSION_LOG) == 1


def test_both_directions_of_one_symbol_are_separate_rows() -> None:
    scan_publish.remember(_table(_published(direction="LONG"),
                                 _published(direction="SHORT")), when=NOW)
    assert len(scan_publish._SESSION_LOG) == 2


def test_a_setup_that_later_clears_is_recorded_as_cleared() -> None:
    """THE RULE THAT MATTERS. Blocked at 09:20, clears at 11:00.

    Keeping the first observation would label it blocked and record levels
    nobody would have traded.
    """
    scan_publish.remember(
        _table(_published(actionable=False, score=0.0, entry=100.0)),
        when=NOW)
    scan_publish.remember(
        _table(_published(actionable=True, score=0.71, entry=101.5)),
        when=NOW + timedelta(hours=1))
    row = list(scan_publish._SESSION_LOG.values())[0]
    assert row["taken"] is True
    assert row["score"] == 0.71
    assert row["entry"] == 101.5


def test_the_first_cleared_observation_wins_not_the_last() -> None:
    """Once it qualifies, that is the moment it would have been acted on."""
    scan_publish.remember(_table(_published(actionable=True, entry=101.5)),
                          when=NOW)
    scan_publish.remember(_table(_published(actionable=True, entry=107.0)),
                          when=NOW + timedelta(hours=2))
    row = list(scan_publish._SESSION_LOG.values())[0]
    assert row["entry"] == 101.5


def test_a_cleared_setup_is_not_downgraded_by_a_later_block() -> None:
    scan_publish.remember(_table(_published(actionable=True, entry=101.5)),
                          when=NOW)
    scan_publish.remember(_table(_published(actionable=False, entry=99.0)),
                          when=NOW + timedelta(hours=1))
    row = list(scan_publish._SESSION_LOG.values())[0]
    assert row["taken"] is True
    assert row["entry"] == 101.5


def test_a_row_with_no_levels_is_not_logged() -> None:
    """There is no entry, stop or target for outcomes.py to resolve."""
    scan_publish.remember(
        _table(_published(direction="NO SETUP", entry=None, stop=None)),
        when=NOW)
    assert scan_publish._SESSION_LOG == {}


def test_a_new_session_starts_a_new_log() -> None:
    scan_publish.remember(_table(_published()), when=NOW)
    scan_publish.remember(_table(_published(symbol="BBB")),
                          when=NOW + timedelta(days=1))
    assert len(scan_publish._SESSION_LOG) == 1
    assert list(scan_publish._SESSION_LOG)[0][0] == "BBB"


def test_an_empty_table_is_not_an_error() -> None:
    assert scan_publish.remember(pd.DataFrame(), when=NOW) == 0
    assert scan_publish.remember(None, when=NOW) == 0


# --- publishing ----------------------------------------------------------

def test_the_object_is_named_by_day() -> None:
    assert scan_publish.log_object_name(NOW) == "scan_log_20260921.csv"


def test_publishing_writes_a_csv_outcomes_can_read(_bucket) -> None:
    import scan_intraday

    scan_publish.remember(_table(_published(symbol="AAA"),
                                 _published(symbol="BBB")), when=NOW)
    assert scan_publish.publish_log(NOW) == 2

    payload = _bucket["scan_log_20260921.csv"].decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(payload)))
    assert len(rows) == 2
    assert list(rows[0]) == scan_intraday._CSV_FIELDS


def test_publishing_nothing_writes_nothing(_bucket) -> None:
    assert scan_publish.publish_log(NOW) == 0
    assert _bucket == {}


def test_the_whole_session_is_rewritten_each_time(_bucket) -> None:
    """S3 has no append, so a partial rewrite would lose the session."""
    scan_publish.remember(_table(_published(symbol="AAA")), when=NOW)
    scan_publish.publish_log(NOW)
    scan_publish.remember(_table(_published(symbol="BBB")), when=NOW)
    scan_publish.publish_log(NOW)

    payload = _bucket["scan_log_20260921.csv"].decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(payload)))
    assert sorted(r["symbol"] for r in rows) == ["AAA", "BBB"]


# --- outcomes picks the fetched files up ---------------------------------

def test_outcomes_reads_a_fetched_container_log(tmp_path,
                                                monkeypatch) -> None:
    """Otherwise the download is written and never resolved."""
    import config
    import outcomes

    live = tmp_path / "scan_log.csv"
    live.write_text("run_date\n", encoding="utf-8")
    (tmp_path / "scan_log_20260921.csv").write_text("run_date\n",
                                                    encoding="utf-8")
    (tmp_path / "scan_log.csv.superseded").write_text("run_date\n",
                                                      encoding="utf-8")
    monkeypatch.setattr(config, "SCAN_LOG_CSV", live)

    names = [p.name for p in outcomes.log_files()]
    assert "scan_log.csv" in names
    assert "scan_log.csv.superseded" in names
    assert "scan_log_20260921.csv" in names


def test_the_outcomes_file_is_not_mistaken_for_a_scan_log(tmp_path,
                                                          monkeypatch) -> None:
    """The glob must not swallow neighbouring files."""
    import config
    import outcomes

    live = tmp_path / "scan_log.csv"
    live.write_text("run_date\n", encoding="utf-8")
    (tmp_path / "outcomes.csv").write_text("row_id\n", encoding="utf-8")
    (tmp_path / "scan_log_notes.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(config, "SCAN_LOG_CSV", live)

    names = [p.name for p in outcomes.log_files()]
    assert "outcomes.csv" not in names
    assert "scan_log_notes.csv" not in names, "matched a non-dated file"
