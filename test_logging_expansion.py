"""The scan log has to hold the setups that were REFUSED, not just taken.

WHY. The log is meant to become a track record, and a track record of
only the trades you took cannot answer the one question worth asking of
it: did the gates pick the winners? Every row was a survivor, so there
were no negatives, and nothing can be learned from a dataset with one
class in it.

Measured 2026-09-17: one pass gave a direction to 637 of 2,492 symbols
and cleared 48. The log kept 48. The other 589 already have levels and
already have bars to resolve against, so the counterfactual - "would this
have worked if we had taken it?" - costs nothing but a row.

WHAT MUST NOT BREAK. A setup with no direction has no entry, stop or
target, so outcomes.py has nothing to resolve it against; logging one
adds a row that is counted malformed on every nightly pass forever. And
both files rotate when their columns change, because DictWriter writes in
FIELDS order regardless of the header on disk - appending new columns to
an old file misaligns every row from that point on, which is how `True`
once ended up under `symbol`.
"""
from __future__ import annotations

import csv
from datetime import datetime
from zoneinfo import ZoneInfo

import outcomes
import scan_intraday

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 17, 10, 30, tzinfo=IST)


class _Levels:
    entry, stop, target = 100.0, 98.0, 104.0
    quantity, lot_risk = 10, 20.0
    stop_pct, target_pct = 2.0, 4.0
    breakeven_pct, cost_rupees = 0.12, 12.0
    required_win_rate = 0.34


class _Readings:
    def __init__(self, symbol):
        self.symbol = symbol
        self.rvol = 1.5
        self.relative_strength = 0.8
        self.oi_change_pct = None
        self.futures_share = None
        self.vwap = 99.5
        self.turnover_20d = 1_000_000.0


class _Setup:
    def __init__(self, symbol, direction="LONG", passed=True, levels=True,
                 reasons=None):
        self.readings = _Readings(symbol)
        self.direction = direction
        self.levels = _Levels() if levels else None
        self.passed = passed
        self.rank_score = 0.5
        self.reasons = reasons or []

    @property
    def actionable(self):
        return self.passed and self.levels is not None


# --- what gets written ---------------------------------------------------

def test_a_blocked_setup_is_recorded_as_not_taken() -> None:
    row = scan_intraday._row(_Setup("AAA", passed=False), NOW)
    assert row["taken"] is False


def test_a_cleared_setup_is_recorded_as_taken() -> None:
    row = scan_intraday._row(_Setup("AAA", passed=True), NOW)
    assert row["taken"] is True


def test_the_blocking_gate_is_recorded() -> None:
    setup = _Setup("AAA", passed=False, reasons=[
        "relative volume 1.50x vs floor 1.20x [PASS]",
        "relative strength -0.30pp vs Nifty, needs to outperform [FAIL]",
        "cost 0.30 of a typical day [FAIL]",
    ])
    row = scan_intraday._row(setup, NOW)
    assert row["blocked_by"].startswith("relative strength")


def test_only_the_first_failing_gate_is_recorded() -> None:
    """Gates run in order, so a setup stopped early was never judged on
    the later ones. Counting every failure overstates how often the last
    gate binds."""
    setup = _Setup("AAA", passed=False, reasons=[
        "gate one [FAIL]", "gate two [FAIL]"])
    assert scan_intraday._row(setup, NOW)["blocked_by"] == "gate one"


def test_a_cleared_setup_names_no_blocker() -> None:
    setup = _Setup("AAA", passed=True, reasons=["everything [PASS]"])
    assert scan_intraday._row(setup, NOW)["blocked_by"] == ""


def test_the_new_columns_are_in_the_field_list() -> None:
    """extrasaction="ignore" drops anything absent here, silently."""
    assert "taken" in scan_intraday._CSV_FIELDS
    assert "blocked_by" in scan_intraday._CSV_FIELDS


def test_every_written_key_is_a_declared_field() -> None:
    row = scan_intraday._row(_Setup("AAA"), NOW)
    assert set(row) <= set(scan_intraday._CSV_FIELDS), (
        set(row) - set(scan_intraday._CSV_FIELDS))


# --- the blocker parser --------------------------------------------------

def test_no_reasons_at_all_is_not_an_error() -> None:
    assert scan_intraday.first_blocker(_Setup("AAA")) == ""


def test_a_setup_without_a_reasons_attribute_is_tolerated() -> None:
    class _Bare:
        pass
    assert scan_intraday.first_blocker(_Bare()) == ""


def test_a_very_long_reason_is_truncated() -> None:
    setup = _Setup("AAA", reasons=["x" * 500 + " [FAIL]"])
    assert len(scan_intraday.first_blocker(setup)) <= 120


# --- the outcome join ----------------------------------------------------

def _outcome():
    return outcomes.Outcome(
        row_id="k", outcome="TARGET", exit_time="10:45", exit_price=104.0,
        r_multiple=2.0, bars_held=5, mfe_r=2.1, mae_r=-0.3)


def test_the_labels_reach_the_outcomes_file() -> None:
    logged = {"run_date": "2026-09-17", "run_time": "10:30", "symbol": "AAA",
              "direction": "LONG", "taken": "False",
              "blocked_by": "relative strength"}
    row = outcomes.to_row(logged, _outcome())
    assert row["taken"] == "False"
    assert row["blocked_by"] == "relative strength"


def test_the_outcome_columns_include_the_labels() -> None:
    assert "taken" in outcomes.FIELDS
    assert "blocked_by" in outcomes.FIELDS


def test_a_row_logged_before_the_columns_existed_is_blank_not_guessed(
) -> None:
    """Blank is distinguishable from both True and False, which matters -
    an old row is not evidence that the setup was refused."""
    row = outcomes.to_row({"symbol": "AAA"}, _outcome())
    assert row["taken"] == ""
    assert row["blocked_by"] == ""


# --- rotation, and not re-scoring what was already resolved --------------

def test_a_changed_header_rotates_rather_than_misaligning(tmp_path) -> None:
    target = tmp_path / "outcomes.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "symbol"])
        writer.writerow(["old-key", "AAA"])

    outcomes.append([dict.fromkeys(outcomes.FIELDS, "")], path=target)

    retired = tmp_path / "outcomes.csv.superseded"
    assert retired.exists(), "overwrote a file with different columns"
    with target.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == outcomes.FIELDS


def test_a_rotated_file_is_still_counted_as_resolved(tmp_path) -> None:
    """Otherwise every pre-rotation row is resolved again and double
    counts in every hit rate computed afterwards."""
    retired = tmp_path / "outcomes.csv.superseded"
    with retired.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "symbol"])
        writer.writerow(["already-done", "AAA"])

    assert "already-done" in outcomes.load_resolved(tmp_path / "outcomes.csv")


def test_the_live_and_rotated_files_are_both_read(tmp_path) -> None:
    live = tmp_path / "outcomes.csv"
    for path, key in ((live, "fresh"),
                      (tmp_path / "outcomes.csv.superseded", "old")):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id"])
            writer.writerow([key])
    assert outcomes.load_resolved(live) == {"fresh", "old"}


def test_a_matching_header_is_appended_to_not_rotated(tmp_path) -> None:
    target = tmp_path / "outcomes.csv"
    outcomes.append([dict.fromkeys(outcomes.FIELDS, "a")], path=target)
    outcomes.append([dict.fromkeys(outcomes.FIELDS, "b")], path=target)
    assert not (tmp_path / "outcomes.csv.superseded").exists()
    with target.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
