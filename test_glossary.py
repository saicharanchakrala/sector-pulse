"""Every column shown in the UI must carry a plain-English tooltip.

WHY THIS IS A TEST AND NOT A CONVENTION. The tooltips all come from one
helper, so the mechanism was never the problem - the problem was columns
added later that quietly had no glossary entry and rendered bare. Eleven
of them had accumulated on the sector tab before this test existed.

Two layers. The first builds the frames that CAN be built - the two
importable frame builders - and demands full coverage on their real output.
The second scans the source for dict literals that look like table rows and
demands the same, so a column added in app.py fails here rather than
shipping unexplained.
"""
from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

import glossary
import horizons
import instrument_report

REPO = pathlib.Path(__file__).resolve().parent

# Dict keys that look like columns but are not: widget option labels,
# config maps, and the like. Each one is here because it was checked, so
# adding to this list should be a deliberate act rather than a shortcut.
NOT_COLUMNS = {
    "Live (now)", "Replay a past instant",          # radio options
    # scan-scope selectbox options, taken verbatim from app._SCAN_SCOPES
    "F&O single stocks (default)",
    "All listed equities, first 300",
    "All listed equities (very slow)",
    # Pivot-ladder names. These are ROW VALUES under the "Level" column,
    # not columns of their own, and the "Level" tooltip explains the whole
    # ladder in one place.
    "Pivot", "R1", "R2", "R3", "S1", "S2", "S3",
    "Assessment | None", "float | None", "int | None", "str | None",
    "TradeLevels | None", "datetime | None", "date | None",
    "list | None", "dict | None", "bool | None", "PivotLadder | None",
    "CentralPivotRange | None", "OpeningRange | None", "Match | None",
    "Report", "Readings", "Setup",                  # type names in strings
    "CE", "PE", "FUT",                              # instrument types
    "NSE", "NFO", "BSE",                            # exchanges
}


def assessment(horizon: str = "mid"):
    """One Assessment, for to_frame."""
    return horizons.Assessment(
        symbol="TEST", horizon=horizon, price=100.0, direction="LONG",
        trend=5.0, relative=1.5, volatility=25.0, drawdown=-8.0,
        position_in_range=0.7, expected_move=9.0, cost_pct=0.23,
        cost_multiple=39.0, score=0.5, reasons=[], blocked="")


def test_every_column_of_the_horizon_table_has_a_tooltip() -> None:
    frame = horizons.to_frame([assessment()])
    assert not frame.empty
    bare = [c for c in frame.columns if not glossary.column_help(str(c))]
    assert bare == [], f"columns with no tooltip: {bare}"


def test_the_horizon_table_shows_both_price_levels() -> None:
    # The two the user has to act on. A percentage move is not something
    # you can put in an order.
    frame = horizons.to_frame([assessment()])
    assert "Stop loss at" in frame.columns
    assert "Exit price" in frame.columns
    assert frame["Stop loss at"].iloc[0] == pytest.approx(95.5)   # half of 9%
    assert frame["Exit price"].iloc[0] == pytest.approx(109.0)    # the full 9%


def test_the_levels_are_two_to_one_reward_to_risk() -> None:
    # The same convention the intraday sizer uses. If these two disagreed,
    # the required-win-rate arithmetic would be wrong for one of them.
    a = assessment()
    risk = abs(a.price - a.stop_price)
    reward = abs(a.target_price - a.price)
    assert reward / risk == pytest.approx(2.0)


def test_a_short_view_puts_the_stop_above_and_the_exit_below() -> None:
    a = horizons.Assessment(
        symbol="T", horizon="mid", price=100.0, direction="SHORT",
        trend=-5.0, relative=-1.5, volatility=25.0, drawdown=-8.0,
        position_in_range=0.3, expected_move=9.0, cost_pct=0.23,
        cost_multiple=39.0)
    assert a.stop_price > 100.0
    assert a.target_price < 100.0


def test_unusable_inputs_give_no_levels_rather_than_a_wrong_one() -> None:
    for price, move in ((0.0, 9.0), (100.0, float("nan"))):
        a = horizons.Assessment(
            symbol="T", horizon="mid", price=price, direction="LONG",
            trend=0.0, relative=0.0, volatility=25.0, drawdown=0.0,
            position_in_range=0.5, expected_move=move, cost_pct=0.23,
            cost_multiple=1.0)
        assert a.stop_price != a.stop_price      # NaN
        assert a.target_price != a.target_price


def test_every_column_of_the_instrument_summary_has_a_tooltip() -> None:
    report = instrument_report.Report(
        symbol="TEST", found=True, price=100.0,
        verdicts={"mid": instrument_report.HorizonVerdict(
            horizon="mid", verdict="BUY", assessment=assessment())})
    frame = instrument_report.summary_frame(report)
    assert not frame.empty
    bare = [c for c in frame.columns if not glossary.column_help(str(c))]
    assert bare == [], f"columns with no tooltip: {bare}"
    assert "Stop loss at" in frame.columns and "Exit price" in frame.columns


def _column_like_keys(path: pathlib.Path) -> set:
    """String keys of dict literals that look like table rows.

    A row dict is identified by having at least three string keys whose
    first character is upper-case - which is what every display frame in
    this project looks like and what no config map does.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if len(keys) < 3:
            continue
        if sum(1 for k in keys if k[:1].isupper()) < 3:
            continue
        found.update(k for k in keys if k[:1].isupper())
    return found


@pytest.mark.parametrize("name", ["app.py", "horizons.py",
                                  "instrument_report.py"])
def test_no_source_file_builds_a_table_column_without_a_tooltip(name) -> None:
    keys = _column_like_keys(REPO / name)
    bare = sorted(k for k in keys
                  if k not in NOT_COLUMNS and not glossary.column_help(k))
    assert bare == [], (
        f"{name} builds these table columns with no glossary entry: {bare}. "
        f"Add them to glossary.COLUMNS, or to NOT_COLUMNS if they are not "
        f"really columns.")


def test_the_glossary_says_something_useful_rather_than_restating_the_name() -> None:
    # A tooltip that repeats the header helps nobody. Every entry has to be
    # a sentence about what the number MEANS for a decision.
    for name, text in glossary.COLUMNS.items():
        assert len(text) > len(name) + 15, f"{name!r} tooltip is too thin"
        assert text.strip().endswith((".", "?")), f"{name!r} is not a sentence"


def test_the_concepts_are_all_reachable() -> None:
    assert glossary.CONCEPTS
    for name, text in glossary.CONCEPTS.items():
        assert len(text) > 60, f"{name!r} explanation is too thin"
