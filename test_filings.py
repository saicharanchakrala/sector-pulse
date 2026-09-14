"""Tests for the recent-filings context panel.

This module sits at the edge of a NEGATIVE result, and the tests exist as
much to pin down what it must not claim as to check what it does. A filing
measured slightly WORSE than no filing at all, so any wording that reads as
a buy reason is a defect, not a style preference.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import filings

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A small announcement store standing in for the real parquet."""
    def build(rows):
        frame = pd.DataFrame(rows)
        frame["public_at"] = pd.to_datetime(
            frame["public_at"]).dt.tz_localize("Asia/Kolkata")
        path = tmp_path / "announcements.parquet"
        frame.to_parquet(path)
        monkeypatch.setattr(filings, "STORE", path)
        return path
    return build


# --- look-ahead ----------------------------------------------------------

def test_a_filing_at_or_after_the_moment_asked_about_is_not_shown(store) -> None:
    # The display must show only what the trader could already have seen.
    # A filing stamped at the decision moment has not reached them yet, and
    # showing it is the same leak the store was built to avoid.
    store([{"symbol": "X", "desc": "Dividend",
            "public_at": "2026-09-10 11:00:00"}])
    at = datetime(2026, 9, 10, 11, 0, tzinfo=IST)
    assert filings.recent_for("X", now=at) == []
    later = datetime(2026, 9, 10, 11, 1, tzinfo=IST)
    assert len(filings.recent_for("X", now=later)) == 1


def test_a_filing_older_than_the_window_drops_out(store) -> None:
    store([{"symbol": "X", "desc": "Dividend",
            "public_at": "2026-09-10 10:00:00"}])
    at = datetime(2026, 9, 10, 10, 45, tzinfo=IST)
    assert filings.recent_for("X", now=at, minutes=30) == []
    assert len(filings.recent_for("X", now=at, minutes=60)) == 1


def test_a_naive_timestamp_from_a_caller_is_treated_as_ist(store) -> None:
    # Bars are tz-aware but a caller may still hand over a naive datetime.
    # Comparing it raw would raise; assuming UTC would shift the window by
    # five and a half hours and show filings from the wrong part of the day.
    store([{"symbol": "X", "desc": "Dividend",
            "public_at": "2026-09-10 11:00:00"}])
    assert len(filings.recent_for("X", now=datetime(2026, 9, 10, 11, 10))) == 1


# --- what it must not claim ---------------------------------------------

def test_the_note_never_reads_as_a_reason_to_take_the_trade(store) -> None:
    # Measured: 29.6% after a material filing against 30.6% without. A
    # caption implying otherwise would be actively misleading, so the
    # wording is pinned by a test rather than left to future editing.
    store([{"symbol": "X", "desc": "Credit Rating",
            "public_at": "2026-09-10 11:00:00"}])
    note = filings.note_for("X", now=datetime(2026, 9, 10, 11, 5, tzinfo=IST))
    lowered = note.lower()
    assert "not a direction signal" in lowered
    assert "worse" in lowered
    for word in ("buy", "bullish", "opportunity", "strong signal"):
        assert word not in lowered, f"the note must not imply {word!r}"


def test_the_note_names_volatility_which_is_what_replicated(store) -> None:
    # The only finding that survived the holdout was the volatility one, so
    # it is the only thing the caption is allowed to assert.
    store([{"symbol": "X", "desc": "Acquisition",
            "public_at": "2026-09-10 11:00:00"}])
    note = filings.note_for("X", now=datetime(2026, 9, 10, 11, 5, tzinfo=IST))
    assert "volatility" in note.lower()


def test_nothing_recent_produces_no_note_at_all(store) -> None:
    store([{"symbol": "X", "desc": "Dividend",
            "public_at": "2026-09-10 09:30:00"}])
    assert filings.note_for(
        "X", now=datetime(2026, 9, 10, 14, 0, tzinfo=IST)) == ""


# --- category handling ---------------------------------------------------

def test_routine_filings_are_excluded_by_default(store) -> None:
    # Copy of Newspaper Publication is the single largest in-session
    # category and carries nothing. Showing it would bury the real ones.
    store([{"symbol": "X", "desc": "Copy of Newspaper Publication",
            "public_at": "2026-09-10 11:00:00"}])
    at = datetime(2026, 9, 10, 11, 5, tzinfo=IST)
    assert filings.recent_for("X", now=at) == []
    assert len(filings.recent_for("X", now=at, material_only=False)) == 1


def test_the_material_list_is_not_the_one_that_scored_best() -> None:
    # "Credit Rating 41.7%" came out of a seven-way search whose shuffled
    # null produced a 37.0% median best. The list must stay chosen on
    # meaning, so the categories that scored WORST are still in it.
    for weak in ("Bagging/Receiving of orders/contracts", "Acquisition",
                 "Change in Management"):
        assert weak in filings.MATERIAL


@pytest.mark.parametrize("desc,want", [
    ("Outcome of Board Meeting", "material"),
    ("Trading Window", "routine"),
    ("Something Unheard Of", "other"),
    (None, "other"),
])
def test_categories_map_as_expected(desc, want) -> None:
    assert filings.kind_of(desc) == want


# --- degradation ---------------------------------------------------------

def test_an_absent_store_is_silent_rather_than_an_error(monkeypatch,
                                                       tmp_path) -> None:
    # A fresh checkout has no store: it is gitignored cache. The panel must
    # render nothing rather than breaking the scan tab.
    monkeypatch.setattr(filings, "STORE", tmp_path / "missing.parquet")
    assert filings.recent_for("X") == []
    assert filings.note_for("X") == ""
    assert filings.store_age() is None


def test_store_age_reports_how_far_behind_it_is(store) -> None:
    # Silence only means "no news" if the store is current. A caller needs
    # to be able to tell the two apart.
    recent = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=9)
    store([{"symbol": "X", "desc": "Dividend",
            "public_at": recent.tz_localize(None).floor("s")}])
    newest, behind = filings.store_age()
    assert behind >= 8


def test_several_filings_are_counted_and_the_newest_leads(store) -> None:
    store([{"symbol": "X", "desc": "Dividend",
            "public_at": "2026-09-10 11:00:00"},
           {"symbol": "X", "desc": "Acquisition",
            "public_at": "2026-09-10 11:10:00"}])
    at = datetime(2026, 9, 10, 11, 15, tzinfo=IST)
    found = filings.recent_for("X", now=at)
    assert [f["desc"] for f in found] == ["Acquisition", "Dividend"]
    assert "+1 more" in filings.note_for("X", now=at)


def test_another_symbols_filing_is_not_shown(store) -> None:
    store([{"symbol": "Y", "desc": "Dividend",
            "public_at": "2026-09-10 11:00:00"}])
    assert filings.recent_for(
        "X", now=datetime(2026, 9, 10, 11, 5, tzinfo=IST)) == []
