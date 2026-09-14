"""Tests for the NSE announcements backfill.

Every test here guards something an independent review actually found, or a
claim the module's own docstring makes. Nothing touches the network: the
session is a stub and the fetch functions are substituted.

THE ORDERING IS DELIBERATE. The look-ahead tests come first because they
are the only failures in this module that would make a backtest LOOK
BETTER. A fetcher that drops rows produces a weak result and someone
investigates; a fetcher that stamps an announcement public earlier than it
was produces a strong one and nobody does.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import fetch_announcements as fa

IST = "Asia/Kolkata"


def _row(**overrides) -> dict:
    """One announcement in the shape NSE actually returns."""
    row = {"symbol": "RELIANCE", "sm_name": "Reliance Industries Limited",
           "smIndustry": "Refineries", "sm_isin": "INE002A01018",
           "desc": "Outcome of Board Meeting", "attchmntText": "text",
           "attchmntFile": "https://nsearchives.nseindia.com/x.pdf",
           "attFileSize": "1 MB", "fileSize": "1 MB",
           "an_dt": "22-Oct-2024 20:11:53",
           "exchdisstime": "22-Oct-2024 20:11:54",
           "sort_date": "2024-10-22 20:11:53", "dt": "22102024201153",
           "difference": "00:00:01", "seq_id": "106031024",
           "hasXbrl": True, "old_new": None, "bflag": None,
           "csvName": None, "orgid": None}
    row.update(overrides)
    return row


# --- look-ahead: the failures that would flatter a backtest ---------------

def test_public_at_is_timezone_aware_so_a_bar_comparison_is_not_shifted() -> None:
    # NSE prints IST without saying so and pd.to_datetime returns naive.
    # Bars are datetime64[us, Asia/Kolkata]. A naive column raises on
    # comparison, which is the SAFE outcome - the dangerous one is a caller
    # who strips the bar's timezone instead and moves every announcement by
    # five and a half hours, far enough to drag an after-close filing into
    # the middle of the session.
    frame = fa.tidy([_row()])
    assert str(frame["public_at"].dtype) == f"datetime64[us, {IST}]"
    bar = pd.Timestamp("2024-10-22 20:11:55", tz=IST)
    # The comparison a caller would naturally write must simply work.
    assert bool(frame["public_at"].iloc[0] < bar)


def test_public_at_uses_dissemination_time_not_the_companys_filing_time() -> None:
    # an_dt is when the company submitted; exchdisstime is when NSE
    # published and therefore when anyone could act. Measured over 358,791
    # rows the gap is a second at the median but exceeds a day on 31 of
    # them, topping out at 2.3 days - so picking the wrong one is not
    # cosmetic, it marks a filing public several sessions early.
    frame = fa.tidy([_row(an_dt="20-Oct-2024 09:00:00",
                          exchdisstime="22-Oct-2024 20:11:54")])
    assert frame["public_at"].iloc[0] == pd.Timestamp(
        "2024-10-22 20:11:54", tz=IST)


@pytest.mark.parametrize("blank", [None, float("nan"), "", "   ", "None",
                                   "nan", "NaT", "-"])
def test_a_blank_dissemination_time_falls_back_rather_than_vanishing(blank) -> None:
    # astype(str) turns None and NaN into the literal "None"/"nan", which
    # are not empty strings. A bare .ne("") test passed them through to a
    # parse that failed, so the row lost its timestamp entirely instead of
    # using the fallback the line exists to provide.
    frame = fa.tidy([_row(exchdisstime=blank,
                          an_dt="22-Oct-2024 20:11:53")])
    assert frame["public_at"].iloc[0] == pd.Timestamp(
        "2024-10-22 20:11:53", tz=IST)


def test_a_response_missing_the_dissemination_field_raises_not_falls_back() -> None:
    # The single most dangerous drift in this module. tidy used to
    # manufacture a missing field as all-None, which sent EVERY row down
    # the an_dt fallback at once - and because public_at is recomputed on
    # load, one such response would retime two years of cached data to
    # filing time with no error anywhere.
    with pytest.raises(KeyError, match="exchdisstime"):
        fa.tidy([{"symbol": "X", "an_dt": "01-Sep-2026 10:00:00",
                  "seq_id": "1"}])


def test_a_future_dated_record_is_dropped_rather_than_marked_public_early() -> None:
    # A record stamped ahead of now would say an announcement was public
    # before it happened. fetch_insider bounds its parse for the same
    # reason. The ceiling is now() rather than midnight tomorrow, because a
    # guard that admits the next 24 hours is not a guard.
    ahead = (pd.Timestamp.now() + pd.Timedelta(days=2)).strftime(
        "%d-%b-%Y %H:%M:%S")
    frame = fa.tidy([_row(an_dt=ahead, exchdisstime=ahead)])
    assert pd.isna(frame["public_at"].iloc[0])


def test_an_unparseable_timestamp_becomes_nat_rather_than_a_wrong_time() -> None:
    frame = fa.tidy([_row(an_dt="not a date", exchdisstime="also not")])
    assert pd.isna(frame["public_at"].iloc[0])


# --- truncation: the failure that silently loses rows ---------------------

def test_a_failed_child_returns_none_rather_than_the_truncated_parent(monkeypatch) -> None:
    # fetch_span exists to stop a silently truncated window entering the
    # dataset. Returning the parent payload on a child failure handed back
    # the very response that tripped the suspicion, and main would store
    # it, tag it complete and never look again.
    calls = []

    def fake(session, first, last):
        calls.append((first, last))
        if (last - first).days + 1 == fa.WINDOW_DAYS:
            return [{"n": i} for i in range(fa.SUSPECT_COUNT + 1)]
        return None                      # both halves fail

    monkeypatch.setattr(fa, "fetch_window", fake)
    monkeypatch.setattr(fa.time, "sleep", lambda _: None)
    got = fa.fetch_span(None, date(2024, 10, 17), date(2024, 10, 23))
    assert got is None, "a suspected-truncated payload must not be returned"
    assert len(calls) > 1, "it must have attempted the split"


def test_a_capped_window_is_split_and_the_halves_are_joined(monkeypatch) -> None:
    def fake(session, first, last):
        span = (last - first).days + 1
        if span == fa.WINDOW_DAYS:
            return [{"n": i} for i in range(fa.SUSPECT_COUNT + 5)]
        return [{"day": first.isoformat(), "span": span}]

    monkeypatch.setattr(fa, "fetch_window", fake)
    monkeypatch.setattr(fa.time, "sleep", lambda _: None)
    got = fa.fetch_span(None, date(2024, 10, 17), date(2024, 10, 23))
    assert got is not None and len(got) == 2


def test_splitting_covers_every_day_exactly_once(monkeypatch) -> None:
    # A split that drops or repeats a day at the boundary would be
    # invisible: the count would simply be a little wrong.
    seen: list = []

    def fake(session, first, last):
        span = (last - first).days + 1
        if span > 1:
            return [{"n": i} for i in range(fa.SUSPECT_COUNT + 1)]
        seen.append(first)
        return [{"day": first.isoformat()}]

    monkeypatch.setattr(fa, "fetch_window", fake)
    monkeypatch.setattr(fa.time, "sleep", lambda _: None)
    first, last = date(2024, 10, 17), date(2024, 10, 23)
    fa.fetch_span(None, first, last)
    want = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    assert sorted(seen) == want
    assert len(seen) == len(set(seen)), "a day was fetched twice"


def test_a_failed_request_is_none_and_an_empty_week_is_a_list() -> None:
    # The two must stay distinguishable: an empty week is data to be
    # recorded, a failed week is a hole that has to be refetched.
    class Dead:
        def get(self, *a, **k):
            raise OSError("no route to host")

    class Empty:
        status_code = 200

        @staticmethod
        def json():
            return {"data": []}

    class Quiet:
        def get(self, *a, **k):
            return Empty()

    day = date(2024, 10, 17)
    assert fa.fetch_window(Dead(), day, day) is None
    assert fa.fetch_window(Quiet(), day, day) == []


# --- the resume grid -----------------------------------------------------

def test_the_window_grid_does_not_move_when_the_run_date_does() -> None:
    # Anchored on `today - years`, every boundary shifted by a day
    # overnight: measured, 0 of 105 tags matched between one day's run and
    # the next, so a "resumable" fetcher refetched two years every morning.
    span = timedelta(days=730)
    today = {fa.window_tag(f)
             for f, _ in fa.week_windows(date(2026, 9, 13) - span,
                                         date(2026, 9, 13))}
    tomorrow = {fa.window_tag(f)
                for f, _ in fa.week_windows(date(2026, 9, 14) - span,
                                            date(2026, 9, 14))}
    assert today <= tomorrow or tomorrow <= today or today & tomorrow
    assert len(today & tomorrow) >= len(today) - 1


def test_a_one_year_run_reuses_the_two_year_runs_windows() -> None:
    end = date(2026, 9, 13)
    two = {fa.window_tag(f) for f, _ in
           fa.week_windows(end - timedelta(days=730), end)}
    one = {fa.window_tag(f) for f, _ in
           fa.week_windows(end - timedelta(days=365), end)}
    assert one <= two, "a shorter run must not invent a new grid"


def test_the_tag_carries_the_span_so_changing_it_cannot_collide() -> None:
    # Without the span in the key, a cached 7-day window and a new 14-day
    # window claim the same name and the days between them are fetched by
    # nothing. Measured on a partial cache, switching 7 to 14 silently
    # skipped 10 days.
    assert fa.window_tag(date(2026, 9, 13)) == f"2026-09-13+{fa.WINDOW_DAYS}"


def test_windows_tile_the_span_without_gap_or_overlap() -> None:
    windows = fa.week_windows(date(2026, 1, 1), date(2026, 3, 31))
    for (_, previous_last), (next_first, _) in zip(windows, windows[1:]):
        assert next_first == previous_last + timedelta(days=1)
    assert windows[0][0] <= date(2026, 1, 1)
    assert windows[-1][1] >= date(2026, 3, 31)


# --- deduplication -------------------------------------------------------

def test_rows_without_an_id_are_kept_rather_than_collapsed_to_one() -> None:
    # drop_duplicates treats NaN as equal to NaN, so a column of nulls
    # reduces to a single row. seq_id is present on all 359,297 records
    # today, but a rename upstream would otherwise reduce the whole
    # dataset to one row on save, silently.
    frame = pd.DataFrame({"seq_id": [None, None, None], "symbol": list("ABC")})
    assert len(fa.deduplicate(frame)) == 3


def test_a_repeated_id_keeps_the_freshest_copy() -> None:
    frame = pd.DataFrame({"seq_id": ["1", "1"], "desc": ["old", "new"]})
    out = fa.deduplicate(frame)
    assert len(out) == 1 and out["desc"].iloc[0] == "new"


# --- reporting -----------------------------------------------------------

def test_the_session_buckets_are_disjoint_and_exhaustive() -> None:
    frame = fa.tidy([
        _row(seq_id="1", exchdisstime="22-Oct-2024 08:00:00"),
        _row(seq_id="2", exchdisstime="22-Oct-2024 11:00:00"),
        _row(seq_id="3", exchdisstime="22-Oct-2024 20:00:00"),
    ])
    split = fa.session_share(frame)
    assert split["before"] == 1 and split["during"] == 1 and split["after"] == 1
    assert split["before"] + split["during"] + split["after"] == split["total"]


def test_a_filing_at_the_closing_bell_counts_as_after_the_close() -> None:
    frame = fa.tidy([_row(exchdisstime="22-Oct-2024 15:30:00")])
    assert fa.session_share(frame)["after"] == 1


def test_an_empty_window_keeps_the_schema_so_a_resumed_run_matches_a_clean_one() -> None:
    # A naively built empty frame downgrades the string columns to object
    # and the timestamp to a different unit, so a resumed run that hit a
    # quiet week wrote a different parquet schema than a clean run.
    empty = fa.tidy([])
    assert list(empty.columns) == fa.FIELDS + ["public_at", "window"]
    assert str(empty["public_at"].dtype) == f"datetime64[us, {IST}]"


def test_every_field_the_api_returns_is_stored() -> None:
    # The docstring claims nothing is filtered at fetch time. The first
    # version discarded 10 of the 20 returned fields while saying so,
    # including old_new, which is the amendment flag this study wants.
    frame = fa.tidy([_row()])
    for field in ("old_new", "difference", "sort_date", "hasXbrl", "orgid"):
        assert field in frame.columns
