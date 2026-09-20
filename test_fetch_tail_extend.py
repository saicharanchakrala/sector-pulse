"""The nightly job has to move existing symbols forward, not just backfill.

WHAT WENT WRONG. fetch_tail's only job was `listed - have` - symbols the
consolidated daily store had NEVER seen. Nothing advanced a symbol
already present. What advanced them was an accident: the UI's
longer-horizon tables re-downloaded three months of daily bars every
morning, refreshing the per-symbol day files that the fold consolidated.

Switching that off on 2026-09-18 - to stop the UI hammering Kite and
exhausting the machine - removed the only thing advancing the store. Four
days later:

    2026-09-09   2,518 symbols
    2026-09-11   2,315
    2026-09-15   1,055
    2026-09-16      12

prev_close was therefore a DIFFERENT DATE per symbol. That is worse than
a uniform lag: the relative-strength gate compares a symbol's day change
against the index across a universe whose members were measuring from
different days, and nothing anywhere reported it. The nightly log said
"daily store now holds 2,521 symbols", which was true throughout.

So the tests below pin the extension, the grouping that makes it one
request per symbol, and the coverage report - because the missing report
is what let it hide.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import fetch_tail

IST = "Asia/Kolkata"


def _frame(through, sessions=5):
    index = pd.date_range(end=pd.Timestamp(through, tz=IST),
                          periods=sessions, freq="D")
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                         "Close": 100.0, "Volume": 1000.0}, index=index)


# --- the settled end -----------------------------------------------------

def test_the_end_is_yesterday_on_a_weekday() -> None:
    # 2026-09-17 is a Thursday, so yesterday is Wednesday the 16th.
    assert fetch_tail.settled_end(date(2026, 9, 17)) == date(2026, 9, 16)


@pytest.mark.parametrize("today,expected", [
    (date(2026, 9, 20), date(2026, 9, 18)),   # Sunday  -> Friday
    (date(2026, 9, 21), date(2026, 9, 18)),   # Monday  -> Friday
    (date(2026, 9, 19), date(2026, 9, 18)),   # Saturday-> Friday
])
def test_the_end_walks_back_off_a_weekend(today, expected) -> None:
    """Otherwise every Sunday and Monday refetches the whole universe for
    a session that never existed."""
    assert fetch_tail.settled_end(today) == expected


# --- which symbols are behind --------------------------------------------

def test_a_symbol_reaching_the_end_is_left_alone() -> None:
    groups = fetch_tail.stale_groups({"AAA": date(2026, 9, 18)},
                                     date(2026, 9, 18))
    assert groups == {}


def test_a_symbol_behind_the_end_is_grouped_by_where_it_reaches() -> None:
    newest = {"AAA": date(2026, 9, 11), "BBB": date(2026, 9, 11),
              "CCC": date(2026, 9, 15)}
    groups = fetch_tail.stale_groups(newest, date(2026, 9, 18))
    assert groups == {date(2026, 9, 11): ["AAA", "BBB"],
                      date(2026, 9, 15): ["CCC"]}


def test_grouping_means_one_request_per_symbol() -> None:
    """market_source.bars takes ONE span for a list, so the grouping is
    what keeps this to a handful of calls over short spans rather than a
    two-year backfill each."""
    newest = {f"S{i}": date(2026, 9, 11) for i in range(2000)}
    newest.update({f"T{i}": date(2026, 9, 15) for i in range(500)})
    groups = fetch_tail.stale_groups(newest, date(2026, 9, 18))
    assert len(groups) == 2
    assert sum(len(v) for v in groups.values()) == 2500


def test_a_symbol_ahead_of_the_end_is_not_refetched() -> None:
    """A store written after the close already has today's bar."""
    groups = fetch_tail.stale_groups({"AAA": date(2026, 9, 19)},
                                     date(2026, 9, 18))
    assert groups == {}


# --- the fetch -----------------------------------------------------------

@pytest.fixture
def _store(monkeypatch):
    def install(newest):
        frames = {s: _frame(d) for s, d in newest.items()}
        monkeypatch.setattr(fetch_tail.bar_store, "load",
                            lambda interval="day", **kw: dict(frames))
    return install


def test_every_group_is_fetched_from_where_it_reaches(monkeypatch,
                                                      _store) -> None:
    _store({"AAA": date(2026, 9, 11), "CCC": date(2026, 9, 15)})
    asked = []

    def fake(symbols, start, end, interval="day", **kw):
        asked.append((start, end, tuple(sorted(symbols))))
        return {s: _frame(end) for s in symbols}

    monkeypatch.setattr(fetch_tail.market_source, "bars", fake)
    refreshed = fetch_tail.extend_stale(end=date(2026, 9, 18))

    assert refreshed == 2
    assert (date(2026, 9, 11), date(2026, 9, 18), ("AAA",)) in asked
    assert (date(2026, 9, 15), date(2026, 9, 18), ("CCC",)) in asked


def test_the_span_starts_on_the_day_already_held(monkeypatch,
                                                 _store) -> None:
    """The one-bar overlap rewrites a last session that was itself
    partial rather than building on it."""
    _store({"AAA": date(2026, 9, 15)})
    spans = []
    monkeypatch.setattr(fetch_tail.market_source, "bars",
                        lambda s, start, end, **kw: spans.append(start) or {})
    fetch_tail.extend_stale(end=date(2026, 9, 18))
    assert spans == [date(2026, 9, 15)]


def test_nothing_behind_means_no_requests(monkeypatch, _store) -> None:
    _store({"AAA": date(2026, 9, 18), "BBB": date(2026, 9, 18)})

    def explode(*args, **kwargs):
        raise AssertionError("fetched when everything was current")

    monkeypatch.setattr(fetch_tail.market_source, "bars", explode)
    assert fetch_tail.extend_stale(end=date(2026, 9, 18)) == 0


def test_an_expired_token_reports_rather_than_crashing(monkeypatch, capsys,
                                                       _store) -> None:
    """THE REGRESSION TEST.

    Kite tokens die about 06:00 and this job runs after the close, so a
    session that has not been refreshed is the ordinary case for a job run
    a day late - not a crash. Letting NoSession escape aborted the whole
    of fetch_tail, including the fold, and printed a traceback into the
    nightly log where the only useful content is which action fixes it.
    """
    _store({"AAA": date(2026, 9, 11)})

    def expired(*args, **kwargs):
        raise fetch_tail.market_source.NoSession(
            "No usable Kite session, so no prices can be fetched. "
            "Run: .venv\\Scripts\\python -m kite_login")

    monkeypatch.setattr(fetch_tail.market_source, "bars", expired)
    refreshed = fetch_tail.extend_stale(end=date(2026, 9, 18))

    assert refreshed == 0
    out = capsys.readouterr().out
    assert "CANNOT EXTEND" in out
    assert "kite_login" in out, "did not say which action fixes it"
    assert "prev_close remains older" in out


def test_a_token_expiring_mid_run_keeps_what_was_already_fetched(
        monkeypatch, _store) -> None:
    """The first group's bars are cached and real; only the rest is lost."""
    _store({"AAA": date(2026, 9, 11), "CCC": date(2026, 9, 15)})
    calls = {"n": 0}

    def flaky(symbols, start, end, interval="day", **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {s: _frame(end) for s in symbols}
        raise fetch_tail.market_source.NoSession("expired")

    monkeypatch.setattr(fetch_tail.market_source, "bars", flaky)
    assert fetch_tail.extend_stale(end=date(2026, 9, 18)) == 1


def test_an_empty_store_is_not_an_error(monkeypatch) -> None:
    monkeypatch.setattr(fetch_tail.bar_store, "load",
                        lambda interval="day", **kw: {})
    assert fetch_tail.extend_stale(end=date(2026, 9, 18)) == 0


def test_an_unreadable_store_is_not_an_error(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("gone")

    monkeypatch.setattr(fetch_tail.bar_store, "load", boom)
    assert fetch_tail.newest_sessions() == {}


def test_an_empty_frame_is_not_counted_as_coverage(monkeypatch) -> None:
    monkeypatch.setattr(fetch_tail.bar_store, "load",
                        lambda interval="day", **kw: {"AAA": pd.DataFrame(),
                                                      "BBB": None})
    assert fetch_tail.newest_sessions() == {}


# --- the report that would have caught it --------------------------------

def test_the_report_names_the_session_and_how_many_reach_it(
        monkeypatch, capsys, _store) -> None:
    """"now holds 2,521 symbols" was true for four stale days.

    What was missing was how far the store REACHES, so the report has to
    print the date and the count that got there.
    """
    _store({"AAA": date(2026, 9, 18), "BBB": date(2026, 9, 11),
            "CCC": date(2026, 9, 11)})
    monkeypatch.setattr(fetch_tail, "settled_end",
                        lambda today=None: date(2026, 9, 18))
    fetch_tail.report_coverage()
    out = capsys.readouterr().out
    assert "1 reach 2026-09-18" in out
    assert "BEHIND" in out
    assert "2 symbols did not reach" in out


def test_a_level_store_reports_no_laggards(monkeypatch, capsys,
                                           _store) -> None:
    _store({"AAA": date(2026, 9, 18), "BBB": date(2026, 9, 18)})
    monkeypatch.setattr(fetch_tail, "settled_end",
                        lambda today=None: date(2026, 9, 18))
    fetch_tail.report_coverage()
    out = capsys.readouterr().out
    assert "2 reach 2026-09-18" in out
    assert "BEHIND" not in out
    assert "did not reach" not in out
