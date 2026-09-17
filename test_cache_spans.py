"""A cached span that CONTAINS the requested one must serve it.

WHAT BROKE. The cache key is the exact span - symbol, interval, start,
end - and every horizon the app offers is a window that rolls forward
daily. The mid-term table asked for 2026-06-16..2026-09-15 yesterday and
2026-06-17..2026-09-16 today: the same three months of daily bars but for
one session at either end, a different filename, and a total miss.

So the first page load of every trading day refetched months of daily
bars for the whole scope. Measured 2026-09-17: 222 day-spans redownloaded
in one morning, a burst of Kite calls during the session, which is what
the SSLEOFError drops in the console were. bar_cache had accumulated
12,042 files - one per symbol per day, all holding nearly the same bars.

WHAT MUST NOT BREAK. The cache is what the whole app's correctness rests
on, so the tests below pin the refusals as hard as the hits: a span that
does not cover the request, a mismatched interval or oi flag, and the
staleness rule - which has to be judged against the file actually read,
not the one that was asked for.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import market_source as ms

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """A cache directory of our own, and never a network fetch.

    token_for returning None makes a miss end in "Not listed on Kite" and
    an omitted symbol, so "was it served from cache?" is simply "is the
    symbol in the result?" - with no way for a test to reach Kite.
    """
    monkeypatch.setattr(ms, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ms, "token_for", lambda symbol: None)
    monkeypatch.setattr(ms, "_SPANS_MEMO", {})
    return tmp_path


def _write(tmp_path, symbol, interval, start, end, oi=False, sessions=None,
           mtime=None):
    """A cache file exactly where _cache_path would put one."""
    monkey = ms._cache_path(symbol, interval, start, end, oi=oi)
    path = tmp_path / monkey.name
    span = sessions if sessions is not None else (start, end)
    index = pd.date_range(span[0], span[1], freq="D", tz=IST)
    frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                          "Close": 100.0, "Volume": 1000.0}, index=index)
    if oi:
        frame["OpenInterest"] = 5.0
    frame.to_parquet(path)
    if mtime is not None:
        stamp = mtime.timestamp()
        import os
        os.utime(path, (stamp, stamp))
    return path


# A span that ended before today is settled, so the file only has to have
# been written after that date's close.
SETTLED = datetime(2026, 9, 16, 18, 0, tzinfo=IST)


# --- the hit ------------------------------------------------------------

def test_a_wider_cached_span_serves_a_narrower_request(_isolated_cache) -> None:
    _write(_isolated_cache, "HDFCBANK", "day", date(2026, 6, 17),
           date(2026, 9, 15), mtime=SETTLED)
    got = ms.bars(["HDFCBANK"], date(2026, 6, 18), date(2026, 9, 14),
                  interval="day")
    assert "HDFCBANK" in got, "the bars were on disk and were refetched"


def test_the_served_frame_is_sliced_to_what_was_asked_for(
        _isolated_cache) -> None:
    """Serving the wider frame whole would hand back a different range."""
    _write(_isolated_cache, "HDFCBANK", "day", date(2026, 6, 17),
           date(2026, 9, 15), mtime=SETTLED)
    start, end = date(2026, 7, 1), date(2026, 7, 10)
    frame = ms.bars(["HDFCBANK"], start, end, interval="day")["HDFCBANK"]
    assert frame.index[0].date() == start
    assert frame.index[-1].date() == end
    assert len(frame) == 10


def test_the_request_may_match_the_cached_span_exactly(
        _isolated_cache) -> None:
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           mtime=SETTLED)
    got = ms.bars(["X"], date(2026, 6, 17), date(2026, 9, 15), interval="day")
    assert "X" in got


def test_a_symbol_whose_safe_name_holds_underscores_still_matches(
        _isolated_cache) -> None:
    """NIFTY 50 becomes NIFTY_50 on disk, and NIFTY_MEDIA is a real ETF.

    The filename is split, not pattern-matched, because a greedy match
    over an underscored symbol picks the wrong field.
    """
    _write(_isolated_cache, "NIFTY 50", "day", date(2026, 6, 17),
           date(2026, 9, 15), mtime=SETTLED)
    got = ms.bars(["NIFTY 50"], date(2026, 6, 18), date(2026, 9, 14),
                  interval="day")
    assert "NIFTY 50" in got


def test_the_narrowest_containing_span_is_preferred(_isolated_cache) -> None:
    """A tighter file means a smaller read and less slicing."""
    _write(_isolated_cache, "X", "day", date(2025, 1, 1), date(2026, 9, 15),
           mtime=SETTLED)
    tight = _write(_isolated_cache, "X", "day", date(2026, 6, 1),
                   date(2026, 9, 15), mtime=SETTLED)
    spans = ms._cache_spans("day", False)
    chosen = ms._covering_span(spans, "X", date(2026, 7, 1), date(2026, 8, 1))
    assert chosen == tight


# --- the refusals -------------------------------------------------------

@pytest.mark.parametrize("cached_start,cached_end", [
    (date(2026, 7, 1), date(2026, 9, 15)),    # starts too late
    (date(2026, 6, 1), date(2026, 8, 1)),     # ends too early
    (date(2026, 7, 1), date(2026, 8, 1)),     # inside the request
])
def test_a_span_that_does_not_cover_the_request_is_refused(
        _isolated_cache, cached_start, cached_end) -> None:
    _write(_isolated_cache, "X", "day", cached_start, cached_end,
           mtime=SETTLED)
    got = ms.bars(["X"], date(2026, 6, 18), date(2026, 9, 14), interval="day")
    assert "X" not in got, "served bars it does not have"


def test_another_intervals_file_is_never_used(_isolated_cache) -> None:
    """3-minute bars are not daily bars, whatever the dates say."""
    _write(_isolated_cache, "X", "3minute", date(2026, 6, 17),
           date(2026, 9, 15), mtime=SETTLED)
    got = ms.bars(["X"], date(2026, 6, 18), date(2026, 9, 14), interval="day")
    assert "X" not in got


def test_a_frame_without_open_interest_is_not_served_to_an_oi_caller(
        _isolated_cache) -> None:
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           oi=False, mtime=SETTLED)
    got = ms.bars(["X"], date(2026, 6, 18), date(2026, 9, 14), interval="day",
                  oi=True)
    assert "X" not in got, "the missing column is invisible downstream"


def test_an_oi_file_is_not_offered_to_a_plain_caller_by_accident(
        _isolated_cache) -> None:
    """Not a correctness failure, but the key must stay symmetrical."""
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           oi=True, mtime=SETTLED)
    spans = ms._cache_spans("day", False)
    assert ms._covering_span(spans, "X", date(2026, 6, 18),
                             date(2026, 9, 14)) is None


def test_refresh_ignores_the_cache_entirely(_isolated_cache) -> None:
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           mtime=SETTLED)
    got = ms.bars(["X"], date(2026, 6, 18), date(2026, 9, 14), interval="day",
                  refresh=True)
    assert "X" not in got


def test_a_stale_wider_span_is_refused_like_an_exact_one(
        _isolated_cache) -> None:
    """Staleness is judged on the file ACTUALLY read.

    A settled span is only trustworthy if the file was written after that
    date's close; one captured mid-session stops wherever the fetch
    reached. Applying the rule to the requested key rather than the source
    file would exempt every wider span from it.
    """
    mid_session = datetime(2026, 9, 14, 9, 23, tzinfo=IST)
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           mtime=mid_session)
    got = ms.bars(["X"], date(2026, 6, 18), date(2026, 9, 14), interval="day")
    assert "X" not in got, "a truncated capture was served as history"


def test_an_untimestamped_frame_is_refused_rather_than_served_whole(
        _isolated_cache) -> None:
    path = ms._cache_path("X", "day", date(2026, 6, 17), date(2026, 9, 15))
    pd.DataFrame({"Close": [1.0, 2.0]}).to_parquet(_isolated_cache / path.name)
    assert ms._slice_span(pd.DataFrame({"Close": [1.0]}), date(2026, 1, 1),
                          date(2026, 12, 31)) is None


def test_a_slice_with_no_rows_in_range_does_not_masquerade_as_data(
        _isolated_cache) -> None:
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           sessions=(date(2026, 6, 17), date(2026, 6, 20)), mtime=SETTLED)
    got = ms.bars(["X"], date(2026, 8, 1), date(2026, 8, 10), interval="day")
    assert "X" not in got


# --- the filename round trip --------------------------------------------

@pytest.mark.parametrize("symbol", ["HDFCBANK", "NIFTY 50", "NIFTY_MEDIA",
                                    "M&M", "GVT&D"])
@pytest.mark.parametrize("interval,oi", [("day", False), ("3minute", False),
                                         ("day", True)])
def test_a_cache_name_parses_back_to_what_built_it(symbol, interval,
                                                   oi) -> None:
    start, end = date(2026, 6, 17), date(2026, 9, 16)
    name = ms._cache_path(symbol, interval, start, end, oi=oi).name
    parsed = ms._parse_cache_name(name)
    assert parsed is not None, name
    _, got_interval, got_start, got_end, got_oi = parsed
    assert got_interval == interval
    assert (got_start, got_end, got_oi) == (start, end, oi)


@pytest.mark.parametrize("name", [
    "notkite__X__day__20260617_20260916.parquet",
    "kite__X__day__20260617_20260916.txt",
    "kite__X__day__badspan.parquet",
    "kite__day__20260617_20260916.parquet",
    "random.parquet",
])
def test_a_foreign_filename_is_ignored_rather_than_guessed_at(name) -> None:
    assert ms._parse_cache_name(name) is None


# --- the cost of the fallback -------------------------------------------

def test_the_directory_is_not_listed_when_every_exact_key_hits(
        _isolated_cache, monkeypatch) -> None:
    """The fallback must cost nothing on the path it does not help.

    Listing bar_cache takes 1.74 seconds at its real size, and bars() runs
    several times per render.
    """
    start, end = date(2026, 6, 17), date(2026, 9, 15)
    _write(_isolated_cache, "X", "day", start, end, mtime=SETTLED)
    listed = {"n": 0}
    real = ms._cache_spans

    def counting(interval, oi):
        listed["n"] += 1
        return real(interval, oi)

    monkeypatch.setattr(ms, "_cache_spans", counting)
    got = ms.bars(["X"], start, end, interval="day")
    assert "X" in got
    assert listed["n"] == 0, "listed the directory for nothing"


def test_the_index_is_rebuilt_when_the_directory_changes(
        _isolated_cache) -> None:
    """The memo must not outlive the files it describes."""
    first = ms._cache_spans("day", False)
    assert first == {}
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           mtime=SETTLED)
    # Directory mtimes have coarse resolution on some filesystems, so the
    # memo is cleared rather than raced against.
    ms._SPANS_MEMO.clear()
    again = ms._cache_spans("day", False)
    assert "X" in again


def test_a_second_call_reuses_the_index(_isolated_cache) -> None:
    _write(_isolated_cache, "X", "day", date(2026, 6, 17), date(2026, 9, 15),
           mtime=SETTLED)
    first = ms._cache_spans("day", False)
    second = ms._cache_spans("day", False)
    assert second is first, "rebuilt an unchanged directory"
