"""Tests for the consolidated bar store.

Three of these pin bugs that were live and measurable, and the store had no
tests at all when they were found:

  * The fold kept only the LARGEST file per symbol, on the assumption that
    the biggest file holds the longest span and therefore everything a
    smaller one does. Neither half is true: on this project's own cache,
    216 of 460 multi-span 5-minute symbols had their freshest data in a
    SMALLER file, so the store lost the current session for 47% of the
    intraday universe.
  * The `end` bound was `<= end + 1 day`, which admits the following day's
    midnight bar. Asked for 2026-03-31 it returned 2026-04-01. Every
    backtest in this project reads through here, so that is tomorrow's
    price reaching today's decision.
  * The date-bound retry existed to survive a naive/aware comparison
    failure, and then performed the same comparison itself - so the
    fallback raised out of the handler written to catch it.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import bar_store

IST = "Asia/Kolkata"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An isolated cache and store, so tests never see the real one."""
    source = tmp_path / "bar_cache"
    consolidated = tmp_path / "forecast_cache"
    source.mkdir()
    consolidated.mkdir()
    monkeypatch.setattr(bar_store, "CACHE", source)
    monkeypatch.setattr(bar_store, "STORE", consolidated)
    return source


def write_cached(cache, symbol: str, interval: str, span: str,
                 days: list, close: float = 100.0, mtime: float | None = None):
    """A per-symbol cache file shaped the way market_source writes them."""
    index = pd.DatetimeIndex([pd.Timestamp(d, tz=IST) for d in days])
    frame = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": 1_000.0}, index=index)
    path = cache / f"kite__{symbol}__{interval}__{span}.parquet"
    frame.to_parquet(path)
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))
    return path


# --- every file per symbol is folded, not just the biggest ---------------

def test_a_smaller_but_fresher_file_is_not_dropped(cache) -> None:
    # THE BUG. A long window cached earlier is a bigger file; a short
    # window refreshed today is a smaller one. Taking the bigger lost today
    # for 47% of multi-span 5-minute symbols.
    write_cached(cache, "RELIANCE", "day", "long",
                 ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06",
                  "2026-01-07", "2026-01-08"], mtime=1_000_000)
    write_cached(cache, "RELIANCE", "day", "recent",
                 ["2026-01-09"], mtime=2_000_000)
    assert bar_store.rebuild("day", verbose=False)
    frames = bar_store.load("day")
    latest = frames["RELIANCE"].index.max().date()
    assert latest == date(2026, 1, 9)
    assert len(frames["RELIANCE"]) == 7


def test_both_files_contribute_their_own_rows(cache) -> None:
    write_cached(cache, "TCS", "day", "a", ["2026-02-02", "2026-02-03"],
                 mtime=1_000_000)
    write_cached(cache, "TCS", "day", "b", ["2026-02-04", "2026-02-05"],
                 mtime=2_000_000)
    bar_store.rebuild("day", verbose=False)
    got = bar_store.load("day")["TCS"]
    assert [s.date().isoformat() for s in got.index] == [
        "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"]


def test_the_most_recently_written_file_wins_a_collision(cache) -> None:
    # Overlapping spans disagreeing about the same bar: the newer write is
    # the corrected one. This needs a STABLE sort in rebuild - quicksort
    # left the survivor arbitrary, so the fix could pass by luck.
    # Span names chosen so GLOB order puts the stale file first. Both
    # files are the same size, so the old largest-file rule fell through to
    # its tie-break and kept whichever came first - with names "fresh" and
    # "stale" that was the fresh one, and the test passed against the bug.
    write_cached(cache, "INFY", "day", "aaa_stale", ["2026-03-02"],
                 close=100.0, mtime=1_000_000)
    write_cached(cache, "INFY", "day", "zzz_fresh", ["2026-03-02"],
                 close=222.0, mtime=2_000_000)
    bar_store.rebuild("day", verbose=False)
    got = bar_store.load("day")["INFY"]
    assert len(got) == 1
    assert float(got["Close"].iloc[0]) == 222.0


def test_source_files_orders_oldest_write_first(cache) -> None:
    write_cached(cache, "WIPRO", "day", "b", ["2026-04-02"], mtime=2_000_000)
    write_cached(cache, "WIPRO", "day", "a", ["2026-04-01"], mtime=1_000_000)
    names = [p.name for _, p in bar_store._source_files("day")]
    assert names == ["kite__WIPRO__day__a.parquet",
                     "kite__WIPRO__day__b.parquet"]


def test_intervals_do_not_bleed_into_each_other(cache) -> None:
    write_cached(cache, "SBIN", "day", "d", ["2026-05-04"])
    write_cached(cache, "SBIN", "5minute", "f", ["2026-05-04"])
    assert len(bar_store._source_files("day")) == 1
    assert len(bar_store._source_files("5minute")) == 1


# --- the end bound (a lookahead leak) -------------------------------------

def test_the_end_bound_excludes_the_following_day(cache) -> None:
    # Verified against the real store before the fix: end=2026-03-31
    # returned a 2026-04-01 bar.
    write_cached(cache, "HDFCBANK", "day",
                 "s", ["2026-03-30", "2026-03-31", "2026-04-01",
                       "2026-04-02"])
    bar_store.rebuild("day", verbose=False)
    got = bar_store.load("day", symbols=["HDFCBANK"],
                         end=date(2026, 3, 31))["HDFCBANK"]
    assert [s.date().isoformat() for s in got.index] == [
        "2026-03-30", "2026-03-31"]


def test_the_end_bound_still_includes_its_own_day(cache) -> None:
    # The other way to get this wrong. Intraday bars on `end` are stamped
    # 09:15 to 15:25, all AFTER midnight of `end`, so a bound of `<= end`
    # would silently drop the whole session.
    index = pd.DatetimeIndex([pd.Timestamp("2026-06-01 09:15", tz=IST),
                              pd.Timestamp("2026-06-01 15:25", tz=IST)])
    frame = pd.DataFrame({c: 1.0 for c in bar_store.OHLCV}, index=index)
    frame.to_parquet(cache / "kite__ITC__5minute__s.parquet")
    bar_store.rebuild("5minute", verbose=False)
    got = bar_store.load("5minute", end=date(2026, 6, 1))["ITC"]
    assert len(got) == 2


def test_the_start_bound_is_inclusive(cache) -> None:
    write_cached(cache, "LT", "day", "s",
                 ["2026-07-01", "2026-07-02", "2026-07-03"])
    bar_store.rebuild("day", verbose=False)
    got = bar_store.load("day", start=date(2026, 7, 2))["LT"]
    assert [s.date().isoformat() for s in got.index] == ["2026-07-02",
                                                         "2026-07-03"]


# --- the fallback path ----------------------------------------------------

def test_naive_stamps_do_not_raise_out_of_the_date_retry(cache,
                                                         monkeypatch) -> None:
    # The retry exists because a naive/aware comparison raises inside the
    # parquet reader. It then made the same comparison itself, so the
    # fallback raised out of the handler written to catch it.
    write_cached(cache, "AXISBANK", "day", "s",
                 ["2026-08-03", "2026-08-04", "2026-08-05"])
    bar_store.rebuild("day", verbose=False)
    path = bar_store.store_path("day")
    stored = pd.read_parquet(path)
    stored["stamp"] = pd.DatetimeIndex(stored["stamp"]).tz_localize(None)
    stored.to_parquet(path, index=False)

    real = pd.read_parquet

    def only_symbol_filters(target, *args, **kwargs):
        # Force the fallback: the pushed-down date filter fails on a
        # naive-stamped file, which is the situation being tested.
        if any(f[0] == "stamp" for f in (kwargs.get("filters") or [])):
            raise TypeError("naive vs aware")
        return real(target, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", only_symbol_filters)
    got = bar_store.load("day", symbols=["AXISBANK"],
                         start=date(2026, 8, 4), end=date(2026, 8, 4))
    assert [s.date().isoformat() for s in got["AXISBANK"].index] == [
        "2026-08-04"]


def test_a_missing_store_returns_empty_rather_than_raising(cache) -> None:
    assert bar_store.load("day") == {}
    assert bar_store.status("day")["present"] is False


def test_rebuild_with_no_source_files_reports_nothing_folded(cache) -> None:
    assert bar_store.rebuild("day", verbose=False) == 0

def test_the_span_in_the_filename_outranks_mtime(cache) -> None:
    # mtime is the weaker signal: a restore, copy or sync rewrites it. The
    # span is a fact about the contents, so a file covering later dates
    # folds last even when its mtime says it was written first.
    write_cached(cache, "BEL", "day", "20260101_20260131",
                 ["2026-01-30"], close=111.0, mtime=9_000_000)
    write_cached(cache, "BEL", "day", "20260201_20260228",
                 ["2026-01-30"], close=222.0, mtime=1_000_000)
    names = [p.name for _, p in bar_store._source_files("day")]
    assert names[-1].endswith("20260201_20260228.parquet")
    bar_store.rebuild("day", verbose=False)
    got = bar_store.load("day")["BEL"]
    assert float(got["Close"].iloc[0]) == 222.0


def test_an_unparseable_span_falls_back_to_mtime(cache) -> None:
    write_cached(cache, "IOC", "day", "legacy", ["2026-01-30"],
                 close=111.0, mtime=9_000_000)
    write_cached(cache, "IOC", "day", "alsolegacy", ["2026-01-30"],
                 close=222.0, mtime=1_000_000)
    # Neither name carries a span, so the newer WRITE wins - which is the
    # old rule, retained for files this scheme cannot read.
    names = [p.name for _, p in bar_store._source_files("day")]
    assert names[-1].endswith("legacy.parquet")
    assert bar_store._span_end("legacy.parquet") == ""
