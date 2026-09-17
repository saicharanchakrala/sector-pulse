"""Pruning the bar cache must never take a bar something still reads.

WHAT THIS GUARDS. bar_store.rebuild folds the consolidated store from the
per-symbol files ALONE - it does not read the store it replaces. So a file
deleted here is gone from bars_3minute.parquet after the next nightly run,
and from bars_day.parquet if the daily interval were ever included.

That makes two mistakes unrecoverable rather than merely wasteful:

  * pruning the DAILY interval would delete history back to about 2001;
  * pruning intraday to "yesterday" - which is what was asked for - would
    leave relative_volume with no baseline. It medians today's cumulative
    volume against the same clock time across roughly a dozen prior
    sessions, and a missing baseline reads as a low ratio, not as an
    error.

So the tests below are mostly refusals, and the retention window is
asserted against the config the readers actually use rather than against
a number written here.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

import config
import market_source
import prune_cache

TODAY = date(2026, 9, 17)


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(market_source, "CACHE_DIR", tmp_path)
    return tmp_path


def _write(tmp_path, symbol, interval, start, end, oi=False, body=b"x" * 100):
    name = market_source._cache_path(symbol, interval, start, end, oi=oi).name
    path = tmp_path / name
    path.write_bytes(body)
    return path


# --- what must never be deleted -----------------------------------------

def test_daily_bars_are_never_pruned(_cache_dir) -> None:
    """bars_day.parquet is rebuilt from these, and holds ~25 years."""
    old = _write(_cache_dir, "RELIANCE", "day", date(2001, 1, 1),
                 date(2001, 3, 1))
    prune_cache.prune(today=TODAY, apply=True)
    assert old.exists(), "deleted daily history"


def test_the_prunable_list_excludes_day() -> None:
    assert "day" not in prune_cache.PRUNABLE


def test_a_file_inside_the_window_survives(_cache_dir) -> None:
    recent = _write(_cache_dir, "RELIANCE", "3minute",
                    TODAY - timedelta(days=5), TODAY - timedelta(days=1))
    prune_cache.prune(today=TODAY, apply=True)
    assert recent.exists()


def test_a_wide_span_reaching_recently_survives(_cache_dir) -> None:
    """Keyed on the END, not the start.

    A span that begins a year ago but runs to yesterday is exactly the
    file the wider-span cache reuse depends on.
    """
    wide = _write(_cache_dir, "RELIANCE", "3minute", date(2025, 1, 1),
                  TODAY - timedelta(days=1))
    prune_cache.prune(today=TODAY, apply=True)
    assert wide.exists()


def test_the_window_covers_every_reader() -> None:
    """Derived from config, so changing a lookback cannot orphan the rule."""
    keep = prune_cache.retention_days()
    assert keep > config.SCAN_REPLAY_LOOKBACK_DAYS
    assert keep > market_source.calendar_days(config.SCAN_BAR_LOOKBACK, 12)
    import live_bars
    assert keep > live_bars.history_days()


def test_the_relative_volume_baseline_is_still_covered() -> None:
    """A dozen prior SESSIONS is the gate's real requirement.

    Calendar days are not sessions, so the window has to clear a dozen
    trading days with weekends and holidays in it.
    """
    assert prune_cache.retention_days() >= 12 * 7 / 5


def test_a_foreign_file_is_left_alone(_cache_dir) -> None:
    stray = _cache_dir / "notes.txt"
    stray.write_text("keep me")
    prune_cache.prune(today=TODAY, apply=True)
    assert stray.exists()


# --- what should be deleted ---------------------------------------------

def test_an_intraday_file_past_the_window_is_pruned(_cache_dir) -> None:
    old = _write(_cache_dir, "RELIANCE", "3minute", date(2026, 1, 1),
                 date(2026, 1, 20))
    count, freed = prune_cache.prune(today=TODAY, apply=True)
    assert not old.exists()
    assert count == 1
    assert freed == 100


@pytest.mark.parametrize("interval", ["minute", "3minute", "5minute",
                                      "15minute", "60minute"])
def test_every_intraday_interval_is_covered(_cache_dir, interval) -> None:
    old = _write(_cache_dir, "X", interval, date(2026, 1, 1),
                 date(2026, 1, 20))
    prune_cache.prune(today=TODAY, apply=True)
    assert not old.exists()


def test_an_oi_variant_is_pruned_too(_cache_dir) -> None:
    old = _write(_cache_dir, "X", "3minute", date(2026, 1, 1),
                 date(2026, 1, 20), oi=True)
    prune_cache.prune(today=TODAY, apply=True)
    assert not old.exists()


# --- the dry run ---------------------------------------------------------

def test_nothing_is_deleted_without_apply(_cache_dir) -> None:
    old = _write(_cache_dir, "X", "3minute", date(2026, 1, 1),
                 date(2026, 1, 20))
    count, freed = prune_cache.prune(today=TODAY, apply=False)
    assert old.exists(), "a dry run deleted a file"
    assert count == 1
    assert freed == 100


def test_the_cli_defaults_to_reporting(_cache_dir, capsys) -> None:
    old = _write(_cache_dir, "X", "3minute", date(2026, 1, 1),
                 date(2026, 1, 20))
    assert prune_cache.main([]) == 0
    assert old.exists()
    assert "would delete" in capsys.readouterr().out


def test_the_cli_deletes_with_apply(_cache_dir, capsys) -> None:
    old = _write(_cache_dir, "X", "3minute", date(2026, 1, 1),
                 date(2026, 1, 20))
    assert prune_cache.main(["--apply"]) == 0
    assert not old.exists()
    assert "deleted" in capsys.readouterr().out


def test_a_zero_retention_window_is_refused(_cache_dir, capsys) -> None:
    """The literal version of the request, refused at the door.

    "Delete 3-minute bars once the day is over" empties the consolidated
    store on the next nightly fold.
    """
    recent = _write(_cache_dir, "X", "3minute", TODAY - timedelta(days=1),
                    TODAY - timedelta(days=1))
    assert prune_cache.main(["--apply", "--keep-days", "0"]) == 1
    assert recent.exists()


def test_an_explicit_override_is_honoured(_cache_dir) -> None:
    old = _write(_cache_dir, "X", "3minute", date(2026, 9, 1),
                 date(2026, 9, 5))
    prune_cache.prune(today=TODAY, keep_days=5, apply=True)
    assert not old.exists()


# --- subsumed spans, which is where the files actually are ---------------

def _age(path, days_old):
    """Backdate a file so containment can be ordered by write time."""
    import os
    import time
    stamp = time.time() - days_old * 86_400
    os.utime(path, (stamp, stamp))
    return path


def test_a_contained_span_is_pruned(_cache_dir) -> None:
    wide = _age(_write(_cache_dir, "RELIANCE", "3minute",
                       TODAY - timedelta(days=17), TODAY), 0)
    narrow = _age(_write(_cache_dir, "RELIANCE", "3minute",
                         TODAY - timedelta(days=16),
                         TODAY - timedelta(days=1)), 1)
    prune_cache.prune(today=TODAY, apply=True)
    assert wide.exists(), "deleted the container"
    assert not narrow.exists(), "kept a span already covered"


def test_a_container_written_EARLIER_does_not_justify_deleting(
        _cache_dir) -> None:
    """Freshness is the whole reason the container is safe to keep.

    An older wide file may predate a correction the newer narrow one has.
    """
    _age(_write(_cache_dir, "X", "3minute", TODAY - timedelta(days=17),
                TODAY), 9)
    narrow = _age(_write(_cache_dir, "X", "3minute",
                         TODAY - timedelta(days=16),
                         TODAY - timedelta(days=1)), 0)
    prune_cache.prune(today=TODAY, apply=True)
    assert narrow.exists(), "deleted a file fresher than its container"


def test_partially_overlapping_spans_are_both_kept(_cache_dir) -> None:
    """Neither contains the other, so neither can stand in for it."""
    left = _age(_write(_cache_dir, "X", "3minute", date(2026, 9, 1),
                       date(2026, 9, 10)), 0)
    right = _age(_write(_cache_dir, "X", "3minute", date(2026, 9, 5),
                        date(2026, 9, 15)), 0)
    prune_cache.prune(today=TODAY, apply=True)
    assert left.exists() and right.exists()


def test_another_symbols_span_never_covers_this_one(_cache_dir) -> None:
    _age(_write(_cache_dir, "WIDE", "3minute", date(2026, 9, 1),
                date(2026, 9, 17)), 0)
    other = _age(_write(_cache_dir, "NARROW", "3minute", date(2026, 9, 5),
                        date(2026, 9, 10)), 1)
    prune_cache.prune(today=TODAY, apply=True)
    assert other.exists()


def test_another_interval_never_covers_this_one(_cache_dir) -> None:
    _age(_write(_cache_dir, "X", "5minute", date(2026, 9, 1),
                date(2026, 9, 17)), 0)
    other = _age(_write(_cache_dir, "X", "3minute", date(2026, 9, 5),
                        date(2026, 9, 10)), 1)
    prune_cache.prune(today=TODAY, apply=True)
    assert other.exists()


def test_an_oi_span_never_covers_a_plain_one(_cache_dir) -> None:
    """The oi flag is part of the key for a reason: the column's absence
    is invisible downstream."""
    _age(_write(_cache_dir, "X", "3minute", date(2026, 9, 1),
                date(2026, 9, 17), oi=True), 0)
    plain = _age(_write(_cache_dir, "X", "3minute", date(2026, 9, 5),
                        date(2026, 9, 10)), 1)
    prune_cache.prune(today=TODAY, apply=True)
    assert plain.exists()


def test_daily_spans_are_not_subsumption_pruned_either(_cache_dir) -> None:
    _age(_write(_cache_dir, "X", "day", date(2001, 1, 1), TODAY), 0)
    inner = _age(_write(_cache_dir, "X", "day", date(2010, 1, 1),
                        date(2015, 1, 1)), 1)
    prune_cache.prune(today=TODAY, apply=True)
    assert inner.exists(), "pruned daily history by subsumption"


def test_subsumption_can_be_turned_off(_cache_dir) -> None:
    _age(_write(_cache_dir, "X", "3minute", TODAY - timedelta(days=17),
                TODAY), 0)
    narrow = _age(_write(_cache_dir, "X", "3minute",
                         TODAY - timedelta(days=16),
                         TODAY - timedelta(days=1)), 1)
    prune_cache.prune(today=TODAY, apply=True, subsumed=False)
    assert narrow.exists()


def test_a_file_is_not_counted_twice(_cache_dir) -> None:
    """Old AND subsumed. Counting it twice would misreport the saving."""
    _age(_write(_cache_dir, "X", "3minute", date(2026, 1, 1),
                date(2026, 1, 31)), 0)
    _age(_write(_cache_dir, "X", "3minute", date(2026, 1, 5),
                date(2026, 1, 20)), 1)
    count, _ = prune_cache.prune(today=TODAY, apply=False)
    assert count == 2, count


def test_an_unreadable_cache_directory_is_not_an_exception(
        monkeypatch) -> None:
    class _Missing:
        def iterdir(self):
            raise OSError("gone")

    monkeypatch.setattr(market_source, "CACHE_DIR", _Missing())
    assert prune_cache.stale_files(today=TODAY) == []
