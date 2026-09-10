"""Tests for building intraday bars out of live ticks.

The volume tests are the point of this file. Kite's tick carries CUMULATIVE
day volume, so the obvious implementation - summing it across a bar - would
multiply the true figure by the tick count. That error is invisible in the
output (bars look normal, just busier) and would push relative volume, one
of the scanner's nine gates, permanently through its floor.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import config
import live_bars

IST = ZoneInfo("Asia/Kolkata")


def tick(token: int, price: float, cumulative: int, when: datetime):
    """A stand-in for kite_ticker.Tick carrying only what the builder reads."""
    return SimpleNamespace(instrument_token=token, last_price=price,
                           volume=cumulative, exchange_timestamp=when,
                           received_at=when)


def at(hour: int, minute: int, second: int = 0) -> datetime:
    """An explicit wall-clock instant. Used where the exact time matters."""
    return datetime(2026, 9, 9, hour, minute, second, tzinfo=IST)


# Bucket width comes from config, so these tests follow the configured bar
# size instead of hardcoding one. They used to assume five minutes, and
# every one of them broke the day the intraday size moved to three.
STEP = live_bars.BAR_SECONDS // 60


def nth(index: int, second: int = 0) -> datetime:
    """Start of the `index`-th bucket from 10:00, plus optional seconds.

    10:00 is a bucket boundary for every size this project supports, so
    nth(0) is always a clean bucket start and nth(k) is always k buckets
    later - which is what these tests actually mean when they say "the
    next bar".
    """
    return (datetime(2026, 9, 9, 10, 0, tzinfo=IST)
            + timedelta(minutes=STEP * index, seconds=second))


def sess(index: int, second: int = 0) -> datetime:
    """`index` buckets after 09:15, plus optional seconds.

    NOT necessarily a bucket boundary: 09:15 is one at every size up to
    15 minutes, but at 30m and 60m it floors to 09:00. The tests using
    this are about the session open rather than about bucket alignment.
    """
    return (datetime(2026, 9, 9, 9, 15, tzinfo=IST)
            + timedelta(minutes=STEP * index, seconds=second))


def last_second_of(index: int) -> datetime:
    """One second before bucket `index` ends - still inside it."""
    return nth(index + 1) - timedelta(seconds=1)


def near_close(buckets_back: int = 0, second: int = 0) -> datetime:
    """`buckets_back` + 1 buckets before the 15:30 close, plus seconds.

    A true bucket start for every size that divides 30 minutes, which is
    every size this project offers except 60m.
    """
    close = datetime(2026, 9, 9, 15, 30, tzinfo=IST)
    aligned = close - timedelta(minutes=STEP * (buckets_back + 1))
    return aligned + timedelta(seconds=second)


# --- bucketing ------------------------------------------------------------

def test_bars_are_stamped_at_the_start_of_their_bucket() -> None:
    # Kite's historical candles are stamped at the START, and every
    # point-in-time guard in this project relies on that: a bar labelled
    # 10:00 covers 10:00 to 10:00 + one bar.
    assert live_bars.bucket_start(at(10, 0, 1)) == nth(0)
    assert live_bars.bucket_start(nth(1) - timedelta(seconds=1)) == nth(0)
    assert live_bars.bucket_start(nth(1)) == nth(1)
    assert live_bars.bucket_start(nth(1, 1)) == nth(1)
    # Half a bucket past the open is still inside the opening bucket, at
    # any size.
    assert live_bars.bucket_start(sess(0, live_bars.BAR_SECONDS // 2)) == sess(0)


def test_bucketing_is_done_in_ist_whatever_the_tick_carries() -> None:
    utc = datetime(2026, 9, 9, 4, 32, 0, tzinfo=ZoneInfo("UTC"))  # 10:02 IST
    assert live_bars.bucket_start(utc) == nth(0)


# --- volume is a delta ----------------------------------------------------

def test_bar_volume_is_the_delta_not_the_sum_of_cumulative_readings() -> None:
    builder = live_bars.BarBuilder()
    # First bucket establishes the baseline; second is measurable.
    builder.add([tick(1, 100.0, 1_000, nth(0, 5)),
                 tick(1, 101.0, 1_400, nth(0, 60)),
                 tick(1, 99.0, 1_900, last_second_of(0))])
    builder.add([tick(1, 100.5, 2_500, nth(1, 5)),
                 tick(1, 102.0, 3_100, nth(1, 60)),
                 tick(1, 101.5, 3_400, last_second_of(1))])
    builder.add([tick(1, 101.0, 3_500, nth(2, 1))])
    frame = builder.snapshot().sort_values("Date")
    second = frame[frame["Date"] == nth(1)].iloc[0]
    # Cumulative went 1,900 -> 3,400 across the bar, so 1,500 traded.
    # Summing the readings would have given 2,500+3,100+3,400 = 9,000.
    assert second["Volume"] == pytest.approx(1_500.0)
    assert second["Volume"] != pytest.approx(9_000.0)


def test_the_first_bar_has_unknown_volume_rather_than_the_whole_day() -> None:
    # Joining mid-session, the cumulative figure includes everything traded
    # before we were listening. Reporting that as one bar's volume would
    # read as a colossal spike.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 5_000_000, at(11, 7, 30))])
    builder.add([tick(1, 100.0, 5_010_000, at(11, 10, 1))])
    frame = builder.snapshot().sort_values("Date")
    first = frame.iloc[0]
    assert pd.isna(first["Volume"])
    assert bool(first["partial"]) is True


def test_a_later_bar_is_complete_and_not_flagged_partial() -> None:
    builder = live_bars.BarBuilder()
    # The feed joins mid-bucket, so bucket 0 has no volume baseline and is
    # the only partial one. Buckets after it are whole, whatever the
    # configured bar size is.
    builder.add([tick(1, 100.0, 100, nth(0, 60))])
    for index, cumulative in ((1, 250), (2, 400), (3, 600)):
        builder.add([tick(1, 100.0, cumulative, nth(index))])
    frame = builder.snapshot().sort_values("Date")
    later = frame[frame["Date"] == nth(2)].iloc[0]
    assert bool(later["partial"]) is False
    assert later["Volume"] == pytest.approx(150.0)


def test_a_cumulative_figure_that_goes_backwards_cannot_make_volume_negative() -> None:
    # Should not happen, but a reconnect replaying an older tick would do
    # it, and a negative volume would corrupt every downstream average.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 5_000, nth(0)),
                 tick(1, 100.0, 5_000, nth(0, 150))])
    builder.add([tick(1, 100.0, 4_000, nth(1)),
                 tick(1, 100.0, 4_100, nth(1, 30))])
    builder.add([tick(1, 100.0, 6_000, nth(2))])
    volumes = builder.snapshot()["Volume"].dropna()
    assert (volumes >= 0).all()


# --- OHLC -----------------------------------------------------------------

def test_ohlc_comes_from_the_prices_seen_in_that_bucket() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 10, nth(0)),
                 tick(1, 105.0, 20, nth(0, 60)),
                 tick(1, 95.0, 30, nth(0, 120)),
                 tick(1, 102.0, 40, nth(0, 150))])
    builder.add([tick(1, 103.0, 50, nth(1))])
    bar = builder.snapshot().iloc[0]
    assert bar["Open"] == pytest.approx(100.0)
    assert bar["High"] == pytest.approx(105.0)
    assert bar["Low"] == pytest.approx(95.0)
    assert bar["Close"] == pytest.approx(102.0)
    assert bar["ticks"] == 4


def test_a_zero_or_negative_price_is_ignored() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 0.0, 10, nth(0)), tick(1, 100.0, 20, nth(0, 60))])
    builder.add([tick(1, 101.0, 30, nth(1))])
    bar = builder.snapshot().iloc[0]
    assert bar["Open"] == pytest.approx(100.0)
    assert bar["ticks"] == 1


def test_instruments_are_kept_apart() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 10, nth(0)), tick(2, 500.0, 70, nth(0))])
    builder.add([tick(1, 110.0, 20, nth(1)), tick(2, 550.0, 90, nth(1))])
    builder.add([tick(1, 111.0, 30, nth(2)), tick(2, 551.0, 95, nth(2))])
    frame = builder.snapshot()
    assert set(frame["instrument_token"]) == {1, 2}
    one = frame[(frame["instrument_token"] == 1) & (frame["Date"] == nth(1))]
    two = frame[(frame["instrument_token"] == 2) & (frame["Date"] == nth(1))]
    assert one.iloc[0]["Volume"] == pytest.approx(10.0)
    assert two.iloc[0]["Volume"] == pytest.approx(20.0)


# --- the forming bar ------------------------------------------------------

def test_the_forming_bar_is_excluded_by_default() -> None:
    # Every indicator here treats a bar as a finished period. Half a bar
    # served as a whole one understates range and volume.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 10, nth(0)), tick(1, 101.0, 20, nth(0, 120))])
    assert builder.snapshot().empty
    with_forming = builder.snapshot(include_forming=True)
    assert len(with_forming) == 1
    assert bool(with_forming.iloc[0]["partial"]) is True


def test_an_empty_builder_returns_a_frame_with_the_right_columns() -> None:
    frame = live_bars.BarBuilder().snapshot()
    assert frame.empty
    for column in ("instrument_token", "Date", "Open", "High", "Low",
                   "Close", "Volume"):
        assert column in frame.columns


# --- persistence and reading back ----------------------------------------

def test_flush_then_load_round_trips_and_maps_tokens_to_symbols(monkeypatch,
                                                                tmp_path) -> None:
    monkeypatch.setattr(live_bars, "STORE", tmp_path)
    builder = live_bars.BarBuilder()
    for index, cumulative in enumerate((100, 300, 450, 600)):
        builder.add([tick(738561, 100.0 + index, cumulative, nth(index))])
    written = builder.flush(when=nth(0).date())
    assert written >= 3
    loaded = live_bars.load_today({"RELIANCE": 738561}, when=nth(0).date())
    assert list(loaded) == ["RELIANCE"]
    frame = loaded["RELIANCE"]
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(frame.index.tz) == "Asia/Kolkata"
    assert frame.index.is_monotonic_increasing


def test_load_drops_partial_bars_unless_asked_to_keep_them(monkeypatch,
                                                           tmp_path) -> None:
    monkeypatch.setattr(live_bars, "STORE", tmp_path)
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 9_999, nth(0, 60))])   # partial, no baseline
    builder.add([tick(1, 101.0, 10_100, nth(1))])      # one whole bucket
    builder.add([tick(1, 102.0, 10_200, nth(2))])      # closes it; still open
    builder.flush(when=nth(0).date())
    kept = live_bars.load_today({"X": 1}, when=nth(0).date())
    assert len(kept["X"]) == 1                          # only the clean bar
    everything = live_bars.load_today({"X": 1}, when=nth(0).date(),
                                      drop_partial=False)
    assert len(everything["X"]) == 2


def test_load_returns_empty_when_the_feed_has_written_nothing(monkeypatch,
                                                              tmp_path) -> None:
    monkeypatch.setattr(live_bars, "STORE", tmp_path)
    assert live_bars.load_today({"X": 1}) == {}
    assert live_bars.status()["present"] is False


# --- the history window ---------------------------------------------------

def test_the_history_window_ends_yesterday_so_it_caches_all_day() -> None:
    # A window ending TODAY changes every session and misses the parquet
    # cache every morning, which is the delay this module exists to remove.
    start, end = live_bars.history_window(12)
    assert end < datetime.now(IST).date()
    assert (end - start) == timedelta(days=12)

# --- volume when a bar sees no volumetric tick (B1) -----------------------

def quiet(token: int, price: float, when: datetime):
    """A packet with no volume field at all, which is what an index sends.

    Kite's index packets carry price only. An LTP-sized packet for a stock
    does not carry volume either. The builder must be able to tell this
    apart from a genuine cumulative of zero.
    """
    return SimpleNamespace(instrument_token=token, last_price=price,
                           exchange_timestamp=when, received_at=when)


def test_a_bar_with_no_volumetric_tick_reports_unknown_not_zero() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, nth(0)),
                 tick(1, 100.0, 1_000, nth(0, 150))])
    builder.add([quiet(1, 101.0, nth(1)), quiet(1, 102.0, nth(1, 60))])
    builder.add([tick(1, 103.0, 1_600, nth(2))])
    bars = builder.snapshot().set_index("Date")
    # Not zero: nothing was measured, so nothing may be claimed.
    assert pd.isna(bars.loc[nth(1), "Volume"])
    # But NOT flagged partial. The bar covered its whole bucket and its
    # OHLC is sound - only the volume is unknown, which the NaN already
    # says. Flagging it partial made load_today(drop_partial=True) discard
    # every index bar, and with it the benchmark the scanner needs.
    assert not bool(bars.loc[nth(1), "partial"])
    # And the tick count proves the bar was not simply empty.
    assert bars.loc[nth(1), "ticks"] == 2


def test_a_volumeless_bar_does_not_make_the_next_bar_report_the_whole_day() -> None:
    # THE BUG. `if tick.volume:` would not overwrite the baseline, and
    # _finish advanced the mark to it regardless, so the mark became 0 and
    # the following bar reported cumulative-since-open as its own volume.
    # Measured in production as a bar of 15,926,996 against a real few
    # thousand.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000_000, nth(0))])
    builder.add([tick(1, 100.0, 0, nth(1))])          # cumulative of 0
    builder.add([tick(1, 100.0, 1_006_000, nth(2))])
    # An ordinary later tick closes the third bar, so this test needs no
    # method HEAD lacks and therefore fails on the DEFECT rather than on
    # an AttributeError that would mis-report the cause.
    builder.add([tick(1, 100.0, 1_007_000, nth(3))])
    bars = builder.snapshot().set_index("Date")
    assert bars.loc[nth(2), "Volume"] == 6_000


def test_an_index_never_reports_a_volume_it_could_not_have_measured() -> None:
    builder = live_bars.BarBuilder()
    for index in range(4):
        builder.add([quiet(256265, 25_000.0 + index, nth(index))])
    builder.add([quiet(256265, 25_100.0, nth(4))])
    builder.close_open_bars()
    # The loop above used to step five minutes while nth() stepped three,
    # so this closing tick landed in an EARLIER bucket than the open one
    # and was refused as a replay - the test passed on close_open_bars
    # instead of on the path it describes.
    assert not builder.refused()["out_of_order"]
    snapshot = builder.snapshot()
    assert snapshot["Volume"].isna().all()
    # An index has no volume field ever, so if unknown volume implied
    # partial then the whole benchmark series would be dropped by
    # load_today's default. Only the FIRST bar is partial, for the usual
    # reason: it had no prior bar to measure against.
    assert int(snapshot["partial"].sum()) == 1
    assert bool(snapshot.sort_values("Date")["partial"].iloc[0])


# --- ticks outside the session (B5) ---------------------------------------

def test_the_session_bounds_are_inclusive_of_the_open_and_exclusive_of_the_close() -> None:
    assert live_bars.in_session(sess(0))
    assert live_bars.in_session(near_close(0))
    # A bucket STARTING at 15:30 would cover post-close time.
    assert not live_bars.in_session(at(15, 30))
    assert not live_bars.in_session(at(9, 10))
    assert not live_bars.in_session(at(18, 0))


def test_ticks_after_the_close_do_not_become_bars() -> None:
    # 3,024 of 4,320 bars in one measured session were stamped after 15:30,
    # with frozen prices and no volume, and fed ATR and the opening range.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 5_000, near_close(1))])
    builder.add([tick(1, 100.0, 5_500, near_close(0))])
    builder.add([tick(1, 100.0, 5_500, at(15, 40)),
                 tick(1, 100.0, 5_500, at(16, 10)),
                 tick(1, 100.0, 5_500, at(18, 45))])
    # No close_open_bars here: the post-close ticks must not close the
    # last bucket of the session either. Asserting on the completed bars alone keeps this
    # failing on the defect rather than on a missing method.
    stamps = [s.astimezone(IST) for s in builder.snapshot()["Date"]]
    assert stamps == [near_close(1)]
    assert builder.refused()["outside_session"] == 3


def test_ticks_before_the_open_do_not_become_bars() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 0, at(9, 5)), tick(1, 100.0, 0, at(9, 12))])
    builder.add([tick(1, 100.0, 100, sess(0))])
    # A later tick closes the opening bar the ordinary way, so the assertion
    # below is reached against HEAD too.
    builder.add([tick(1, 100.0, 300, sess(1))])
    stamps = [s.astimezone(IST) for s in builder.snapshot()["Date"]]
    assert stamps == [sess(0)]


# --- out-of-order buckets (B2) -------------------------------------------

def test_a_replayed_older_bucket_does_not_reopen_a_finished_bar() -> None:
    # A reconnect replays. Treating an OLDER bucket as "a new bar" closed
    # the newer one, reopened the older, and rewound the volume baseline -
    # so the same period appeared twice and the delta went wrong.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, nth(0))])
    builder.add([tick(1, 105.0, 2_000, nth(1))])
    builder.add([tick(1, 99.0, 1_500, nth(0, 30))])   # the replay
    builder.add([tick(1, 106.0, 3_000, nth(2))])
    bars = builder.snapshot()
    stamps = [s.astimezone(IST) for s in bars["Date"]]
    # No duplicate 10:00, and the 10:05 bar was not truncated by the replay.
    assert stamps == [nth(0), nth(1)]
    assert builder.refused()["out_of_order"] == 1
    closed = bars.set_index("Date")
    assert closed.loc[nth(1), "Volume"] == 1_000
    assert closed.loc[nth(1), "High"] == 105.0


def test_a_second_tick_in_the_same_bucket_still_extends_it() -> None:
    # The guard must reject only STRICTLY older buckets: `!=` was the bug,
    # but `<=` would be a worse one, refusing every tick after the first.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, nth(0)),
                 tick(1, 110.0, 1_200, nth(0, 60)),
                 tick(1, 95.0, 1_400, last_second_of(0))])
    builder.close_open_bars()
    bar = builder.snapshot().set_index("Date").loc[nth(0)]
    assert bar["High"] == 110.0 and bar["Low"] == 95.0
    assert bar["ticks"] == 3
    assert builder.refused()["out_of_order"] == 0


# --- the last bar of the session (B4) ------------------------------------

def test_the_final_bar_of_a_session_is_written_rather_than_lost() -> None:
    # _finish runs only when a LATER tick arrives, and after 15:30 none
    # does, so the closing bucket stayed in _open and flush() - which
    # writes completed bars only - never saw it.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, near_close(1))])
    builder.add([tick(1, 104.0, 1_800, near_close(0)),
                 tick(1, 103.0, 2_000, near_close(0, 30))])
    assert len(builder.snapshot()) == 1          # the last bar is still open
    assert builder.close_open_bars() == 1
    bars = builder.snapshot().set_index("Date")
    assert near_close(0) in [s.astimezone(IST) for s in bars.index]
    assert bars.loc[near_close(0), "Volume"] == 1_000
    assert bars.loc[near_close(0), "Close"] == 103.0
    # Idempotent: calling it again must not duplicate the bar.
    assert builder.close_open_bars() == 0
    assert len(builder.snapshot()) == 2


def test_closing_open_bars_covers_every_instrument() -> None:
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, nth(0)),
                 tick(2, 200.0, 2_000, nth(0)),
                 tick(3, 300.0, 3_000, nth(0))])
    assert builder.close_open_bars() == 3
    assert set(builder.snapshot()["instrument_token"]) == {1, 2, 3}

# --- a bar closed early is not a finished bar (round 2) ------------------

def test_a_bar_closed_mid_bucket_is_flagged_partial() -> None:
    # close_open_bars had no notion of closing EARLY, so a Ctrl-C thirty
    # seconds into a bucket wrote a thirty-second bar recorded as a
    # finished full-length one - and combined() keeps `last`, so on restart
    # it overrode the correct backfilled bar for that stamp.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, nth(1, 10)),
                 tick(1, 101.0, 1_200, nth(1, 40))])
    # Closed while that bucket is still the CURRENT one, so the bar
    # covers 30 seconds of its BAR_SECONDS and must say so.
    assert builder.close_open_bars(now=nth(1, 40)) == 1
    row = builder.snapshot().set_index("Date").loc[nth(1)]
    assert bool(row["partial"])


def test_the_last_bar_of_a_finished_session_is_not_flagged_partial() -> None:
    # The case close_open_bars exists for. At 15:30 the session's last
    # bucket is genuinely over, so its bar is complete and not flagged.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, near_close(1))])
    builder.add([tick(1, 104.0, 1_800, near_close(0)),
                 tick(1, 103.0, 2_000, near_close(0, 30))])
    # The session is over: 15:25 is no longer the current bucket.
    assert builder.close_open_bars(now=at(15, 31)) == 1
    row = builder.snapshot().set_index("Date").loc[near_close(0)]
    assert not bool(row["partial"])
    assert row["Volume"] == 1_000


def test_a_replay_cannot_reopen_a_bucket_already_written() -> None:
    # The out-of-order guard compared against the OPEN bar, and
    # close_open_bars empties that - so a tick arriving afterwards
    # reopened a bucket that had already been emitted.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, nth(0))])
    builder.add([tick(1, 105.0, 2_000, nth(1))])
    builder.close_open_bars()
    builder.add([tick(1, 99.0, 1_500, nth(0, 30))])
    stamps = [s.astimezone(IST) for s in builder.snapshot()["Date"]]
    assert stamps == [nth(0), nth(1)]
    assert builder.refused()["out_of_order"] == 1


# --- a genuine cumulative of zero is information (round 2) ---------------

def test_a_true_cumulative_of_zero_is_a_usable_baseline() -> None:
    # `reading > 0` was the same truthiness test the comment beside it
    # claimed to have replaced, so the first genuinely measurable bar of
    # an illiquid name was thrown away.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 0, sess(0)),
                 tick(1, 100.0, 0, sess(0, 60))])
    builder.add([tick(1, 101.0, 5_000, sess(1))])
    builder.add([tick(1, 102.0, 9_000, sess(2))])
    bars = builder.snapshot().set_index("Date")
    # 09:15 has no prior mark, so its own volume is still unknowable - but
    # it establishes the baseline of 0 that makes 09:20 measurable.
    assert bars.loc[sess(1), "Volume"] == 5_000


# --- a naive stamp is refused, not guessed (round 2) ---------------------

def test_a_naive_timestamp_is_refused_rather_than_assumed_local() -> None:
    # astimezone() on a naive datetime assumes the PROCESS timezone. With
    # the session gate in place, a UTC-configured host would have refused
    # every tick and written no bars at all, silently.
    builder = live_bars.BarBuilder()
    naive = datetime(2026, 9, 9, 10, 0)
    builder.add([SimpleNamespace(instrument_token=1, last_price=100.0,
                                 volume=1_000, exchange_timestamp=naive,
                                 received_at=naive)])
    assert builder.snapshot().empty
    assert builder.refused()["naive_timestamp"] == 1


def test_a_weekend_stamp_is_not_inside_trading_hours() -> None:
    # 2026-09-13 is a Sunday. in_session was time-of-day only.
    sunday = datetime(2026, 9, 13, 11, 0, tzinfo=IST)
    assert not live_bars.in_session(sunday)
    assert live_bars.in_session(datetime(2026, 9, 11, 11, 0, tzinfo=IST))


# --- the staleness limit must not refuse a healthy feed (round 2) --------

def test_the_staleness_limit_exceeds_the_peak_age_of_a_healthy_feed() -> None:
    # A bar closes only when a tick from the NEXT bucket arrives, so the
    # bar starting at S stays the newest in the file until its successor
    # lands at S+2*BAR+flush. The limit was 600s against a peak of 620s,
    # which refused the live path for the first 20s of every bucket.
    flush_default = 20
    peak = live_bars.BAR_SECONDS * 2 + flush_default
    limit = config.SCAN_LIVE_MAX_AGE_SECONDS
    assert limit > peak, (limit, peak)
    # And an UPPER bound, which the hardcoded 900 quietly lost when the bar
    # size changed: 900 against a 380s peak tolerated nearly three missing
    # bars, so a dead feed went on passing as live.
    assert limit <= peak + live_bars.BAR_SECONDS, (limit, peak)


# --- snapshot builds incrementally (round 3) -----------------------------

def test_snapshot_is_incremental_and_still_returns_everything() -> None:
    # It used to rebuild the whole frame from the list of completed bars on
    # every call: 1,805ms at 193,000 bars, 3,061ms at 400,000, every
    # 15-second flush, competing for the GIL with tick handling. Correct
    # but wasteful, and the waste grew all day.
    builder = live_bars.BarBuilder()
    seen = 0
    buckets = 10
    for index in range(buckets):
        builder.add([tick(1, 100.0 + index, 1_000 * (index + 1),
                          nth(index)),
                     tick(2, 200.0 + index, 2_000 * (index + 1),
                          nth(index))])
        frame = builder.snapshot()
        # Monotonic: a cached frame must never lose rows it already had.
        assert len(frame) >= seen
        seen = len(frame)
    builder.close_open_bars()
    final = builder.snapshot()
    assert len(final) == buckets * 2             # every bucket, both tokens
    assert set(final["instrument_token"]) == {1, 2}
    # Repeated calls are stable rather than accumulating duplicates.
    assert len(builder.snapshot()) == 20
    assert len(builder.snapshot()) == 20


def test_the_forming_bars_are_never_cached() -> None:
    # Completed bars are frozen so they can be cached; a forming bar is
    # still moving, so caching it would freeze a price mid-bucket.
    builder = live_bars.BarBuilder()
    builder.add([tick(1, 100.0, 1_000, nth(0))])
    first = builder.snapshot(include_forming=True)
    assert float(first[first["partial"]]["Close"].iloc[0]) == 100.0
    builder.add([tick(1, 123.0, 1_500, nth(0, 120))])
    second = builder.snapshot(include_forming=True)
    assert float(second[second["partial"]]["Close"].iloc[0]) == 123.0
    # And the forming row must not have been appended twice.
    assert len(second) == len(first)


def test_an_empty_builder_still_has_the_right_columns() -> None:
    frame = live_bars.BarBuilder().snapshot()
    assert frame.empty
    assert list(frame.columns) == live_bars.BarBuilder.EMPTY_COLUMNS
