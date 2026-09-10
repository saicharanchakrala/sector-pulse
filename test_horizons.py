"""Tests for the multi-horizon assessment and its cross-sectional ranking.

The ranking tests pin a bug that made the composite do the OPPOSITE of what
it documented. `cost_multiple` is `expected_move / cost_pct`, and `cost_pct`
is a constant per horizon, so cost_multiple is volatility times a constant -
measured on 372 real names at a ratio of exactly 2.173913, with
spearman(multiple, -volatility) = -1.000000.

Scoring both, 0.30 on rank(multiple) and 0.10 on rank(-volatility), summed
to `0.10 + 0.20 * rank(multiple)`: a constant plus a POSITIVE loading on
volatility. Controlling for the other two inputs, the old composite's
marginal correlation with volatility was +1.0000 - a perfect reward where
the weight exists to impose a penalty.

None of this makes the ranking predictive. Four horizons were measured on
this project's own data and none showed skill. These tests only hold the
code to what it claims to compute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import horizons

IST = "Asia/Kolkata"


def daily(closes: list, start: str = "2025-01-01") -> pd.DataFrame:
    """A daily frame with business-day stamps, as bar_store hands back."""
    index = pd.date_range(start, periods=len(closes), freq="B", tz=IST)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": [1e6] * len(closes)},
                        index=index)


def assessment(relative: float, volatility: float, position: float = 0.5,
               multiple: float = 10.0, symbol: str = "X"):
    """An Assessment with only the fields rank() reads."""
    sessions = horizons.HORIZONS["mid"]["sessions"]
    cost = float(horizons.HORIZONS["mid"]["cost_pct"])
    expected = volatility / np.sqrt(252.0) * np.sqrt(sessions)
    return horizons.Assessment(
        symbol=symbol, horizon="mid", direction="LONG", price=100.0,
        trend=relative, relative=relative, volatility=volatility,
        drawdown=-5.0, position_in_range=position,
        expected_move=expected, cost_pct=cost,
        cost_multiple=(expected / cost if cost > 0 else 0.0),
        reasons=[], blocked="")


# --- the volatility penalty must be able to fire --------------------------

def test_two_names_alike_but_for_volatility_are_ordered_by_it() -> None:
    # The whole point of the weight. Identical relative strength and range
    # position, so volatility is the only thing left to separate them, and
    # the calmer name must win.
    calm = assessment(10.0, 20.0, symbol="CALM")
    wild = assessment(10.0, 60.0, symbol="WILD")
    ordered = horizons.rank([wild, calm], top=2)
    assert [a.symbol for a in ordered] == ["CALM", "WILD"]
    assert calm.score > wild.score


def test_the_composite_does_not_reward_volatility_at_the_margin() -> None:
    # The measured failure, reproduced as a test. Holding the other two
    # inputs fixed, score must fall as volatility rises. The old composite
    # gave a marginal correlation of +1.0 here.
    fixed = [assessment(5.0, vol, position=0.5, symbol=f"V{vol:.0f}")
             for vol in (15.0, 25.0, 35.0, 45.0, 55.0)]
    horizons.rank(list(fixed), top=len(fixed))
    scores = [a.score for a in fixed]
    assert scores == sorted(scores, reverse=True), scores


def test_relative_strength_still_dominates_the_ordering() -> None:
    # The penalty must not have become the main term. A much stronger name
    # should still outrank a calmer but weaker one.
    strong = assessment(40.0, 50.0, symbol="STRONG")
    calm = assessment(1.0, 15.0, symbol="CALM")
    ordered = horizons.rank([calm, strong], top=2)
    assert [a.symbol for a in ordered][0] == "STRONG"


def test_cost_coverage_is_a_hard_partition_not_a_score_term() -> None:
    # Dropping `multiple` from the composite must not lose the cost rule:
    # a name that cannot cover its costs ranks below every name that can,
    # however strong it looks.
    payable = assessment(1.0, 30.0, symbol="PAYS")
    unpayable = assessment(90.0, 30.0, symbol="CANNOT")
    unpayable.cost_multiple = 0.5           # below MIN_COST_MULTIPLE of 3
    assert not unpayable.pays_for_itself and payable.pays_for_itself
    ordered = horizons.rank([unpayable, payable], top=2)
    assert [a.symbol for a in ordered] == ["PAYS", "CANNOT"]


def test_ranking_an_empty_list_returns_nothing() -> None:
    assert horizons.rank([], top=5) == []
    assert horizons.rank([None, None], top=5) == []


# --- prices that would break the arithmetic ------------------------------

def test_a_zero_price_at_the_start_of_the_window_is_refused() -> None:
    # `trend` divides by the window's FIRST close and only the last was
    # checked, so this gave an infinite trend that then led every
    # cross-sectional rank in the run.
    lookback = horizons._lookback_for("short")
    closes = [100.0 + i for i in range(80)]
    # The window is the LAST lookback+1 closes, so the divisor is here.
    closes[-(lookback + 1)] = 0.0
    got = horizons.assess_daily("ZERO", daily(closes), "short", None)
    assert got is None or np.isfinite(got.trend)


def test_a_negative_price_does_not_flip_the_trend_sign() -> None:
    lookback = horizons._lookback_for("short")
    closes = [100.0 + i for i in range(80)]
    closes[-(lookback + 1)] = -50.0
    # Prices rose across the window, so a NEGATIVE trend here could only
    # come from the sign flip a negative divisor produces.
    got = horizons.assess_daily("NEG", daily(closes), "short", None)
    assert got is None or got.trend > 0


def test_volatility_ignores_non_positive_closes_rather_than_clamping() -> None:
    # Clamping the divisor to 1e-9 turned one bad price into a return of
    # about 1e11, which took the top of the volatility column outright.
    good = np.array([100.0 + i * 0.5 for i in range(80)])
    with_hole = good.copy()
    with_hole[40] = 0.0
    clean = horizons._annualised_volatility(good)
    holed = horizons._annualised_volatility(with_hole)
    assert np.isfinite(holed)
    assert holed < clean * 3.0, (clean, holed)


def test_too_little_history_returns_nothing_rather_than_a_guess() -> None:
    assert horizons.assess_daily("SHORT", daily([100.0] * 8), "long",
                                 None) is None
    assert horizons.assess_daily("NONE", None, "mid", None) is None


# --- the benchmark must be compared over the same dates -------------------

def test_the_benchmark_is_aligned_on_dates_not_row_positions() -> None:
    # A benchmark with extra leading history is the common case - the index
    # has no listing gaps and individual names do. Positionally,
    # bench[-(lookback+1)] then sits on a different DATE, so the excess
    # return subtracted a different span from the one it measured.
    lookback = horizons._lookback_for("short")
    n = lookback + 20
    stock = daily([100.0 * (1.02 ** i) for i in range(n)])
    # Same dates, same path, so the excess return must be ~0 whatever
    # history precedes it.
    long_bench = daily([100.0 * (1.02 ** i) for i in range(n)])
    got = horizons.assess_daily("S", stock, "short", long_bench)
    assert got is not None
    assert abs(got.relative) < 1e-6, got.relative

    # Now drop five sessions from INSIDE the benchmark trailing window, as
    # a holiday the index observed and the stock did not, or a missing bar.
    # The window-start date itself survives, so a date-aligned lookup is
    # exact. Positionally, bench[-(lookback+1)] now sits five sessions
    # EARLIER than the stock window it is subtracted from.
    gapped = long_bench.drop(long_bench.index[-(lookback - 3):-(lookback - 8)])
    assert len(gapped) == len(long_bench) - 5
    aligned = horizons.assess_daily("S", stock, "short", gapped)
    assert aligned is not None
    # Date-aligned the answer is unchanged. Positionally it was not: five
    # extra sessions of 2% compounding is about 10 percentage points of
    # phantom excess return.
    assert abs(aligned.relative) < 1e-6, aligned.relative


def test_a_missing_benchmark_leaves_relative_equal_to_trend() -> None:
    lookback = horizons._lookback_for("short")
    stock = daily([100.0 + i for i in range(lookback + 20)])
    got = horizons.assess_daily("S", stock, "short", None)
    assert got is not None
    assert got.relative == got.trend


def test_a_benchmark_without_a_close_column_is_ignored_safely() -> None:
    lookback = horizons._lookback_for("short")
    stock = daily([100.0 + i for i in range(lookback + 20)])
    got = horizons.assess_daily("S", stock, "short",
                                pd.DataFrame({"Open": [1.0, 2.0]}))
    assert got is not None
    assert got.relative == got.trend

# --- price hygiene must reach every scored reading (round 2) -------------

def test_a_zero_close_does_not_poison_the_range_position() -> None:
    # Cleaning inside _annualised_volatility alone left position_in_range
    # and drawdown - both scored - reading the raw array, where a single
    # zero sets `low` and therefore the whole range position.
    lookback = horizons._lookback_for("short")
    closes = [100.0 + i for i in range(80)]
    clean = horizons.assess_daily("C", daily(list(closes)), "short", None)
    closes[-3] = 0.0
    holed = horizons.assess_daily("H", daily(closes), "short", None)
    assert clean is not None and holed is not None
    assert abs(holed.position_in_range - clean.position_in_range) < 0.05
    assert abs(holed.drawdown - clean.drawdown) < 1.0


# --- the benchmark must actually reach the end of the window (round 2) ---

def test_a_benchmark_that_stops_early_is_refused_not_carried_forward() -> None:
    # reindex(..., method="ffill") has no staleness limit, so an index
    # frame ending before the stock's last date would carry its last known
    # close forward and be compared against the stock's current price - a
    # fabricated excess return for every symbol at once.
    lookback = horizons._lookback_for("short")
    n = lookback + 20
    stock = daily([100.0 * (1.02 ** i) for i in range(n)])
    bench = daily([100.0 * (1.02 ** i) for i in range(n)])
    stale = bench.iloc[:-10]
    got = horizons.assess_daily("S", stock, "short", stale)
    assert got is not None
    # No usable benchmark, so relative falls back to the raw trend rather
    # than to a number computed against a stale index close.
    assert got.relative == got.trend


# --- the live price anchor ------------------------------------------------
#
# Measured on RELIANCE at 09:36 on 2026-09-10: the daily store's newest
# close was 1288.70 while the live price was 1274.50. Every level came off
# that stale anchor, so the table offered a 2:1 stop and target that became
# 6.7:1 for anyone placing it at the live price - risk 0.71% where 1.81%
# was intended. The required-win-rate arithmetic assumes 2:1, so it was
# quoting a figure for a trade nobody could place.

def test_a_live_price_re_anchors_the_levels() -> None:
    lookback = horizons._lookback_for("short")
    frame = daily([100.0 + i * 0.2 for i in range(lookback + 20)])
    close_anchored = horizons.assess_daily("S", frame, "short", None)
    live_anchored = horizons.assess_daily("S", frame, "short", None,
                                          live_price=90.0)
    assert close_anchored is not None and live_anchored is not None
    assert live_anchored.price == pytest.approx(90.0)
    assert live_anchored.stop_price != close_anchored.stop_price
    assert live_anchored.target_price != close_anchored.target_price


def test_the_live_anchor_keeps_reward_to_risk_at_two_to_one() -> None:
    # The whole point. A stale anchor silently changed this ratio, and the
    # cost arithmetic depends on it being 2:1.
    lookback = horizons._lookback_for("short")
    frame = daily([100.0 + i * 0.2 for i in range(lookback + 20)])
    for price in (80.0, 95.0, 110.0, 130.0):
        a = horizons.assess_daily("S", frame, "short", None, live_price=price)
        assert a is not None
        risk = abs(a.price - a.stop_price)
        reward = abs(a.target_price - a.price)
        assert reward / risk == pytest.approx(2.0), price


def test_trend_and_relative_do_NOT_move_with_the_live_price() -> None:
    # Deliberate. `relative` subtracts the benchmark's move over the same
    # span, so advancing this stock's end point while the index stays on
    # its last close would manufacture excess return from a timing
    # mismatch. Both sides move together or neither does.
    lookback = horizons._lookback_for("short")
    n = lookback + 20
    stock = daily([100.0 * (1.01 ** i) for i in range(n)])
    bench = daily([100.0 * (1.005 ** i) for i in range(n)])
    plain = horizons.assess_daily("S", stock, "short", bench)
    live = horizons.assess_daily("S", stock, "short", bench, live_price=50.0)
    assert plain is not None and live is not None
    assert live.trend == pytest.approx(plain.trend)
    assert live.relative == pytest.approx(plain.relative)
    # And volatility is a property of the window, not of one tick.
    assert live.volatility == pytest.approx(plain.volatility)
    assert live.expected_move == pytest.approx(plain.expected_move)


def test_a_live_price_above_the_trailing_range_is_clamped() -> None:
    # A live price CAN sit outside the trailing range where the last close
    # could not. Uncapped, a new high read as a range position of 1.04 and
    # a "below recent peak" of +3.8%, both nonsense.
    lookback = horizons._lookback_for("short")
    frame = daily([100.0 + i * 0.2 for i in range(lookback + 20)])
    a = horizons.assess_daily("S", frame, "short", None, live_price=500.0)
    assert a is not None
    assert a.position_in_range == pytest.approx(1.0)
    assert a.drawdown == pytest.approx(0.0)


def test_a_live_price_below_the_trailing_range_is_clamped() -> None:
    lookback = horizons._lookback_for("short")
    frame = daily([100.0 + i * 0.2 for i in range(lookback + 20)])
    a = horizons.assess_daily("S", frame, "short", None, live_price=1.0)
    assert a is not None
    assert a.position_in_range == pytest.approx(0.0)
    assert a.drawdown < 0.0


def test_an_unusable_live_price_falls_back_to_the_close() -> None:
    # A missing quote must degrade to the stale anchor, not to zero.
    lookback = horizons._lookback_for("short")
    frame = daily([100.0 + i * 0.2 for i in range(lookback + 20)])
    base = horizons.assess_daily("S", frame, "short", None)
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        a = horizons.assess_daily("S", frame, "short", None, live_price=bad)
        assert a is not None, bad
        assert a.price == pytest.approx(base.price), bad


def test_live_prices_for_returns_empty_rather_than_raising(monkeypatch) -> None:
    # With no session the horizons must still render, anchored on the last
    # close, rather than the whole tab failing.
    import market_source

    def boom(symbols, session=None):
        raise market_source.NoSession("no session")

    monkeypatch.setattr(market_source, "last_prices", boom)
    assert horizons.live_prices_for(["RELIANCE"]) == {}
    assert horizons.live_prices_for([]) == {}


# --- the lookup must anchor the same way the tables do -------------------
#
# The gap that made a lookup disagree with the broker's screen. MOLBIO
# closed at 1,253.00 on 9 Sep 2026 and traded at 1,509.40 the next
# morning; the horizon tables had been re-anchored on the live price and
# instrument_report had not, so it reported the close as "Price" and
# derived every stop and exit from a figure 20% away from the market.

def test_the_lookup_passes_a_live_price_through_to_the_levels(monkeypatch) -> None:
    import instrument_report

    monkeypatch.setattr(horizons, "live_prices_for",
                        lambda symbols: {"FAKE": 1509.40})
    lookback = horizons._lookback_for("short")
    # The stock must OUTPACE the benchmark, or relative is 0, the view is
    # SHORT, and the levels straddle the other way round.
    frame = daily([1000.0 * (1.01 ** i) for i in range(lookback + 20)])
    flat = daily([1000.0] * (lookback + 20))
    frames = {"FAKE": frame, "NIFTY 50": flat}
    monkeypatch.setattr(instrument_report.bar_store, "load",
                        lambda *a, **k: frames)
    monkeypatch.setattr(instrument_report, "resolve",
                        lambda q: ("FAKE", False, 0))
    report = instrument_report.analyse("FAKE", include_intraday=False)
    assert report.found
    assert report.live_anchored is True
    assert report.price == pytest.approx(1509.40)
    short = report.verdicts["short"].assessment
    assert short is not None
    assert short.price == pytest.approx(1509.40)
    # And the levels straddle the LIVE price, not the stale close.
    assert short.stop_price < 1509.40 < short.target_price


def test_the_lookup_falls_back_to_the_close_and_says_so(monkeypatch) -> None:
    import instrument_report

    monkeypatch.setattr(horizons, "live_prices_for", lambda symbols: {})
    lookback = horizons._lookback_for("short")
    closes = [1000.0 + i for i in range(lookback + 20)]
    frame = daily(closes)
    frames = {"FAKE": frame, "NIFTY 50": frame}
    monkeypatch.setattr(instrument_report.bar_store, "load",
                        lambda *a, **k: frames)
    monkeypatch.setattr(instrument_report, "resolve",
                        lambda q: ("FAKE", False, 0))
    report = instrument_report.analyse("FAKE", include_intraday=False)
    assert report.found
    # Degrades to the close rather than to nothing - but flags it, so the
    # UI can warn instead of presenting a stale price as current.
    assert report.live_anchored is False
    assert report.price == pytest.approx(closes[-1])


def test_use_live_false_never_quotes(monkeypatch) -> None:
    # The CLI and the tests must be able to run without a session.
    import instrument_report

    def boom(symbols):
        raise AssertionError("should not have quoted")

    monkeypatch.setattr(horizons, "live_prices_for", boom)
    lookback = horizons._lookback_for("short")
    frame = daily([1000.0 + i for i in range(lookback + 20)])
    frames = {"FAKE": frame, "NIFTY 50": frame}
    monkeypatch.setattr(instrument_report.bar_store, "load",
                        lambda *a, **k: frames)
    monkeypatch.setattr(instrument_report, "resolve",
                        lambda q: ("FAKE", False, 0))
    report = instrument_report.analyse("FAKE", include_intraday=False,
                                       use_live=False)
    assert report.found and report.live_anchored is False


# --- realised volatility for the short horizon ---------------------------
#
# The daily path cannot estimate volatility from ten returns, so it
# silently widened its window to sixty sessions - meaning the plausible
# move for a two-week hold was borrowed from two months of data. Ten
# sessions of intraday bars is roughly 1,250 returns at three minutes,
# over the window actually being assessed.

def intraday(prices_by_day: dict) -> pd.DataFrame:
    """An intraday frame from {date-string: [closes]}.

    Bars are spaced at the configured size, so these tests follow the
    project's bar size instead of pinning the one it happened to use when
    they were written. The estimator sums squared returns per session and
    does not read the spacing, so the figures do not move with it.
    """
    import config

    step = pd.Timedelta(seconds=config.SCAN_BAR_SECONDS)
    stamps, closes = [], []
    for day, series in prices_by_day.items():
        base = pd.Timestamp(f"{day} 09:15", tz=IST)
        for i, price in enumerate(series):
            stamps.append(base + step * i)
            closes.append(price)
    frame = pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(stamps))
    for column in ("Open", "High", "Low"):
        frame[column] = frame["Close"]
    frame["Volume"] = 1e5
    return frame


def test_realised_volatility_is_annualised_and_positive() -> None:
    rng = np.random.default_rng(7)
    days = {f"2026-09-{d:02d}": list(100.0 * np.cumprod(
        1 + rng.normal(0, 0.001, 75))) for d in range(1, 11)}
    got = horizons.realised_volatility(intraday(days), 10)
    assert np.isfinite(got)
    # sqrt(74 * 1e-6 * 252) is 13.7% from the in-session path alone; the
    # gaps this generator leaves between days take the measured figure to
    # about 21%. The band is wide because the point of this test is
    # "annualised and finite", not the value.
    assert 5.0 < got < 80.0, got


def test_the_overnight_gap_raises_the_estimate() -> None:
    # THE TRAP THIS AVOIDS. Intraday returns alone miss every close-to-open
    # move. Left out, the estimate comes in low, the plausible move with
    # it, and the cost-multiple gate lets through setups whose move cannot
    # cover the round trip.
    # The control arm must have NO gap at all, or this compares a big gap
    # with a small one instead of with none. Each day therefore opens
    # exactly where the last one closed: the old version restarted every
    # day at 100.0 after closing at 100.74, a -0.74% gap every night.
    calm, level = {}, 100.0
    for d in range(1, 11):
        calm[f"2026-09-{d:02d}"] = [level + i * 0.01 for i in range(75)]
        level = level + 0.74
    gappy, level = {}, 100.0
    for d in range(1, 11):
        gappy[f"2026-09-{d:02d}"] = [level + i * 0.01 for i in range(75)]
        level = (level + 0.74) * 1.02
    quiet = horizons.realised_volatility(intraday(calm), 10)
    jumpy = horizons.realised_volatility(intraday(gappy), 10)
    assert jumpy > quiet * 2, (quiet, jumpy)
    # And the gap term must be the whole difference: identical in-session
    # paths, so anything else would mean the sessions were not comparable.
    assert quiet == pytest.approx(
        horizons.realised_volatility(intraday(calm), 10))


def test_too_little_intraday_history_returns_nan_not_a_guess() -> None:
    assert horizons.realised_volatility(None, 10) != horizons.realised_volatility(None, 10)
    thin = intraday({"2026-09-01": [100.0, 100.1, 100.2]})
    assert np.isnan(horizons.realised_volatility(thin, 10))
    assert np.isnan(horizons.realised_volatility(pd.DataFrame(), 10))


def test_the_short_horizon_uses_intraday_bars_when_given_them() -> None:
    lookback = horizons._lookback_for("short")
    daily_frame = daily([100.0 + i * 0.2 for i in range(lookback + 20)])
    rng = np.random.default_rng(3)
    # Deliberately far more volatile intraday than the smooth daily path,
    # so the two estimates cannot be confused.
    days = {f"2026-09-{d:02d}": list(100.0 * np.cumprod(
        1 + rng.normal(0, 0.004, 75))) for d in range(1, 13)}
    without = horizons.assess_daily("X", daily_frame, "short", None)
    with_fine = horizons.assess_daily("X", daily_frame, "short", None,
                                      fine=intraday(days))
    assert without is not None and with_fine is not None
    assert with_fine.volatility != pytest.approx(without.volatility)
    assert with_fine.volatility > without.volatility


def test_the_long_horizon_ignores_intraday_bars() -> None:
    # A 252-session view has plenty of daily returns and no business
    # resting on ten sessions of ticks.
    lookback = horizons._lookback_for("long")
    daily_frame = daily([100.0 + i * 0.2 for i in range(lookback + 20)])
    days = {f"2026-09-{d:02d}": [100.0 + i * 0.5 for i in range(75)]
            for d in range(1, 13)}
    plain = horizons.assess_daily("X", daily_frame, "long", None)
    fed = horizons.assess_daily("X", daily_frame, "long", None,
                                fine=intraday(days))
    assert plain is not None and fed is not None
    assert fed.volatility == pytest.approx(plain.volatility)


def test_a_flat_session_counts_as_zero_rather_than_vanishing() -> None:
    # A halt, an illiquid name or a frozen feed produces a session with no
    # variance. Dropping it averaged only the sessions that moved, which
    # biased the estimate UP - in the estimator whose own docstring warns
    # against biasing the plausible move.
    moving = {f"2026-09-{d:02d}": [100.0 + i * 0.05 for i in range(75)]
              for d in range(1, 6)}
    flat = {f"2026-09-{d:02d}": [100.0] * 75 for d in range(6, 11)}
    both = dict(moving)
    both.update(flat)
    busy = horizons.realised_volatility(intraday(moving), 5)
    mixed = horizons.realised_volatility(intraday(both), 10)
    assert np.isfinite(mixed)
    # Ten sessions, five of them still: the average variance must be about
    # half the five-session figure, so the volatility about 1/sqrt(2) of it.
    assert mixed < busy, (mixed, busy)
    assert mixed == pytest.approx(busy / np.sqrt(2), rel=0.35)
    # All flat is zero volatility, which is a measurement and not a gap.
    assert horizons.realised_volatility(intraday(flat), 5) == pytest.approx(0.0)


def test_the_first_session_in_the_window_gets_its_gap_too() -> None:
    # previous_close used to be seeded from inside the window, so one
    # session in ten had no overnight term - a systematic understatement.
    days, level = {}, 100.0
    for d in range(1, 13):
        days[f"2026-09-{d:02d}"] = [level + i * 0.01 for i in range(75)]
        level = (level + 0.74) * 1.03      # a 3% gap every night
    frame = intraday(days)
    # Ask for a window that starts well inside the frame, so a prior
    # session exists to seed from.
    windowed = horizons.realised_volatility(frame, 5)
    # The same five sessions with nothing before them: the first one then
    # has no gap available, so the estimate must come out LOWER.
    cut = frame[frame.index >= pd.Timestamp("2026-09-08 00:00", tz=IST)]
    orphaned = horizons.realised_volatility(cut, 5)
    assert windowed > orphaned, (windowed, orphaned)


def test_fewer_than_three_usable_sessions_is_nan_not_an_average() -> None:
    two = {f"2026-09-{d:02d}": [100.0 + i * 0.02 for i in range(75)]
           for d in range(1, 3)}
    assert np.isnan(horizons.realised_volatility(intraday(two), 10))
    # Sessions with too few bars do not count towards the three, but their
    # close still anchors the next gap.
    thin = {f"2026-09-{d:02d}": [100.0, 100.1] for d in range(1, 9)}
    assert np.isnan(horizons.realised_volatility(intraday(thin), 10))


# --- one estimator per run, never two in one ranking ---------------------

def test_the_intraday_store_is_used_only_if_it_covers_the_universe() -> None:
    # Realised volatility and the 60-session daily estimate do not share a
    # level, so mixing them inside one rank(pct=True) column would let
    # data availability move a symbol's position and flip its cost gate.
    days = {f"2026-09-{d:02d}": [100.0 + i * 0.05 for i in range(75)]
            for d in range(1, 11)}
    frame = intraday(days)
    full = {f"S{i}": frame for i in range(10)}
    symbols = [f"S{i}" for i in range(10)]
    # A filtered copy rather than the same dict: unmeasurable symbols are
    # dropped, so identity is not the thing to assert.
    assert set(horizons._usable_fine(full, symbols)) == set(symbols)
    partial = {f"S{i}": frame for i in range(5)}
    assert horizons._usable_fine(partial, symbols) == {}
    assert horizons._usable_fine({}, symbols) == {}
    assert horizons._usable_fine(full, []) == {}


def test_a_symbol_with_one_session_does_not_count_as_covered() -> None:
    # The real case, from the day the 3-minute store was first built: 121
    # symbols with 14 sessions and one with a single session written by a
    # backfill. Not empty, so counting frames called it covered - and then
    # the estimator returns NaN for it and that symbol alone falls back to
    # the daily figure, inside the same cross-sectional rank.
    full = intraday({f"2026-09-{d:02d}": [100.0 + i * 0.05 for i in range(75)]
                     for d in range(1, 11)})
    one_day = intraday({"2026-09-10": [100.0 + i * 0.05 for i in range(75)]})
    assert horizons._measurable_sessions(full) == 10
    assert horizons._measurable_sessions(one_day) == 1
    assert np.isnan(horizons.realised_volatility(one_day, 10))
    symbols = [f"S{i}" for i in range(10)]
    store = {s: full for s in symbols}
    store["S9"] = one_day
    # One in ten unmeasurable is inside the tolerance, and that symbol is
    # dropped from the store rather than left to fall back silently.
    kept = horizons._usable_fine(store, symbols)
    assert set(kept) == set(symbols[:9])
    # Two in ten is not.
    store["S8"] = one_day
    assert horizons._usable_fine(store, symbols) == {}


def test_sessions_too_thin_to_measure_do_not_count() -> None:
    thin = intraday({f"2026-09-{d:02d}": [100.0, 100.1, 100.2]
                     for d in range(1, 11)})
    assert horizons._measurable_sessions(thin) == 0
    assert horizons._usable_fine({"S0": thin}, ["S0"]) == {}
    assert horizons._measurable_sessions(None) == 0
    assert horizons._measurable_sessions(pd.DataFrame()) == 0


def test_a_stale_intraday_store_is_refused_rather_than_quoted() -> None:
    # Easy to reach by accident: the store is fed by whatever interval the
    # live feed fetches, and that follows config. If the feed moves and
    # this store stops being topped up, its volatility is from another
    # month and must not be ranked against today's daily estimates.
    old = {f"2026-01-{d:02d}": [100.0 + i * 0.05 for i in range(75)]
           for d in range(1, 11)}
    frame = intraday(old)
    symbols = ["S0", "S1"]
    assert horizons._usable_fine({s: frame for s in symbols}, symbols) == {}


def test_the_bar_size_is_declared_once_and_fetched_at_that_size() -> None:
    # live_bars used to hold its own 300, and after that was fixed the
    # FETCH still asked for five minutes - so combined() spliced 5-minute
    # prior sessions onto 3-minute live bars. Both halves are asserted
    # against config rather than against a literal, so this test keeps
    # meaning something after the next size change.
    import inspect

    import config
    import live_bars
    import market_source

    assert live_bars.BAR_SECONDS == config.SCAN_BAR_SECONDS
    kite = market_source.kite_interval(config.SCAN_BAR_INTERVAL)
    assert kite.startswith(str(config.SCAN_BAR_SECONDS // 60))
    # Nothing in the live path may carry its own interval literal.
    for function in (live_bars.prewarm, live_bars.backfill_today):
        default = inspect.signature(function).parameters["interval"].default
        assert default is None, (function.__name__, default)
    source = inspect.getsource(live_bars.prewarm) + inspect.getsource(
        live_bars.backfill_today)
    assert source.count("config.SCAN_BAR_INTERVAL") == 2
    # And the store the SHORT horizon reads must be the one the feed
    # fills. Naming a size nothing fetches leaves that store to go stale:
    # it works on whatever is already cached, then quietly stops being
    # current, and the volatility behind every short-horizon plausible
    # move comes from another month.
    assert config.HORIZON_SHORT_INTERVAL == kite
