"""Tests for the edge harness and the bar cache that feeds it.

Every test here is offline and synthetic. That is the point: the harness
exists to decide whether a rule has an edge, so its own guarantees have to
be pinned against frames whose future is known by construction rather than
against a market. The guarantees under test are the ones a wrong answer
would flatter:

  the opening range does not exist before the bars that form it printed, so
    a rule cannot read it early even if it forgets to check;
  a bar that spans both stop and target counts as a stop;
  a position still open at the bell is marked out at the close, not dropped
    and not scored as flat;
  expectancy arithmetic, on rows small enough to check by hand;
  a frame cached without open interest is never served to a caller that
    asked for open interest.

Run with: .venv\\Scripts\\python -m pytest test_edge_lab.py -q
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import edge_lab
import kite_bars

ORB = edge_lab.BARS_PER_ORB

# A quiet session: eight bars of 0.4 range, so the ATR built from prior
# sessions like this one is about 0.4.
QUIET = [(100.0, 100.2, 99.8, 100.0)] * 8

# A session whose opening range is wide only because of bars 1 and 2. Bar 0
# on its own spans 0.4; bars 0..2 together span 94.0 to 106.0. That gap is
# what a precomputed opening range used to leak to bar 0.
LOUD = [
    (100.0, 100.2, 99.8, 100.0),
    (100.0, 106.0, 99.5, 105.0),
    (105.0, 105.5, 94.0, 95.0),
    (95.0, 96.0, 94.5, 95.5),
    (95.5, 96.5, 95.0, 96.0),
    (96.0, 96.5, 95.5, 96.0),
    (96.0, 96.2, 95.8, 96.0),
    (96.0, 96.4, 95.9, 96.2),
]
LOUD_ORB_HIGH = 106.0
LOUD_ORB_LOW = 94.0


def _frame(sessions: list, first_day: str = "2026-01-05") -> pd.DataFrame:
    """A 5-minute OHLCV frame from per-session lists of (o, h, l, c) tuples.

    One session per calendar day from 09:15 Asia/Kolkata, which is all the
    harness needs: it groups bars by the index's date.
    """
    open_of_day = pd.Timedelta(hours=9, minutes=15)
    stamps: list = []
    data: list = []
    for offset, session in enumerate(sessions):
        base = (pd.Timestamp(first_day, tz="Asia/Kolkata")
                + pd.Timedelta(days=offset) + open_of_day)
        for bar, (open_, high, low, close) in enumerate(session):
            stamps.append(base + pd.Timedelta(minutes=5 * bar))
            data.append((open_, high, low, close, 1000.0))
    return pd.DataFrame(data,
                        columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(stamps, name="Date"))


def _watch(frames: dict, **kwargs) -> list:
    """Every context the harness builds, for a rule that never fires."""
    seen: list = []

    def watcher(ctx):
        """Record the context and decline to trade."""
        seen.append(ctx)
        return None

    rows = edge_lab.run_rule(watcher, frames, warmup=2, **kwargs)
    assert rows == []
    return seen


# --- the opening range cannot be read early ------------------------------

def test_no_precomputed_range_is_handed_to_the_context() -> None:
    """The guarantee is structural: there is no field to leak."""
    fields = edge_lab.Context.__dataclass_fields__
    assert "orb_high" not in fields
    assert "orb_low" not in fields
    assert isinstance(edge_lab.Context.orb_high, property)
    assert isinstance(edge_lab.Context.orb_low, property)


def test_opening_range_is_nan_until_it_has_closed() -> None:
    contexts = _watch({"TEST": _frame([QUIET, QUIET, LOUD])})
    assert contexts, "the last session should have been replayed"
    early = [c for c in contexts if c.i < ORB]
    later = [c for c in contexts if c.i >= ORB]
    assert len(early) == ORB
    assert later
    for ctx in early:
        assert not ctx.orb_closed
        assert math.isnan(ctx.orb_high)
        assert math.isnan(ctx.orb_low)
    for ctx in later:
        assert ctx.orb_closed
        assert ctx.orb_high == pytest.approx(LOUD_ORB_HIGH)
        assert ctx.orb_low == pytest.approx(LOUD_ORB_LOW)


def test_every_context_array_stops_at_the_current_bar() -> None:
    contexts = _watch({"TEST": _frame([QUIET, QUIET, LOUD])})
    for ctx in contexts:
        assert len(ctx.high) == ctx.i + 1
        assert len(ctx.low) == ctx.i + 1
        assert len(ctx.close) == ctx.i + 1
        assert len(ctx.vwap) == ctx.i + 1
        assert ctx.high.max() == pytest.approx(
            max(bar[1] for bar in LOUD[:ctx.i + 1]))
        assert ctx.price == pytest.approx(LOUD[ctx.i][3])


def test_rule_ignoring_orb_closed_cannot_see_future_bars() -> None:
    """A rule that reads the range without checking cannot fire early.

    The rule below is the shape that used to leak: it needs only the width
    of the opening range, which a precomputed value made available at bar 0
    and which at bar 0 is a fact about bars 1 and 2.
    """
    def naive_width(ctx):
        """Fire on a wide opening range, without checking orb_closed."""
        if ctx.orb_high - ctx.orb_low > 2 * ctx.atr:
            return edge_lab.LONG
        return None

    frames = {"TEST": _frame([QUIET, QUIET, LOUD])}
    atr = _watch(frames)[0].atr
    # The counterfactual the fix removes: the full range does clear the
    # threshold, and bar 0 on its own does not, so a leaked value would
    # have fired at bar 0.
    assert LOUD_ORB_HIGH - LOUD_ORB_LOW > 2 * atr
    assert LOUD[0][1] - LOUD[0][2] < 2 * atr

    rows = edge_lab.run_rule(naive_width, frames, warmup=2)
    assert rows, "the rule should still fire once the range has closed"
    assert min(row["i"] for row in rows) == ORB


def test_a_session_with_no_prior_history_is_skipped_not_guessed() -> None:
    """No prior sessions means no ATR, so nothing is measurable.

    With warmup=0 the harness is asked to replay from the very first
    session, which has nothing behind it. It must skip that session rather
    than crash on an empty prior history or size against the session
    itself.
    """
    assert edge_lab._prior_levels({}, []) is None
    frames = {"TEST": _frame([QUIET, QUIET, LOUD])}
    rows = edge_lab.run_rule(lambda ctx: edge_lab.LONG, frames, warmup=0)
    days = {row["day"] for row in rows}
    assert date(2026, 1, 5) not in days     # nothing prior at all
    assert date(2026, 1, 6) not in days     # eight prior bars, no 14-bar ATR
    assert date(2026, 1, 7) in days


# --- first touch ---------------------------------------------------------

def test_first_touch_counts_a_straddling_bar_as_a_stop() -> None:
    high = pd.Series([110.0]).values
    low = pd.Series([90.0]).values
    assert edge_lab.first_touch(high, low, edge_lab.LONG, 100.0, 5.0,
                                5.0) == "STOP"
    assert edge_lab.first_touch(high, low, edge_lab.SHORT, 100.0, 5.0,
                                5.0) == "STOP"


def test_first_touch_resolves_the_unambiguous_paths() -> None:
    rising = pd.DataFrame({"high": [101.0, 106.0], "low": [100.0, 105.0]})
    falling = pd.DataFrame({"high": [100.0, 95.0], "low": [99.0, 94.0]})
    quiet = pd.DataFrame({"high": [100.5], "low": [99.5]})
    up_high, up_low = rising["high"].values, rising["low"].values
    down_high, down_low = falling["high"].values, falling["low"].values
    assert edge_lab.first_touch(up_high, up_low, edge_lab.LONG, 100.0, 5.0,
                                5.0) == "TARGET"
    assert edge_lab.first_touch(down_high, down_low, edge_lab.LONG, 100.0,
                                5.0, 5.0) == "STOP"
    assert edge_lab.first_touch(down_high, down_low, edge_lab.SHORT, 100.0,
                                5.0, 5.0) == "TARGET"
    assert edge_lab.first_touch(quiet["high"].values, quiet["low"].values,
                                edge_lab.LONG, 100.0, 5.0, 5.0) == "NEITHER"
    # A stop distance of zero is refused rather than treated as an instant
    # stop-out.
    assert edge_lab.first_touch(up_high, up_low, edge_lab.LONG, 100.0, 0.0,
                                5.0) == "NEITHER"


# --- unresolved trades ---------------------------------------------------

# Prior sessions wide enough that a 0.5-point drift cannot reach even the
# tightest default stop, so the signal is still open at the bell.
WIDE = [(100.0, 105.0, 95.0, 100.0)] * 8
DRIFT = [
    (100.0, 100.1, 99.9, 100.0),
    (100.0, 100.1, 99.9, 100.0),
    (100.0, 100.1, 99.9, 100.0),
    (100.0, 100.1, 99.9, 100.0),
    (100.0, 100.1, 99.9, 100.0),
    (100.0, 100.2, 100.0, 100.2),
    (100.2, 100.4, 100.1, 100.3),
    (100.3, 100.5, 100.2, 100.4),
]


def _fire_at(bar: int):
    """A rule that fires LONG on one nominated bar and never again."""
    def rule(ctx):
        """Fire LONG at the nominated bar index."""
        return edge_lab.LONG if ctx.i == bar else None
    return rule


def test_unresolved_trade_is_marked_at_the_close_not_at_zero() -> None:
    frames = {"TEST": _frame([WIDE, WIDE, DRIFT])}
    rows = edge_lab.run_rule(_fire_at(4), frames, warmup=2)
    assert len(rows) == 1
    row = rows[0]
    verdicts = [row[(s, t)] for s in edge_lab.DEFAULT_STOPS
                for t in edge_lab.DEFAULT_TARGETS]
    assert set(verdicts) == {"NEITHER"}
    entry = DRIFT[4][3]
    expected = (DRIFT[-1][3] - entry) / row["sigma"]
    assert row["entry"] == pytest.approx(entry)
    assert row["close_move_sigma"] == pytest.approx(expected)
    assert row["close_move_sigma"] != 0.0
    # The coin-flip twin is open at the bell too: a 0.5-point drift reaches
    # neither side's levels.
    assert set(row["flipped"].values()) == {"NEITHER"}
    summary = edge_lab.summarise(rows)
    assert summary["signals"] == 1
    for cell in summary["grid"]:
        assert cell["signals"] == 1
        assert cell["resolved"] == 0
        assert cell["hit_rate"] is None
        # Marked at the close in R - the close move over the stop distance -
        # not scored as flat. This asserted 0.0 while the test's own name
        # said the opposite, which is how the harness shipped the bug.
        assert cell["gross_r"] == pytest.approx(expected / cell["stop"])
        assert cell["gross_r"] > 0.0
        # A coin takes each side half the time, so the drift cancels and
        # the whole gross figure is edge over the coin.
        assert cell["random_gross_r"] == pytest.approx(0.0)
        assert cell["edge_r"] == pytest.approx(cell["gross_r"])


# --- summarise arithmetic ------------------------------------------------

def _row(verdict: str, mfe: float, mae: float, flipped: str = "NEITHER",
         close_move: float = 0.0) -> dict:
    """One hand-built row for a single (stop, target) geometry.

    `flipped` is the coin-flip twin's verdict on the same path, and
    `close_move` the rule side's close move in plausible-move units.
    """
    return {"symbol": "TEST", "day": date(2026, 1, 5), "direction": "LONG",
            "entry": 100.0, "sigma": 10.0, "mfe_sigma": mfe,
            "mae_sigma": mae, "close_move_sigma": close_move,
            (0.5, 1.0): verdict, "flipped": {(0.5, 1.0): flipped}}


def test_summarise_expectancy_arithmetic_by_hand() -> None:
    # Rule side: two targets at +2R, one stop at -1R, one open trade that
    # closed 0.2 up on a 0.5 stop, which is +0.4R.
    # Twin side: each target path stopped the short out (-1R each); the
    # stopped path closed 0.6 down, so the short was open at +1.2R; the
    # open path is open for the short too, at -0.4R.
    rows = [_row("TARGET", 1.2, 0.2, flipped="STOP"),
            _row("TARGET", 1.1, 0.3, flipped="STOP"),
            _row("STOP", 0.1, 0.9, flipped="NEITHER", close_move=-0.6),
            _row("NEITHER", 0.4, 0.4, flipped="NEITHER", close_move=0.2)]
    summary = edge_lab.summarise(rows, stops=(0.5,), targets=(1.0,))
    assert summary["signals"] == 4
    assert summary["sessions"] == 1
    assert summary["symbols"] == 1
    assert summary["longs"] == 4
    assert summary["shorts"] == 0
    assert len(summary["grid"]) == 1
    cell = summary["grid"][0]
    assert cell["signals"] == 4
    assert cell["resolved"] == 3
    assert cell["reward_risk"] == pytest.approx(2.0)
    assert cell["hit_rate"] == pytest.approx(2 / 3)
    # Coin: 2 target hits between the two sides over 3 + 2 resolved trades.
    assert cell["random_rate"] == pytest.approx(2 / 5)
    assert cell["edge_pp"] == pytest.approx((2 / 3 - 2 / 5) * 100)
    # Two targets at 2R, one stop at -1R and the open trade marked at its
    # +0.4R close, over the four signals taken: counted, not dropped, and
    # not scored as flat.
    assert cell["gross_r"] == pytest.approx((2 * 2.0 - 1 + 0.4) / 4)
    # Twin: -1 - 1 + 1.2 - 0.4 = -1.2R. A coin averages the two sides.
    assert cell["random_gross_r"] == pytest.approx((3.4 - 1.2) / 8)
    assert cell["edge_r"] == pytest.approx(3.4 / 4 - 2.2 / 8)
    # Stop distance is 0.5 sigma = 5.0 on a 100 entry, so costs are
    # 0.000824 / 0.05 of one R.
    assert cell["cost_r"] == pytest.approx(edge_lab.COST_FRACTION / 0.05)
    assert cell["net_r"] == pytest.approx(cell["gross_r"] - cell["cost_r"])
    exc = summary["excursions"]
    assert exc["mfe_mean"] == pytest.approx((1.2 + 1.1 + 0.1 + 0.4) / 4)
    assert exc["mae_mean"] == pytest.approx((0.2 + 0.3 + 0.9 + 0.4) / 4)
    assert exc["mfe_over_mae"] == pytest.approx(exc["mfe_mean"]
                                                / exc["mae_mean"])
    assert exc["mfe"]["p50"] == pytest.approx(0.75)
    assert edge_lab.best_geometry(summary) is cell


def test_a_rule_that_is_a_coin_flip_scores_no_edge() -> None:
    """Swap the sides on every path and the rule IS the coin: zero edge.

    The old closed-form baseline could not pass this, because it never
    looked at the paths at all.
    """
    rows = [_row("TARGET", 1.2, 0.2, flipped="STOP"),
            _row("STOP", 0.2, 1.2, flipped="TARGET"),
            _row("NEITHER", 0.3, 0.3, flipped="NEITHER", close_move=0.1),
            _row("NEITHER", 0.3, 0.3, flipped="NEITHER", close_move=-0.1)]
    cell = edge_lab.summarise(rows, stops=(0.5,), targets=(1.0,))["grid"][0]
    assert cell["hit_rate"] == pytest.approx(cell["random_rate"])
    assert cell["edge_pp"] == pytest.approx(0.0)
    assert cell["gross_r"] == pytest.approx(cell["random_gross_r"])
    assert cell["edge_r"] == pytest.approx(0.0)


def _driftless_rows(paths: int, bars: int = 35, sub: int = 8,
                    seed: int = 3) -> list:
    """Rows for a pure random walk, scored at the shipped 0.5 / 1.0 geometry.

    Sub-steps inside each bar give the bars real highs and lows, so the
    same-bar tie rule and the close cut-off both bite exactly as they do on
    market data. The walk's whole-horizon sd is 1/1.4 of one plausible
    move, which is what the harness unit measures (see the UNITS note).
    """
    rng = np.random.default_rng(seed)
    step = (1.0 / 1.4) / math.sqrt(bars * sub)
    fine = np.cumsum(rng.normal(0.0, step, (paths, bars * sub)), axis=1)
    fine = fine.reshape(paths, bars, sub) + 100.0
    highs, lows = fine.max(axis=2), fine.min(axis=2)
    rows = []
    for p in range(paths):
        row = {"symbol": "WALK", "day": p, "direction": "LONG",
               "entry": 100.0, "sigma": 1.0, "mfe_sigma": 0.0,
               "mae_sigma": 0.0,
               "close_move_sigma": float(fine[p, -1, -1] - 100.0)}
        row[(0.5, 1.0)] = edge_lab.first_touch(
            highs[p], lows[p], edge_lab.LONG, 100.0, 0.5, 1.0)
        row["flipped"] = {(0.5, 1.0): edge_lab.first_touch(
            highs[p], lows[p], edge_lab.SHORT, 100.0, 0.5, 1.0)}
        rows.append(row)
    return rows


def test_a_driftless_walk_scores_no_edge_against_the_coin() -> None:
    """The baseline must call a random walk random.

    The closed-form stop / (stop + target) it replaced says 33.3% here,
    but a driftless walk counted this harness's way - hits only before the
    close, ties to the stop - resolves in the target's favour about 24% of
    the time, so the formula read pure chance as about nine points WORSE
    than chance. Measured over 40 seeds at 6,000 paths: 22.7-25.7% hit,
    at least 7.7 points under the formula, with the coin baseline never
    more than 1.6 points from the hit rate and |edge_r| never above 0.03 R.
    The limits below sit at roughly twice those extremes.
    """
    cell = edge_lab.summarise(_driftless_rows(6000), stops=(0.5,),
                              targets=(1.0,))["grid"][0]
    formula = 0.5 / (0.5 + 1.0)
    assert formula - cell["hit_rate"] > 0.05
    assert abs(cell["edge_pp"]) < 4.0
    assert abs(cell["edge_r"]) < 0.06


def test_a_short_left_open_is_marked_at_its_own_close_move() -> None:
    """The SHORT mirror of the unresolved test. close_move_sigma is signed
    for the signal's own direction, so a drift UP is a loss to a short, and
    the twin - a long - gains exactly what the short loses."""
    frames = {"TEST": _frame([WIDE, WIDE, DRIFT])}

    def fire_short(ctx):
        """Fire SHORT at bar 4 and never again."""
        return edge_lab.SHORT if ctx.i == 4 else None

    rows = edge_lab.run_rule(fire_short, frames, warmup=2)
    assert len(rows) == 1
    row = rows[0]
    expected = (DRIFT[4][3] - DRIFT[-1][3]) / row["sigma"]
    assert row["close_move_sigma"] == pytest.approx(expected)
    assert row["close_move_sigma"] < 0.0
    for cell in edge_lab.summarise(rows)["grid"]:
        assert cell["gross_r"] == pytest.approx(expected / cell["stop"])
        assert cell["gross_r"] < 0.0
        assert cell["random_gross_r"] == pytest.approx(0.0)
        assert cell["edge_r"] == pytest.approx(cell["gross_r"])


def test_a_twin_missing_from_any_row_withholds_the_baseline() -> None:
    """Comparing totals would pass one rule-only row plus one twin-only
    row; the baseline has to see a twin on EVERY row it scores."""
    rule_only = _row("TARGET", 1.2, 0.2)
    del rule_only["flipped"]
    twin_only = _row("STOP", 0.1, 0.9, flipped="TARGET")
    del twin_only[(0.5, 1.0)]
    both = _row("STOP", 0.1, 0.9, flipped="TARGET")
    cell = edge_lab.summarise([rule_only, twin_only, both], stops=(0.5,),
                              targets=(1.0,))["grid"][0]
    assert cell["signals"] == 2
    assert cell["random_rate"] is None
    assert cell["edge_r"] is None


def test_the_report_prints_signs_n_a_and_no_negative_zero(capsys) -> None:
    """print_report once crashed on a sign-after-width format spec. A
    missing baseline prints n/a, and a tiny negative rounds to +0.000."""
    rows = [_row("TARGET", 1.2, 0.2), _row("STOP", 0.1, 0.9),
            _row("NEITHER", 0.3, 0.3, close_move=-0.0001)]
    for row in rows:
        del row["flipped"]
    summary = edge_lab.summarise(rows, stops=(0.5,), targets=(1.0,))
    edge_lab.print_report("no twins", summary)
    out = capsys.readouterr().out
    assert "n/a" in out
    assert "+0.333" in out                      # (2 - 1 - 0.0002) / 3
    assert "-0.000" not in out
    tiny = [_row("NEITHER", 0.3, 0.3, close_move=-0.0001, flipped="STOP"),
            _row("TARGET", 1.2, 0.2, flipped="STOP")]
    edge_lab.print_report("tiny", edge_lab.summarise(
        tiny, stops=(0.5,), targets=(1.0,)))
    assert "-0.000" not in capsys.readouterr().out


def test_rows_without_a_twin_report_no_baseline_rather_than_a_wrong_one(
        ) -> None:
    rows = [_row("TARGET", 1.2, 0.2), _row("STOP", 0.1, 0.9)]
    for row in rows:
        del row["flipped"]
    cell = edge_lab.summarise(rows, stops=(0.5,), targets=(1.0,))["grid"][0]
    assert cell["gross_r"] == pytest.approx((2.0 - 1.0) / 2)
    assert cell["random_rate"] is None
    assert cell["random_gross_r"] is None
    assert cell["edge_pp"] is None
    assert cell["edge_r"] is None


def test_summarise_reports_nothing_rather_than_zeroes_on_no_rows() -> None:
    assert edge_lab.summarise([]) == {"signals": 0, "grid": [],
                                      "excursions": {}}


def test_break_even_geometry_outranks_every_loser() -> None:
    """A net_r of exactly 0.0 is a number, not a missing value."""
    grid = [{"net_r": -0.2}, {"net_r": 0.0}, {"net_r": -0.9},
            {"net_r": 0.1}, {"net_r": None}]
    order = [cell["net_r"] for cell in sorted(grid,
                                              key=edge_lab._net_r_desc)]
    assert order == [0.1, 0.0, -0.2, -0.9, None]


# --- the bar cache is keyed on what it holds -----------------------------

SPAN = (date(2025, 9, 8), date(2026, 9, 8))


def _kite_frame(with_oi: bool) -> pd.DataFrame:
    """A one-bar Kite-shaped frame, with or without open interest."""
    index = pd.DatetimeIndex([pd.Timestamp("2025-09-08 09:15",
                                           tz="Asia/Kolkata")], name="Date")
    data = {"Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [1000]}
    if with_oi:
        data["OpenInterest"] = [4200]
    return pd.DataFrame(data, index=index)


def test_cache_path_is_keyed_on_open_interest() -> None:
    plain = kite_bars._cache_path("RELIANCE", "5minute", *SPAN)
    with_oi = kite_bars._cache_path("RELIANCE", "5minute", *SPAN, oi=True)
    assert plain != with_oi
    # The no-OI name is unchanged, so the existing cache stays valid.
    assert plain.name == "kite__RELIANCE__5minute__20250908_20260908.parquet"
    assert with_oi.name.endswith("__oi.parquet")


def test_a_cache_without_open_interest_is_not_served_to_an_oi_caller(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(kite_bars, "CACHE_DIR", tmp_path)
    plain_path = kite_bars._cache_path("RELIANCE", "5minute", *SPAN)
    _kite_frame(with_oi=False).to_parquet(plain_path)
    calls: list = []

    def fake_historical(token, start, end, interval="5minute", oi=False,
                        session=None):
        """Stand-in for kite_client.historical, with no network."""
        calls.append(oi)
        return _kite_frame(with_oi=oi)

    monkeypatch.setattr(kite_bars.kite_client, "historical", fake_historical)
    frame = kite_bars.fetch_symbol("RELIANCE", 738561, *SPAN, oi=True)
    assert frame is not None
    assert "OpenInterest" in frame.columns
    assert calls == [True], "the non-OI cache must not satisfy an oi call"
    # The OI frame is cached under its own key, leaving the plain one alone.
    assert kite_bars._cache_path("RELIANCE", "5minute", *SPAN,
                                 oi=True).exists()
    assert "OpenInterest" not in pd.read_parquet(plain_path).columns


def test_a_plain_cache_still_serves_a_plain_call(tmp_path,
                                                 monkeypatch) -> None:
    monkeypatch.setattr(kite_bars, "CACHE_DIR", tmp_path)
    _kite_frame(with_oi=False).to_parquet(
        kite_bars._cache_path("RELIANCE", "5minute", *SPAN))

    def refuse(*args, **kwargs):
        """Any fetch here would mean the cache was ignored."""
        raise AssertionError("cached frame should have been served")

    monkeypatch.setattr(kite_bars.kite_client, "historical", refuse)
    frame = kite_bars.fetch_symbol("RELIANCE", 738561, *SPAN)
    assert frame is not None
    assert len(frame) == 1


def test_a_frame_without_open_interest_is_refused_when_oi_was_asked_for(
        tmp_path, monkeypatch) -> None:
    """Kite serving no OI column is reported, not passed off as an OI frame."""
    monkeypatch.setattr(kite_bars, "CACHE_DIR", tmp_path)

    def no_oi(*args, **kwargs):
        """Kite answering an oi request without the column."""
        return _kite_frame(with_oi=False)

    monkeypatch.setattr(kite_bars.kite_client, "historical", no_oi)
    assert kite_bars.fetch_symbol("RELIANCE", 738561, *SPAN, oi=True) is None
    assert not kite_bars._cache_path("RELIANCE", "5minute", *SPAN,
                                     oi=True).exists()


def test_a_bad_span_is_reported_not_raised(tmp_path, monkeypatch) -> None:
    """A ValueError from historical used to escape and kill the worker pool."""
    monkeypatch.setattr(kite_bars, "CACHE_DIR", tmp_path)

    def bad_span(*args, **kwargs):
        """What kite_client.historical raises when start is after end."""
        raise ValueError("start 2026-09-08 is after end 2025-09-08")

    monkeypatch.setattr(kite_bars.kite_client, "historical", bad_span)
    assert kite_bars.fetch_symbol("RELIANCE", 738561, SPAN[1],
                                  SPAN[0]) is None
