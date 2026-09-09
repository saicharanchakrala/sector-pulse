"""Measure what price actually does after a signal, instead of assuming.

This exists because the live rule failed in a specific, diagnosable way: it
reached its target in 1 of 21 setups where a driftless random walk predicts
about a third. That gap is too large to be luck, and it points at the model
rather than the market. The target was placed at one sigma, where sigma is
the bar ATR grown by sqrt(bars remaining) - which assumes intraday prices
random-walk. They do not: at short horizons they mean-revert, so realised
range grows slower than sqrt(time) and a target set that way sits beyond
where price usually goes.

So nothing here assumes a distribution. For every signal in the available
history it records the maximum favourable and adverse excursion that
followed, then reports, for a grid of stop and target sizes, how often the
target was reached before the stop. That turns the choice of geometry into
a measurement.

Two disciplines the harness keeps:

  no lookahead     Readings at bar i use bars 0..i only. The forward path is
                   bars i+1 onward, and never touches the readings.
  one signal       A breakout is recorded on the first bar it fires, not on
                   every bar it remains true. Counting each subsequent bar
                   would turn one move into a dozen wins, which is exactly
                   the distortion that makes tipster scorecards meaningless.

Sample-size honesty: 5-minute bars reach back about 59 sessions, and signals
inside one session are correlated. Any edge measured here is provisional.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

import config
import indicators
from levels import LONG, SHORT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    """One recorded breakout and the excursions that followed it."""

    symbol: str
    session: date
    bar_index: int
    direction: str
    entry: float
    atr_bar: float
    bars_left: int
    rvol: "float | None"
    relative_strength: "float | None"
    turnover_20d: "float | None"
    mfe: float                 # max favourable excursion, in rupees
    mae: float                 # max adverse excursion, in rupees
    close_move: float          # entry to session close, signed for direction
    first_touch: dict          # (stop_mult, target_mult) -> "TARGET"/"STOP"/"NEITHER"

    @property
    def sigma(self) -> float:
        """The sqrt-scaled remaining move the live rule assumes."""
        return self.atr_bar * math.sqrt(max(1, self.bars_left))

    def excursion_in_sigma(self, favourable: bool) -> float:
        """MFE or MAE expressed in the live rule's own sigma units."""
        if self.sigma <= 0.0:
            return 0.0
        return (self.mfe if favourable else self.mae) / self.sigma


@dataclass
class Outcome:
    """Aggregate result for one (stop, target) pair, in sigma multiples."""

    stop_mult: float
    target_mult: float
    targets: int = 0
    stops: int = 0
    neither: int = 0

    @property
    def total(self) -> int:
        """Signals evaluated."""
        return self.targets + self.stops + self.neither

    @property
    def decided(self) -> int:
        """Signals that resolved to a target or a stop before the close."""
        return self.targets + self.stops

    @property
    def hit_rate(self) -> "float | None":
        """Fraction of resolved signals that reached the target."""
        if self.decided <= 0:
            return None
        return self.targets / self.decided

    @property
    def reward_risk(self) -> float:
        """Target distance over stop distance."""
        return self.target_mult / self.stop_mult if self.stop_mult else 0.0

    @property
    def random_walk_rate(self) -> float:
        """Hit rate a driftless walk would give this geometry."""
        span = self.stop_mult + self.target_mult
        return self.stop_mult / span if span else 0.0

    def expectancy_r(self) -> "float | None":
        """Expected return per unit risked, before costs.

        Unresolved signals are marked out at the close rather than dropped,
        because a rule that leaves most positions open has to be judged on
        what that actually pays.
        """
        if self.total <= 0:
            return None
        wins = self.targets * self.reward_risk
        return (wins - self.stops) / self.total

    @property
    def edge_over_random(self) -> "float | None":
        """Percentage points of hit rate above a driftless walk."""
        rate = self.hit_rate
        if rate is None:
            return None
        return (rate - self.random_walk_rate) * 100.0


def _forward_paths(session: pd.DataFrame, start: int
                   ) -> tuple[pd.Series, pd.Series]:
    """Highs and lows strictly after `start`, within the same session."""
    tail = session.iloc[start + 1:]
    return tail["High"], tail["Low"]


def _first_touch(highs: pd.Series, lows: pd.Series, direction: str,
                 entry: float, stop_distance: float,
                 target_distance: float) -> str:
    """Which level the path reached first, walking bar by bar.

    When one bar spans both levels the stop is assumed, because a 5-minute
    bar's high and low carry no ordering and assuming the favourable one
    would flatter every result.
    """
    if stop_distance <= 0.0 or target_distance <= 0.0:
        return "NEITHER"
    if direction == LONG:
        stop, target = entry - stop_distance, entry + target_distance
        for high, low in zip(highs, lows):
            hit_stop, hit_target = low <= stop, high >= target
            if hit_stop:
                return "STOP"
            if hit_target:
                return "TARGET"
    else:
        stop, target = entry + stop_distance, entry - target_distance
        for high, low in zip(highs, lows):
            hit_stop, hit_target = high >= stop, low <= target
            if hit_stop:
                return "STOP"
            if hit_target:
                return "TARGET"
    return "NEITHER"


def _direction_at(last: float, vwap: "float | None",
                  orb: "indicators.OpeningRange | None") -> "str | None":
    """The live rule's direction test, applied to one bar."""
    if vwap is None or orb is None:
        return None
    above = last > vwap
    if above and last > orb.high:
        return LONG
    if not above and last < orb.low:
        return SHORT
    return None


def scan_session(symbol: str, history: pd.DataFrame, session_day: date,
                 benchmark_change: "float | None",
                 stop_mults: tuple[float, ...],
                 target_mults: tuple[float, ...],
                 turnover_20d: "float | None" = None) -> list[Signal]:
    """Every first-touch breakout in one session, with forward excursions.

    `history` must contain the session plus enough prior sessions for ATR
    and relative volume. Only bars up to and including the signal bar are
    used for readings.
    """
    session = indicators.session_bars(history, day=session_day)
    if session is None or len(session) < 2:
        return []
    prior = history.loc[[t for t in history.index if t.date() < session_day]]
    prev_close = None
    if not prior.empty:
        closes = prior["Close"].dropna()
        if not closes.empty:
            prev_close = float(closes.iloc[-1])

    out: list[Signal] = []
    fired: set[str] = set()
    per_session = max(1, len(session))
    for index in range(len(session)):
        upto = session.iloc[:index + 1]
        if not indicators.opening_range_closed(upto):
            continue
        # Readings use only what had printed by this bar.
        window = pd.concat([prior, upto])
        orb = indicators.opening_range(upto)
        vwap = indicators.vwap(upto)
        last = float(upto["Close"].iloc[-1])
        direction = _direction_at(last, vwap, orb)
        if direction is None or direction in fired:
            continue
        atr = indicators.atr(window)
        if atr is None or atr <= 0.0:
            continue
        fired.add(direction)
        bars_left = per_session - (index + 1)
        if bars_left <= 0:
            continue
        sigma = atr * math.sqrt(bars_left)
        highs, lows = _forward_paths(session, index)
        if highs.empty:
            continue
        if direction == LONG:
            mfe = float(highs.max()) - last
            mae = last - float(lows.min())
            close_move = float(session["Close"].iloc[-1]) - last
        else:
            mfe = last - float(lows.min())
            mae = float(highs.max()) - last
            close_move = last - float(session["Close"].iloc[-1])
        touches = {}
        for stop_mult in stop_mults:
            for target_mult in target_mults:
                touches[(stop_mult, target_mult)] = _first_touch(
                    highs, lows, direction, last,
                    sigma * stop_mult, sigma * target_mult)
        day_change = indicators.percent_change(last, prev_close)
        out.append(Signal(
            symbol=symbol, session=session_day, bar_index=index,
            direction=direction, entry=last, atr_bar=atr, bars_left=bars_left,
            rvol=indicators.relative_volume(window),
            relative_strength=indicators.relative_strength(day_change,
                                                           benchmark_change),
            turnover_20d=turnover_20d,
            mfe=max(0.0, mfe), mae=max(0.0, mae), close_move=close_move,
            first_touch=touches))
    return out


def aggregate(signals: list[Signal], stop_mults: tuple[float, ...],
              target_mults: tuple[float, ...]) -> list[Outcome]:
    """Roll signals into one Outcome per (stop, target) pair."""
    grid = {(s, t): Outcome(s, t) for s in stop_mults for t in target_mults}
    for signal in signals:
        for key, verdict in signal.first_touch.items():
            outcome = grid.get(key)
            if outcome is None:
                continue
            if verdict == "TARGET":
                outcome.targets += 1
            elif verdict == "STOP":
                outcome.stops += 1
            else:
                outcome.neither += 1
    return sorted(grid.values(), key=lambda o: (o.stop_mult, o.target_mult))


def excursion_percentiles(signals: list[Signal]) -> dict[str, float]:
    """Where MFE and MAE actually land, in sigma units.

    This is the number the live rule got wrong. If the median MFE is well
    below one sigma, a one-sigma target is beyond the typical move and no
    amount of gate tuning will rescue it.
    """
    if not signals:
        return {}
    mfe = pd.Series([s.excursion_in_sigma(True) for s in signals])
    mae = pd.Series([s.excursion_in_sigma(False) for s in signals])
    out: dict[str, float] = {}
    for label, series in (("mfe", mfe), ("mae", mae)):
        for pct in (10, 25, 50, 75, 90):
            out[f"{label}_p{pct}"] = float(series.quantile(pct / 100.0))
        out[f"{label}_mean"] = float(series.mean())
    return out
