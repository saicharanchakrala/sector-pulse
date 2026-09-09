"""Pure intraday indicator maths over pandas bar frames.

Every function here is deterministic and network-free: give it bars, get a
number. That is deliberate, because these are the only parts of the scanner
that can be tested exactly rather than eyeballed against a live market.

Conventions used throughout:
  * A "bar frame" is a DataFrame indexed by tz-aware timestamps with the
    columns Open, High, Low, Close, Volume. Missing columns are tolerated
    where a sensible fallback exists and refused where none does.
  * Functions return None rather than raising or inventing a value when the
    input cannot support the calculation. A caller that treats None as zero
    will fail open, so callers must gate on it explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

IST = ZoneInfo("Asia/Kolkata")

_REQUIRED = ("High", "Low", "Close")


def _clean(frame: pd.DataFrame, columns: tuple[str, ...] = _REQUIRED
           ) -> "pd.DataFrame | None":
    """Drop rows missing any required column; None if nothing usable remains."""
    if frame is None or frame.empty:
        return None
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        return None
    usable = frame.dropna(subset=list(columns))
    return None if usable.empty else usable


def _finite(value: float) -> "float | None":
    """Guard against NaN and inf leaking out of a division."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number


def session_bars(frame: pd.DataFrame, day: "date | None" = None
                 ) -> "pd.DataFrame | None":
    """Rows belonging to one session, defaulting to the frame's last session.

    Intraday frames arrive with several days concatenated, and almost every
    intraday measure is meaningless across a session boundary: an overnight
    gap is not a price move that could have been traded.
    """
    usable = _clean(frame)
    if usable is None:
        return None
    try:
        dates = [ts.date() for ts in usable.index]
    except AttributeError:
        return None
    target = dates[-1] if day is None else day
    keep = [ts for ts, own in zip(usable.index, dates) if own == target]
    if not keep:
        return None
    return usable.loc[keep]


def typical_price(bars: pd.DataFrame) -> pd.Series:
    """(High + Low + Close) / 3, the standard VWAP price input."""
    return (bars["High"] + bars["Low"] + bars["Close"]) / 3.0


def vwap(bars: pd.DataFrame) -> "float | None":
    """Session volume-weighted average price.

    Returns None when volume is absent or sums to zero. A zero-volume VWAP
    would silently collapse to a simple mean, which is a different indicator
    wearing VWAP's name, and every gate downstream would then compare price
    against the wrong line.
    """
    usable = _clean(bars)
    if usable is None or "Volume" not in usable.columns:
        return None
    volumes = usable["Volume"].fillna(0.0)
    total = float(volumes.sum())
    if total <= 0.0:
        return None
    weighted = float((typical_price(usable) * volumes).sum())
    return _finite(weighted / total)


@dataclass(frozen=True)
class OpeningRange:
    """High and low of the session's first N minutes."""

    high: float
    low: float
    bars: int
    minutes: int

    @property
    def width(self) -> float:
        """Absolute span of the opening range."""
        return self.high - self.low

    def width_pct(self, reference: float) -> "float | None":
        """Opening-range width as a percent of a reference price."""
        if reference <= 0.0:
            return None
        return _finite(self.width / reference * 100.0)


def opening_range(bars: pd.DataFrame,
                  minutes: int = config.SCAN_OPENING_RANGE_MINUTES
                  ) -> "OpeningRange | None":
    """Opening range over the first `minutes` of the session in `bars`.

    Bars are selected by timestamp rather than by position, so a session
    missing its 09:15 print yields a shorter range instead of silently
    sliding the window later into the day.
    """
    usable = session_bars(bars)
    if usable is None or minutes <= 0:
        return None
    start = usable.index[0]
    cutoff = start + timedelta(minutes=minutes)
    window = usable.loc[[ts for ts in usable.index if ts < cutoff]]
    if window.empty:
        return None
    high = _finite(window["High"].max())
    low = _finite(window["Low"].min())
    if high is None or low is None or high < low:
        return None
    return OpeningRange(high=high, low=low, bars=len(window), minutes=minutes)


def opening_range_closed(bars: pd.DataFrame,
                         minutes: int = config.SCAN_OPENING_RANGE_MINUTES
                         ) -> bool:
    """Whether the session has traded past its opening range.

    This is a precondition for any breakout reading, not a nicety. While the
    latest bar is still inside the opening-range window, that window's high
    and low are computed from the same bar as the current price, so

        orb.low <= last <= orb.high

    holds identically and no break can ever be represented. A scan run
    before the range closes therefore returns "no setup" for arithmetic
    reasons, on every symbol, regardless of what the market is doing.
    """
    session = session_bars(bars)
    if session is None or minutes <= 0:
        return False
    start = session.index[0]
    return session.index[-1] >= start + timedelta(minutes=minutes)


@dataclass(frozen=True)
class CentralPivotRange:
    """Previous-session CPR: the pivot plus its top and bottom band.

    A narrow CPR relative to price is the conventional read for a trending
    day and a wide one for a range-bound day. This module computes it and
    says nothing about whether that read holds.
    """

    pivot: float
    top: float
    bottom: float

    @property
    def width(self) -> float:
        """Distance between the two bands."""
        return self.top - self.bottom

    def width_pct(self, reference: float) -> "float | None":
        """CPR width as a percent of a reference price."""
        if reference <= 0.0:
            return None
        return _finite(self.width / reference * 100.0)


def central_pivot_range(prev_high: float, prev_low: float,
                        prev_close: float) -> "CentralPivotRange | None":
    """CPR from one previous session's high, low and close."""
    for value in (prev_high, prev_low, prev_close):
        if value is None or _finite(value) is None or value <= 0.0:
            return None
    if prev_high < prev_low:
        return None
    pivot = (prev_high + prev_low + prev_close) / 3.0
    bc = (prev_high + prev_low) / 2.0
    tc = pivot - bc + pivot
    return CentralPivotRange(pivot=pivot, top=max(tc, bc), bottom=min(tc, bc))


@dataclass(frozen=True)
class PivotLadder:
    """Previous-session floor pivots: the pivot plus three levels each way.

    The standard construction, the one every Indian broker's screen shows:

        pivot = (H + L + C) / 3
        R1 = 2P - L          S1 = 2P - H
        R2 = P + (H - L)     S2 = P - (H - L)
        R3 = H + 2(P - L)    S3 = L - 2(H - P)

    Computed from the PREVIOUS session only, so every level is fixed before
    the open and cannot move during the day.

    FOR DISPLAY, NOT FOR PLACEMENT. Using these as stops was built and
    then measured on 79,725 candidate stops across 249 dates: paired
    within the same session, a stop sitting on a pivot was hit +0.445 pp
    MORE often than one at the same distance elsewhere, CI [-1.30, +2.22],
    sign test p 0.37. See levels._structural_levels for the full figures
    and why an unpaired version of the test misleadingly read -2.19 pp.
    So these levels are shown because they are worth seeing and cost
    nothing, not because they predict where price turns.
    """

    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float

    @property
    def supports(self) -> tuple:
        """S1, S2, S3 - nearest first."""
        return (self.s1, self.s2, self.s3)

    @property
    def resistances(self) -> tuple:
        """R1, R2, R3 - nearest first."""
        return (self.r1, self.r2, self.r3)

    def below(self, price: float) -> tuple:
        """Every level under `price`, nearest first.

        Deliberately not just the supports: after a strong move up, R1 and
        even R2 sit BELOW the current price and are then the nearest real
        structure beneath it. Treating only S1-S3 as support would reach
        past them for a level much further away.
        """
        levels = sorted((v for v in self._all if v < price), reverse=True)
        return tuple(levels)

    def above(self, price: float) -> tuple:
        """Every level over `price`, nearest first."""
        return tuple(sorted(v for v in self._all if v > price))

    @property
    def _all(self) -> tuple:
        return (self.s3, self.s2, self.s1, self.pivot,
                self.r1, self.r2, self.r3)


def pivot_ladder(prev_high: float, prev_low: float,
                 prev_close: float) -> "PivotLadder | None":
    """The pivot ladder from one previous session's high, low and close.

    None rather than a partial answer on unusable input, matching
    central_pivot_range: a ladder built from a zero low would put S3 at a
    negative price and quietly become the nearest "level" beneath
    everything.
    """
    for value in (prev_high, prev_low, prev_close):
        if value is None or _finite(value) is None or value <= 0.0:
            return None
    if prev_high < prev_low:
        return None
    span = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3.0
    return PivotLadder(
        pivot=pivot,
        r1=2.0 * pivot - prev_low,
        r2=pivot + span,
        r3=prev_high + 2.0 * (pivot - prev_low),
        s1=2.0 * pivot - prev_high,
        s2=pivot - span,
        s3=prev_low - 2.0 * (prev_high - pivot),
    )


def true_range(bars: pd.DataFrame) -> "pd.Series | None":
    """Wilder true range per bar: the widest of the three standard spans.

    The previous close matters because a gap is real risk; using only
    High-Low would understate the stop distance on exactly the bars where
    getting the stop right matters most.
    """
    usable = _clean(bars)
    if usable is None or len(usable) < 2:
        return None
    prior_close = usable["Close"].shift(1)
    spans = pd.concat([
        usable["High"] - usable["Low"],
        (usable["High"] - prior_close).abs(),
        (usable["Low"] - prior_close).abs(),
    ], axis=1)
    return spans.max(axis=1).dropna()


def true_range_by_session(frame: pd.DataFrame) -> "pd.Series | None":
    """True ranges computed inside each session and concatenated.

    Each session's opening bar keeps its own high-low span instead of being
    measured against yesterday's close. An overnight gap is not a move that
    could have been traded intraday, and letting one in inflates ATR badly:
    with a constant
    1.0 intraday range and a 20-rupee gap, ATR read 2.21 three bars into the
    session against a true 1.0, because Wilder smoothing decays a shock only
    as (13/14)^k. That doubled every stop and target in the first hour, and
    no gate could catch it because both gates scale from the same ATR.

    One numpy pass rather than a loop per session. The earlier version
    iterated the index to find the days and again per day to slice it, then
    built a dropna, a shift, a three-column concat and a row-wise max for
    each session - 44.5ms per symbol, about 9.6s across the F&O universe.
    The session boundary is now a boolean mask, which is where the
    gap-exclusion actually happens: the first bar of each session takes
    high-low only, exactly as a per-session concat did. Verified equal to
    the previous implementation on real cached frames before replacing it.
    """
    if frame is None or frame.empty:
        return None
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        return None
    if any(name not in frame.columns for name in _REQUIRED):
        return None
    usable = frame.dropna(subset=list(_REQUIRED))
    if len(usable) < 2:
        return None
    high = usable["High"].to_numpy(dtype=float)
    low = usable["Low"].to_numpy(dtype=float)
    close = usable["Close"].to_numpy(dtype=float)
    days = usable.index.date
    # True where a bar opens a new session, so its predecessor's close
    # belongs to a different day and must not enter the range.
    opens_session = np.empty(len(days), dtype=bool)
    opens_session[0] = True
    opens_session[1:] = days[1:] != days[:-1]

    span = high - low
    prior = np.empty_like(close)
    prior[0] = np.nan
    prior[1:] = close[:-1]
    with np.errstate(invalid="ignore"):
        ranges = np.maximum(span, np.maximum(np.abs(high - prior),
                                             np.abs(low - prior)))
    ranges[opens_session] = span[opens_session]

    # A session of one bar contributed nothing before, because true_range
    # required two rows. Preserved so the ATR sample size is unchanged.
    counts = {}
    for day in days:
        counts[day] = counts.get(day, 0) + 1
    keep = np.array([counts[day] >= 2 for day in days], dtype=bool)
    if not keep.any():
        return None
    return pd.Series(ranges[keep], index=usable.index[keep])


def atr(bars: pd.DataFrame, period: int = config.SCAN_ATR_BARS
        ) -> "float | None":
    """Wilder's ATR over per-session true ranges, gaps excluded.

    Requires a full `period` of true ranges. A shorter series returns None:
    an ATR computed over three bars is not a smaller ATR, it is noise, and
    it would size stops wrongly precisely on thin names.
    """
    ranges = true_range_by_session(bars)
    if ranges is None or period <= 0 or len(ranges) < period:
        return None
    # Wilder smoothing is sequential and cannot be vectorised, but walking
    # a numpy array is far cheaper than iterating a Series, which builds a
    # scalar object per element.
    values = ranges.to_numpy(dtype=float)
    value = float(values[:period].mean())
    for span in values[period:]:
        value = (value * (period - 1) + float(span)) / period
    return _finite(value)


def cumulative_volume_by_time(frame: pd.DataFrame
                              ) -> dict[date, "pd.Series | None"]:
    """Per session, cumulative volume indexed by clock time.

    Keyed by time-of-day so today's pace can be compared against the same
    point in earlier sessions. Comparing raw totals instead would make every
    stock look quiet at 10am and busy at 3pm.
    """
    if frame is None or frame.empty or "Volume" not in frame.columns:
        return {}
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        return {}
    out: dict[date, "pd.Series | None"] = {}
    # One vectorised grouping, and index.time rather than a comprehension.
    # Profiled at 2.25s of an 8.64s scan across forty symbols before this.
    for day, rows in frame.groupby(index.date, sort=True):
        volumes = rows["Volume"].fillna(0.0)
        if volumes.empty:
            out[day] = None
            continue
        series = volumes.cumsum()
        series.index = rows.index.time
        out[day] = series
    return out


def relative_volume(frame: pd.DataFrame, asof: "time | None" = None
                    ) -> "float | None":
    """Today's cumulative volume over the median of prior sessions at the same time.

    1.0 means today is tracking a typical session; 2.0 means twice the usual
    participation by this point. The median is taken across prior sessions
    rather than the mean so one frenzied day does not set the baseline.
    """
    curves = cumulative_volume_by_time(frame)
    usable = {day: series for day, series in curves.items() if series is not None}
    if len(usable) < 2:
        return None
    today = max(usable)
    current = usable[today]
    clock = current.index[-1] if asof is None else asof
    done = _finite(current.asof(clock)) if len(current) else None
    if done is None or done <= 0.0:
        return None
    baseline: list[float] = []
    for day, series in usable.items():
        if day == today:
            continue
        # asof() returns a short session's final total, which reads as "quiet
        # at this hour" and inflates today's ratio. Measured: one prior
        # session truncated to 20 of 75 bars pushed rvol from 2.0 to 3.16,
        # through the 1.2 floor on nothing but missing data.
        if len(series) == 0 or series.index[-1] < clock:
            continue
        at_time = _finite(series.asof(clock))
        if at_time is not None and at_time > 0.0:
            baseline.append(at_time)
    if not baseline:
        return None
    median = float(pd.Series(baseline).median())
    if median <= 0.0:
        return None
    return _finite(done / median)


def percent_change(last: float, reference: float) -> "float | None":
    """Percent change of `last` against `reference`."""
    if reference is None or reference <= 0.0 or last is None:
        return None
    return _finite((last / reference - 1.0) * 100.0)


def relative_strength(symbol_change_pct: float,
                      benchmark_change_pct: float) -> "float | None":
    """Simple spread of a symbol's day change against the benchmark's.

    Positive means outperforming today. It is a spread of returns, not a
    beta-adjusted alpha, so a high-beta name will read strong in any up
    market. Read it as "leading today", nothing more.
    """
    if symbol_change_pct is None or benchmark_change_pct is None:
        return None
    return _finite(symbol_change_pct - benchmark_change_pct)


def minutes_left_in_session(now: datetime) -> int:
    """Minutes from `now` to the NSE equity close, clamped at zero.

    An aware datetime is converted to IST first, so a Sydney-aware 13:00 is
    not mistaken for 13:00 in Mumbai. A naive datetime is read as already
    being exchange-local, which is what unit tests pass.
    """
    close_h, close_m = config.SCAN_SESSION_CLOSE
    if now.tzinfo is not None:
        now = now.astimezone(IST)
    close = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return max(0, int((close - now).total_seconds() // 60))
