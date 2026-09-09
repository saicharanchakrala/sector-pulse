"""Assess every instrument across four holding periods, not just intraday.

THE FLAW THIS FIXES. The scanner only ever assessed intraday, and its gates
are intraday gates. Chief among them: "45 minutes left vs minimum" - which
correctly refuses a same-session trade near the close, and has nothing
whatever to say about a ten-day or one-year view. At 15:22 the scanner
reported nothing actionable across 216 names, which reads as "no
opportunities" when it actually meant "too late for MY horizon only".

So each horizon now gets its own inputs, its own gates and its own clock:

    intraday   entry to today's close      5-minute bars, session-time gated
    short      about 10 sessions           daily bars, no session gate
    mid        about 63 sessions           daily bars, no session gate
    long       about 252 sessions          daily bars, no session gate

WHAT THE RANKING IS AND IS NOT. Read the honesty note below before using
the output. Every one of these four horizons has been measured on this
project's own data and none showed predictive skill; at short, mid and long
the model's selection was measurably WORSE than buying the same universe
equally weighted (spread -0.26%, -0.25% and -7.36% respectively). What the
ranking here therefore is: a consistent, cost-aware ordering of what the
data currently looks like. What it is not: a forecast, or evidence that the
top of the list will outperform the bottom. The columns that survived
measurement are the cost ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import bar_store
import config
import trade_costs

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Sessions held, and the round-trip cost that horizon actually pays.
# Intraday is MIS: brokerage capped at Rs 20 a leg, STT on the sell only.
# Everything overnight is CNC delivery: zero brokerage but STT on BOTH
# legs, which is dearer in absolute terms and cheaper as a share of a
# larger expected move.
HORIZONS = {
    "intraday": {"sessions": 1, "cost_pct": 0.082, "daily": False},
    "short": {"sessions": 10, "cost_pct": 0.23, "daily": True},
    "mid": {"sessions": 63, "cost_pct": 0.23, "daily": True},
    "long": {"sessions": 252, "cost_pct": 0.23, "daily": True},
}

HONESTY = (
    "**This list is not a prediction, and we tested that properly.** We ran "
    "the same method on about 855,000 past examples. It picked winners no "
    "better than the identical method fed deliberately scrambled answers - "
    "and over a one-year holding period, the scrambled version actually did "
    "better. Simply buying every stock equally beat our picks at all three "
    "holding periods. So read the order as 'what the numbers look like "
    "today', never as 'these are the good ones'."
)

# The same finding for anyone who wants the figures rather than the plain
# summary. Kept beside it so the two can never drift apart.
HONESTY_DETAIL = (
    "Out-of-sample results on a survivorship-corrected universe of 2,254 "
    "companies. Short: -0.21% excess over the index, selection spread "
    "-0.26%, permutation p 0.55. Mid: +1.65% excess but equal-weight gave "
    "+1.90%, spread -0.25%, permutation p 0.55. Long: +3.64% excess against "
    "equal-weight +11.01%, spread -7.36%, permutation p 1.00 - all 30 "
    "shuffled-label runs beat the real one. Ranking accuracy (AUC) was "
    "0.5130, 0.4740 and 0.4671 where 0.50 is a coin flip."
)


@dataclass
class Assessment:
    """One symbol at one horizon: what is measurable, and what it costs."""

    symbol: str
    horizon: str
    price: float
    direction: str
    trend: float                  # return over the horizon's own lookback, %
    relative: float               # same, minus the benchmark, %
    volatility: float             # annualised, %
    drawdown: float               # from the trailing high, %
    position_in_range: float      # 0..1 within the trailing window
    expected_move: float          # volatility scaled to the horizon, %
    cost_pct: float               # round-trip charges, %
    cost_multiple: float          # expected move over cost
    score: float = 0.0
    reasons: list = field(default_factory=list)
    blocked: str = ""

    @property
    def pays_for_itself(self) -> bool:
        """Whether the plausible move covers the round trip at least 3x."""
        return self.cost_multiple >= 3.0


def _sessions_for(horizon: str) -> int:
    return int(HORIZONS[horizon]["sessions"])


def _lookback_for(horizon: str) -> int:
    """Trailing window used to describe a horizon.

    Deliberately the same length as the holding period: describing a
    one-year view from a five-day trend would be measuring something the
    horizon cannot act on.
    """
    return max(5, _sessions_for(horizon))


def _annualised_volatility(closes: np.ndarray) -> float:
    if closes.size < 10:
        return float("nan")
    returns = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
    return float(np.std(returns) * np.sqrt(252) * 100.0)


def assess_daily(symbol: str, frame: pd.DataFrame, horizon: str,
                 benchmark: "pd.DataFrame | None") -> "Assessment | None":
    """One symbol at one daily horizon, from cached daily bars.

    No session-time gate: whether 40 minutes remain today is irrelevant to
    a position meant to be held for ten sessions or a year.
    """
    if frame is None or "Close" not in frame.columns:
        return None
    closes = frame["Close"].dropna().to_numpy(dtype=float)
    lookback = _lookback_for(horizon)
    if closes.size < lookback + 5:
        return None
    price = float(closes[-1])
    if price <= 0:
        return None
    window = closes[-(lookback + 1):]
    trend = (price / window[0] - 1.0) * 100.0
    volatility = _annualised_volatility(closes[-max(lookback, 60):])
    trailing = closes[-lookback:]
    high, low = float(trailing.max()), float(trailing.min())
    span = max(high - low, 1e-9)

    relative = trend
    if benchmark is not None and "Close" in benchmark.columns:
        bench = benchmark["Close"].dropna().to_numpy(dtype=float)
        if bench.size >= lookback + 1:
            bench_move = (float(bench[-1]) / float(bench[-(lookback + 1)])
                          - 1.0) * 100.0
            relative = trend - bench_move

    sessions = _sessions_for(horizon)
    # Volatility scaled to the horizon by root time, which is the same
    # sigma convention the intraday sizer uses.
    expected = (volatility / np.sqrt(252.0) * np.sqrt(sessions)
                if volatility == volatility else float("nan"))
    cost = float(HORIZONS[horizon]["cost_pct"])
    multiple = (expected / cost) if cost > 0 and expected == expected else 0.0

    direction = "LONG" if relative > 0 else "SHORT"
    reasons = [
        f"{lookback}-session trend {trend:+.2f}% "
        f"({'ahead of' if relative > 0 else 'behind'} the index by "
        f"{abs(relative):.2f}pp)",
        f"annualised volatility {volatility:.1f}%, so a {sessions}-session "
        f"move of about {expected:.2f}% is plausible",
        f"round trip costs {cost:.3f}%, which the plausible move covers "
        f"{multiple:.1f}x [{'PASS' if multiple >= 3.0 else 'FAIL'}]",
    ]
    return Assessment(
        symbol=symbol, horizon=horizon, price=price, direction=direction,
        trend=trend, relative=relative, volatility=volatility,
        drawdown=(price / high - 1.0) * 100.0 if high > 0 else float("nan"),
        position_in_range=(price - low) / span,
        expected_move=expected, cost_pct=cost, cost_multiple=multiple,
        reasons=reasons,
        blocked="" if multiple >= 3.0 else "expected move too small for costs")


# Daily-horizon gates. Deliberately fewer and blunter than the intraday
# set: a position held for weeks is not helped by a five-minute reading,
# and inventing gates to look thorough would be nine ways of saying the
# same thing. Each returns (passed, line) so the report can show the number
# behind every verdict rather than just its outcome.
MIN_TURNOVER = 50_000_000.0        # Rs 5 crore a day, same floor as intraday
MIN_PRICE = 20.0
MIN_COST_MULTIPLE = 3.0
MAX_DRAWDOWN = -35.0               # deep below the trailing high, %


def gate_lines(assessment: "Assessment", turnover: "float | None"
               ) -> tuple:
    """(passed, reasons) for one daily-horizon assessment."""
    reasons = []
    checks = []

    ok = assessment.price >= MIN_PRICE
    checks.append(ok)
    reasons.append(f"price {assessment.price:,.2f} vs floor "
                   f"{MIN_PRICE:,.2f} [{'PASS' if ok else 'FAIL'}]")

    if turnover is None:
        reasons.append("20-session turnover unknown, so liquidity fails "
                       "closed rather than passing on missing data [FAIL]")
        checks.append(False)
    else:
        ok = turnover >= MIN_TURNOVER
        checks.append(ok)
        reasons.append(f"20-session turnover {turnover:,.0f} vs floor "
                       f"{MIN_TURNOVER:,.0f} [{'PASS' if ok else 'FAIL'}]")

    ok = assessment.cost_multiple >= MIN_COST_MULTIPLE
    checks.append(ok)
    reasons.append(
        f"plausible {_sessions_for(assessment.horizon)}-session move "
        f"{assessment.expected_move:.2f}% covers the {assessment.cost_pct:.3f}% "
        f"round trip {assessment.cost_multiple:.1f}x, needs "
        f"{MIN_COST_MULTIPLE:.0f}x [{'PASS' if ok else 'FAIL'}]")

    ok = assessment.relative > 0
    checks.append(ok)
    reasons.append(
        f"relative to the index {assessment.relative:+.2f}pp over "
        f"{_lookback_for(assessment.horizon)} sessions, needs to be ahead "
        f"[{'PASS' if ok else 'FAIL'}]")

    deep = assessment.drawdown == assessment.drawdown and         assessment.drawdown < MAX_DRAWDOWN
    checks.append(not deep)
    reasons.append(
        f"{assessment.drawdown:+.2f}% from the trailing high, floor "
        f"{MAX_DRAWDOWN:.0f}% [{'FAIL' if deep else 'PASS'}]")

    if assessment.volatility != assessment.volatility:
        reasons.append("volatility not computable, so the move estimate "
                       "cannot be trusted [FAIL]")
        checks.append(False)

    return all(checks), reasons


def turnover_20d(frame: "pd.DataFrame | None") -> "float | None":
    """Median rupee turnover over the last 20 sessions, or None."""
    if frame is None or frame.empty:
        return None
    if "Volume" not in frame.columns or "Close" not in frame.columns:
        return None
    tail = frame.tail(20)
    values = (tail["Close"] * tail["Volume"]).dropna()
    if values.empty:
        return None
    return float(values.median())


def rank(assessments: list, top: int = 20) -> list:
    """Score cross-sectionally, then return the best `top`.

    Scored against the other names in the SAME run rather than on absolute
    thresholds, because "strong relative trend" only means anything next to
    what the rest of the market did. Names whose plausible move cannot
    cover costs are ranked last whatever else they show.
    """
    usable = [a for a in assessments if a is not None]
    if not usable:
        return []
    frame = pd.DataFrame([{
        "i": i, "relative": a.relative, "multiple": a.cost_multiple,
        "position": a.position_in_range,
        "volatility": -(a.volatility if a.volatility == a.volatility else 0.0),
    } for i, a in enumerate(usable)])
    for column, weight in (("relative", 0.45), ("multiple", 0.30),
                           ("position", 0.15), ("volatility", 0.10)):
        ranks = frame[column].rank(pct=True).fillna(0.0)
        frame[f"w_{column}"] = ranks * weight
    frame["score"] = frame[[c for c in frame.columns
                            if c.startswith("w_")]].sum(axis=1)
    for _, row in frame.iterrows():
        usable[int(row["i"])].score = round(float(row["score"]), 4)
    payable = [a for a in usable if a.pays_for_itself]
    rest = [a for a in usable if not a.pays_for_itself]
    payable.sort(key=lambda a: a.score, reverse=True)
    rest.sort(key=lambda a: a.score, reverse=True)
    return (payable + rest)[:top]


def assess_universe(symbols: list, horizons: "list | None" = None,
                    benchmark_symbol: str = "NIFTY 50",
                    top: int = 20, progress=None) -> dict:
    """Every symbol across the daily horizons, ranked, top `top` each.

    `progress` is called as progress(horizon, done, total) so a UI can fill
    its table as results arrive rather than waiting for the whole sweep.
    """
    wanted = [h for h in (horizons or list(HORIZONS))
              if HORIZONS.get(h, {}).get("daily")]
    if not wanted:
        return {}
    longest = max(_lookback_for(h) for h in wanted)
    # Calendar days for the deepest lookback, with slack for holidays.
    start = date.today() - timedelta(days=int(longest * 7 / 5) + 30)
    frames = bar_store.load("day", symbols=list(symbols) + [benchmark_symbol],
                            start=start)
    if not frames:
        logger.warning("No consolidated daily bars; run bar_store first")
        return {}
    benchmark = frames.get(benchmark_symbol)
    out = {}
    for horizon in wanted:
        results = []
        total = len(symbols)
        for done, symbol in enumerate(symbols, start=1):
            frame = frames.get(symbol)
            if frame is None:
                continue
            try:
                assessment = assess_daily(symbol, frame, horizon, benchmark)
            except Exception as exc:
                logger.warning("%s at %s: %s", symbol, horizon, exc)
                continue
            if assessment is not None:
                results.append(assessment)
            if progress is not None and done % 25 == 0:
                progress(horizon, done, total)
        out[horizon] = rank(results, top=top)
        if progress is not None:
            progress(horizon, total, total)
    return out


def to_frame(assessments: list) -> pd.DataFrame:
    """Ranked assessments as a display table, cost columns to the front."""
    if not assessments:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Symbol": a.symbol,
        "View": a.direction,
        "Price": round(a.price, 2),
        "Cost %": round(a.cost_pct, 3),
        "Plausible move %": round(a.expected_move, 2),
        "Move vs fees": f"{a.cost_multiple:.1f}x",
        "Pays for itself?": "yes" if a.pays_for_itself else "no",
        "Trend %": round(a.trend, 2),
        "Beat index by": round(a.relative, 2),
        "Volatility %": round(a.volatility, 1),
        "Below recent peak": round(a.drawdown, 2),
        "Where in range": round(a.position_in_range, 2),
        "Our ranking": a.score,
    } for a in assessments])
