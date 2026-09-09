"""One instrument, every horizon, with the reasoning shown.

Search a symbol and get: the intraday setup if the session supports one,
the short, mid and long assessments, every gate with the number behind it,
the round-trip cost arithmetic, and a BUY or NO BUY per horizon.

WHAT THE VERDICT MEANS, precisely, because it would be easy to read more
into it than is there. BUY means "every gate passed and the plausible move
covers the round trip at least three times". NO BUY names the first gate
that failed. Both are statements about TODAY'S MEASURABLE CONDITIONS and
about arithmetic, not forecasts:

  * The cost columns are arithmetic and have held up under measurement.
  * The ranking has not. Four horizons were tested on this project's own
    data across roughly 855,000 samples and none showed predictive skill;
    at short, mid and long the model's selection scored WORSE than buying
    the same universe equally weighted.

So a BUY here is best read as "nothing measurable rules this out", which is
a much weaker claim than "this will go up", and NO BUY is the more
informative of the two verdicts because it rests on a failed check rather
than on an absence of one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import bar_store
import config
import horizons

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

BUY = "BUY"
NO_BUY = "NO BUY"


@dataclass
class HorizonVerdict:
    """One horizon's outcome for one instrument."""

    horizon: str
    verdict: str
    blocker: str = ""
    reasons: list = field(default_factory=list)
    assessment: object = None
    setup: object = None

    @property
    def is_buy(self) -> bool:
        return self.verdict == BUY


@dataclass
class Report:
    """Everything known about one instrument, per horizon."""

    symbol: str
    found: bool
    price: float = 0.0
    as_of: "datetime | None" = None
    in_fo: bool = False
    lot_size: int = 0
    note: str = ""
    verdicts: dict = field(default_factory=dict)

    @property
    def buys(self) -> list:
        """Horizons whose every gate passed."""
        return [name for name, v in self.verdicts.items() if v.is_buy]


def resolve(query: str) -> tuple:
    """(symbol, in_fo, lot_size) for a user's search string, or ("", ...).

    Matched against the instrument snapshot rather than accepted verbatim,
    so a typo returns nothing instead of an empty analysis that looks like
    a real one.
    """
    import instruments
    import market_source

    wanted = market_source.canonical(query)
    if not wanted:
        return "", False, 0
    snapshot = instruments.load_latest()
    if snapshot is None:
        return wanted, False, 0
    lots = {i.symbol: (i.lot_size or 0) for i in snapshot.fo_underlyings}
    if wanted in lots:
        return wanted, True, lots[wanted]
    known = {i.symbol for i in snapshot.equities}
    if wanted in known:
        return wanted, False, 0
    # Unknown to the snapshot, but the master may still list it - a cash
    # name outside the F&O universe is perfectly analysable.
    return (wanted, False, 0) if market_source.token_for(wanted) else ("", False, 0)


def suggest(query: str, limit: int = 12) -> list:
    """Symbols containing the query, for a search box."""
    import instruments

    text = (query or "").strip().upper()
    snapshot = instruments.load_latest()
    if snapshot is None:
        return []
    pool = sorted({i.symbol for i in snapshot.fo_underlyings}
                  | {i.symbol for i in snapshot.equities})
    if not text:
        return pool[:limit]
    starts = [s for s in pool if s.startswith(text)]
    contains = [s for s in pool if text in s and s not in starts]
    return (starts + contains)[:limit]


def _daily_verdicts(symbol: str, frames: dict) -> dict:
    """Short, mid and long verdicts from the consolidated daily store."""
    frame = frames.get(symbol)
    benchmark = frames.get("NIFTY 50")
    turnover = horizons.turnover_20d(frame)
    out = {}
    for horizon in ("short", "mid", "long"):
        assessment = horizons.assess_daily(symbol, frame, horizon, benchmark)
        if assessment is None:
            out[horizon] = HorizonVerdict(
                horizon=horizon, verdict=NO_BUY,
                blocker=f"not enough history for a "
                        f"{horizons.HORIZONS[horizon]['sessions']}-session view",
                reasons=["fewer daily bars than the horizon needs [FAIL]"])
            continue
        passed, lines = horizons.gate_lines(assessment, turnover)
        blocker = ""
        if not passed:
            failed = [l for l in lines if "[FAIL]" in l]
            blocker = failed[0] if failed else "a gate failed"
        out[horizon] = HorizonVerdict(
            horizon=horizon, verdict=BUY if passed else NO_BUY,
            blocker=blocker, reasons=lines, assessment=assessment)
    return out


def _intraday_verdict(symbol: str) -> HorizonVerdict:
    """Today's intraday verdict, from live or cached 5-minute bars."""
    import instruments
    import scan_data
    import setups

    ticker = instruments.to_ticker(symbol)
    try:
        bars = scan_data.fetch_bars([ticker, config.SCAN_BENCHMARK])
    except Exception as exc:
        return HorizonVerdict(
            horizon="intraday", verdict=NO_BUY,
            blocker=f"no intraday bars ({exc})",
            reasons=[f"bar fetch failed: {exc} [FAIL]"])
    frame = bars.intraday.get(ticker)
    if frame is None or frame.empty:
        return HorizonVerdict(
            horizon="intraday", verdict=NO_BUY,
            blocker="no intraday bars for this symbol",
            reasons=["no 5-minute bars available [FAIL]"])
    benchmark = scan_data.benchmark_change_pct(bars)
    reading = setups.measure(symbol, ticker, frame, bars.daily.get(ticker),
                             benchmark, datetime.now(IST))
    if reading is None:
        return HorizonVerdict(
            horizon="intraday", verdict=NO_BUY,
            blocker="intraday readings not computable",
            reasons=["measure returned nothing [FAIL]"])
    setup = setups.evaluate(reading)
    blocker = ""
    if not setup.actionable:
        failed = [r for r in setup.reasons if "[FAIL]" in r]
        blocker = failed[0] if failed else "no directional agreement"
    return HorizonVerdict(
        horizon="intraday", verdict=BUY if setup.actionable else NO_BUY,
        blocker=blocker, reasons=list(setup.reasons), setup=setup)


def analyse(query: str, include_intraday: bool = True) -> Report:
    """Full report for one instrument across all four horizons."""
    symbol, in_fo, lot = resolve(query)
    if not symbol:
        return Report(symbol=(query or "").strip().upper(), found=False,
                      note="Not found in the instrument snapshot or Kite's "
                           "master. Check the spelling, or sync instruments.")
    frames = bar_store.load("day", symbols=[symbol, "NIFTY 50"])
    daily = frames.get(symbol)
    # Explicit None/empty rather than truthiness: a DataFrame raises on
    # bool() instead of answering, so `if not frame` is a crash rather than
    # the guard it looks like.
    if daily is None or daily.empty:
        return Report(symbol=symbol, found=False, in_fo=in_fo, lot_size=lot,
                      note="No daily history in the consolidated store. "
                           "Run `python -m bar_store` after fetching, or "
                           "fetch this symbol's history first.")
    price = float(daily["Close"].dropna().iloc[-1])
    verdicts = _daily_verdicts(symbol, frames)
    if include_intraday:
        verdicts = {"intraday": _intraday_verdict(symbol), **verdicts}
    return Report(symbol=symbol, found=True, price=price,
                  as_of=daily.index[-1].to_pydatetime(), in_fo=in_fo,
                  lot_size=lot, verdicts=verdicts)


def summary_frame(report: Report) -> pd.DataFrame:
    """The verdict per horizon as a compact table."""
    if not report.found or not report.verdicts:
        return pd.DataFrame()
    rows = []
    for name, verdict in report.verdicts.items():
        assessment = verdict.assessment
        setup = verdict.setup
        if assessment is not None:
            move = f"{assessment.expected_move:.2f}%"
            cost = f"{assessment.cost_pct:.3f}%"
            covers = f"{assessment.cost_multiple:.1f}x"
            view = assessment.direction
            extra = f"{assessment.relative:+.2f}pp vs index"
        elif setup is not None and setup.levels is not None:
            move = f"{setup.levels.target_pct:.2f}%"
            cost = f"{setup.levels.breakeven_pct:.3f}%"
            covers = f"{setup.levels.cost_multiple:.1f}x"
            view = setup.direction
            extra = f"needs {setup.levels.required_win_rate * 100:.1f}% win rate"
        else:
            move = cost = covers = view = "-"
            extra = ""
        rows.append({
            "Horizon": name,
            "Held": f"{horizons.HORIZONS[name]['sessions']} session(s)",
            "Verdict": verdict.verdict,
            "View": view,
            "Plausible move": move,
            "Round trip": cost,
            "Move vs fees": covers,
            "Detail": extra,
            "Blocked by": (verdict.blocker.split(" [")[0]
                           if verdict.blocker else ""),
        })
    return pd.DataFrame(rows)


def render_text(report: Report) -> str:
    """The whole report as plain text, for a CLI or a copy-paste."""
    if not report.found:
        return f"{report.symbol}: {report.note}"
    lines = [f"{report.symbol}  price {report.price:,.2f}"
             + (f"  (F&O, lot {report.lot_size})" if report.in_fo
                else "  (cash only, no derivatives)")]
    if report.as_of:
        lines.append(f"latest daily bar {report.as_of:%Y-%m-%d}")
    lines.append("")
    for name, verdict in report.verdicts.items():
        sessions = horizons.HORIZONS[name]["sessions"]
        lines.append(f"--- {name.upper()} ({sessions} session(s)): "
                     f"{verdict.verdict}")
        if verdict.blocker:
            lines.append(f"    blocked by: {verdict.blocker}")
        for reason in verdict.reasons:
            lines.append(f"    {reason}")
        lines.append("")
    lines.append(f"BUY at: {', '.join(report.buys) if report.buys else 'no horizon'}")
    lines.append("")
    lines.append("A BUY means every gate passed and the plausible move covers")
    lines.append("the round trip. It is not a forecast: four horizons were")
    lines.append("measured on this data and none showed predictive skill.")
    return "\n".join(lines)


def main(argv=None) -> int:
    """CLI: python -m instrument_report RELIANCE"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse one instrument across all four horizons")
    parser.add_argument("symbol", help="NSE symbol, e.g. RELIANCE")
    parser.add_argument("--no-intraday", action="store_true",
                        help="skip the intraday leg, which needs bars for today")
    args = parser.parse_args(argv)
    print(render_text(analyse(args.symbol,
                              include_intraday=not args.no_intraday)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
