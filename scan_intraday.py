"""Intraday scanner CLI: python scan_intraday.py [--options] [--top N].

Ranks the F&O universe by a transparent intraday rule and prints, for every
setup that clears each gate, an entry, a stop, a target and a size. With
--options it also picks the near-the-money contract that is actually
tradeable for the top names.

Read this before using it:

  * Nothing here is backtested. The gates and weights are conventional
    technical-analysis choices, not measured edges. yfinance caps 5-minute
    history near 60 days, which is far too little to establish whether this
    rule makes money, so the scan reports what it measured and stops there.
  * Bars are delayed, not live. Every level is computed from the last bar
    yfinance served, which can be 15 minutes stale. Do not treat the entry
    price as executable.
  * It places no orders and never will.

Educational decision support. Not financial advice.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

import config
import instruments
import options_chain
import scan_data
import setups
from levels import LONG

logger = logging.getLogger("scan_intraday")

IST = ZoneInfo("Asia/Kolkata")

DISCLAIMER = (
    "Educational decision support only - NOT financial advice. Levels come "
    "from delayed bars and an unbacktested rule. No orders are placed."
)

_CSV_FIELDS = [
    "run_date", "run_time", "replayed", "symbol", "direction", "score",
    "entry", "stop",
    "target", "quantity", "risk_rupees", "stop_pct", "target_pct",
    "breakeven_pct", "cost_rupees", "required_win_rate", "rvol",
    "relative_strength", "oi_change_pct", "futures_share", "vwap",
    "turnover_20d",
]


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Rank NSE intraday setups with entry, stop and target")
    parser.add_argument("--top", type=int, default=10,
                        help="how many ranked setups to detail (default 10)")
    parser.add_argument("--capital", type=float, default=config.SCAN_CAPITAL,
                        help="intraday capital the sizer assumes")
    parser.add_argument("--risk-pct", type=float,
                        default=config.SCAN_RISK_PCT_PER_TRADE,
                        help="percent of capital risked per trade")
    parser.add_argument("--options", action="store_true",
                        help="also pick a tradeable option contract per setup")
    parser.add_argument("--discover", action="store_true",
                        help="re-discover every listed instrument from NSE "
                             "before scanning, instead of using the last "
                             "saved snapshot")
    parser.add_argument("--all-equities", action="store_true",
                        help="scan every listed equity (about 2,570) rather "
                             "than the F&O single stocks. Adds roughly 5 "
                             "minutes and most names have no 5-minute bars")
    parser.add_argument("--symbols", default="",
                        help="comma-separated symbols to scan instead of the universe")
    parser.add_argument("--limit", type=int, default=0,
                        help="scan only the first N of the F&O list, which is "
                             "alphabetical, not ranked by liquidity (0 = all)")
    parser.add_argument("--as-of", default="",
                        help="replay at a past instant: 'HH:MM' for today, or "
                             "'YYYY-MM-DD HH:MM' for an earlier session. Bars "
                             "after that instant are discarded, so the scan "
                             "sees only what was knowable then. Limited to the "
                             f"last {config.SCAN_MAX_REPLAY_DAYS} days, which "
                             "is where yfinance stops serving 5-minute bars")
    args = parser.parse_args(argv)
    if args.capital <= 0:
        parser.error(f"--capital must be positive, got {args.capital}")
    if args.risk_pct <= 0:
        parser.error(f"--risk-pct must be positive, got {args.risk_pct}")
    if args.top <= 0:
        parser.error(f"--top must be positive, got {args.top}")
    if args.as_of.strip():
        try:
            moment = parse_as_of(args.as_of)
        except ValueError as exc:
            parser.error(str(exc))
        age = (datetime.now(IST).date() - moment.date()).days
        if age > config.SCAN_MAX_REPLAY_DAYS:
            parser.error(
                f"{moment.date()} is {age} days back. yfinance stops serving "
                f"5-minute bars near 60 days, so replays are capped at "
                f"{config.SCAN_MAX_REPLAY_DAYS} days. Nothing older can be "
                f"replayed from free data at all.")
        if moment > datetime.now(IST):
            parser.error(
                f"{moment.strftime('%Y-%m-%d %H:%M')} is in the future. A "
                f"future instant passes every replay guard - the option "
                f"refusal, the open-interest timestamp check and the "
                f"session-time gate all read it as live - so it is refused "
                f"outright.")
    return args


def parse_as_of(text: str) -> datetime:
    """Parse 'HH:MM' (today) or 'YYYY-MM-DD HH:MM' into an IST datetime."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        raise ValueError("--as-of is empty")
    for fmt, dated in (("%Y-%m-%d %H:%M", True), ("%Y-%m-%dT%H:%M", True),
                       ("%H:%M", False)):
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        if dated:
            return parsed.replace(tzinfo=IST)
        today = datetime.now(IST)
        return today.replace(hour=parsed.hour, minute=parsed.minute,
                             second=0, microsecond=0)
    raise ValueError(
        f"--as-of must be 'HH:MM' or 'YYYY-MM-DD HH:MM', got {text!r}")


def load_universe(refresh: bool) -> instruments.Universe:
    """The last discovered instrument set, or a fresh discovery pass.

    Discovery is the only source of symbols. There is no fallback constant:
    a hardcoded list is what previously left NIFTYFPI treated as an equity
    on every run, and a scan that cannot say which instruments were listed
    when it ran cannot be audited afterwards.
    """
    if not refresh:
        cached = instruments.load_latest()
        if cached is not None:
            return cached
        logger.info("No instrument snapshot on disk; discovering now")
    return instruments.discover()


def resolve_symbols(args: argparse.Namespace,
                    universe: instruments.Universe) -> tuple[list[str], str]:
    """Symbols to scan and a description of where they came from.

    The default is the F&O single stocks. Those are the names with the depth
    to enter and exit inside a session and the only ones with options, so
    scanning them is the whole tradeable intraday universe. The full listed
    set is available on a flag; most of it has no 5-minute bars and would be
    rejected on turnover anyway.
    """
    captured = universe.captured_at.strftime("%Y-%m-%d %H:%M")
    if args.symbols.strip():
        picked = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        return picked, f"--symbols ({len(picked)})"
    if args.all_equities:
        picked = [inst.symbol for inst in universe.equities]
        source = f"all listed equities ({len(picked)}, discovered {captured})"
    else:
        picked = [inst.symbol for inst in universe.fo_stocks]
        source = f"F&O single stocks ({len(picked)}, discovered {captured})"
    if args.limit > 0:
        picked = picked[:args.limit]
        source += f", limited to {len(picked)}"
    return picked, source


def build_setups(symbols: list[str], bars: scan_data.BarSet,
                 benchmark: "float | None", now: datetime, capital: float,
                 risk_pct: float,
                 fo_state: "dict | None" = None) -> list[setups.Setup]:
    """Measure and gate every symbol that returned bars."""
    out: list[setups.Setup] = []
    states = fo_state or {}
    for symbol in symbols:
        ticker = instruments.to_ticker(symbol)
        frame = bars.intraday.get(ticker)
        if frame is None:
            continue
        reading = setups.measure(symbol, ticker, frame, bars.daily.get(ticker),
                                 benchmark, now, fo_state=states.get(symbol))
        if reading is None:
            continue
        out.append(setups.evaluate(reading, capital=capital, risk_pct=risk_pct))
    return out


def _row(setup: setups.Setup, now: datetime,
         replayed: bool = False) -> dict[str, object]:
    """Flatten one actionable setup into a scan_log.csv row.

    `replayed` is recorded because the log is meant to become a track
    record, and a replayed row stamped with a past time is otherwise
    indistinguishable from a live scan made at that time.
    """
    trade = setup.levels
    reading = setup.readings
    return {
        "run_date": now.strftime("%Y-%m-%d"),
        "run_time": now.strftime("%H:%M:%S"),
        "replayed": replayed,
        "symbol": reading.symbol,
        "direction": setup.direction,
        "score": setup.rank_score,
        "entry": round(trade.entry, 2),
        "stop": round(trade.stop, 2),
        "target": round(trade.target, 2),
        "quantity": trade.quantity,
        "risk_rupees": round(trade.lot_risk, 2),
        "stop_pct": round(trade.stop_pct, 3),
        "target_pct": round(trade.target_pct, 3),
        "breakeven_pct": round(trade.breakeven_pct, 4),
        "cost_rupees": round(trade.cost_rupees, 2),
        "required_win_rate": round(trade.required_win_rate, 4),
        "rvol": round(reading.rvol, 3) if reading.rvol is not None else "",
        "relative_strength": (round(reading.relative_strength, 3)
                              if reading.relative_strength is not None else ""),
        "oi_change_pct": (round(reading.oi_change_pct, 3)
                          if reading.oi_change_pct is not None else ""),
        "futures_share": (round(reading.futures_share, 4)
                          if reading.futures_share is not None else ""),
        "vwap": round(reading.vwap, 2) if reading.vwap is not None else "",
        "turnover_20d": (round(reading.turnover_20d, 0)
                         if reading.turnover_20d is not None else ""),
    }


def _existing_header(path) -> "list[str] | None":
    """The header already in the log file, or None when there is none."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            first = next(csv.reader(handle), None)
    except (OSError, csv.Error, StopIteration):
        return None
    return first or None


def append_log(rows: list[dict[str, object]]) -> None:
    """Append actionable setups to scan_log.csv, header-aware.

    A file whose header predates a change to _CSV_FIELDS is rotated aside
    rather than appended to. Appending would write the new field order under
    the old column names and silently misalign every row - adding the
    'replayed' column did exactly that, putting True under 'symbol' - which
    corrupts the log this file exists to be.
    """
    if not rows:
        return
    path = config.SCAN_LOG_CSV
    try:
        header = _existing_header(path) if path.exists() else None
        if header is not None and header != _CSV_FIELDS:
            retired = path.with_suffix(".csv.superseded")
            counter = 1
            while retired.exists():
                counter += 1
                retired = path.with_suffix(f".csv.superseded{counter}")
            path.rename(retired)
            logger.warning("%s had an older column set (%d fields vs %d); "
                           "moved it to %s and started a new log rather than "
                           "misaligning the rows", path.name, len(header),
                           len(_CSV_FIELDS), retired.name)
            header = None
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            if header is None:
                writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)


def _table_header() -> str:
    """Column header for the ranked table."""
    return (f"{'Symbol':<14}{'Dir':<6}{'Score':>6}{'Entry':>10}{'Stop':>10}"
            f"{'Target':>10}{'Qty':>7}{'Risk':>9}{'Win%':>7}{'RVOL':>6}"
            f"{'RS':>7}{'OI%':>8}{'Cost x':>8}")


def _oi_cell(change: "float | None") -> str:
    """Open-interest change as a cell; 'n/a' when the name has no derivatives."""
    return "n/a" if change is None else f"{change:+.2f}"


def _cost_cell(multiple: float) -> str:
    """Cost multiple as a fixed-width cell, with inf spelled out."""
    return "inf" if math.isinf(multiple) else f"{multiple:.1f}"


def _table_row(setup: setups.Setup) -> str:
    """One ranked-table line for an actionable setup."""
    trade = setup.levels
    reading = setup.readings
    cost = trade.cost_multiple
    return (f"{reading.symbol:<14}{setup.direction:<6}{setup.rank_score:>6.3f}"
            f"{trade.entry:>10.2f}{trade.stop:>10.2f}{trade.target:>10.2f}"
            f"{trade.quantity:>7}{trade.lot_risk:>9,.0f}"
            f"{trade.required_win_rate * 100:>7.1f}{reading.rvol or 0.0:>6.2f}"
            f"{reading.relative_strength or 0.0:>+7.2f}"
            f"{_oi_cell(reading.oi_change_pct):>8}"
            f"{_cost_cell(cost):>8}")


def print_report(ranked: list[setups.Setup], bars: scan_data.BarSet,
                 source: str, benchmark: "float | None", now: datetime,
                 top: int, capital: float = config.SCAN_CAPITAL
                 ) -> list[setups.Setup]:
    """Print the full scan report; return the actionable setups shown."""
    actionable = [s for s in ranked if s.actionable]
    print(f"\n{'=' * 100}")
    print("SECTOR PULSE INTRADAY SCAN")
    print(f"Generated {now.strftime('%Y-%m-%d %H:%M:%S %Z')}  |  universe: {source}")
    nifty = (f"Nifty today {benchmark:+.2f}%" if benchmark is not None
             else "Nifty change unavailable, so relative strength fails closed")
    print(f"Bars for {bars.covered}/{bars.requested} symbols  |  {nifty}")
    print("=" * 100)

    if not actionable:
        print()
        if bars.covered == 0:
            print(f"NO BARS CAME BACK AT ALL ({bars.covered}/{bars.requested}).")
            print()
            print("This is a data failure, not an absence of setups. The "
                  "source may be rate limiting, or the chosen")
            print("date may have had no session. Re-run before drawing any "
                  "conclusion from an empty result.")
            print(f"{chr(10)}{DISCLAIMER}{chr(10)}")
            return []
        premature = [s for s in ranked if not s.readings.range_closed]
        if premature and len(premature) == len(ranked):
            open_h, open_m = config.SCAN_SESSION_OPEN
            closes_at = open_h * 60 + open_m + config.SCAN_OPENING_RANGE_MINUTES
            print(f"NO SIGNAL IS COMPUTABLE YET. The opening range covers the "
                  f"first {config.SCAN_OPENING_RANGE_MINUTES} minutes of the "
                  f"session and does not close until "
                  f"{closes_at // 60:02d}:{closes_at % 60:02d}.")
            print()
            print("Until then the latest bar is itself one of the bars that "
                  "define the range, so the range high and low")
            print("bracket the current price by construction and no breakout "
                  "can be represented. This is arithmetic,")
            print("not a reading of the market: the same empty output would "
                  "appear on the sharpest gap-up morning on")
            print("record. Re-run once the range has closed.")
            print(f"{chr(10)}{DISCLAIMER}{chr(10)}")
            return []
        print("NO SETUP cleared every gate. On most days that is the expected "
              "answer: the direction rule needs VWAP and the opening "
              "range to agree, which they usually do not.")
        _print_near_misses(ranked)
        print(f"\n{DISCLAIMER}\n")
        return []

    print()
    print(f"{len(actionable)} setup(s) cleared every gate, best first:")
    print()
    print(_table_header())
    print("-" * len(_table_header()))
    for setup in actionable[:top]:
        print(_table_row(setup))

    total_risk = sum(s.levels.lot_risk for s in actionable)
    total_notional = sum(s.levels.entry * s.levels.quantity for s in actionable)
    print()
    print(f"AGGREGATE EXPOSURE: taking all {len(actionable)} risks "
          f"{total_risk:,.0f} rupees ({total_risk / capital * 100:.1f}% of "
          f"the {capital:,.0f} assumed) across {total_notional:,.0f} of "
          f"notional,")
    print(f"which is {total_notional / capital:.1f}x that capital. Each row "
          f"is sized independently against its own stop, so the")
    print("per-trade cap does not bound the total and nothing above flags it.")
    best = actionable[0]
    print()
    print(f"TOP SETUP: {best.symbol} {best.direction} - score {best.rank_score:.3f}")
    print()
    print("WHY:")
    for line in textwrap.wrap(setups.explain(best), width=92):
        print(f"  {line}")

    print("\nGate detail for the top setups:")
    for setup in actionable[:min(top, 3)]:
        print(f"\n  {setup.symbol} -> {setup.direction} "
              f"(score {setup.rank_score:.3f})")
        for reason in setup.reasons:
            print(f"    - {reason}")
    print(f"\n{DISCLAIMER}\n")
    return actionable[:top]


def _print_near_misses(ranked: list[setups.Setup], limit: int = 5) -> None:
    """Show symbols that got a direction but failed a later gate."""
    directional = [s for s in ranked
                   if s.direction != setups.NO_SETUP and not s.actionable]
    if not directional:
        return
    print()
    print(f"Closest {min(limit, len(directional))} that had a direction but "
          f"failed a gate:")
    for setup in directional[:limit]:
        failures = [r.replace(" [FAIL]", "") for r in setup.reasons
                    if "[FAIL]" in r]
        print(f"  {setup.symbol:<14}{setup.direction:<6}"
              f"{'; '.join(failures) if failures else 'no failure recorded'}")


def print_options(shown: list[setups.Setup], lot_sizes: dict[str, int],
                  index_symbols: "set[str] | None" = None) -> None:
    """Pick and print a tradeable contract for each shown setup."""
    if not shown:
        return
    print(f"{'-' * 100}")
    print("OPTION CONTRACTS for the setups above")
    print("Direction comes from the equity setup. What is assessed here is "
          "only whether a contract\nis liquid and cheap enough to express it. "
          "Buying only: selling naked options is never proposed.")
    print("-" * 100)
    session = options_chain.open_session()
    for setup in shown:
        symbol = setup.symbol
        lot_size = lot_sizes.get(symbol, 0)
        if lot_size <= 0:
            print(f"\n  {symbol}: no lot size known, so option costs cannot "
                  f"be computed. Skipped.")
            continue
        contracts, spot, expiry = options_chain.fetch_chain(
            symbol, symbol in (index_symbols or set()), session=session)
        if not contracts or spot is None:
            print(f"\n  {symbol}: no option chain returned.")
            continue
        metrics = options_chain.summarise(symbol, contracts, spot, expiry)
        contract, reasons = options_chain.pick_contract(
            contracts, spot, setup.direction, lot_size)
        print(f"\n  {symbol} {setup.direction} | expiry {expiry} | "
              f"spot {spot:.2f} | lot {lot_size}")
        if metrics is not None:
            pcr = metrics.put_call_ratio
            pcr_text = f", PCR {pcr:.2f}" if pcr is not None else ""
            print(f"    chain: ATM {metrics.atm_strike:.0f}, IV call "
                  f"{metrics.atm_call_iv:.1f} / put {metrics.atm_put_iv:.1f}"
                  f"{pcr_text}")
            if metrics.max_pain is not None:
                print(f"    max pain {metrics.max_pain:.0f} (descriptive only, "
                      f"no predictive record)")
        if contract is None:
            print(f"    no tradeable contract: {reasons[0]}")
            for line in reasons[1:4]:
                print(f"      {line}")
            continue
        side = "CALL" if setup.direction == LONG else "PUT"
        print(f"    BUY {contract.strike:.0f} {side} at mid {contract.mid:.2f} "
              f"(bid {contract.bid:.2f} / ask {contract.ask:.2f})")
        for reason in reasons:
            print(f"      - {reason}")


def run(args: argparse.Namespace) -> int:
    """Execute one scan; always returns 0 so a scheduler never flags an outage."""
    actual_now = datetime.now(IST)
    now = actual_now
    replaying = False
    if args.as_of.strip():
        now = parse_as_of(args.as_of)
        # ANY earlier instant is a replay, not merely an earlier date. The
        # previous test compared dates, so replaying 10:00 of today passed
        # straight through and every guard below was skipped: an audit found
        # the open-interest column carrying a snapshot captured at 20:36 and
        # the option chain quoting closing prices, both presented as 10:00
        # readings.
        replaying = now < actual_now
    universe = load_universe(args.discover)
    if universe.gaps:
        for gap in universe.gaps:
            logger.info("Discovery gap: %s", gap)
    symbols, source = resolve_symbols(args, universe)
    if not symbols:
        print("No symbols to scan.")
        return 0
    logger.info("Scanning %d symbols (%s)", len(symbols), source)
    tickers = [instruments.to_ticker(s) for s in symbols] + [config.SCAN_BENCHMARK]
    bars = scan_data.fetch_bars(
        tickers, target=now.date() if now.date() != actual_now.date() else None)
    if args.as_of.strip():
        bars = scan_data.truncate(bars, now)
        source += f", replayed as of {now.strftime('%Y-%m-%d %H:%M')} IST"
    if not bars.intraday:
        print("No intraday bars came back at all - market holiday, or the "
              "data source is down.")
        return 0
    benchmark = scan_data.benchmark_change_pct(bars)
    if benchmark is None:
        logger.warning("No benchmark change: every relative-strength gate "
                       "will fail closed")
    # NSE publishes no open-interest history, so the snapshot on disk
    # describes whenever it was captured. It may only be used when it was
    # captured at or before the instant being scanned. Comparing timestamps
    # rather than dates is the fix: the snapshot used for a 10:00 replay had
    # captured_at 20:36 the same evening, and its "+24.57% (building)" was
    # the entire day's positioning build shown as a 10:00 reading.
    snapshot_is_contemporaneous = universe.captured_at <= now
    fo_state = universe.fo_state if snapshot_is_contemporaneous else None
    if not snapshot_is_contemporaneous:
        logger.warning("Instrument snapshot was captured %s, after the %s "
                       "instant being scanned, so open interest is dropped "
                       "rather than backfilled with later figures",
                       universe.captured_at.strftime("%Y-%m-%d %H:%M"),
                       now.strftime("%Y-%m-%d %H:%M"))
    evaluated = build_setups(symbols, bars, benchmark, now, args.capital,
                             args.risk_pct, fo_state=fo_state)
    ranked = setups.rank(evaluated)
    shown = print_report(ranked, bars, source, benchmark, now, args.top,
                         capital=args.capital)
    append_log([_row(s, now, replayed=replaying) for s in shown])
    if args.options and shown:
        if replaying:
            print()
            print("-" * 100)
            print("OPTION CONTRACTS cannot be shown for a replayed instant.")
            print()
            print("NSE serves only a live option chain and publishes no "
                  "archive, so a chain fetched now describes now. Pinning it")
            print("onto an earlier signal changes the strike, the premium and "
                  "the spread: an audit of a 10:00 replay found every printed")
            print("spot equal to that day's CLOSE, ten for ten, and one "
                  "underlying whose at-the-money strike differed by a full")
            print("step between the two instants. Stored chain snapshots make "
                  "this answerable - capture them with the dashboard's sync")
            print("button or discover.py, and replays after the first capture "
                  "can read a real chain.")
        else:
            lot_sizes = {inst.symbol: inst.lot_size
                         for inst in universe.fo_underlyings if inst.lot_size}
            index_symbols = {inst.symbol for inst in universe.fo_indices}
            print_options(shown, lot_sizes, index_symbols)
    return 0


def main(argv: "list[str] | None" = None) -> int:
    """Configure logging and run one scan."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
