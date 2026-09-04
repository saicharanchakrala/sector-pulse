"""Build holdings.csv from one or more Zerodha tradebook exports.

The tradebook is the authoritative record of what you bought and sold, so net
quantity and average cost are derived from it rather than hand-maintained.
Average cost uses the running weighted-average convention: a sell removes
quantity at the prevailing average and leaves that average unchanged.

Corporate actions (splits, bonuses, consolidations) never appear in a
tradebook, so a replay cannot see them. If one has affected a holding, the
quantity and average cost for that symbol will be wrong and nothing here can
detect it; check such symbols against Kite by hand.
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

import config
from holdings import HoldingsError, parse_number

logger = logging.getLogger(__name__)

HOLDINGS_HEADER = ("Symbol", "ISIN", "Quantity Available", "Average Price", "LTP")
REQUIRED_COLUMNS = ("symbol", "trade_date", "trade_type", "quantity", "price")
# Only cash equity produces a holding. A blank segment is tolerated so that
# hand-made test files stay usable.
EQUITY_SEGMENTS = {"EQ", ""}
# Quantities are integral in practice, so anything at this scale is float noise.
EPSILON = 1e-9


@dataclass(frozen=True)
class Trade:
    """One executed trade from a tradebook export."""

    symbol: str
    isin: str
    stamp: datetime
    trade_date: str
    exchange: str
    trade_type: str
    quantity: float
    price: float
    trade_id: str
    order_id: str
    timed: bool

    @property
    def sort_key(self) -> tuple[datetime, int, str]:
        """Total, deterministic ordering: time, then buys before sells."""
        return (self.stamp, 0 if self.trade_type == "buy" else 1, self.trade_id)

    @property
    def identity(self) -> tuple:
        """Key that de-duplicates overlapping exports.

        trade_id is only unique per exchange per day, so it cannot be used
        alone. Rows without one fall back to their full content rather than
        bypassing de-duplication.
        """
        if self.trade_id:
            return (self.trade_id, self.exchange, self.symbol, self.trade_date)
        return (self.symbol, self.trade_date, self.stamp, self.trade_type,
                self.quantity, self.price, self.order_id)


@dataclass
class Position:
    """A running position built up from trades."""

    symbol: str
    isin: str = ""
    quantity: float = 0.0
    cost_pool: float = 0.0
    sold: float = 0.0
    realised: float = 0.0
    oversold: float = 0.0

    @property
    def is_open(self) -> bool:
        """True when a real quantity remains, ignoring float noise."""
        return self.quantity > EPSILON

    @property
    def avg_cost(self) -> float:
        """Weighted-average cost per unit of the remaining quantity."""
        return self.cost_pool / self.quantity if self.is_open else 0.0


def _parse_stamp(trade_date: str, executed_at: str) -> tuple[datetime, bool]:
    """Resolve a trade's timestamp, flagging rows that carried no time.

    Dates and times are parsed rather than compared as strings: a non-ISO
    date sorts by day-of-month and scrambles the whole replay.
    """
    try:
        day = date.fromisoformat(trade_date)
    except ValueError as exc:
        raise HoldingsError(
            f"Unrecognised trade_date {trade_date!r}; expected YYYY-MM-DD. "
            f"Re-export the tradebook without reformatting it.") from exc
    text = (executed_at or "").strip()
    if not text:
        # No time means the row cannot be placed within its day. Put it last
        # so a same-day buy is never replayed after the sell it funded.
        return datetime.combine(day, time(23, 59, 59)), False
    try:
        return datetime.fromisoformat(text), True
    except ValueError:
        pass
    try:
        return datetime.combine(day, time.fromisoformat(text)), True
    except ValueError as exc:
        raise HoldingsError(
            f"Unrecognised order_execution_time {text!r} for {trade_date}") from exc


def _row_to_trade(row: dict) -> "Trade | None":
    """Convert one tradebook row into a Trade, or None if unusable."""
    symbol = (row.get("symbol") or "").strip().upper()
    trade_type = (row.get("trade_type") or "").strip().lower()
    segment = (row.get("segment") or "").strip().upper()
    quantity = parse_number(row.get("quantity"))
    price = parse_number(row.get("price"))
    if not symbol:
        return None
    if segment not in EQUITY_SEGMENTS:
        logger.warning("Ignoring %s: segment %s is not cash equity", symbol, segment)
        return None
    if trade_type not in {"buy", "sell"}:
        logger.warning("Ignoring %s trade with unknown type %r", symbol, trade_type)
        return None
    if quantity <= 0.0 or price <= 0.0:
        # A zero price would silently halve the average cost.
        logger.warning("Ignoring %s row: quantity=%s price=%s",
                       symbol, quantity, price)
        return None
    trade_date = (row.get("trade_date") or "").strip()
    stamp, timed = _parse_stamp(trade_date, row.get("order_execution_time") or "")
    return Trade(symbol=symbol, isin=(row.get("isin") or "").strip(),
                 stamp=stamp, trade_date=trade_date,
                 exchange=(row.get("exchange") or "").strip().upper(),
                 trade_type=trade_type, quantity=quantity, price=price,
                 trade_id=(row.get("trade_id") or "").strip(),
                 order_id=(row.get("order_id") or "").strip(), timed=timed)


def read_trades(paths: list[str]) -> list[Trade]:
    """Read every tradebook file, de-duplicating repeated trades."""
    trades: dict[tuple, Trade] = {}
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise HoldingsError(f"Tradebook not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = {(h or "").strip().lower() for h in (reader.fieldnames or [])}
            missing = [c for c in REQUIRED_COLUMNS if c not in headers]
            if missing:
                raise HoldingsError(
                    f"{csv_path} is not a Zerodha tradebook: missing columns "
                    f"{missing}. Saw {sorted(headers)}")
            for row in reader:
                trade = _row_to_trade(row)
                if trade is not None:
                    _add_trade(trades, trade, csv_path)
    if not trades:
        raise HoldingsError("No usable trades found in the given tradebooks")
    untimed = sum(1 for trade in trades.values() if not trade.timed)
    if untimed:
        logger.warning("%d trade(s) carry no execution time and were placed at "
                       "end of day; same-day ordering may be approximate", untimed)
    return sorted(trades.values(), key=lambda trade: trade.sort_key)


def _add_trade(trades: dict, trade: Trade, source: Path) -> None:
    """Insert a trade, refusing to overwrite a genuinely different one."""
    prior = trades.get(trade.identity)
    if prior is not None and (prior.quantity, prior.price,
                              prior.trade_type) != (trade.quantity, trade.price,
                                                    trade.trade_type):
        raise HoldingsError(
            f"{source}: two different trades share the identity "
            f"{trade.identity}: {prior.quantity}@{prior.price} "
            f"({prior.trade_type}) vs {trade.quantity}@{trade.price} "
            f"({trade.trade_type}). Refusing to guess which is real.")
    trades[trade.identity] = trade


def build_positions(trades: list[Trade]) -> dict[str, Position]:
    """Replay trades chronologically into per-symbol positions."""
    positions: dict[str, Position] = {}
    for trade in trades:
        position = positions.setdefault(trade.symbol, Position(trade.symbol))
        if trade.isin and not position.isin:
            position.isin = trade.isin
        if trade.trade_type == "buy":
            position.quantity += trade.quantity
            position.cost_pool += trade.quantity * trade.price
            continue
        sold = min(trade.quantity, position.quantity)
        if trade.quantity > position.quantity + EPSILON:
            # Selling more than the tradebooks account for means the opening
            # position predates these files.
            position.oversold += trade.quantity - position.quantity
        if sold > 0:
            average = position.avg_cost
            position.realised += sold * (trade.price - average)
            position.cost_pool -= sold * average
            position.quantity -= sold
        if abs(position.quantity) <= EPSILON:      # clear float residue
            position.quantity = 0.0
            position.cost_pool = 0.0
        position.sold += trade.quantity
    return positions


def open_positions(positions: dict[str, Position]) -> list[Position]:
    """Positions still held, in symbol order."""
    return [position for _, position in sorted(positions.items())
            if position.is_open]


def write_holdings(path: "str | Path", positions: dict[str, Position],
                   prices: dict[str, float]) -> list[Position]:
    """Write open positions to a holdings CSV the planner can load."""
    held = open_positions(positions)
    for position in held:
        if abs(position.quantity - round(position.quantity)) > EPSILON:
            raise HoldingsError(
                f"{position.symbol}: fractional quantity "
                f"{position.quantity}. ETF units are whole; refusing to round.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HOLDINGS_HEADER)
        for position in held:
            price = prices.get(position.symbol, 0.0)
            writer.writerow([position.symbol, position.isin,
                             f"{round(position.quantity):d}",
                             f"{position.avg_cost:.2f}",
                             f"{price:.2f}" if price > 0 else ""])
    return held


def _print_table(held: list[Position], prices: dict[str, float]) -> None:
    """Print the per-symbol holdings reconciliation."""
    header = (f"{'SYMBOL':<12}{'QTY':>7}{'AVG COST':>10}{'INVESTED':>12}"
              f"{'LTP':>10}{'VALUE':>12}{'P/L':>9}")
    print()
    print(header)
    print("-" * len(header))
    invested_total = value_total = 0.0
    priced = False
    for position in held:
        invested = position.quantity * position.avg_cost
        price = prices.get(position.symbol, 0.0)
        invested_total += invested
        left = (f"{position.symbol:<12}{position.quantity:>7,.0f}"
                f"{position.avg_cost:>10,.2f}{invested:>12,.0f}")
        if price > 0:
            priced = True
            value = position.quantity * price
            value_total += value
            pnl = (value / invested - 1.0) * 100.0 if invested else 0.0
            print(f"{left}{price:>10,.2f}{value:>12,.0f}{pnl:>+8.1f}%")
        else:
            print(f"{left}{'-':>10}{'-':>12}{'-':>9}")
    print("-" * len(header))
    total_value = f"{value_total:>12,.0f}" if priced else f"{'-':>12}"
    print(f"{'TOTAL':<12}{'':>7}{'':>10}{invested_total:>12,.0f}"
          f"{'':>10}{total_value}")


def _print_realised(positions: dict[str, Position]) -> None:
    """Report realised profit on every sell, whether the position survives or not."""
    sellers = [position for _, position in sorted(positions.items())
               if position.sold > 0 and not position.oversold]
    if not sellers:
        return
    print("\nRealised profit/loss on sells in these tradebooks:")
    for position in sellers:
        state = (f"{position.quantity:,.0f} units still held" if position.is_open
                 else "flat")
        print(f"  {position.symbol:<12}{position.realised:>+12,.0f}   "
              f"{position.sold:,.0f} units sold, {state}")
    print(f"  {'TOTAL':<12}{sum(p.realised for p in sellers):>+12,.0f}")
    print("  A flat position is absent from holdings.csv only because you hold "
          "none of it.\n  Add it to targets.yaml if you want contributions to "
          "buy it again.")


def _print_incomplete(positions: dict[str, Position]) -> None:
    """Warn about symbols whose opening position predates these tradebooks."""
    incomplete = {position.symbol: position.oversold
                  for _, position in sorted(positions.items())
                  if position.oversold}
    if not incomplete:
        return
    print(f"\nWARNING: these symbols sold more than the supplied tradebooks "
          f"account for:\n  {incomplete}\nThe opening position predates these "
          f"files, so their quantity and average cost are\nwrong. Add the "
          f"earlier tradebook, or pass --force to write anyway.")


def _parse_args() -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(
        description="Build holdings.csv from Zerodha tradebook exports.")
    parser.add_argument("tradebooks", nargs="+",
                        help="one or more tradebook CSV files")
    parser.add_argument("--out", default=str(config.HOLDINGS_CSV),
                        help="holdings CSV to write (default: %(default)s)")
    parser.add_argument("--no-prices", action="store_true",
                        help="skip the live price lookup and leave LTP blank")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the summary without writing the file")
    parser.add_argument("--force", action="store_true",
                        help="write even when the history looks incomplete")
    return parser.parse_args()


def _refusal(positions: dict[str, Position], out: Path,
             force: bool) -> "str | None":
    """Reason to refuse writing over an existing holdings file, if any."""
    if force:
        return None
    if any(position.oversold for position in positions.values()):
        return ("Refusing to write: the tradebook history is incomplete "
                "(see the warning above).")
    if not open_positions(positions) and out.exists() and out.stat().st_size > 0:
        return (f"Refusing to write: this would replace {out.name} with an "
                f"empty file.\nDid you pass only a sells-only tradebook?")
    return None


def main() -> int:
    """Entry point. Returns 2 on unusable input or a refused write, else 0."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        trades = read_trades(args.tradebooks)
    except HoldingsError as exc:
        print(f"ERROR: {exc}")
        return 2
    positions = build_positions(trades)
    prices: dict[str, float] = {}
    if not args.no_prices:
        from plan_investment import fetch_prices
        prices = fetch_prices([p.symbol for p in open_positions(positions)])
    print(f"\nParsed {len(trades)} trades, "
          f"{trades[0].trade_date} to {trades[-1].trade_date}")
    _print_table(open_positions(positions), prices)
    _print_realised(positions)
    _print_incomplete(positions)
    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0
    refusal = _refusal(positions, Path(args.out), args.force)
    if refusal:
        print(f"\n{refusal}")
        return 2
    try:
        written = write_holdings(args.out, positions, prices)
    except (HoldingsError, OSError) as exc:
        print(f"\nERROR: {exc}")
        return 2
    print(f"\nWrote {len(written)} positions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
