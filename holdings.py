"""Loading of holdings exports and target-weight files."""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from portfolio_models import Holding

logger = logging.getLogger(__name__)

TARGET_SUM_TOLERANCE = 0.5   # percentage points

# Header candidates in priority order. Zerodha Console and Kite exports differ,
# and both have changed shape over time, so match generously.
_SYMBOL_HEADERS = ("symbol", "instrument", "tradingsymbol", "scrip", "name")
_QUANTITY_HEADERS = ("quantity available", "qty", "quantity", "net quantity",
                     "holdings quantity", "shares")
_AVG_COST_HEADERS = ("average price", "avg cost", "average cost price",
                     "avg price", "buy average", "buy avg", "average")
_PRICE_HEADERS = ("ltp", "last price", "last traded price", "cur price",
                  "market price", "previous closing price", "closing price",
                  "close price")

# Zerodha ships several quantity columns; only the tradeable one is wanted.
_QUANTITY_EXCLUDE = ("discrepant", "pledged", "long term", "collateral",
                     "t1", "authorised", "authorized")
_PRICE_EXCLUDE = ("change", "chg", "average", "avg")


class HoldingsError(ValueError):
    """Raised when a holdings file cannot be understood."""


class TargetsError(ValueError):
    """Raised when a targets file is missing, malformed, or does not sum to 100."""


def _normalise(header: str) -> str:
    """Lowercase a CSV header and strip punctuation so variants compare equal."""
    cleaned = header.strip().lower().replace(".", " ").replace("_", " ")
    cleaned = re.sub(r"[^a-z0-9 %]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _pick_column(headers: list[str], candidates: tuple[str, ...],
                 exclude: tuple[str, ...] = ()) -> "str | None":
    """Return the original header best matching candidates, or None."""
    normalised = {_normalise(h): h for h in headers if h}
    allowed = {norm: original for norm, original in normalised.items()
               if not any(bad in norm for bad in exclude)}
    for candidate in candidates:
        if candidate in allowed:
            return allowed[candidate]
    for candidate in candidates:
        for norm, original in allowed.items():
            if candidate in norm:
                return original
    return None


def _to_float(raw: "str | None") -> float:
    """Parse a possibly comma-grouped, currency-prefixed number; blank is 0.0."""
    if raw is None:
        return 0.0
    text = str(raw).strip()
    if not text or text in {"-", "--", "NA", "N/A", "nan"}:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return 0.0
    try:
        value = float(text)
    except ValueError as exc:
        raise HoldingsError(f"Could not parse number: {raw!r}") from exc
    return -value if negative else value


def load_holdings(path: "str | Path") -> list[Holding]:
    """Read a Zerodha-style holdings CSV into Holding records."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise HoldingsError(f"Holdings file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise HoldingsError(f"Holdings file has no header row: {csv_path}")
        symbol_col = _pick_column(headers, _SYMBOL_HEADERS)
        quantity_col = _pick_column(headers, _QUANTITY_HEADERS, _QUANTITY_EXCLUDE)
        avg_col = _pick_column(headers, _AVG_COST_HEADERS)
        price_col = _pick_column(headers, _PRICE_HEADERS, _PRICE_EXCLUDE)
        if symbol_col is None or quantity_col is None:
            raise HoldingsError(
                f"Could not find symbol and quantity columns in {csv_path}. "
                f"Saw headers: {headers}")
        if price_col is None:
            logger.warning("No last-price column in %s; prices must be fetched",
                           csv_path)
        holdings = _rows_to_holdings(reader, symbol_col, quantity_col,
                                     avg_col, price_col)
    if not holdings:
        raise HoldingsError(f"No usable holdings rows in {csv_path}")
    return holdings


def _rows_to_holdings(reader: "csv.DictReader", symbol_col: str,
                      quantity_col: str, avg_col: "str | None",
                      price_col: "str | None") -> list[Holding]:
    """Convert CSV rows into Holdings, skipping blank and zero-quantity rows."""
    holdings: list[Holding] = []
    seen: set[str] = set()
    for row in reader:
        symbol = (row.get(symbol_col) or "").strip().upper()
        if not symbol:
            continue
        quantity = _to_float(row.get(quantity_col))
        if quantity <= 0.0:
            logger.warning("Skipping %s: quantity is %s", symbol, quantity)
            continue
        if symbol in seen:
            raise HoldingsError(f"Duplicate symbol in holdings file: {symbol}")
        seen.add(symbol)
        avg_cost = _to_float(row.get(avg_col)) if avg_col else 0.0
        last_price = _to_float(row.get(price_col)) if price_col else 0.0
        holdings.append(Holding(symbol=symbol, quantity=quantity,
                                avg_cost=avg_cost, last_price=last_price))
    return holdings


def load_targets(path: "str | Path") -> dict[str, float]:
    """Read a flat 'SYMBOL: weight' targets file and validate it sums to 100."""
    targets_path = Path(path)
    if not targets_path.exists():
        raise TargetsError(f"Targets file not found: {targets_path}")
    targets: dict[str, float] = {}
    with targets_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            entry = line.split("#", 1)[0].strip()
            if not entry:
                continue
            if ":" not in entry:
                raise TargetsError(
                    f"{targets_path}:{line_no}: expected 'SYMBOL: weight', "
                    f"got {line.strip()!r}")
            symbol, _, weight_text = entry.partition(":")
            symbol = symbol.strip().strip("\"'").upper()
            if symbol in targets:
                raise TargetsError(f"{targets_path}:{line_no}: duplicate {symbol}")
            try:
                weight = float(weight_text.strip())
            except ValueError as exc:
                raise TargetsError(
                    f"{targets_path}:{line_no}: bad weight for {symbol}: "
                    f"{weight_text.strip()!r}") from exc
            if not 0.0 <= weight <= 100.0:
                raise TargetsError(
                    f"{targets_path}:{line_no}: {symbol} weight {weight} "
                    f"is outside 0-100")
            targets[symbol] = weight
    if not targets:
        raise TargetsError(f"No target weights defined in {targets_path}")
    total = sum(targets.values())
    if abs(total - 100.0) > TARGET_SUM_TOLERANCE:
        raise TargetsError(
            f"Target weights in {targets_path} sum to {total:.2f}, not 100. "
            f"Fix the weights before planning a contribution.")
    return targets
