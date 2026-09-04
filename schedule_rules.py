"""Contribution cadence: decide whether today is a buying day at all."""
from __future__ import annotations

import csv
import logging
from datetime import date, timedelta
from pathlib import Path

import config
from allocator import max_abs_drift
from portfolio_models import DriftRow, Plan

logger = logging.getLogger(__name__)

CONTRIBUTIONS_HEADER = ("date", "action", "symbol", "units", "last_price",
                        "amount", "portfolio_value", "mode")


def load_last_contribution(path: "str | Path") -> "date | None":
    """Return the date of the most recent recorded contribution, or None."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    latest: "date | None" = None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("date") or "").strip()
            if not raw:
                continue
            try:
                parsed = date.fromisoformat(raw)
            except ValueError:
                logger.warning("Ignoring unparseable contribution date %r", raw)
                continue
            if latest is None or parsed > latest:
                latest = parsed
    return latest


def record_contribution(path: "str | Path", plan: Plan) -> None:
    """Append one row per order to the contributions log, creating it if needed."""
    csv_path = Path(path)
    is_new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(CONTRIBUTIONS_HEADER)
        if not plan.orders:
            writer.writerow([plan.asof.isoformat(), plan.action, "", 0, "",
                             0.0, round(plan.portfolio_value, 2), plan.mode])
            return
        for order in plan.orders:
            writer.writerow([plan.asof.isoformat(), plan.action, order.symbol,
                             order.units, round(order.last_price, 2),
                             round(order.amount, 2),
                             round(plan.portfolio_value, 2), plan.mode])


def evaluate_cadence(
    today: date,
    last_contribution: "date | None",
    rows: list[DriftRow],
    interval_days: int = config.CONTRIBUTION_INTERVAL_DAYS,
    band_pp: float = config.REBALANCE_BAND_PP,
    min_days_between_buys: int = config.MIN_DAYS_BETWEEN_BUYS,
) -> tuple[str, list[str], "date | None"]:
    """Decide INVEST or HOLD for today and explain every gate that was checked."""
    drift = max_abs_drift(rows)
    if last_contribution is None:
        return "INVEST", ["PASS: no contribution on record, this is the first run"], None
    days_since = (today - last_contribution).days
    next_due = last_contribution + timedelta(days=interval_days)
    reasons: list[str] = [
        f"last contribution {last_contribution.isoformat()} "
        f"({days_since} day(s) ago)"
    ]
    if days_since < 0:
        reasons.append("FAIL: last contribution is dated in the future, "
                       "check contributions.csv")
        return "HOLD", reasons, next_due
    if days_since >= interval_days:
        reasons.append(f"PASS: scheduled contribution due "
                       f"({days_since} >= {interval_days} days)")
        return "INVEST", reasons, next_due
    reasons.append(f"FAIL: scheduled contribution not due until "
                   f"{next_due.isoformat()} ({days_since} < {interval_days} days)")
    if drift < band_pp:
        reasons.append(f"FAIL: largest drift {drift:.2f}pp is inside the "
                       f"{band_pp:.2f}pp rebalance band")
        return "HOLD", reasons, next_due
    reasons.append(f"PASS: largest drift {drift:.2f}pp breaches the "
                   f"{band_pp:.2f}pp rebalance band")
    if days_since < min_days_between_buys:
        reasons.append(f"FAIL: only {days_since} day(s) since the last buy, "
                       f"minimum spacing is {min_days_between_buys}")
        return "HOLD", reasons, next_due
    reasons.append(f"PASS: {days_since} day(s) since the last buy meets the "
                   f"{min_days_between_buys}-day minimum spacing")
    return "INVEST", reasons, next_due
