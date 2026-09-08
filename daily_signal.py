"""Daily end-of-day trade signal CLI: python daily_signal.py [--market IN].

Runs news -> momentum -> intraday -> decision for one market, prints a
report, appends per-sector rows to signals.csv, and writes last_signal.json
for the dashboard. Always exits 0 so schedulers never flag a data outage.
Educational tool - not financial advice; it never places orders.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import textwrap
from dataclasses import asdict
from datetime import datetime

import config
import decision
import market_data
import news_archive
import news_fetcher
from decision import TradeSignal, decide, top_pick
from intraday import get_intraday_snapshots
from profiles import MarketProfile, get_profile

logger = logging.getLogger("daily_signal")

_CSV_FIELDS = [
    "run_date", "run_time", "market", "sector", "etf", "action", "top_pick",
    "rank_score", "news_today", "news_count_today", "day_change_pct",
    "last_hour_change_pct", "momentum", "illiquid",
]

DISCLAIMER = (
    "Educational decision support only - NOT financial advice. Intraday data "
    "may be delayed; this rule is unvalidated. No orders are placed."
)


def _csv_row(signal: TradeSignal, now: datetime, market: str,
             is_top: bool) -> dict[str, object]:
    """Flatten one TradeSignal into a signals.csv row."""
    snap = signal.intraday
    return {
        "run_date": now.strftime("%Y-%m-%d"),
        "run_time": now.strftime("%H:%M:%S"),
        "market": market,
        "sector": signal.sector,
        "etf": signal.etf or "",
        "action": signal.action,
        "top_pick": is_top,
        "rank_score": round(signal.rank_score, 4),
        "news_today": round(signal.news_today, 4),
        "news_count_today": signal.news_count_today,
        "day_change_pct": round(snap.day_change_pct, 3) if snap else "",
        "last_hour_change_pct": round(snap.last_hour_change_pct, 3) if snap else "",
        "momentum": round(signal.momentum, 4),
        "illiquid": signal.illiquid,
    }


def append_csv_rows(rows: list[dict[str, object]]) -> None:
    """Append rows to signals.csv, writing the header when the file is new."""
    path = config.SIGNALS_CSV
    try:
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)


def signal_to_dict(signal: TradeSignal) -> dict[str, object]:
    """Serialize a TradeSignal (and nested snapshot) to JSON-safe plain dicts."""
    payload = asdict(signal)
    if payload["intraday"] is not None:
        payload["intraday"]["asof"] = signal.intraday.asof.isoformat()
    payload["explanation"] = decision.explain(signal)
    return payload


def write_last_signal(signals: list[TradeSignal], market: str,
                      now: datetime) -> None:
    """Write last_signal.json for the Streamlit app to display."""
    pick = top_pick(signals)
    payload = {
        "generated_at": now.isoformat(),
        "market": market,
        "top_pick": signal_to_dict(pick) if pick is not None else None,
        "signals": [signal_to_dict(s) for s in signals],
    }
    try:
        with config.LAST_SIGNAL_JSON.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except OSError as exc:
        logger.warning("Could not write %s: %s", config.LAST_SIGNAL_JSON, exc)


def _signal_line(signal: TradeSignal) -> str:
    """Format one '<ACTION> ETF (sector) - rank' fragment for the headline."""
    return (f"{signal.action} {signal.etf} ({signal.sector}) - "
            f"rank {signal.rank_score:+.3f}")


def print_top_signals(signals: list[TradeSignal]) -> None:
    """Print the headline buy, the reasoning behind it, and any runners-up."""
    pick = top_pick(signals)
    if pick is None:
        print()
        print("NO BUY today - no sector passed every check.")
        weakest = [s for s in signals if decision.is_declining(s)]
        if weakest:
            print("Under real pressure today: "
                  + ", ".join(f"{s.sector} ({s.etf})" for s in weakest))
        return
    print()
    print(f"TOP SIGNAL: {_signal_line(pick)}")
    print()
    print("WHY:")
    for line in textwrap.wrap(decision.explain(pick), width=74):
        print(f"  {line}")
    others = [s for s in signals
              if s.action == decision.ACTION_BUY and s is not pick]
    if others:
        print()
        print("Also cleared every check: "
              + ", ".join(f"{s.etf} ({s.sector})" for s in others))


def _table_row(signal: TradeSignal) -> str:
    """One per-sector line of the report table."""
    snap = signal.intraday
    day = f"{snap.day_change_pct:+.2f}" if snap else "n/a"
    hour = f"{snap.last_hour_change_pct:+.2f}" if snap else "n/a"
    illiq = "YES" if signal.illiquid else ""
    return (f"{signal.sector:<22}{signal.etf or '-':<15}{signal.action:<11}"
            f"{signal.rank_score:>+7.3f}{signal.news_today:>+7.2f}"
            f"{signal.news_count_today:>4}{day:>8}{hour:>9}"
            f"{signal.momentum:>+7.2f}{illiq:>7}")


def print_report(signals: list[TradeSignal], profile: MarketProfile,
                 now: datetime) -> None:
    """Print the human-readable end-of-day signal report."""
    print(f"\n{'=' * 78}")
    print(f"SECTOR PULSE DAILY SIGNAL - {profile.label} ({profile.key})")
    print(f"Generated {now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}")
    print("=" * 78)
    print_top_signals(signals)
    header = (f"{'Sector':<22}{'ETF':<15}{'Action':<11}{'Rank':>7}{'News':>7}"
              f"{'N':>4}{'Day%':>8}{'LastHr%':>9}{'Mom':>7}{'Illiq':>7}")
    print()
    print(header)
    print("-" * len(header))
    # A sector with no tradeable ETF can never be actionable, so ranking it
    # among the rest makes a structural impossibility look like a near miss.
    actionable = [s for s in signals if s.etf is not None]
    untradeable = [s for s in signals if s.etf is None]
    for signal in actionable:
        print(_table_row(signal))
    if untradeable:
        print()
        print("Not tradeable - no ETF in this profile, so never actionable:")
        for signal in untradeable:
            print(f"  {_table_row(signal)}")
    pick = top_pick(signals)
    detail: list[TradeSignal] = []
    ranked = [s for s in signals if s.etf is not None] or signals
    for candidate in ([pick] if pick is not None else []) + ranked[:2] + ranked[-2:]:
        if candidate not in detail:
            detail.append(candidate)
    print("\nGate detail - top pick plus top 2 and bottom 2 ranked sectors:")
    for signal in detail:
        print(f"\n  {signal.sector} ({signal.etf or 'no ETF'}) -> {signal.action}")
        for reason in signal.reasons:
            print(f"    - {reason}")
    print(f"\n{DISCLAIMER}\n")


def run(market: str) -> int:
    """Execute the full daily-signal pipeline for one market; always returns 0."""
    profile = get_profile(market)
    now = datetime.now().astimezone()
    logger.info("Fetching news, momentum and intraday data for %s", profile.key)
    items = news_fetcher.fetch_all_news(config.SIGNAL_NEWS_HOURS * 2, profile)
    # Archive before anything else: an unarchived headline is gone in 48h,
    # and with it any chance of ever backtesting the news gate.
    written = news_archive.archive_news(items)
    logger.info("Archived %d new headline(s) of %d fetched", written, len(items))
    momentum = market_data.get_sector_momentum(profile)
    snapshots = get_intraday_snapshots(list(profile.trade_etfs.values()))
    if not snapshots:
        print("Market closed today - no signal.")
        append_csv_rows([{
            "run_date": now.strftime("%Y-%m-%d"),
            "run_time": now.strftime("%H:%M:%S"),
            "market": profile.key,
            "sector": "",
            "etf": "",
            "action": "MARKET_CLOSED",
            "top_pick": False,
            "rank_score": "",
            "news_today": "",
            "news_count_today": "",
            "day_change_pct": "",
            "last_hour_change_pct": "",
            "momentum": "",
            "illiquid": "",
        }])
        return 0
    signals = decide(items, momentum, snapshots, profile)
    print_report(signals, profile, now)
    pick = top_pick(signals)
    append_csv_rows([
        _csv_row(signal, now, profile.key, signal is pick) for signal in signals
    ])
    write_last_signal(signals, profile.key, now)
    return 0


def main() -> int:
    """Parse CLI args, configure logging, and run the pipeline."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Sector Pulse daily trade signal")
    parser.add_argument("--market", default="IN", help="Market profile key (default IN)")
    args = parser.parse_args()
    return run(args.market)


if __name__ == "__main__":
    sys.exit(main())
