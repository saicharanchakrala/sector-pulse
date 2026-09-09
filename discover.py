"""Instrument discovery CLI: python discover.py [--expiries] [--show N].

Fetches every tradeable NSE instrument as it stands at this moment and
persists it with a timestamp, so a later scan can be driven entirely from
what was actually listed rather than from a hand-maintained constant.

    python discover.py                 # equities, F&O master, F&O state
    python discover.py --expiries      # also every option expiry per name
    python discover.py --show 20       # print the 20 largest OI increases

What comes back, and what does not, is printed rather than assumed. In
particular there is no bulk per-contract stock-futures feed available from
NSE right now, so futures activity is reported per underlying as turnover
and open interest instead of contract by contract.

Educational tooling. It reads public market data and places no orders.
"""
from __future__ import annotations

import argparse
import logging
import sys

import instruments


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Discover every listed NSE instrument at this moment")
    parser.add_argument("--expiries", action="store_true",
                        help="also fetch option expiries per underlying "
                             "(about 216 paced requests, roughly a minute)")
    parser.add_argument("--show", type=int, default=10,
                        help="how many underlyings to detail (default 10)")
    args = parser.parse_args(argv)
    if args.show < 0:
        parser.error(f"--show cannot be negative, got {args.show}")
    return args


def _print_counts(universe: instruments.Universe) -> None:
    """Print what the discovery pass actually retrieved."""
    print(f"\n{'=' * 78}")
    print("NSE INSTRUMENT DISCOVERY")
    print(f"Captured {universe.captured_at.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}")
    print("=" * 78)
    print()
    labels = {
        "equities": "listed equities (EQUITY_L master)",
        "fo_indices": "F&O indices",
        "fo_stocks": "F&O single stocks",
        "fo_state": "underlyings with live OI/turnover",
        "futures_contracts": "futures contracts (per-contract feed)",
        "option_expiries": "underlyings with expiries fetched",
    }
    for key, value in universe.counts().items():
        print(f"  {labels.get(key, key):<40}{value:>8,}")
    lots = [i for i in universe.fo_stocks if i.lot_size]
    print(f"  {'F&O stocks with a known lot size':<40}{len(lots):>8,}")


def _print_gaps(universe: instruments.Universe) -> None:
    """Print anything that could not be retrieved, plainly."""
    if not universe.gaps:
        return
    print()
    print("Not available in this run:")
    for gap in universe.gaps:
        print(f"  - {gap}")


def _print_activity(universe: instruments.Universe, limit: int) -> None:
    """Print the largest open-interest increases with their turnover mix."""
    states = [s for s in universe.fo_state.values()
              if s.oi_change_pct is not None and s.total_turnover_lakh > 0]
    if not states or limit <= 0:
        return
    broken = [s.symbol for s in universe.fo_state.values()
              if not s.totals_reconcile]
    print()
    print(f"Largest open-interest increases ({min(limit, len(states))} of "
          f"{len(states)} underlyings):")
    header = (f"  {'Symbol':<14}{'Spot':>11}{'OI chg %':>10}{'OI chg':>12}"
              f"{'Fut share':>11}{'Deriv turnover':>17}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    ranked = sorted(states, key=lambda s: s.oi_change_pct, reverse=True)
    for state in ranked[:limit]:
        share = state.futures_share
        share_text = f"{share * 100:.1f}%" if share is not None else "n/a"
        print(f"  {state.symbol:<14}{state.spot:>11,.1f}"
              f"{state.oi_change_pct:>+10.2f}{state.oi_change:>12,.0f}"
              f"{share_text:>11}"
              f"{state.derivatives_turnover_rupees / 1e7:>14,.0f} cr")
    print()
    print("  Open interest rising with price is conventionally read as fresh")
    print("  positioning and falling OI as unwinding. That is a convention,")
    print("  not a measured edge, and nothing here is backtested.")
    if broken:
        print()
        print(f"  WARNING: turnover columns stopped reconciling for "
              f"{len(broken)} underlying(s) ({', '.join(broken[:5])}). The "
              f"futures/options split may have changed units upstream and "
              f"should not be trusted until checked.")


def run(args: argparse.Namespace) -> int:
    """Execute one discovery pass; always returns 0."""
    logging.getLogger(__name__).info("Discovering instruments from NSE")
    universe = instruments.discover(with_expiries=args.expiries)
    _print_counts(universe)
    _print_gaps(universe)
    _print_activity(universe, args.show)
    path = instruments.save(universe)
    print()
    if path is None:
        print("Could not persist the snapshot; the scan will fall back to "
              "whatever it finds on disk.")
    else:
        print(f"Saved to {path}")
        print("Run the scanner against it with: "
              "python scan_intraday.py --universe discovered")
    print()
    return 0


def main(argv: "list[str] | None" = None) -> int:
    """Configure logging and run discovery."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
