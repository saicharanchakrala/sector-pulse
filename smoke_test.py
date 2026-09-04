"""End-to-end smoke test for Sector Pulse (news -> momentum -> analyzer) per profile."""
from __future__ import annotations

import logging
import sys
from collections import Counter
from datetime import datetime, timezone

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

import analyzer  # noqa: E402
import claude_insights  # noqa: E402
import config  # noqa: E402
import decision  # noqa: E402
import market_data  # noqa: E402
import models  # noqa: E402
import news_fetcher  # noqa: E402
import profiles  # noqa: E402
from intraday import IntradaySnapshot  # noqa: E402
from models import NewsItem, SectorMomentum, SectorScore  # noqa: E402
from profiles import PROFILES, MarketProfile  # noqa: E402


def run_profile(profile: MarketProfile) -> tuple[list[NewsItem], dict[str, SectorMomentum]]:
    """Run the news -> momentum -> analyzer pipeline checks for one profile."""
    print(f"\n{'=' * 70}\nPROFILE {profile.key}: {profile.label} ({profile.currency})\n{'=' * 70}")

    # --- 1. News -------------------------------------------------------------
    items = news_fetcher.fetch_all_news(48, profile)
    per_source = Counter(item.source for item in items)
    print("\nPer-source item counts:")
    for source, count in per_source.most_common():
        print(f"  {source:<35} {count}")
    assert len(items) >= 5, f"[{profile.key}] Expected >= 5 news items, got {len(items)}"
    assert len(per_source) >= 2, (
        f"[{profile.key}] Expected items from >= 2 distinct sources, got {len(per_source)}"
    )
    print(f"News OK: {len(items)} items from {len(per_source)} sources")

    # --- 2. Momentum ---------------------------------------------------------
    momentum = market_data.get_sector_momentum(profile)
    if not momentum:
        print("WARNING: momentum data unavailable (network?); continuing news-only")
    else:
        print(f"Momentum OK: {len(momentum)} sectors with usable data")

    # --- 3. Analyzer ---------------------------------------------------------
    scores = analyzer.analyze(items, momentum, 0.5, profile)
    expected = len(profile.sectors)
    assert len(scores) == expected, (
        f"[{profile.key}] Expected exactly {expected} SectorScores, got {len(scores)}"
    )
    assert all(isinstance(s, SectorScore) for s in scores), "Non-SectorScore in results"
    for s in scores:
        assert -1.0 <= s.composite <= 1.0, (
            f"[{profile.key}] Composite out of [-1, 1] for {s.sector}: {s.composite}"
        )
    composites = [s.composite for s in scores]
    assert composites == sorted(composites, reverse=True), (
        f"[{profile.key}] Scores are not sorted descending by composite"
    )
    print(f"Analyzer OK: {expected} sectors, composites in [-1, 1], sorted descending")

    # --- 4. Ranked table -----------------------------------------------------
    header = (
        f"{'Rank':<5}{'Sector':<25}{'Composite':>10}{'News':>8}"
        f"{'Articles':>10}{'Momentum':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for rank, s in enumerate(scores, start=1):
        mom = f"{s.momentum.score:+.3f}" if s.momentum is not None else "n/a"
        print(
            f"{rank:<5}{s.sector:<25}{s.composite:>+10.3f}{s.news_score:>+8.2f}"
            f"{s.news_count:>10}{mom:>10}"
        )
    print(f"\nTOP 3 ({profile.key}): {scores[0].sector}, {scores[1].sector}, {scores[2].sector}")
    return items, momentum


def _synthetic_snapshot(ticker: str, day_change: float, last_hour: float,
                        volume: float) -> IntradaySnapshot:
    """Build one synthetic intraday snapshot for the decision-engine check."""
    return IntradaySnapshot(
        ticker=ticker,
        last_price=100.0,
        asof=datetime.now(timezone.utc),
        day_change_pct=day_change,
        last_hour_change_pct=last_hour,
        range_position=0.8,
        avg_volume_20d=volume,
    )


def run_decision_check(items: list[NewsItem],
                       momentum: dict[str, SectorMomentum]) -> None:
    """Offline decision-engine check with synthetic snapshots (market need not be open)."""
    print(f"\n{'=' * 70}\nDECISION ENGINE CHECK (IN, synthetic intraday)\n{'=' * 70}")
    profile = PROFILES["IN"]
    now = datetime.now(timezone.utc)
    boosted = list(items) + [
        NewsItem(
            title=f"TCS and Infosys soar as IT stocks rally on upgrade {i}",
            summary="Indian IT services shares surge on strong deal wins.",
            link=f"https://example.com/synthetic-it-{i}",
            source="synthetic",
            published=now,
        )
        for i in range(8)
    ] + [
        NewsItem(
            title=f"Tata Steel and metal stocks crash as prices collapse {i}",
            summary="Indian metal shares plunge on weak demand and heavy losses.",
            link=f"https://example.com/synthetic-metal-{i}",
            source="synthetic",
            published=now,
        )
        for i in range(10)
    ]
    forced_momentum = dict(momentum)
    forced_momentum["IT"] = SectorMomentum(
        sector="IT", etf="^CNXIT", returns={"5d": 2.0}, score=0.5
    )
    forced_momentum["Metal"] = SectorMomentum(
        sector="Metal", etf="^CNXMETAL", returns={"5d": -2.5}, score=-0.5
    )
    snapshots = {
        "ITBEES.NS": _synthetic_snapshot("ITBEES.NS", 1.5, 0.3, 500_000.0),
        "BANKBEES.NS": _synthetic_snapshot("BANKBEES.NS", -1.2, -0.4, 800_000.0),
        "METALIETF.NS": _synthetic_snapshot("METALIETF.NS", -1.4, -0.5, 600_000.0),
        "PHARMABEES.NS": _synthetic_snapshot("PHARMABEES.NS", 0.9, 0.1, 1_000.0),
    }
    signals = decision.decide(boosted, forced_momentum, snapshots, profile)
    assert len(signals) == len(profile.sectors), (
        f"Expected {len(profile.sectors)} TradeSignals, got {len(signals)}"
    )
    ranks = [s.rank_score for s in signals]
    assert ranks == sorted(ranks, reverse=True), "Signals not sorted by rank desc"
    assert all(s.action in ("BUY", "SELL", "NO ACTION") for s in signals), (
        "Invalid action"
    )
    by_sector = {s.sector: s for s in signals}
    it_signal = by_sector["IT"]
    assert it_signal.action == "BUY", (
        f"Engineered IT signal should be BUY, got {it_signal.action}: {it_signal.reasons}"
    )
    metal_signal = by_sector["Metal"]
    assert metal_signal.action == "SELL", (
        f"Engineered Metal signal should be SELL, got {metal_signal.action}: "
        f"{metal_signal.reasons}"
    )
    assert by_sector["Pharma"].illiquid, "Pharma snapshot should be flagged illiquid"
    assert by_sector["Pharma"].action == "NO ACTION", (
        "Illiquid Pharma must not be actionable"
    )
    assert any("--- SELL gates ---" in r for r in by_sector["Pharma"].reasons), (
        "NO ACTION reasons must include the SELL-gate separator"
    )
    assert by_sector["Realty"].etf is None, "Realty should have no tradeable ETF"
    actionable = [s for s in signals if s.action != "NO ACTION"]
    assert {"BUY", "SELL"} <= {s.action for s in actionable}, (
        "Mixed case should produce both a BUY and a SELL"
    )
    expected_pick = max(actionable, key=lambda s: abs(s.rank_score))
    pick = decision.top_pick(signals)
    assert pick is not None and pick is expected_pick, (
        f"Top pick must be the largest |rank_score| actionable signal "
        f"(expected {expected_pick.sector}, got {pick.sector if pick else None})"
    )
    print(f"Decision OK: {len(signals)} signals, sorted, IT -> BUY, Metal -> SELL "
          f"(top pick {pick.action} {pick.etf} / {pick.sector}, "
          f"rank {pick.rank_score:+.3f})")


def main() -> int:
    """Run the end-to-end smoke test for every registered profile."""
    logger.info("Modules imported OK: %s", ", ".join(
        m.__name__ for m in (analyzer, claude_insights, config, decision,
                             market_data, models, news_fetcher, profiles)))
    in_items: list[NewsItem] = []
    in_momentum: dict[str, SectorMomentum] = {}
    for profile in PROFILES.values():
        items, momentum = run_profile(profile)
        if profile.key == "IN":
            in_items, in_momentum = items, momentum
    run_decision_check(in_items, in_momentum)
    print("\nAll profiles passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
