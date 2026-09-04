"""End-of-day BUY / SELL / NO ACTION decision engine.

Correlates today's news trend (today-only analyzer run) with each tradeable
ETF's intraday traded trend, sanity-checked by multi-day momentum. BUY and
SELL are symmetric gate sets; a sector that passes neither is NO ACTION.
SELL means exit/avoid/underweight — short-selling is not modelled. Pure
computation — no network, never raises.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import analyzer
import config
from intraday import IntradaySnapshot
from models import NewsItem, SectorMomentum
from profiles import MarketProfile, get_profile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeSignal:
    """One sector's daily trade decision and the inputs behind it."""

    sector: str
    etf: str | None                 # None = no tradeable ETF for this sector
    action: str                     # "BUY" | "SELL" | "NO ACTION"
    rank_score: float
    news_today: float               # news score computed on today-only items
    news_count_today: int
    momentum: float                 # multi-day momentum score (0.0 if unavailable)
    intraday: IntradaySnapshot | None
    illiquid: bool                  # avg_volume_20d < config.SIGNAL_MIN_AVG_VOLUME
    reasons: list[str]              # human-readable pass/fail per gate


def _today_news_scores(
    items: list[NewsItem],
    momentum: dict[str, SectorMomentum],
    profile: MarketProfile,
) -> dict[str, tuple[float, int]]:
    """Run the analyzer on the last SIGNAL_NEWS_HOURS of news only."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.SIGNAL_NEWS_HOURS)
    fresh = [item for item in items if item.published >= cutoff]
    scores = analyzer.analyze(fresh, momentum, profile=profile)
    return {s.sector: (s.news_score, s.news_count) for s in scores}


def _tradeable_reason(etf: str | None,
                      snapshot: IntradaySnapshot | None) -> tuple[bool, str]:
    """Gate 1 (shared by BUY and SELL): tradeable ETF with intraday data."""
    if etf is None:
        return False, "no tradeable ETF for sector [FAIL]"
    if snapshot is None:
        return False, f"no intraday snapshot for {etf} [FAIL]"
    return True, f"tradeable ETF {etf} with intraday data [PASS]"


def _liquidity_reason(snapshot: IntradaySnapshot | None,
                      illiquid: bool) -> "str | None":
    """Gate 5 (shared by BUY and SELL): 20-session average volume floor."""
    if snapshot is None:
        return None
    return (f"avg 20d volume {snapshot.avg_volume_20d:,.0f} >= "
            f"{config.SIGNAL_MIN_AVG_VOLUME:,} "
            f"[{'FAIL' if illiquid else 'PASS'}]")


def _buy_gates(
    etf: str | None,
    snapshot: IntradaySnapshot | None,
    news_today: float,
    news_count: int,
    momentum_score: float,
    illiquid: bool,
) -> tuple[bool, list[str]]:
    """Evaluate all BUY gates; return (all_passed, reason strings)."""
    reasons: list[str] = []
    tradeable, reason = _tradeable_reason(etf, snapshot)
    reasons.append(reason)
    news_ok = (news_today >= config.SIGNAL_MIN_NEWS
               and news_count >= config.SIGNAL_MIN_ARTICLES)
    reasons.append(
        f"news {news_today:+.2f} >= {config.SIGNAL_MIN_NEWS} and "
        f"n={news_count} >= {config.SIGNAL_MIN_ARTICLES} "
        f"[{'PASS' if news_ok else 'FAIL'}]"
    )
    if snapshot is not None:
        intraday_ok = (snapshot.day_change_pct >= config.SIGNAL_MIN_INTRADAY_PCT
                       and snapshot.last_hour_change_pct >= 0.0)
        reasons.append(
            f"day {snapshot.day_change_pct:+.2f}% >= "
            f"{config.SIGNAL_MIN_INTRADAY_PCT}% and last hour "
            f"{snapshot.last_hour_change_pct:+.2f}% >= 0 "
            f"[{'PASS' if intraday_ok else 'FAIL'}]"
        )
    else:
        intraday_ok = False
    momentum_ok = momentum_score >= config.SIGNAL_MIN_MOMENTUM
    reasons.append(
        f"momentum {momentum_score:+.2f} >= {config.SIGNAL_MIN_MOMENTUM} "
        f"[{'PASS' if momentum_ok else 'FAIL'}]"
    )
    liquidity = _liquidity_reason(snapshot, illiquid)
    if liquidity is not None:
        reasons.append(liquidity)
    passed = tradeable and news_ok and intraday_ok and momentum_ok and not illiquid
    return passed, reasons


def _sell_gates(
    etf: str | None,
    snapshot: IntradaySnapshot | None,
    news_today: float,
    news_count: int,
    momentum_score: float,
    illiquid: bool,
) -> tuple[bool, list[str]]:
    """Evaluate all SELL gates (mirror of BUY); return (all_passed, reasons)."""
    reasons: list[str] = []
    tradeable, reason = _tradeable_reason(etf, snapshot)
    reasons.append(reason)
    news_ok = (news_today <= -config.SIGNAL_MIN_NEWS
               and news_count >= config.SIGNAL_MIN_ARTICLES)
    reasons.append(
        f"news {news_today:+.2f} <= {-config.SIGNAL_MIN_NEWS} and "
        f"n={news_count} >= {config.SIGNAL_MIN_ARTICLES} "
        f"[{'PASS' if news_ok else 'FAIL'}]"
    )
    if snapshot is not None:
        intraday_ok = (snapshot.day_change_pct <= -config.SIGNAL_MIN_INTRADAY_PCT
                       and snapshot.last_hour_change_pct <= 0.0)
        reasons.append(
            f"day {snapshot.day_change_pct:+.2f}% <= "
            f"{-config.SIGNAL_MIN_INTRADAY_PCT}% and last hour "
            f"{snapshot.last_hour_change_pct:+.2f}% <= 0 "
            f"[{'PASS' if intraday_ok else 'FAIL'}]"
        )
    else:
        intraday_ok = False
    momentum_ok = momentum_score <= -config.SIGNAL_MIN_MOMENTUM
    reasons.append(
        f"momentum {momentum_score:+.2f} <= {-config.SIGNAL_MIN_MOMENTUM} "
        f"[{'PASS' if momentum_ok else 'FAIL'}]"
    )
    liquidity = _liquidity_reason(snapshot, illiquid)
    if liquidity is not None:
        reasons.append(liquidity)
    passed = tradeable and news_ok and intraday_ok and momentum_ok and not illiquid
    return passed, reasons


def _classify(
    etf: str | None,
    snapshot: IntradaySnapshot | None,
    news_today: float,
    news_count: int,
    momentum_score: float,
    illiquid: bool,
) -> tuple[str, list[str]]:
    """Resolve a sector's action: BUY first, then SELL, else NO ACTION."""
    args = (etf, snapshot, news_today, news_count, momentum_score, illiquid)
    buy_ok, buy_reasons = _buy_gates(*args)
    if buy_ok:
        return "BUY", buy_reasons
    sell_ok, sell_reasons = _sell_gates(*args)
    if sell_ok:
        return "SELL", sell_reasons
    return "NO ACTION", buy_reasons + ["--- SELL gates ---"] + sell_reasons


def decide(
    items: list[NewsItem],
    momentum: dict[str, SectorMomentum],
    snapshots: dict[str, IntradaySnapshot],
    profile: MarketProfile | None = None,
) -> list[TradeSignal]:
    """Score every sector and gate BUY/SELL decisions; sorted by rank desc."""
    resolved = get_profile() if profile is None else profile
    today_news = _today_news_scores(items, momentum, resolved)
    signals: list[TradeSignal] = []
    for sector in resolved.sectors:
        etf = resolved.trade_etfs.get(sector)
        snapshot = snapshots.get(etf) if etf is not None else None
        news_today, news_count = today_news.get(sector, (0.0, 0))
        sector_momentum = momentum.get(sector)
        momentum_score = sector_momentum.score if sector_momentum is not None else 0.0
        illiquid = (snapshot is not None
                    and snapshot.avg_volume_20d < config.SIGNAL_MIN_AVG_VOLUME)
        day_change = snapshot.day_change_pct if snapshot is not None else 0.0
        rank_score = (0.4 * news_today
                      + 0.4 * math.tanh(day_change / 1.0)
                      + 0.2 * momentum_score)
        action, reasons = _classify(
            etf, snapshot, news_today, news_count, momentum_score, illiquid
        )
        signals.append(
            TradeSignal(
                sector=sector,
                etf=etf,
                action=action,
                rank_score=rank_score,
                news_today=news_today,
                news_count_today=news_count,
                momentum=momentum_score,
                intraday=snapshot,
                illiquid=illiquid,
                reasons=reasons,
            )
        )
    signals.sort(key=lambda signal: signal.rank_score, reverse=True)
    return signals


def top_pick(signals: list[TradeSignal]) -> "TradeSignal | None":
    """Actionable signal with the largest |rank_score|, or None if none pass."""
    actionable = [s for s in signals if s.action != "NO ACTION"]
    if not actionable:
        return None
    return max(actionable, key=lambda signal: abs(signal.rank_score))
