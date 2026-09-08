"""End-of-day BUY / DON'T BUY decision engine.

Correlates today's news trend (today-only analyzer run) with each tradeable
ETF's intraday traded trend, sanity-checked by multi-day momentum. A sector
is a BUY only when every gate passes; everything else is DON'T BUY. The
engine never advises selling a holding. A mirror set of gates runs purely as
a diagnostic, so the explanation can distinguish a sector under real
pressure from one merely having a quiet day. Pure computation - no network,
never raises.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import analyzer
import config
import news_archive
from intraday import IntradaySnapshot
from models import NewsItem, SectorMomentum
from profiles import MarketProfile, get_profile

logger = logging.getLogger(__name__)

# Two outcomes only. The engine never tells you to sell something you hold:
# it either clears every buy check or it does not.
ACTION_BUY = "BUY"
ACTION_NO_BUY = "DON'T BUY"


@dataclass(frozen=True)
class TradeSignal:
    """One sector's daily trade decision and the inputs behind it."""

    sector: str
    etf: str | None                 # None = no tradeable ETF for this sector
    action: str                     # "BUY" | "DON'T BUY"
    rank_score: float
    news_today: float               # news score computed on today-only items
    news_count_today: int
    min_articles: int               # the floor actually applied to this sector
    momentum: float                 # multi-day momentum score (0.0 if unavailable)
    momentum_weight: float          # share of MOMENTUM_WINDOWS weight actually used
    intraday: IntradaySnapshot | None
    illiquid: bool                  # 20d mean turnover < SIGNAL_MIN_AVG_TURNOVER
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
    turnover = snapshot.avg_volume_20d * snapshot.last_price
    return (f"avg 20d turnover {turnover:,.0f} >= "
            f"{config.SIGNAL_MIN_AVG_TURNOVER:,} "
            f"[{'FAIL' if illiquid else 'PASS'}]")


def _momentum_reason(momentum_score: float,
                     momentum_weight: float) -> tuple[bool, str]:
    """Gate 4: multi-day trend floor, and only when the trend is measurable.

    A missing momentum score used to arrive as 0.0, and 0.0 >= -0.2, so a data
    outage made the falling-knife check PASS. A gate that blocks must fail when
    its input is absent, not wave the trade through.
    """
    if momentum_weight <= 0.0:
        return False, "momentum unavailable, no usable price windows [FAIL]"
    if momentum_weight < config.SIGNAL_MIN_MOMENTUM_WEIGHT:
        return False, (f"momentum {momentum_score:+.2f} computed on only "
                       f"{momentum_weight:.0%} of window weight, below "
                       f"{config.SIGNAL_MIN_MOMENTUM_WEIGHT:.0%} [FAIL]")
    ok = momentum_score >= config.SIGNAL_MIN_MOMENTUM
    return ok, (f"momentum {momentum_score:+.2f} >= {config.SIGNAL_MIN_MOMENTUM} "
                f"on {momentum_weight:.0%} of window weight "
                f"[{'PASS' if ok else 'FAIL'}]")


def _buy_gates(
    etf: str | None,
    snapshot: IntradaySnapshot | None,
    news_today: float,
    news_count: int,
    min_articles: int,
    momentum_score: float,
    momentum_weight: float,
    illiquid: bool,
) -> tuple[bool, list[str]]:
    """Evaluate all BUY gates; return (all_passed, reason strings)."""
    reasons: list[str] = []
    tradeable, reason = _tradeable_reason(etf, snapshot)
    reasons.append(reason)
    news_ok = (news_today >= config.SIGNAL_MIN_NEWS
               and news_count >= min_articles)
    reasons.append(
        f"news {news_today:+.2f} >= {config.SIGNAL_MIN_NEWS} and "
        f"n={news_count} >= {min_articles} "
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
    momentum_ok, momentum_reason = _momentum_reason(momentum_score, momentum_weight)
    reasons.append(momentum_reason)
    liquidity = _liquidity_reason(snapshot, illiquid)
    if liquidity is not None:
        reasons.append(liquidity)
    passed = tradeable and news_ok and intraday_ok and momentum_ok and not illiquid
    return passed, reasons


def _decline_gates(
    etf: str | None,
    snapshot: IntradaySnapshot | None,
    news_today: float,
    news_count: int,
    min_articles: int,
    momentum_score: float,
    momentum_weight: float,
    illiquid: bool,
) -> tuple[bool, list[str]]:
    """Evaluate the mirror of the buy gates; return (all_passed, reasons).

    This is a diagnostic, not an action. It distinguishes a sector that is
    actively weak from one that merely failed to qualify, which is worth
    saying in the explanation even though the verdict is the same.
    """
    reasons: list[str] = []
    tradeable, reason = _tradeable_reason(etf, snapshot)
    reasons.append(reason)
    news_ok = (news_today <= -config.SIGNAL_MIN_NEWS
               and news_count >= min_articles)
    reasons.append(
        f"news {news_today:+.2f} <= {-config.SIGNAL_MIN_NEWS} and "
        f"n={news_count} >= {min_articles} "
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
    reliable = momentum_weight >= config.SIGNAL_MIN_MOMENTUM_WEIGHT
    # SIGNAL_MIN_MOMENTUM is itself negative, so negating it inverts the test.
    # These gates began life as SELL gates, where `<= +0.2` deliberately meant
    # "do not sell into a strong uptrend". Read as an actively-declining
    # diagnostic that admits mild uptrends, so compare against the floor itself.
    momentum_ok = reliable and momentum_score <= config.SIGNAL_MIN_MOMENTUM
    reasons.append(
        f"momentum {momentum_score:+.2f} <= {config.SIGNAL_MIN_MOMENTUM} "
        f"on {momentum_weight:.0%} of window weight "
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
    min_articles: int,
    momentum_score: float,
    momentum_weight: float,
    illiquid: bool,
) -> tuple[str, list[str]]:
    """Resolve a sector's action: BUY when every gate passes, else DON'T BUY."""
    args = (etf, snapshot, news_today, news_count, min_articles,
            momentum_score, momentum_weight, illiquid)
    buy_ok, buy_reasons = _buy_gates(*args)
    if buy_ok:
        return ACTION_BUY, buy_reasons
    _, decline_reasons = _decline_gates(*args)
    return (ACTION_NO_BUY,
            buy_reasons + ["--- decline check (diagnostic) ---"] + decline_reasons)


def decide(
    items: list[NewsItem],
    momentum: dict[str, SectorMomentum],
    snapshots: dict[str, IntradaySnapshot],
    profile: MarketProfile | None = None,
    baselines: "dict[str, float] | None" = None,
) -> list[TradeSignal]:
    """Score every sector and gate the buy decision; sorted by rank desc."""
    resolved = get_profile() if profile is None else profile
    today_news = _today_news_scores(items, momentum, resolved)
    # Article floors scale with each sector's own coverage once the archive is
    # deep enough; until then this is {} and the flat floor applies.
    if baselines is None:
        baselines = news_archive.sector_article_baselines(resolved)
    signals: list[TradeSignal] = []
    for sector in resolved.sectors:
        etf = resolved.trade_etfs.get(sector)
        snapshot = snapshots.get(etf) if etf is not None else None
        news_today, news_count = today_news.get(sector, (0.0, 0))
        sector_momentum = momentum.get(sector)
        momentum_score = sector_momentum.score if sector_momentum is not None else 0.0
        # How much of the configured window weight actually produced a return.
        # Absent data must not masquerade as a neutral 0.0 score.
        momentum_weight = (
            sum(weight for label, (weight, _) in config.MOMENTUM_WINDOWS.items()
                if label in sector_momentum.returns)
            if sector_momentum is not None else 0.0
        )
        illiquid = (snapshot is not None
                    and snapshot.avg_volume_20d * snapshot.last_price
                    < config.SIGNAL_MIN_AVG_TURNOVER)
        day_change = snapshot.day_change_pct if snapshot is not None else 0.0
        rank_score = (0.4 * news_today
                      + 0.4 * math.tanh(day_change / 1.0)
                      + 0.2 * momentum_score)
        min_articles = news_archive.required_articles(sector, baselines)
        action, reasons = _classify(
            etf, snapshot, news_today, news_count, min_articles,
            momentum_score, momentum_weight, illiquid
        )
        signals.append(
            TradeSignal(
                sector=sector,
                etf=etf,
                action=action,
                rank_score=rank_score,
                news_today=news_today,
                news_count_today=news_count,
                min_articles=min_articles,
                momentum=momentum_score,
                momentum_weight=momentum_weight,
                intraday=snapshot,
                illiquid=illiquid,
                reasons=reasons,
            )
        )
    signals.sort(key=lambda signal: signal.rank_score, reverse=True)
    return signals


def is_declining(signal: TradeSignal) -> bool:
    """True when the sector is actively weak, not merely unremarkable."""
    snap = signal.intraday
    return bool(
        snap is not None
        and signal.news_today <= -config.SIGNAL_MIN_NEWS
        and signal.news_count_today >= signal.min_articles
        and snap.day_change_pct <= -config.SIGNAL_MIN_INTRADAY_PCT
        and signal.momentum_weight >= config.SIGNAL_MIN_MOMENTUM_WEIGHT
        and signal.momentum <= config.SIGNAL_MIN_MOMENTUM
    )


def _trend_words(momentum_score: float) -> str:
    """Describe a momentum score without using the number alone."""
    if momentum_score >= 0.3:
        return "clearly rising"
    if momentum_score >= 0.05:
        return "drifting up"
    if momentum_score > -0.05:
        return "roughly flat"
    return "still slightly down, though inside the tolerance"


def _buy_story(signal: TradeSignal, name: str) -> list[str]:
    """Plain-English reasons a sector cleared every buy gate."""
    snap = signal.intraday
    parts = [
        f"{name} is today's strongest candidate because all five checks "
        f"passed."
    ]
    parts.append(
        f"Coverage over the last {config.SIGNAL_NEWS_HOURS} hours leaned "
        f"positive: {signal.news_count_today} stories averaging "
        f"{signal.news_today:+.2f} on a scale from -1 to +1, where the rule "
        f"wants at least {config.SIGNAL_MIN_NEWS:+.2f} from "
        f"{config.SIGNAL_MIN_ARTICLES} stories."
    )
    if snap is not None:
        parts.append(
            f"The price agreed rather than contradicting the news: the ETF is "
            f"up {snap.day_change_pct:+.2f}% on the day and was still firm in "
            f"the final stretch ({snap.last_hour_change_pct:+.2f}% over the "
            f"last hour), so buyers stayed rather than fading it into the "
            f"close."
        )
    parts.append(
        f"Zooming out, the multi-week trend is "
        f"{_trend_words(signal.momentum)} ({signal.momentum:+.2f}, measured on "
        f"{signal.momentum_weight:.0%} of the price windows) - this is the "
        f"check that stops the rule buying something in free fall just because "
        f"it looks cheap today."
    )
    if snap is not None:
        parts.append(
            f"And it is liquid enough to act on, turning over about "
            f"{snap.avg_volume_20d * snap.last_price:,.0f} rupees a day "
            f"against a floor of {config.SIGNAL_MIN_AVG_TURNOVER:,}."
        )
    return parts


def _no_buy_story(signal: TradeSignal, name: str) -> list[str]:
    """Plain-English reasons a sector failed at least one buy gate."""
    snap = signal.intraday
    if signal.etf is None:
        return [f"{name} has no listed ETF in this profile, so there is "
                f"nothing to buy even when the sector looks good."]
    if snap is None:
        return [f"No intraday prices came back for {name} today, so the rule "
                f"cannot confirm what the news is claiming."]
    failures: list[str] = []
    if not (signal.news_today >= config.SIGNAL_MIN_NEWS
            and signal.news_count_today >= config.SIGNAL_MIN_ARTICLES):
        failures.append(
            f"news flow was only {signal.news_today:+.2f} across "
            f"{signal.news_count_today} stories, short of "
            f"{config.SIGNAL_MIN_NEWS:+.2f} from "
            f"{signal.min_articles}")
    if not (snap.day_change_pct >= config.SIGNAL_MIN_INTRADAY_PCT
            and snap.last_hour_change_pct >= 0.0):
        failures.append(
            f"the price did not back it up ({snap.day_change_pct:+.2f}% on "
            f"the day, {snap.last_hour_change_pct:+.2f}% in the last hour)")
    if signal.momentum_weight <= 0.0:
        failures.append(
            "the multi-week trend could not be measured at all, so the "
            "falling-knife check cannot be cleared")
    elif signal.momentum_weight < config.SIGNAL_MIN_MOMENTUM_WEIGHT:
        failures.append(
            f"the multi-week trend rests on only "
            f"{signal.momentum_weight:.0%} of its price windows, too little "
            f"to trust")
    elif signal.momentum < config.SIGNAL_MIN_MOMENTUM:
        failures.append(
            f"the multi-week trend is falling too hard "
            f"({signal.momentum:+.2f})")
    if signal.illiquid:
        failures.append(
            f"it is too thinly traded ("
            f"{snap.avg_volume_20d * snap.last_price:,.0f} rupees a day) "
            f"to get in and out cleanly")
    joined = "; ".join(failures) if failures else "one of the checks failed"
    parts = [f"{name} is not a buy today because {joined}."]
    if is_declining(signal):
        parts.append(
            "It is not merely unremarkable either - negative news, a falling "
            "price and a downward trend all line up, so this is a sector "
            "under real pressure rather than one having a quiet day.")
    return parts


def explain(signal: TradeSignal) -> str:
    """Explain a signal in layman's terms, as one prose paragraph."""
    name = f"{signal.sector} ({signal.etf})" if signal.etf else signal.sector
    if signal.action == ACTION_BUY:
        parts = _buy_story(signal, name)
    else:
        parts = _no_buy_story(signal, name)
    parts.append(
        "Treat this as a description of what the rule measured, not a "
        "recommendation: the thresholds have no backtest behind them.")
    return " ".join(parts)


def top_pick(signals: list[TradeSignal]) -> "TradeSignal | None":
    """Highest-ranked buy, or None when nothing cleared every gate."""
    buys = [s for s in signals if s.action == ACTION_BUY]
    if not buys:
        return None
    return max(buys, key=lambda signal: signal.rank_score)
