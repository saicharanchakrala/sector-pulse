"""News classification, finance-tuned sentiment, and composite sector scoring."""
from __future__ import annotations

import html
import logging
import math
import re
import unicodedata
from datetime import datetime, timezone

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from models import NewsItem, ScoredNewsItem, SectorMomentum, SectorScore
from profiles import MarketProfile, get_profile

logger = logging.getLogger(__name__)

FINANCE_LEXICON: dict[str, float] = {
    "plunge": -2.5,
    "plunges": -2.5,
    "soar": 2.2,
    "soars": 2.2,
    "slump": -2.0,
    "rally": 2.0,
    "rallies": 2.0,
    "downgrade": -1.8,
    "upgrade": 1.5,
    "bankruptcy": -3.0,
    "layoffs": -2.0,
    "tariff": -1.2,
    "tariffs": -1.2,
    "sanctions": -1.5,
    "recession": -2.2,
    "stimulus": 1.5,
    "bullish": 2.0,
    "bearish": -2.0,
    "default": -2.0,
    "surge": 2.0,
    "surges": 2.0,
    "crash": -3.0,
    "selloff": -2.2,
    "rebound": 1.8,
    "dividend": 0.8,
    "buyback": 1.2,
    "inflation": -1.0,
    "shortfall": -1.5,
    "beats": 1.5,
    "misses": -1.5,
    "outage": -1.5,
    "probe": -1.0,
    "lawsuit": -1.2,
}

_PHRASE_BOOST_DIVISOR = 8.0

_VADER_CACHE: dict[str, SentimentIntensityAnalyzer] = {}
_PATTERN_CACHE: dict[str, dict[str, "re.Pattern[str]"]] = {}


def _get_vader(profile: MarketProfile) -> SentimentIntensityAnalyzer:
    """Return a cached VADER analyzer with base finance + profile lexicons."""
    vader = _VADER_CACHE.get(profile.key)
    if vader is None:
        vader = SentimentIntensityAnalyzer()
        vader.lexicon.update(FINANCE_LEXICON)
        vader.lexicon.update(profile.lexicon)
        _VADER_CACHE[profile.key] = vader
    return vader


def _sector_patterns(profile: MarketProfile) -> dict[str, "re.Pattern[str]"]:
    """Build (once per profile) one case-insensitive word-boundary regex per sector."""
    patterns = _PATTERN_CACHE.get(profile.key)
    if patterns is None:
        patterns = {
            name: re.compile(
                r"\b(?:" + "|".join(re.escape(kw) for kw in sector.keywords) + r")\b",
                re.IGNORECASE,
            )
            for name, sector in profile.sectors.items()
        }
        _PATTERN_CACHE[profile.key] = patterns
    return patterns


def _phrase_boost(text_lower: str, phrases: dict[str, float]) -> float:
    """Sum of value/8 adjustments for each profile phrase found in the text."""
    return sum(
        value / _PHRASE_BOOST_DIVISOR
        for phrase, value in phrases.items()
        if phrase.lower() in text_lower
    )


def _recency_weight(published: datetime, now: datetime) -> float:
    """Half-life decay weight for an article's age, clamping future timestamps to 0."""
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - published).total_seconds() / 3600.0)
    return 0.5 ** (age_hours / config.RECENCY_HALF_LIFE_HOURS)


def _normalize_match_text(text: str) -> str:
    """Normalize text for keyword matching: unescape HTML entities, straighten quotes, fold accents."""
    text = html.unescape(text).replace("’", "'").replace("‘", "'")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _score_items(items: list[NewsItem], profile: MarketProfile) -> list[ScoredNewsItem]:
    """Classify each item into sectors and attach sentiment and recency weight."""
    patterns = _sector_patterns(profile)
    vader = _get_vader(profile)
    now = datetime.now(timezone.utc)
    scored: list[ScoredNewsItem] = []
    for item in items:
        summary = item.summary or ""
        match_text = _normalize_match_text(f"{item.title} {summary}")
        matched = [name for name, pattern in patterns.items() if pattern.search(match_text)]
        sentiment = vader.polarity_scores(f"{item.title}. {summary[:300]}")["compound"]
        if profile.phrases:
            sentiment += _phrase_boost(f"{item.title} {summary}".lower(), profile.phrases)
            sentiment = max(-1.0, min(1.0, sentiment))
        scored.append(
            ScoredNewsItem(
                item=item,
                sectors=matched,
                sentiment=sentiment,
                weight=_recency_weight(item.published, now),
            )
        )
    return scored


def analyze(
    items: list[NewsItem],
    momentum: dict[str, SectorMomentum],
    news_weight: float = config.DEFAULT_NEWS_WEIGHT,
    profile: MarketProfile | None = None,
) -> list[SectorScore]:
    """Rank the profile's sectors by a composite of news sentiment and momentum."""
    resolved = get_profile() if profile is None else profile
    scored = _score_items(items, resolved)
    by_sector: dict[str, list[ScoredNewsItem]] = {name: [] for name in resolved.sectors}
    for scored_item in scored:
        for name in scored_item.sectors:
            by_sector[name].append(scored_item)
    total_assignments = sum(len(group) for group in by_sector.values())

    results: list[SectorScore] = []
    for name, sector_def in resolved.sectors.items():
        group = by_sector[name]
        n = len(group)
        weight_sum = sum(s.weight for s in group)
        raw = sum(s.weight * s.sentiment for s in group) / weight_sum if weight_sum > 0 else 0.0
        news_score = raw * math.sqrt(min(1.0, n / config.MIN_ARTICLES_FULL_CONFIDENCE))
        buzz = n / total_assignments if total_assignments else 0.0
        sector_momentum = momentum.get(name)
        if sector_momentum is not None:
            composite = news_weight * news_score + (1.0 - news_weight) * sector_momentum.score
        else:
            composite = news_score
        top_items = sorted(group, key=lambda s: s.weight * abs(s.sentiment), reverse=True)
        results.append(
            SectorScore(
                sector=name,
                etf=sector_def.etf,
                news_score=news_score,
                news_count=n,
                buzz=buzz,
                momentum=sector_momentum,
                composite=composite,
                top_items=top_items[: config.TOP_HEADLINES_PER_SECTOR],
            )
        )
    results.sort(key=lambda score: score.composite, reverse=True)
    return results
