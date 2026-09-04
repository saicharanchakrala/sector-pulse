"""Shared data models for Sector Pulse."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SectorDef:
    """Static definition of a GICS sector tracked by the app."""

    name: str
    etf: str                      # SPDR sector ETF ticker
    keywords: list[str]           # lowercase phrases for news classification


@dataclass
class NewsItem:
    """A single news article pulled from an RSS feed."""

    title: str
    summary: str
    link: str
    source: str
    published: datetime           # timezone-aware UTC


@dataclass
class ScoredNewsItem:
    """A news item after sector classification and sentiment scoring."""

    item: NewsItem
    sectors: list[str]            # matched sector names (may be empty)
    sentiment: float              # -1..1 finance-adjusted VADER compound
    weight: float                 # 0..1 recency weight


@dataclass
class SectorMomentum:
    """Price momentum for one sector's ETF."""

    sector: str
    etf: str
    returns: dict[str, float]     # window label -> percent return, e.g. {"5d": 1.2}
    score: float                  # weighted tanh-squashed momentum, roughly -1..1


@dataclass
class SectorScore:
    """Final composite ranking entry for one sector."""

    sector: str
    etf: str
    news_score: float             # -1..1 confidence-damped weighted sentiment
    news_count: int
    buzz: float                   # 0..1 share of sector-classified article matches
    momentum: SectorMomentum | None
    composite: float              # -1..1
    top_items: list[ScoredNewsItem] = field(default_factory=list)
