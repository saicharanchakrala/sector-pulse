"""Market profile registry for Sector Pulse.

A :class:`MarketProfile` bundles everything market-specific — RSS feeds,
sector definitions, and sentiment-lexicon additions — so the rest of the
app stays market-agnostic. Add a new market by creating one module in this
package and registering its profile in :data:`PROFILES`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config
from models import SectorDef

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketProfile:
    """Static configuration bundle describing one tracked market."""

    key: str                                  # short registry key, e.g. "US"
    label: str                                # human-readable market name
    currency: str                             # ISO currency code, e.g. "USD"
    feeds: list[dict]                         # {"name", "url", "category"} dicts
    sectors: dict[str, SectorDef]             # sector name -> definition
    lexicon: dict[str, float] = field(default_factory=dict)   # extra VADER unigrams
    phrases: dict[str, float] = field(default_factory=dict)   # multi-word boosts (-4..4)
    trade_etfs: dict[str, str] = field(default_factory=dict)  # sector -> tradeable ETF ticker


from profiles.india import IN_PROFILE  # noqa: E402  (needs MarketProfile above)
from profiles.us import US_PROFILE  # noqa: E402

PROFILES: dict[str, MarketProfile] = {
    US_PROFILE.key: US_PROFILE,
    IN_PROFILE.key: IN_PROFILE,
}


def get_profile(key: str | None = None) -> MarketProfile:
    """Resolve a profile by key (None -> config.DEFAULT_MARKET); unknown keys fall back to US."""
    resolved = key if key is not None else config.DEFAULT_MARKET
    profile = PROFILES.get(resolved)
    if profile is None:
        logger.warning("Unknown market profile %s; falling back to US", resolved)
        return PROFILES["US"]
    return profile
