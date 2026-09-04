"""Concurrent RSS fetching, parsing, and deduplication for Sector Pulse."""
from __future__ import annotations

import calendar
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import feedparser
import requests

import config
from models import NewsItem
from profiles import MarketProfile, get_profile

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def _clean_text(raw: str) -> str:
    """Strip HTML tags, unescape entities, and collapse whitespace runs."""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", raw))).strip()


def _normalize_title(title: str) -> str:
    """Lowercase a title, keeping only alphanumerics and spaces, for dedupe."""
    return _WS_RE.sub(" ", _NON_ALNUM_RE.sub(" ", title.lower())).strip()


def _entry_published(entry: feedparser.FeedParserDict) -> datetime:
    """Return the entry's tz-aware UTC publish time (clamped to now), or now."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            published = datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
            return min(published, datetime.now(timezone.utc))
    return datetime.now(timezone.utc)


def _fetch_feed(feed: dict, cutoff: datetime) -> list[NewsItem]:
    """Fetch and parse one feed into recent NewsItems (may raise on failure)."""
    with requests.get(
        feed["url"],
        timeout=config.REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": config.USER_AGENT},
    ) as response:
        response.raise_for_status()
        content = response.content
    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        logger.warning(
            "Feed %s unparseable: %s",
            feed.get("name", "?"),
            parsed.get("bozo_exception"),
        )
        return []
    items: list[NewsItem] = []
    for entry in parsed.entries:
        try:
            title = _clean_text(entry.get("title", ""))
            link = entry.get("link", "")
            if not title or not link:
                continue
            published = _entry_published(entry)
        except (ValueError, TypeError, OverflowError, OSError) as exc:
            logger.debug(
                "Skipping malformed entry in feed %s: %s", feed.get("name", "?"), exc
            )
            continue
        if published < cutoff:
            continue
        items.append(
            NewsItem(
                title=title,
                summary=_clean_text(entry.get("summary", "")),
                link=link,
                source=feed["name"],
                published=published,
            )
        )
        if len(items) >= config.MAX_ITEMS_PER_FEED:
            break
    return items


def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    """Drop duplicates by exact link and by normalized title, keeping the first."""
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        norm_title = _normalize_title(item.title)
        if item.link in seen_links or (norm_title and norm_title in seen_titles):
            continue
        seen_links.add(item.link)
        if norm_title:
            seen_titles.add(norm_title)
        unique.append(item)
    return unique


def fetch_all_news(
    max_age_hours: int = config.NEWS_MAX_AGE_HOURS,
    profile: MarketProfile | None = None,
) -> list[NewsItem]:
    """Fetch the profile's feeds concurrently; return deduped news, newest first."""
    resolved = get_profile() if profile is None else profile
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    collected: list[NewsItem] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_fetch_feed, feed, cutoff): feed
            for feed in resolved.feeds
        }
        for future in as_completed(futures):
            feed = futures[future]
            try:
                collected.extend(future.result())
            except (
                requests.RequestException,
                KeyError,
                ValueError,
                TypeError,
                OverflowError,
                OSError,
            ) as exc:
                logger.warning("Feed %s failed: %s", feed.get("name", "?"), exc)
    collected.sort(key=lambda item: item.published, reverse=True)
    return _dedupe(collected)
