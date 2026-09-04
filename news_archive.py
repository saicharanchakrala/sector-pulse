"""Durable daily archive of fetched headlines.

RSS feeds serve roughly a 48-hour window. A headline not captured on the day
it appears is gone permanently, which means the news half of the signal can
never be backtested for any day that was not archived. This module writes
every fetched item to a dated JSONL file so that a sentiment backtest becomes
possible from the first day it runs, and no earlier.

The archive is append-only and deduplicated by link within each day's file, so
running the signal several times a day costs nothing and loses nothing.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import config
from models import NewsItem

logger = logging.getLogger(__name__)


def archive_path(day: date, directory: "str | Path | None" = None) -> Path:
    """Path of the archive file for one calendar day."""
    root = Path(directory) if directory is not None else config.NEWS_ARCHIVE_DIR
    return root / f"{day.isoformat()}.jsonl"


def _existing_links(path: Path) -> set[str]:
    """Links already archived in this file, so re-runs do not duplicate."""
    if not path.exists():
        return set()
    links: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    links.add(json.loads(line).get("link", ""))
                except (json.JSONDecodeError, AttributeError):
                    continue
    except OSError as exc:
        logger.warning("Could not read news archive %s: %s", path, exc)
    return links


def archive_news(items: list[NewsItem], day: "date | None" = None,
                 directory: "str | Path | None" = None) -> int:
    """Append unseen items to the day's archive; return how many were written.

    Never raises: a failed archive must not take the signal run down with it.
    """
    if not items:
        return 0
    target_day = day if day is not None else datetime.now(timezone.utc).date()
    path = archive_path(target_day, directory)
    seen = _existing_links(path)
    fetched_at = datetime.now(timezone.utc).isoformat()
    fresh = [item for item in items if item.link and item.link not in seen]
    if not fresh:
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for item in fresh:
                handle.write(json.dumps({
                    "fetched_at": fetched_at,
                    "title": item.title,
                    "summary": item.summary,
                    "link": item.link,
                    "source": item.source,
                    "published": item.published.isoformat(),
                }, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not write news archive %s: %s", path, exc)
        return 0
    return len(fresh)


def load_archived_news(day: date,
                       directory: "str | Path | None" = None) -> list[NewsItem]:
    """Read one day's archived items back as NewsItems, newest first.

    This is what makes a sentiment backtest possible: feed the result straight
    into analyzer.analyze for the day being replayed.
    """
    path = archive_path(day, directory)
    if not path.exists():
        return []
    items: list[NewsItem] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    items.append(NewsItem(
                        title=row["title"],
                        summary=row.get("summary", ""),
                        link=row["link"],
                        source=row.get("source", ""),
                        published=datetime.fromisoformat(row["published"]),
                    ))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("Skipping bad archive row %s:%d: %s",
                                   path.name, line_no, exc)
    except OSError as exc:
        logger.warning("Could not read news archive %s: %s", path, exc)
        return []
    items.sort(key=lambda item: item.published, reverse=True)
    return items


def archived_days(directory: "str | Path | None" = None) -> list[date]:
    """Every day for which an archive file exists, oldest first."""
    root = Path(directory) if directory is not None else config.NEWS_ARCHIVE_DIR
    if not root.is_dir():
        return []
    days: list[date] = []
    for path in root.glob("*.jsonl"):
        try:
            days.append(date.fromisoformat(path.stem))
        except ValueError:
            logger.warning("Ignoring unexpected archive filename %s", path.name)
    return sorted(days)
