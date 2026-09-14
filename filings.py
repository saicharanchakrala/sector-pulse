"""Recent exchange filings for a symbol, as CONTEXT and never as a signal.

WHAT THE MEASUREMENT ACTUALLY SAYS, because this module exists at the edge
of a negative result and the distinction is easy to lose.

A filing does NOT predict direction. Tested pre-registered over 942 flagged
bars against a 683,660-bar control, with the last six months held back:

    after a material filing   29.6%      reached target 27.9%
    no filing (control)       30.6%      reached target 27.6%
                                         stopped out    66.5% vs 62.5%

Slightly WORSE, and its interval excludes even the 33.3% a coin flip gives.
Going looking for a better subset produced "Credit Rating 41.7%" on 60
trades - and shuffling the flags into noise and repeating the same search
produced a best-looking category of 37.0% on the median and 41.7% or better
one run in ten. There is no direction signal here and this module must not
be used to imply one.

WHAT DOES REPLICATE is volatility. A filing raises the bar range and the
relative volume in BOTH halves of the sample:

                   n     bar range    relative volume
    train        358        2.03x              1.94x
    holdout      584        1.22x              1.33x

The sign holds out of sample; the magnitude does not, so treat it as
"noticeably jumpier" and not as a number. That matters here because the
whole cost-and-reachability half of this app runs off an ATR measured over
the prior fourteen sessions, which is calm by construction - so on exactly
these bars the plausible move is understated and the stop is tighter than
it looks. Knowing that is a reason for a human to be careful. It is NOT
wired into the geometry: changing stop distance on this basis would need
its own pre-registered test, and the ten-policy exit sweep is evidence
against it working.

So the contract of this module is narrow and deliberate: tell the reader
something happened, when, and what kind. Let them decide.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

IST = "Asia/Kolkata"
STORE = config.PROJECT_ROOT / "forecast_cache" / "announcements.parquet"

# Filings a reasonable person would call news. Fixed by name before any
# outcome was measured, and NOT re-chosen afterwards - the per-category
# scores above are noise and must not be allowed to edit this list.
MATERIAL = frozenset({
    "Outcome of Board Meeting", "Financial Result Updates",
    "Bagging/Receiving of orders/contracts", "Credit Rating",
    "Acquisition", "Dividend", "Change in Management",
    "Disclosure under SEBI Takeover Regulations",
    "Clarification - Financial Results", "Scheme of Arrangement",
})
# Filed constantly and carrying nothing. Excluded for what they are, not
# for how they scored: Copy of Newspaper Publication is a regulatory
# formality, Trading Window a compliance notice, and so on.
ROUTINE = frozenset({
    "Copy of Newspaper Publication", "Trading Window",
    "Certificate under SEBI (Depositories and Participants) Regulations",
    "General Updates", "Updates", "ESOP/ESOS/ESPS", "Shareholders meeting",
    "News Verification",
})

# How far back a filing still counts as "just now" for the display. Half an
# hour, matching the window the test used, so the thing shown to a reader
# is the thing that was measured.
RECENT_MINUTES = 30


def _load() -> "pd.DataFrame | None":
    """The announcement store, or None when it has not been built.

    Absent is the normal state for a fresh checkout - the store is
    gitignored cache. Callers render nothing rather than erroring.
    """
    if not STORE.exists():
        return None
    try:
        frame = pd.read_parquet(STORE, columns=["symbol", "desc",
                                                "public_at"])
    except Exception as exc:
        logger.warning("Announcement store unreadable, showing no "
                       "filings: %s", exc)
        return None
    return frame.dropna(subset=["public_at"])


def kind_of(description: "str | None") -> str:
    """'material', 'routine' or 'other' for one filing description."""
    if not description:
        return "other"
    if description in MATERIAL:
        return "material"
    if description in ROUTINE:
        return "routine"
    return "other"


def recent_for(symbol: str, now: "datetime | None" = None,
               minutes: int = RECENT_MINUTES,
               material_only: bool = True) -> list:
    """Filings for `symbol` published in the last `minutes`, newest first.

    Strictly BEFORE `now`. A filing stamped at or after the moment being
    asked about has not happened yet from the caller's point of view, and
    including it would be the same look-ahead the store was built to avoid
    - the display would show a filing the trader could not have seen.
    """
    frame = _load()
    if frame is None or not symbol:
        return []
    when = now or datetime.now(tz=pd.Timestamp.now(tz=IST).tzinfo)
    cut = pd.Timestamp(when)
    if cut.tzinfo is None:
        cut = cut.tz_localize(IST)
    window = cut - pd.Timedelta(minutes=minutes)
    rows = frame[(frame["symbol"] == symbol)
                 & (frame["public_at"] < cut)
                 & (frame["public_at"] >= window)]
    if material_only:
        rows = rows[rows["desc"].isin(MATERIAL)]
    rows = rows.sort_values("public_at", ascending=False)
    return [{"desc": str(row.desc), "at": row.public_at,
             "kind": kind_of(row.desc),
             "minutes_ago": int((cut - row.public_at).total_seconds() // 60)}
            for row in rows.itertuples(index=False)]


def note_for(symbol: str, now: "datetime | None" = None,
             minutes: int = RECENT_MINUTES) -> str:
    """One line of context, or "" when there is nothing to say.

    Deliberately phrased as an observation rather than a recommendation.
    The measured effect is on VOLATILITY, so that is what the sentence
    says; it does not hint at direction, because there is none.
    """
    found = recent_for(symbol, now=now, minutes=minutes)
    if not found:
        return ""
    first = found[0]
    extra = (f" (+{len(found) - 1} more)" if len(found) > 1 else "")
    return (f"**{first['desc']}** filed {first['minutes_ago']} min ago"
            f"{extra}. Not a direction signal - filings measured slightly "
            f"WORSE than no filing. They do reliably raise volatility, so "
            f"the plausible move and the stop are both tighter than they "
            f"look right now.")


def store_age() -> "tuple[str, int] | None":
    """(newest date, days behind) for the store, or None if absent.

    The store is a backfill and does not refresh itself. A caller showing
    filings from a store three weeks stale would be showing silence and
    calling it "no news", which is worse than showing nothing.
    """
    frame = _load()
    if frame is None or frame.empty:
        return None
    newest = frame["public_at"].max()
    behind = (pd.Timestamp.now(tz=IST) - newest).days
    return newest.strftime("%Y-%m-%d"), int(behind)
