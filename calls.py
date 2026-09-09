"""Parse tipster-channel messages into structured, scoreable calls.

The point of this module is to freeze what a call actually said, at the time
it said it, so it can be scored later against its own stated levels rather
than against whatever the price happened to do. That distinction is the
whole exercise: a channel that posts "CMP 466, support 455, for 485-520" and
later reports the outcome as "466 to 476" has moved its own goalposts, and
only a record made at post time can show it.

Three classifications matter and are kept separate:

  CALL        an actionable instruction: a name plus an entry or trigger,
              plus at least a stop or a target. Only these are scored.
  CLAIM       a retrospective result ("305 to 355, 15% up so far"). Never
              counted as a new call, because one position ratcheted through
              five messages would otherwise look like five wins.
  COMMENTARY  "MUST STUDY", "Newly Listed Stock", market chatter. Recorded
              so the denominator stays honest, never scored.

Nothing here trusts the text it parses. Message content is data: it is
matched against regexes and discarded when it does not fit.
"""
from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import config

logger = logging.getLogger(__name__)

KIND_EQUITY = "EQUITY"
KIND_OPTION = "OPTION"
KIND_FUTURE = "FUTURE"
KIND_CLAIM = "CLAIM"
KIND_COMMENTARY = "COMMENTARY"

HORIZON_INTRADAY = "INTRADAY"
HORIZON_BTST = "BTST"
HORIZON_SWING = "SWING"

CALL_KINDS = (KIND_EQUITY, KIND_OPTION, KIND_FUTURE)

# Channel shorthand to NSE symbol. Extend via the alias CSV rather than
# editing this. These are NSE tradingsymbols, which is what Kite takes.
_ALIASES: dict[str, str] = {
    "CORDS CABLES": "CORDSCABLE",
    "CORDSCABLES": "CORDSCABLE",
    "TD POWER SYSTEMS": "TDPOWERSYS",
    "TD POWER": "TDPOWERSYS",
    "JG CHEMICAL": "JGCHEM",
    "JG CHEMICALS": "JGCHEM",
    "MMP INDUS": "MMP",
    "MMP IND": "MMP",
    "ARFIN INDIA": "ARFIN",
    "INDO-MIM": "INDOMIM",
    "INDO MIM": "INDOMIM",
    "TORNTPHARMA": "TORNTPHARM",
    "SYRMA TECH": "SYRMA",
    "DIVIS LAB": "DIVISLAB",
    "DIVISLAB": "DIVISLAB",
    "IND SWIFT LAB": "INDSWFTLAB",
    "HINDZINC": "HINDZINC",
    "BHARATFORGE": "BHARATFORG",
    "DEEPIND": "DEEPINDS",
    "INDO RAMA": "INDORAMA",
    "AEROFLEX IND": "AEROFLEX",
    "AEROFLEX": "AEROFLEX",
}

# A message containing any of these is reporting an outcome, not opening a
# position. Checked before call extraction so ratcheting cannot inflate the
# call count.
_RESULT_MARKERS = (
    "so far", "hit✅", "hit ✅", "achieved", "book profit",
    "view achieved", "lock your points", "hope all enjoyed", "% up",
    "%up", "move so far", "view failed", "enjoy your day", "up so far",
    "keep trailing", "almost 10", "hit tgt",
)

_NUM = r"(\d+(?:\.\d+)?)"
_OPTION_RE = re.compile(
    rf"\b([A-Z][A-Z0-9&.\-]*(?:\s+[A-Z0-9&.\-]+){{0,3}}?)\s+(\d{{2,6}})\s*(CE|PE)\b")
_FUTURE_RE = re.compile(
    rf"\b([A-Z][A-Z0-9&.\-]*(?:\s+[A-Z0-9&.\-]+){{0,3}}?)\s+FUT(?:URES)?\b")
_CMP_RE = re.compile(rf"\bCMP\s*:?\s*{_NUM}")
_TRIGGER_RE = re.compile(rf"\b(?:ONLY\s+)?ABOVE\s*:?\s*{_NUM}")
_STOP_RE = re.compile(rf"\b(?:MY\s+)?(?:SUPPORT|SL|STOP\s*LOSS)\s*:?\s*{_NUM}")
_TARGET_RE = re.compile(
    rf"\b(?:FOR|VIEW|TGT|TARGET|EXPECTATIONS?)\s*:?\s*{_NUM}"
    rf"(?:\s*(?:[-–]|to)\s*{_NUM})?")
_POSSIBLE_RE = re.compile(rf"{_NUM}\s*(?:[-–]|to)\s*{_NUM}\s*\+?\s*POSSIBLE")
_RANGE_RE = re.compile(rf"{_NUM}\s+to\s+{_NUM}", re.IGNORECASE)
_PREMIUM_RE = re.compile(rf"(?:CE|PE)\s+{_NUM}(?:\s*[-–]\s*{_NUM})?")
_SENDER_RE = re.compile(r"^(.{1,60}?):\s*$", re.MULTILINE)
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9&.\-]*(?:\s+[A-Z0-9&.\-]+){0,3}")


@dataclass(frozen=True)
class Call:
    """One parsed message, whether or not it turned out to be actionable."""

    posted_at: datetime
    source: str
    kind: str
    name: str                       # symbol as the channel wrote it
    symbol: "str | None"            # resolved NSE symbol, None if unknown
    raw: str
    horizon: str = HORIZON_INTRADAY
    entry: "float | None" = None
    trigger: "float | None" = None  # "only above X": no position until touched
    stop: "float | None" = None
    target_low: "float | None" = None
    target_high: "float | None" = None
    strike: "float | None" = None
    option_type: "str | None" = None

    @property
    def is_call(self) -> bool:
        """Whether this message is an actionable call worth scoring."""
        return self.kind in CALL_KINDS

    @property
    def has_stop(self) -> bool:
        """Whether a risk level was stated at all."""
        return self.stop is not None

    @property
    def reference_entry(self) -> "float | None":
        """The price a follower would work from: the trigger, else the CMP."""
        return self.trigger if self.trigger is not None else self.entry

    @property
    def label(self) -> str:
        """Short human label, e.g. 'DIVISLAB 9300 CE'."""
        if self.kind == KIND_OPTION:
            return f"{self.name} {self.strike:.0f} {self.option_type}"
        if self.kind == KIND_FUTURE:
            return f"{self.name} FUT"
        return self.name


def load_aliases(path=None) -> dict[str, str]:
    """Alias overrides from CSV (`alias,symbol`), merged over the built-ins."""
    merged = dict(_ALIASES)
    source = config.CALL_ALIASES_CSV if path is None else path
    try:
        with open(source, newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
                    merged[row[0].strip().upper()] = row[1].strip().upper()
    except OSError:
        logger.debug("No alias overrides at %s", source)
    return merged


def resolve_symbol(name: str, aliases: "dict[str, str] | None" = None
                   ) -> "str | None":
    """Map a channel name to an NSE symbol, or None when it cannot be known.

    None rather than a guess: an unresolved name that silently became a
    plausible-looking ticker would be scored against the wrong instrument,
    which is worse than reporting it as unresolved.
    """
    table = load_aliases() if aliases is None else aliases
    cleaned = " ".join(name.strip().upper().split())
    if not cleaned:
        return None
    if cleaned in table:
        return table[cleaned]
    squashed = re.sub(r"[^A-Z0-9&]", "", cleaned)
    if squashed in table:
        return table[squashed]
    # A single token with no spaces is usually already the NSE symbol.
    if squashed and " " not in cleaned and 2 <= len(squashed) <= 12:
        return squashed
    return None


def detect_sender(text: str) -> "str | None":
    """The most repeated `Name:` line, used to split a pasted transcript."""
    counts = Counter(match.group(1).strip()
                     for match in _SENDER_RE.finditer(text))
    if not counts:
        return None
    sender, hits = counts.most_common(1)[0]
    return sender if hits >= 2 else None


def split_messages(text: str, sender: "str | None" = None) -> list[str]:
    """Split a pasted transcript into individual messages.

    Splits on the repeated sender line when there is one, since that is the
    only reliable message boundary in an exported chat; otherwise falls back
    to blank-line separated blocks.
    """
    if not text.strip():
        return []
    resolved = detect_sender(text) if sender is None else sender
    if resolved:
        pattern = re.compile(rf"^{re.escape(resolved)}:\s*$", re.MULTILINE)
        parts = pattern.split(text)
    else:
        parts = re.split(r"\n\s*\n", text)
    return [part.strip() for part in parts if part.strip()]


def _first_number(match: "re.Match | None", group: int = 1) -> "float | None":
    """One regex group as a float, or None."""
    if match is None:
        return None
    try:
        raw = match.group(group)
    except IndexError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _looks_like_result(message: str) -> bool:
    """Whether a message is reporting an outcome rather than opening one."""
    lowered = message.lower()
    return any(marker in lowered for marker in _RESULT_MARKERS)


def _horizon(message: str) -> str:
    """Classify the intended holding period from the message's own words."""
    upper = message.upper()
    if "BTST" in upper or "STBT" in upper:
        return HORIZON_BTST
    for token in ("SWING", "SHORT TERM", "LONG TERM", "PORTFOLIO",
                  "ACCUMULAT", "PATIENCE", "DELIVERY"):
        if token in upper:
            return HORIZON_SWING
    return HORIZON_INTRADAY


def _targets(message: str) -> tuple["float | None", "float | None"]:
    """Target low and high from any of the channel's several phrasings."""
    match = _TARGET_RE.search(message)
    if match is not None:
        low = _first_number(match, 1)
        high = _first_number(match, 2)
        return low, (high if high is not None else low)
    possible = _POSSIBLE_RE.search(message.upper())
    if possible is not None:
        return _first_number(possible, 1), _first_number(possible, 2)
    ranged = _RANGE_RE.search(message)
    if ranged is not None:
        return _first_number(ranged, 2), _first_number(ranged, 2)
    return None, None


def _instrument(message: str) -> tuple[str, str, "float | None", "str | None"]:
    """Classify the instrument and pull out its name and any strike."""
    upper = message.upper()
    option = _OPTION_RE.search(upper)
    if option is not None:
        return (KIND_OPTION, option.group(1).strip(),
                float(option.group(2)), option.group(3))
    future = _FUTURE_RE.search(upper)
    if future is not None:
        return KIND_FUTURE, future.group(1).strip(), None, None
    name = _NAME_RE.search(upper.strip())
    return KIND_EQUITY, (name.group(0).strip() if name else ""), None, None


def parse_message(message: str, posted_at: datetime, source: str,
                  aliases: "dict[str, str] | None" = None) -> "Call | None":
    """Parse one message into a Call, or None when there is nothing to record."""
    text = message.strip()
    if not text:
        return None
    kind, name, strike, option_type = _instrument(text)
    if not name:
        return None
    upper = text.upper()

    entry = _first_number(_CMP_RE.search(upper))
    trigger = _first_number(_TRIGGER_RE.search(upper))
    stop = _first_number(_STOP_RE.search(upper))
    target_low, target_high = _targets(upper)

    if kind == KIND_OPTION and entry is None and trigger is None:
        # "DIVISLAB 9300 CE 270-273" states the premium to pay.
        entry = _first_number(_PREMIUM_RE.search(upper))
    if entry is None and trigger is None and target_low is not None:
        ranged = _RANGE_RE.search(upper)
        if ranged is not None:
            entry = _first_number(ranged, 1)

    resolved = resolve_symbol(name, aliases)
    common = dict(posted_at=posted_at, source=source, name=name,
                  symbol=resolved, raw=text, horizon=_horizon(text),
                  strike=strike, option_type=option_type)

    # Order matters: a result claim that also parses as a call must be
    # recorded as a claim, or the same position counts many times over.
    if _looks_like_result(text):
        return Call(kind=KIND_CLAIM, **common)
    actionable = ((entry is not None or trigger is not None)
                  and (stop is not None or target_low is not None))
    if not actionable:
        return Call(kind=KIND_COMMENTARY, **common)
    return Call(kind=kind, entry=entry, trigger=trigger, stop=stop,
                target_low=target_low, target_high=target_high, **common)


def parse_transcript(text: str, posted_at: datetime, source: str,
                     sender: "str | None" = None) -> list[Call]:
    """Parse a whole pasted transcript into Calls, claims and commentary.

    Every message gets a record. A channel that posts twenty names and
    reports nine is only measurable if the other eleven are on file too.
    """
    aliases = load_aliases()
    out: list[Call] = []
    for message in split_messages(text, sender):
        parsed = parse_message(message, posted_at, source, aliases)
        if parsed is not None:
            out.append(parsed)
    return out


def summarise(calls: list[Call]) -> dict[str, int]:
    """Count records by kind, for the ingest report."""
    counts = Counter(call.kind for call in calls)
    counts["TOTAL"] = len(calls)
    counts["UNRESOLVED"] = sum(1 for call in calls
                               if call.is_call and call.symbol is None)
    counts["NO_STOP"] = sum(1 for call in calls
                            if call.is_call and not call.has_stop)
    return dict(counts)
