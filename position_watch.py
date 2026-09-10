"""Positions you have actually taken, checked against the scanner's own gates.

WHY THIS EXISTS, concretely. On 2026-09-10 the scan showed HDFCLIFE SHORT
at 511.20 with a stop at 516.00. It was blocked - relative strength was
+3.06pp against the Nifty, so shorting a name that was outperforming
failed the one gate that mattered - but it was displayed with full levels
and taken as a suggestion. Over the next two hours the scanner's own view
of HDFCLIFE went SHORT, then NO SETUP, then LONG, and it never once said
that the earlier signal had been invalidated. It simply showed something
different. The stop went through inside a single five-minute bar: 515.50 to
520.55 between 12:00 and 12:05.

The scanner is stateless by design - it describes now, for every symbol,
and forgets. That is right for a screen and useless for someone holding a
position, because the question stops being "what looks interesting" and
becomes "does the thing I already did still stand up". Nothing in this
project answered that.

WHAT THIS DOES AND DOES NOT DO. It reports state: where price sits against
the levels you recorded, whether the scanner's current direction
contradicts your position, and whether the gates that justified it still
pass. It does NOT tell you to exit, hold or add. That is a decision about
your money and this module has no business making it - it exists so the
facts reach you while they still matter, which is precisely what failed on
HDFCLIFE.

SEVERITY ORDER, and the reasoning. A breached stop outranks everything
because the money is already at risk beyond what you accepted; a reached
target is next because it is the outcome you were waiting for; a flipped
direction outranks a failed gate because the scanner now positively
disagrees rather than merely declining to agree.

THE FILE IS PERSONAL DATA. Positions live in a gitignored JSON file beside
holdings.csv and the rest, and nothing here is ever committed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent
STORE = ROOT / "watched_positions.json"

LONG = "LONG"
SHORT = "SHORT"

# States, worst first. `rank()` uses this order, so it is the severity
# policy rather than a display detail.
STOP_BREACHED = "STOP BREACHED"
TARGET_REACHED = "TARGET REACHED"
FLIPPED = "DIRECTION FLIPPED"
UNSUPPORTED = "GATES NO LONGER PASS"
SUPPORTED = "STILL SUPPORTED"
UNKNOWN = "NO LIVE PRICE"
SEVERITY = (STOP_BREACHED, TARGET_REACHED, FLIPPED, UNSUPPORTED, UNKNOWN,
            SUPPORTED)


@dataclass
class Position:
    """One position you have taken and want watched.

    `quantity` is optional and used only to turn a percentage into rupees.
    Nothing here is sent anywhere or used to place an order.
    """

    symbol: str
    side: str
    entry: float
    stop: float
    target: float
    quantity: int = 0
    note: str = ""
    taken_at: str = ""

    def __post_init__(self) -> None:
        self.symbol = (self.symbol or "").strip().upper()
        self.side = (self.side or "").strip().upper()
        if not self.taken_at:
            self.taken_at = datetime.now(IST).isoformat(timespec="seconds")

    @property
    def is_valid(self) -> bool:
        """Whether the levels describe a position that could exist.

        A long whose stop sits above its entry, or a short whose stop sits
        below, is a typo rather than a trade - and silently watching one
        would report a breach immediately and permanently.
        """
        if self.side not in (LONG, SHORT):
            return False
        if not all(v > 0 for v in (self.entry, self.stop, self.target)):
            return False
        if self.side == LONG:
            return self.stop < self.entry < self.target
        return self.target < self.entry < self.stop

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)


@dataclass
class Status:
    """What is true about one watched position right now."""

    position: Position
    state: str
    headline: str
    detail: str = ""
    move_pct: float = float("nan")
    lines: list = field(default_factory=list)

    @property
    def severity(self) -> int:
        return SEVERITY.index(self.state) if self.state in SEVERITY else 99

    @property
    def needs_attention(self) -> bool:
        """Whether this is something to look at rather than leave alone."""
        return self.state in (STOP_BREACHED, TARGET_REACHED, FLIPPED,
                              UNSUPPORTED)


def move_pct(position: Position, price: float) -> float:
    """Percent the position is up (+) or down (-), in the position's favour."""
    if position.entry <= 0 or price <= 0:
        return float("nan")
    raw = (price / position.entry - 1.0) * 100.0
    return raw if position.side == LONG else -raw


def assess(position: Position, live_price: "float | None" = None,
           current_direction: "str | None" = None,
           actionable: "bool | None" = None,
           blocker: str = "") -> Status:
    """What is true about `position` now, worst thing first.

    `current_direction` and `actionable` come from the scanner's reading of
    the same symbol, so the caller does the measuring and this stays a pure
    function over the result. Passing neither still gives the level checks,
    which are the ones that matter most.
    """
    lines = []
    if not position.is_valid:
        return Status(position=position, state=UNKNOWN,
                      headline="These levels cannot describe a position",
                      detail=f"{position.side} with entry {position.entry}, "
                             f"stop {position.stop}, target "
                             f"{position.target} - check the numbers.")
    if live_price is None or not (live_price > 0):
        return Status(position=position, state=UNKNOWN,
                      headline="No live price, so nothing can be checked",
                      detail="The market may be closed, or there is no Kite "
                             "session. The levels below are unverified.")
    moved = move_pct(position, live_price)
    lines.append(f"price {live_price:,.2f} against entry "
                 f"{position.entry:,.2f} - {moved:+.2f}% in your favour")

    # Levels first: these are facts about money, not opinions about signals.
    breached = (live_price <= position.stop if position.side == LONG
                else live_price >= position.stop)
    reached = (live_price >= position.target if position.side == LONG
               else live_price <= position.target)
    if breached:
        loss = abs(position.entry - position.stop) / position.entry * 100.0
        rupees = (f" - about {position.risk_per_share * position.quantity:,.0f} "
                  f"rupees on {position.quantity:,} shares"
                  if position.quantity else "")
        return Status(
            position=position, state=STOP_BREACHED,
            headline=f"Price has passed your stop of {position.stop:,.2f}",
            detail=f"That is the level you chose to stop losing at, about "
                   f"{loss:.2f}% from entry{rupees}. It is a statement "
                   f"about where price is, not advice about what to do.",
            move_pct=moved, lines=lines)
    if reached:
        gain = abs(position.target - position.entry) / position.entry * 100.0
        return Status(
            position=position, state=TARGET_REACHED,
            headline=f"Price has reached your target of {position.target:,.2f}",
            detail=f"About {gain:.2f}% from entry - the outcome the position "
                   f"was set up for.",
            move_pct=moved, lines=lines)

    # Then the scanner's current opinion, when the caller supplied one.
    if current_direction and current_direction in (LONG, SHORT):
        if current_direction != position.side:
            return Status(
                position=position, state=FLIPPED,
                headline=f"The scan now reads {current_direction}, and you "
                         f"are {position.side}",
                detail="The gates that pointed one way now point the other. "
                       "This is the scanner disagreeing with the position, "
                       "not merely declining to support it.",
                move_pct=moved, lines=lines)
        lines.append(f"the scan still reads {current_direction}")
    if actionable is False:
        return Status(
            position=position, state=UNSUPPORTED,
            headline="The setup no longer clears the gates",
            detail=(f"Blocked by: {blocker.split(' [')[0]}." if blocker
                    else "One of the checks that justified it now fails."),
            move_pct=moved, lines=lines)
    return Status(
        position=position, state=SUPPORTED,
        headline="Still between your stop and your target",
        detail="Nothing measurable has changed against it.",
        move_pct=moved, lines=lines)


def rank(statuses: list) -> list:
    """Worst first, so what needs attention cannot be scrolled past."""
    return sorted(statuses, key=lambda s: (s.severity, s.position.symbol))


def load(path: "Path | None" = None) -> list:
    """Watched positions, or [] when there are none or the file is broken.

    Never raises: a corrupt watch file must not take the page down, and an
    empty watch list is the correct degraded state.
    """
    target = path or STORE
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Unreadable watch file %s: %s", target.name, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("Watch file %s is not a list; ignoring", target.name)
        return []
    out = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(Position(**{k: row[k] for k in row
                                   if k in Position.__dataclass_fields__}))
        except Exception as exc:
            logger.warning("Skipping unreadable position %r: %s", row, exc)
    return out


def save(positions: list, path: "Path | None" = None) -> int:
    """Persist the watch list atomically. Returns how many were written."""
    target = path or STORE
    payload = [asdict(p) for p in positions]
    temporary = target.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)
    except Exception as exc:
        logger.warning("Could not write %s: %s", target.name, exc)
        return 0
    return len(payload)


def add(position: Position, path: "Path | None" = None) -> list:
    """Add one position, replacing any existing watch on the same symbol+side."""
    existing = [p for p in load(path)
                if not (p.symbol == position.symbol
                        and p.side == position.side)]
    existing.append(position)
    save(existing, path)
    return existing


def remove(symbol: str, side: str, path: "Path | None" = None) -> list:
    """Drop one watch. Returns what is left."""
    wanted = (symbol or "").strip().upper()
    which = (side or "").strip().upper()
    left = [p for p in load(path)
            if not (p.symbol == wanted and p.side == which)]
    save(left, path)
    return left
