"""Store staleness is counted in sessions, not calendar days.

WHAT WAS WRONG. live_bars.from_store refused a store more than
STORE_MAX_STALE_DAYS behind the window's end, measured with calendar
arithmetic. The market does not trade at the weekend, so a store folded
on Friday evening - holding Friday's session, completely current - read
as three days behind on Monday, past the one session of slack, and was
declined. The feed then refetched 17 days for ~2,500 symbols from Kite,
836 seconds, on the one morning of the week when it had everything it
needed.

Measured 2026-09-21, a Monday: the 3-minute store was refused for being
"3 days behind" the window ending Sunday.

Holidays are deliberately not modelled. A store spanning a
holiday-shortened week reads as one session further behind than it truly
is, which costs a refetch rather than a wrong answer.
"""
from __future__ import annotations

from datetime import date

import pytest

import live_bars

# 2026-09-14 Mon, 15 Tue, 16 Wed, 17 Thu, 18 Fri, 19 Sat, 20 Sun, 21 Mon
MON = date(2026, 9, 14)
THU = date(2026, 9, 17)
FRI = date(2026, 9, 18)
SAT = date(2026, 9, 19)
SUN = date(2026, 9, 20)
NEXT_MON = date(2026, 9, 21)


def test_a_friday_store_is_current_on_monday() -> None:
    """THE REGRESSION TEST. Calendar arithmetic made this 3."""
    assert live_bars.sessions_behind(FRI, NEXT_MON) == 1
    assert live_bars.sessions_behind(FRI, SUN) == 0
    assert live_bars.sessions_behind(FRI, SAT) == 0


def test_a_friday_store_read_on_monday_is_within_the_slack() -> None:
    """The whole point: it must not be refused."""
    assert live_bars.sessions_behind(FRI, SUN) <= \
        live_bars.STORE_MAX_STALE_DAYS


def test_consecutive_weekdays_count_one_each() -> None:
    assert live_bars.sessions_behind(MON, date(2026, 9, 15)) == 1
    assert live_bars.sessions_behind(MON, date(2026, 9, 16)) == 2
    assert live_bars.sessions_behind(MON, THU) == 3


def test_the_weekend_itself_counts_for_nothing() -> None:
    assert live_bars.sessions_behind(THU, FRI) == 1
    assert live_bars.sessions_behind(THU, SAT) == 1
    assert live_bars.sessions_behind(THU, SUN) == 1
    assert live_bars.sessions_behind(THU, NEXT_MON) == 2


def test_a_store_that_reaches_the_end_is_not_behind() -> None:
    assert live_bars.sessions_behind(FRI, FRI) == 0


def test_a_store_ahead_of_the_window_is_not_behind() -> None:
    """A store folded after the close already holds today."""
    assert live_bars.sessions_behind(NEXT_MON, FRI) == 0


@pytest.mark.parametrize("newest,end", [(None, FRI), (FRI, None),
                                        (None, None)])
def test_a_missing_bound_is_zero_rather_than_an_error(newest, end) -> None:
    assert live_bars.sessions_behind(newest, end) == 0


def test_a_genuinely_stale_store_is_still_caught() -> None:
    """The slack must not have become unlimited.

    Five sessions behind is what the store actually was on 2026-09-21,
    and that has to keep being refused.
    """
    behind = live_bars.sessions_behind(date(2026, 9, 11), SUN)
    assert behind == 5
    assert behind > live_bars.STORE_MAX_STALE_DAYS


def test_a_long_gap_counts_only_weekdays() -> None:
    """Three full weeks is fifteen sessions, not twenty-one days."""
    assert live_bars.sessions_behind(date(2026, 8, 31),
                                     date(2026, 9, 18)) == 14
