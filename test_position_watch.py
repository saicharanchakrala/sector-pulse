"""Tests for the position watch.

The case these exist for, in full. On 2026-09-10 the scan showed HDFCLIFE
SHORT at 511.20, stop 516.00, target 501.60 - blocked by one gate of
eleven (relative strength +3.06pp, so shorting an outperformer failed),
but displayed with full levels. Over two hours the scanner's own view went
SHORT, then NO SETUP, then LONG, and never said the earlier signal was
invalidated. The stop went through inside ONE five-minute bar: 515.50 to
520.55 between 12:00 and 12:05. The position was closed at 520.00 for
-1.68%.

Every state below is checked against that sequence, because a watch that
would not have caught it is not worth having.
"""
from __future__ import annotations

import json

import pytest

import position_watch as pw

# The real trade, so the tests are anchored on something that happened.
HDFCLIFE = dict(symbol="HDFCLIFE", side=pw.SHORT, entry=511.20,
                stop=516.00, target=501.60, quantity=500)


def test_the_hdfclife_stop_breach_is_caught() -> None:
    # 12:00 bar: open 515.50, high 520.55. Anywhere past 516 is a breach.
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(held, live_price=520.15)
    assert status.state == pw.STOP_BREACHED
    assert "516" in status.headline
    assert status.needs_attention
    # The rupee figure only appears when a quantity was recorded.
    assert "2,400" in status.detail


def test_a_breach_outranks_a_flipped_direction() -> None:
    # Both were true at 12:05 on HDFCLIFE. The money already at risk
    # beyond what was accepted is the more urgent fact.
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(held, live_price=520.15, current_direction=pw.LONG,
                       actionable=True)
    assert status.state == pw.STOP_BREACHED


def test_the_direction_flip_is_caught_before_the_stop_goes() -> None:
    # At 12:05 the scan read LONG while the position was SHORT. Had price
    # not yet passed 516, this is the warning that mattered.
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(held, live_price=514.00, current_direction=pw.LONG)
    assert status.state == pw.FLIPPED
    assert "LONG" in status.headline and "SHORT" in status.headline
    assert status.needs_attention


def test_gates_failing_is_reported_and_names_the_blocker() -> None:
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(
        held, live_price=512.00, current_direction=pw.SHORT, actionable=False,
        blocker="relative strength +3.06pp vs Nifty, needs to underperform [FAIL]")
    assert status.state == pw.UNSUPPORTED
    assert "relative strength" in status.detail
    # The [FAIL] marker is display noise and should not reach the sentence.
    assert "[FAIL]" not in status.detail


def test_a_quiet_position_says_so_without_drama() -> None:
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(held, live_price=509.00, current_direction=pw.SHORT,
                       actionable=True)
    assert status.state == pw.SUPPORTED
    assert not status.needs_attention


def test_the_target_is_caught_for_a_short() -> None:
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(held, live_price=501.00)
    assert status.state == pw.TARGET_REACHED
    assert status.needs_attention


def test_the_same_states_work_for_a_long() -> None:
    held = pw.Position(symbol="RELIANCE", side=pw.LONG, entry=1274.50,
                       stop=1251.48, target=1320.54)
    assert pw.assess(held, live_price=1250.00).state == pw.STOP_BREACHED
    assert pw.assess(held, live_price=1325.00).state == pw.TARGET_REACHED
    assert pw.assess(held, live_price=1280.00,
                     current_direction=pw.SHORT).state == pw.FLIPPED
    assert pw.assess(held, live_price=1280.00, current_direction=pw.LONG,
                     actionable=True).state == pw.SUPPORTED


def test_the_favour_direction_is_signed_per_side() -> None:
    # A short is UP when price falls. Getting this backwards would report
    # every losing short as a winner.
    short = pw.Position(**HDFCLIFE)
    assert pw.move_pct(short, 500.0) > 0
    assert pw.move_pct(short, 520.0) < 0
    long = pw.Position(symbol="X", side=pw.LONG, entry=100.0, stop=95.0,
                       target=110.0)
    assert pw.move_pct(long, 105.0) > 0
    assert pw.move_pct(long, 95.0) < 0


# --- refusing to watch nonsense ------------------------------------------

def test_levels_that_cannot_describe_a_position_are_refused() -> None:
    # A long whose stop sits ABOVE its entry is a typo, and watching it
    # would report a permanent breach from the first tick.
    bad = pw.Position(symbol="X", side=pw.LONG, entry=100.0, stop=110.0,
                      target=120.0)
    assert not bad.is_valid
    assert pw.assess(bad, live_price=105.0).state == pw.UNKNOWN
    worse = pw.Position(symbol="X", side=pw.SHORT, entry=100.0, stop=90.0,
                        target=80.0)
    assert not worse.is_valid


def test_a_valid_position_passes_its_own_check() -> None:
    assert pw.Position(**HDFCLIFE).is_valid
    assert pw.Position(symbol="X", side=pw.LONG, entry=100.0, stop=95.0,
                       target=110.0).is_valid


def test_an_unknown_side_is_refused() -> None:
    assert not pw.Position(symbol="X", side="MAYBE", entry=100.0, stop=95.0,
                           target=110.0).is_valid


def test_no_live_price_means_unverified_rather_than_fine() -> None:
    # Silence here would read as "nothing wrong", which is the failure this
    # module exists to prevent.
    held = pw.Position(**HDFCLIFE)
    for price in (None, 0.0, -5.0):
        status = pw.assess(held, live_price=price)
        assert status.state == pw.UNKNOWN
        assert "no live price" in status.headline.lower()


# --- severity ordering ---------------------------------------------------

def test_the_worst_position_sorts_first() -> None:
    quiet = pw.assess(pw.Position(symbol="A", side=pw.LONG, entry=100.0,
                                  stop=95.0, target=110.0),
                      live_price=101.0, current_direction=pw.LONG,
                      actionable=True)
    stopped = pw.assess(pw.Position(symbol="B", side=pw.LONG, entry=100.0,
                                    stop=95.0, target=110.0),
                        live_price=94.0)
    flipped = pw.assess(pw.Position(symbol="C", side=pw.LONG, entry=100.0,
                                    stop=95.0, target=110.0),
                        live_price=101.0, current_direction=pw.SHORT)
    order = [s.position.symbol for s in pw.rank([quiet, flipped, stopped])]
    assert order == ["B", "C", "A"]


def test_the_assessed_price_is_carried_on_the_status() -> None:
    # The table shows the verdict and the price side by side; taking the
    # price from a second fetch could show a number the state contradicts.
    held = pw.Position(**HDFCLIFE)
    assert pw.assess(held, live_price=509.0).price == pytest.approx(509.0)
    stopped = pw.assess(held, live_price=520.15)
    assert stopped.price == pytest.approx(520.15)
    # No price means no price, not zero.
    blind = pw.assess(held, live_price=None)
    assert blind.price != blind.price


# --- what gets announced, and how often ----------------------------------

def _status(symbol, price, side=pw.SHORT):
    return pw.assess(pw.Position(symbol=symbol, side=side, entry=511.20,
                                 stop=516.00, target=501.60),
                     live_price=price)


def test_only_the_states_worth_interrupting_for_are_announced() -> None:
    quiet = pw.assess(pw.Position(**HDFCLIFE), live_price=509.0,
                      current_direction=pw.SHORT, actionable=True)
    unsupported = pw.assess(pw.Position(**HDFCLIFE), live_price=509.0,
                            current_direction=pw.SHORT, actionable=False)
    blind = pw.assess(pw.Position(**HDFCLIFE), live_price=None)
    fresh, said = pw.alerts_to_announce([quiet, unsupported, blind], set())
    assert fresh == [] and said == set()


def test_a_persisting_breach_is_announced_once() -> None:
    fresh, said = pw.alerts_to_announce([_status("HDFCLIFE", 520.15)], set())
    assert [s.state for s in fresh] == [pw.STOP_BREACHED]
    # Still breached, a different price, twenty seconds later.
    again, said = pw.alerts_to_announce([_status("HDFCLIFE", 523.40)], said)
    assert again == [], "the same breach was announced twice"


def test_a_state_that_clears_and_returns_is_announced_again() -> None:
    # Announce the breach, widen the stop, watch price take out the new
    # one. Remembering keys forever made that second breach SILENT, which
    # is worse than announcing it twice.
    fresh, said = pw.alerts_to_announce([_status("HDFCLIFE", 520.15)], set())
    assert [s.state for s in fresh] == [pw.STOP_BREACHED]
    widened = pw.assess(pw.Position(symbol="HDFCLIFE", side=pw.SHORT,
                                    entry=511.20, stop=525.00, target=501.60),
                        live_price=520.15)
    assert widened.state != pw.STOP_BREACHED
    _, said = pw.alerts_to_announce([widened], said)
    assert said == set(), "the cleared state was still remembered"
    breached_again = pw.assess(pw.Position(symbol="HDFCLIFE", side=pw.SHORT,
                                           entry=511.20, stop=525.00,
                                           target=501.60),
                               live_price=526.00)
    fresh, _ = pw.alerts_to_announce([breached_again], said)
    assert [s.state for s in fresh] == [pw.STOP_BREACHED]


def test_a_removed_position_does_not_stay_suppressed() -> None:
    fresh, said = pw.alerts_to_announce([_status("HDFCLIFE", 520.15)], set())
    assert fresh
    _, said = pw.alerts_to_announce([], said)     # stopped watching it
    assert said == set()
    fresh, _ = pw.alerts_to_announce([_status("HDFCLIFE", 520.15)], said)
    assert fresh, "re-adding the position left it silent"


def test_a_genuinely_new_state_is_announced_again() -> None:
    fresh, said = pw.alerts_to_announce([_status("HDFCLIFE", 503.50)], set())
    assert [s.state for s in fresh] == [pw.NEAR_TARGET]
    fresh, _ = pw.alerts_to_announce([_status("HDFCLIFE", 501.00)], said)
    assert [s.state for s in fresh] == [pw.TARGET_REACHED]


def test_a_batch_of_alerts_comes_worst_first() -> None:
    # The caller plays one sound for the batch, so the head of the list
    # decides which one.
    batch = [_status("A", 501.00), _status("B", 520.15),
             _status("C", 503.50)]
    fresh, _ = pw.alerts_to_announce(batch, set())
    assert [s.state for s in fresh] == [pw.STOP_BREACHED, pw.TARGET_REACHED,
                                        pw.NEAR_TARGET]


def test_a_reversal_outranks_being_nearly_at_target() -> None:
    # THE BUG THIS PINS. Nearing the target returned before the scanner
    # check and outranked it, so a position four fifths of the way to
    # target whose scan had flipped reported NEARING TARGET and the flip
    # appeared nowhere at all - not the state, not the headline, not the
    # lines. Which is the HDFCLIFE failure, inside the module written to
    # prevent it.
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(held, live_price=503.50, current_direction=pw.LONG)
    assert status.state == pw.FLIPPED
    assert "LONG" in status.headline
    # And the near-target progress is not lost, it is demoted to a line.
    assert any("of the way from entry to target" in line
               for line in status.lines)
    # Without the flip it is still NEARING TARGET.
    assert pw.assess(held, live_price=503.50,
                     current_direction=pw.SHORT).state == pw.NEAR_TARGET


def test_a_failed_gate_does_not_hide_a_near_target() -> None:
    # The other side of the same ordering: a merely unsupported setup is
    # less urgent than being about to reach the target.
    held = pw.Position(**HDFCLIFE)
    status = pw.assess(held, live_price=503.50, current_direction=pw.SHORT,
                       actionable=False, blocker="relative strength [FAIL]")
    assert status.state == pw.NEAR_TARGET


def test_every_announced_state_has_something_to_say() -> None:
    # An announced state with an empty sentence would chime and then sit
    # there in silence.
    for status in (_status("A", 520.15), _status("B", 501.00),
                   _status("C", 503.50),
                   pw.assess(pw.Position(**HDFCLIFE), live_price=512.0,
                             current_direction=pw.LONG)):
        assert status.should_announce
        assert status.spoken.startswith(status.position.symbol)
        assert len(status.spoken) > len(status.position.symbol) + 3


# --- persistence ---------------------------------------------------------

def test_positions_survive_a_round_trip(tmp_path) -> None:
    path = tmp_path / "watch.json"
    held = pw.Position(**HDFCLIFE)
    assert pw.save([held], path) == 1
    back = pw.load(path)
    assert len(back) == 1
    assert back[0].symbol == "HDFCLIFE" and back[0].side == pw.SHORT
    assert back[0].entry == pytest.approx(511.20)
    assert back[0].taken_at, "the timestamp must persist"


def test_adding_replaces_the_same_symbol_and_side(tmp_path) -> None:
    path = tmp_path / "watch.json"
    pw.add(pw.Position(**HDFCLIFE), path)
    revised = dict(HDFCLIFE, entry=511.40, stop=515.00)
    left = pw.add(pw.Position(**revised), path)
    assert len(left) == 1, "a second watch on the same position was created"
    assert pw.load(path)[0].stop == pytest.approx(515.00)
    # The opposite side is a different position, not a replacement.
    pw.add(pw.Position(symbol="HDFCLIFE", side=pw.LONG, entry=520.0,
                       stop=515.0, target=530.0), path)
    assert len(pw.load(path)) == 2


def test_the_previous_list_is_kept_beside_the_store(tmp_path) -> None:
    # This file is the only record of what you hold. One generation of
    # history turns a wipe into a file rename.
    path = tmp_path / "watch.json"
    pw.add(pw.Position(**HDFCLIFE), path)
    pw.add(pw.Position(symbol="RELIANCE", side=pw.LONG, entry=1274.5,
                       stop=1251.5, target=1320.5), path)
    previous = path.with_suffix(".prev.json")
    assert previous.exists()
    assert [p.symbol for p in pw.load(previous)] == ["HDFCLIFE"]
    assert len(pw.load(path)) == 2


def test_removing_something_not_there_does_not_rewrite_the_file(
        tmp_path) -> None:
    path = tmp_path / "watch.json"
    pw.add(pw.Position(**HDFCLIFE), path)
    stamp = path.stat().st_mtime_ns
    left = pw.remove("NOTHELD", "LONG", path)
    assert [p.symbol for p in left] == ["HDFCLIFE"]
    assert path.stat().st_mtime_ns == stamp, "the file was rewritten anyway"


def test_removing_leaves_the_others(tmp_path) -> None:
    path = tmp_path / "watch.json"
    pw.add(pw.Position(**HDFCLIFE), path)
    pw.add(pw.Position(symbol="RELIANCE", side=pw.LONG, entry=1274.5,
                       stop=1251.5, target=1320.5), path)
    left = pw.remove("hdfclife", "short", path)      # case-insensitive
    assert [p.symbol for p in left] == ["RELIANCE"]


def test_a_missing_or_corrupt_file_is_empty_rather_than_fatal(tmp_path) -> None:
    # A broken watch file must not take the page down.
    assert pw.load(tmp_path / "absent.json") == []
    broken = tmp_path / "broken.json"
    broken.write_text("{not json at all", encoding="utf-8")
    assert pw.load(broken) == []
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"symbol": "X"}), encoding="utf-8")
    assert pw.load(wrong) == []


def test_unreadable_rows_are_skipped_not_fatal(tmp_path) -> None:
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps([
        {"symbol": "GOOD", "side": "LONG", "entry": 100.0, "stop": 95.0,
         "target": 110.0},
        {"nonsense": True},
        "not even a dict",
    ]), encoding="utf-8")
    back = pw.load(path)
    assert [p.symbol for p in back] == ["GOOD"]


def test_unknown_keys_in_the_file_do_not_break_loading(tmp_path) -> None:
    # A file written by a later version must not break an earlier one.
    path = tmp_path / "future.json"
    path.write_text(json.dumps([
        {"symbol": "X", "side": "LONG", "entry": 100.0, "stop": 95.0,
         "target": 110.0, "some_future_field": 42},
    ]), encoding="utf-8")
    assert [p.symbol for p in pw.load(path)] == ["X"]


# --- turning a table row into a position ---------------------------------

def _row(**cells):
    import pandas as pd
    return pd.Series(cells)


def test_a_scan_row_becomes_the_position_it_describes() -> None:
    held = pw.position_from_row(_row(**{
        "Symbol": "HDFCLIFE", "Side": "SHORT", "Entry price": 511.20,
        "Stop loss at": 516.00, "Exit price": 501.60, "Qty": 500}))
    assert held is not None
    assert (held.symbol, held.side) == ("HDFCLIFE", pw.SHORT)
    assert held.entry == pytest.approx(511.20)
    assert held.stop == pytest.approx(516.00)
    assert held.target == pytest.approx(501.60)
    assert held.quantity == 500
    assert held.is_valid


def test_a_horizon_row_uses_its_own_column_names() -> None:
    # The horizon tables say View and Price where the scan says Side and
    # Entry price. Both must work, or the action appears on one table only.
    held = pw.position_from_row(_row(**{
        "Symbol": "RELIANCE", "View": "LONG", "Price": 1274.50,
        "Stop loss at": 1251.48, "Exit price": 1320.54}))
    assert held is not None and held.side == pw.LONG
    assert held.entry == pytest.approx(1274.50)
    assert held.quantity == 0


def test_the_lookup_supplies_the_symbol_and_price_the_row_lacks() -> None:
    # One row per horizon for a single instrument, so neither the symbol
    # nor the price is in the row itself.
    held = pw.position_from_row(
        _row(**{"Horizon": "short", "View": "LONG", "Verdict": "NO BUY",
                "Stop loss at": 95.0, "Exit price": 110.0}),
        symbol="tcs", entry=100.0, note="from the short horizon")
    assert held is not None
    assert held.symbol == "TCS"                  # upper-cased like any other
    assert held.entry == pytest.approx(100.0)
    assert held.note == "from the short horizon"


def test_a_row_with_nothing_to_watch_is_refused() -> None:
    # No levels, no side, or a side that is not one. Each would produce a
    # watch that reports nonsense from its first tick.
    assert pw.position_from_row(_row(**{
        "Symbol": "X", "Side": "NO SETUP", "Entry price": 100.0,
        "Stop loss at": 95.0, "Exit price": 110.0})) is None
    assert pw.position_from_row(_row(**{
        "Symbol": "X", "Side": "LONG", "Entry price": 100.0})) is None
    assert pw.position_from_row(_row(**{
        "Side": "LONG", "Entry price": 100.0, "Stop loss at": 95.0,
        "Exit price": 110.0})) is None
    assert pw.position_from_row(_row(**{
        "Symbol": "X", "Side": "LONG", "Stop loss at": 95.0,
        "Exit price": 110.0})) is None


def test_blank_and_nan_cells_count_as_absent() -> None:
    import numpy as np
    assert pw.position_from_row(_row(**{
        "Symbol": "X", "Side": "LONG", "Entry price": 100.0,
        "Stop loss at": np.nan, "Exit price": 110.0})) is None
    assert pw.position_from_row(_row(**{
        "Symbol": "X", "Side": "", "Entry price": 100.0,
        "Stop loss at": 95.0, "Exit price": 110.0})) is None


def test_impossible_levels_come_back_for_the_caller_to_object_to() -> None:
    # Refusing here would leave the UI with nothing to explain. The row is
    # returned and Position.is_valid says what is wrong with it.
    held = pw.position_from_row(_row(**{
        "Symbol": "X", "Side": "LONG", "Entry price": 100.0,
        "Stop loss at": 110.0, "Exit price": 120.0}))
    assert held is not None and not held.is_valid
