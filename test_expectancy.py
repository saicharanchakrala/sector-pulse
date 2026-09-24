"""Expectancy in rupees, and the break-even rate the report was not reading.

WHAT IS PINNED HERE.

THE ORIGINAL BUG. calibrate read `required_win_rate` straight off each
outcome row and outcomes.FIELDS does not carry that column - scan_publish
writes it to the scan log and the resolver drops it. `_float(None) or 0.0`
then made `needed` 0.0% for every band, so `edge` was identical to the raw
hit rate in the one column the report called "the only column worth
reading on its own". Every band looked profitable by the exact width of
its own hit rate.

THE SMALLER COPY OF IT. The first fix derived the rate but still added
`or 0.0` into a band total whose denominator counted the row, so twenty
rateable rows beside twenty unrateable ones printed 18.7% instead of
37.4%. One `rateable` gate now decides for the whole file.

THE FLOOR. MIN_SAMPLE was checked against the number of rows while the
statistic was computed over the rows that parsed, so 37 unresolved rows
plus 3 resolved ones printed a per-trade rupee figure from n=3.

THE POPULATION. The resolver keeps blocked setups on purpose as the
counterfactual, and only 5.4% of the current log was taken - so an
unqualified rupee headline describes a population the gates refused.

THE ADDITION. Expectancy per trade in rupees, beside the R figure. One R
is a constant because the sizer makes it one - quantity is derived so the
rupee risk is the same whatever the stop distance - so the conversion is
a multiplication and not an estimate.
"""
from __future__ import annotations

import itertools
import random
from datetime import date, timedelta

import pytest

import calibrate
import config
import levels

# Every fixture used to set stop_pct to 1.0, which made dividing the cost
# by it a no-op and let a net_r with the division removed survive the
# whole suite. It is deliberately not 1.0 here.
STOP_PCT = 1.5
BREAKEVEN = 0.1224
COST_R = BREAKEVEN / STOP_PCT

# EVERY ROW GETS A SESSION AND AN IN-WINDOW SCAN TIME. The report now
# drops rows it cannot place inside the scan window and needs
# MIN_SESSIONS distinct run_dates, so a fixture with no run_date - which
# is what every row here used to be - would be refused by both and every
# readout test below would pass or fail for the wrong reason. Dates
# rotate through MIN_SESSIONS + 5 weekdays, so any 40 consecutive rows
# span all of them whatever test ran first. See test_calibrate_sessions
# for the floors themselves.
_SESSIONS = [day for day in (date(2026, 7, 1) + timedelta(days=offset)
                             for offset in range(80))
             if day.weekday() < 5][:calibrate.MIN_SESSIONS + 5]
_NEXT_SESSION = itertools.count()


def _row(**over):
    """A 2:1 setup with a realistic Indian intraday round trip."""
    session = _SESSIONS[next(_NEXT_SESSION) % len(_SESSIONS)]
    row = {
        "row_id": f"r{random.random()}",
        "run_date": session.isoformat(), "run_time": "10:30:00",
        "replayed": "",
        "taken": "True",
        "score": "0.65",
        "entry": "1000.0", "stop": "990.0", "target": "1020.0",
        "stop_pct": str(STOP_PCT), "breakeven_pct": str(BREAKEVEN),
        "outcome": "CLOSE", "r_multiple": "0.0",
    }
    row.update({k: str(v) for k, v in over.items()})
    return row


# --- the cost, and the rows it cannot be computed for --------------------

def test_the_cost_is_the_round_trip_over_the_stop_distance() -> None:
    assert calibrate.cost_r(_row()) == pytest.approx(COST_R)


def test_a_cost_beyond_the_whole_risk_is_refused() -> None:
    """THE REGRESSION TEST. stop_pct is stored to three decimals, so a
    stop under a tenth of a percent rounds toward zero and the ratio
    explodes. Real rows exist at stop_pct=0.001 with breakeven=0.0847,
    a cost of 84.7R; 74 of 2,750 rows in the live log exceed 1R. One of
    them moved a 40-row headline from +0.078R to -1.919R.
    """
    assert calibrate.cost_r(
        _row(stop_pct=0.001, breakeven_pct=0.0847)) is None


def test_a_zero_stop_pct_is_refused_rather_than_costed_at_zero() -> None:
    """Returning 0.0 reported that row's GROSS R as net, in the file whose
    whole discipline is that costs are never omitted."""
    assert calibrate.cost_r(_row(stop_pct=0.0)) is None
    assert calibrate.net_r(_row(stop_pct=0.0, r_multiple=2.0)) is None


def test_a_cost_exactly_at_the_ceiling_is_kept() -> None:
    """The boundary belongs to the usable side: costs equal to the whole
    risk make a hopeless trade, not a malformed row."""
    row = _row(stop_pct=1.0, breakeven_pct=calibrate.MAX_COST_R)
    assert calibrate.cost_r(row) == pytest.approx(calibrate.MAX_COST_R)


# --- the reward-to-risk ratio, and why it is not differenced ------------

def test_the_ratio_comes_from_config_not_from_the_prices() -> None:
    """Measured on 2,739 real rows: the constant disagreed with the stored
    property by at most 0.19%, while differencing the 2dp prices reached
    21.01% - and needed a risk floor that discarded 43% of rows to beat
    it. The worst single row was MSCI360 at entry 9.9, stop 9.91, target
    9.89, recovered as 1.0000 against a stored 0.7636.
    """
    assert calibrate.reward_risk(_row()) == config.SCAN_REWARD_RISK


def test_a_tick_scale_stop_is_not_mis_rated() -> None:
    """THE REGRESSION TEST for the rounding. Differencing these prices
    gives 1.0; the row was built at the configured ratio."""
    row = _row(entry=9.9, stop=9.91, target=9.89)
    assert calibrate.reward_risk(row) == config.SCAN_REWARD_RISK


def test_prices_that_cannot_be_rounding_override_the_constant() -> None:
    """A row built under a different reward-risk is better described by
    its own levels - and with a stop this wide the differenced ratio is
    precise, which is exactly when it deserves the vote."""
    row = _row(entry=1000.0, stop=900.0, target=1500.0)   # 5:1
    assert calibrate.reward_risk(row) == pytest.approx(5.0)


def test_a_stop_on_top_of_the_entry_has_no_ratio() -> None:
    """THE MUTANT THIS CATCHES. The earlier test passed stop_pct=0.0 and
    exited at the cost gate, so the risk guard it named was untested and
    removing it was a live ZeroDivisionError. stop_pct is valid here so
    the row reaches the guard under test.
    """
    assert calibrate.reward_risk(
        _row(entry=1000.0, stop=1000.0, stop_pct=1.0)) is None
    assert calibrate.required_from_row(
        _row(entry=1000.0, stop=1000.0, stop_pct=1.0)) is None


# --- the break-even rate -------------------------------------------------

def test_the_required_rate_is_not_zero_on_a_real_row() -> None:
    """THE REGRESSION TEST.

    outcomes.FIELDS has no required_win_rate, so reading it yielded None,
    `or 0.0` made it 0.0, and `needed` printed 0.0% for every band - which
    made `edge` equal to the hit rate and every band look like an edge.

    Asserted EXACTLY rather than as `> 0.3`, because 0.3 is also cleared
    by the cost-free 1/(rr+1) = 0.3333, so the loose form did not tell
    the costed formula from the uncosted one.
    """
    assert "required_win_rate" not in _row(), "fixture must match the schema"
    assert calibrate.required_from_row(_row()) == pytest.approx(
        (1.0 + COST_R) / (config.SCAN_REWARD_RISK + 1.0))


def test_the_derived_rate_equals_the_property_it_replaces() -> None:
    """It has to BE levels.required_win_rate, not merely resemble it.

    The derivation assumes cost_rupees/lot_risk == breakeven_pct/stop_pct.
    cost_rupees is a stored field and stop_pct a computed property, so the
    identity is worth measuring rather than asserting. Over 4,000 built
    levels - including structural stops, capital-capped sizes and
    quantities down to 1 - the worst disagreement was 5.5e-15.
    """
    random.seed(7)
    checked = 0
    for _ in range(600):
        direction = random.choice(["LONG", "SHORT"])
        entry = random.uniform(20.0, 12000.0)
        atr = entry * random.uniform(0.0005, 0.02)
        trade = levels.build_levels(direction, entry, atr,
                                    random.randint(5, 110))
        if trade is None:
            continue
        checked += 1
        # The sizer's contract, now that charges come OUT of the budget
        # rather than on top of it: the whole loss at the stop, move plus
        # round trip, fits capital * risk_pct / 100 on every build.
        assert trade.lot_risk + trade.cost_rupees <= levels.risk_budget(
            config.SCAN_CAPITAL, config.SCAN_RISK_PCT_PER_TRADE), trade
        derived = calibrate.required_from_row({
            "entry": trade.entry, "stop": trade.stop,
            "target": trade.target, "stop_pct": trade.stop_pct,
            "breakeven_pct": trade.breakeven_pct,
        })
        assert derived == pytest.approx(trade.required_win_rate, abs=1e-12)
    assert checked > 500, checked


def test_a_short_is_derived_the_same_way() -> None:
    """The levels sit the other way round, so the distances are absolute."""
    assert calibrate.required_from_row(
        _row(entry=1000.0, stop=1010.0, target=980.0)) == pytest.approx(
        calibrate.required_from_row(_row()))


def test_costs_beyond_the_whole_reward_cap_at_certainty() -> None:
    """levels.required_win_rate caps at 1.0 - the trade cannot break even
    at any win rate - and the derivation has to cap identically. No
    randomised case reached this branch, so it is pinned directly. The
    cost has to stay inside MAX_COST_R to reach the cap at all, so the
    ratio is what is squeezed here.
    """
    row = _row(entry=1000.0, stop=900.0, target=910.0,
               stop_pct=10.0, breakeven_pct=10.0)
    assert calibrate.required_from_row(row) == 1.0


def test_a_row_without_levels_is_skipped_not_guessed() -> None:
    for missing in ("entry", "stop", "target"):
        row = _row()
        row[missing] = ""
        assert calibrate.required_from_row(row) is None


# --- what one R is worth -------------------------------------------------

def test_one_r_is_the_configured_risk() -> None:
    assert calibrate.rupees_per_r() == pytest.approx(
        config.SCAN_CAPITAL * config.SCAN_RISK_PCT_PER_TRADE / 100.0)


def test_one_r_tracks_config_rather_than_a_literal(monkeypatch) -> None:
    monkeypatch.setattr(config, "SCAN_CAPITAL", 250_000.0)
    monkeypatch.setattr(config, "SCAN_RISK_PCT_PER_TRADE", 0.5)
    assert calibrate.rupees_per_r() == pytest.approx(1250.0)


# --- net of costs --------------------------------------------------------

def test_net_r_subtracts_the_round_trip_in_r_not_in_percent() -> None:
    """THE MUTANT THIS CATCHES. Every fixture used to set stop_pct=1.0,
    so `gross - breakeven_pct` and `gross - breakeven_pct/stop_pct` were
    the same number and dropping the division survived the whole suite.
    """
    row = _row(r_multiple=2.0, stop_pct=0.5, breakeven_pct=0.1)
    assert calibrate.net_r(row) == pytest.approx(2.0 - 0.2)
    assert calibrate.net_r(row) != pytest.approx(2.0 - 0.1)


def test_net_r_is_strictly_worse_than_gross() -> None:
    assert calibrate.net_r(_row(r_multiple=-1.0)) < -1.0


def test_an_unresolved_row_has_no_net_r() -> None:
    assert calibrate.net_r(_row(r_multiple="")) is None


# --- expectancy ----------------------------------------------------------

def _population():
    """Six rows: two winners, four losers, hand-computable."""
    return [_row(r_multiple=2.0, outcome="TARGET"),
            _row(r_multiple=2.0, outcome="TARGET"),
            _row(r_multiple=-1.0, outcome="STOP"),
            _row(r_multiple=-1.0, outcome="STOP"),
            _row(r_multiple=-1.0, outcome="STOP"),
            _row(r_multiple=-1.0, outcome="STOP")]


def test_expectancy_is_the_mean_net_r() -> None:
    exp = calibrate.expectancy(_population())
    expected = (2 * (2.0 - COST_R) + 4 * (-1.0 - COST_R)) / 6
    assert exp["expectancy_r"] == pytest.approx(expected)


def test_expectancy_in_rupees_is_the_r_figure_converted() -> None:
    """The video's headline number, in the currency actually risked.

    Converted at one R of PRICE risk, which is the budget less the round
    trip because the sizer fits both inside it - not at the whole budget,
    which would overstate every rupee figure by the charges' share.
    """
    exp = calibrate.expectancy(_population())
    one_r = calibrate.rupees_per_r() / (1.0 + COST_R)
    assert exp["rupees_per_r"] == pytest.approx(one_r)
    assert exp["rupees_per_r"] < calibrate.rupees_per_r()
    assert exp["expectancy_rupees"] == pytest.approx(
        exp["expectancy_r"] * one_r)


def test_a_full_stop_out_costs_the_whole_budget_in_rupees() -> None:
    """-1 R of price plus the charges is exactly the budget, no more."""
    stop = _row(r_multiple=-1.0, outcome="STOP")
    loss = calibrate.net_r(stop) * calibrate.price_r_rupees(stop)
    assert loss == pytest.approx(-calibrate.rupees_per_r())


def test_the_rupee_figure_moves_with_the_configured_risk(monkeypatch) -> None:
    """A MUTATION THIS FILE ORIGINALLY MISSED. Every other test runs on
    the default config, where one R happens to be 1,000 - so hardcoding
    1000 inside expectancy passed all of them. Changing the capital is
    the only thing that tells the constant apart from the literal.
    """
    before = calibrate.expectancy(_population())["expectancy_rupees"]
    monkeypatch.setattr(config, "SCAN_CAPITAL", 400_000.0)
    after = calibrate.expectancy(_population())["expectancy_rupees"]
    assert after == pytest.approx(before * 4.0)


def test_expectancy_matches_the_win_times_win_decomposition() -> None:
    """p*avg_win - (1-p)*|avg_loss| has to agree with the realised mean,
    or one of the two numbers on the readout is wrong."""
    exp = calibrate.expectancy(_population())
    rebuilt = (exp["win_rate"] * exp["avg_win"] +
               (1 - exp["win_rate"]) * exp["avg_loss"])
    assert rebuilt == pytest.approx(exp["expectancy_r"])


def test_the_winner_and_loser_averages_are_reported_separately() -> None:
    """A positive expectancy carried by one outsized win is a different
    fact from a steady one, and only these two columns show which."""
    exp = calibrate.expectancy(_population())
    assert exp["avg_win"] > 0 and exp["avg_loss"] < 0


def test_hit_rate_and_win_rate_are_not_the_same_number() -> None:
    """Most setups reach neither level and exit at the close, so the share
    that made money is not the share that reached the target."""
    rows = [*_population(), _row(r_multiple=0.5, outcome="CLOSE")]
    exp = calibrate.expectancy(rows)
    assert exp["hit_rate"] == pytest.approx(2 / 7)
    assert exp["win_rate"] == pytest.approx(3 / 7)


def test_a_trade_that_exactly_breaks_even_is_not_a_win() -> None:
    """THE MUTANT THIS CATCHES. The earlier version used r_multiple=0.0
    with a non-zero cost, so the net was -0.1224 and the boundary was
    never reached - `>= 0` survived. The cost is zero here so the net is
    exactly 0.0 and the comparison is the thing under test.
    """
    exp = calibrate.expectancy([_row(r_multiple=0.0, breakeven_pct=0.0)])
    assert calibrate.net_r(_row(r_multiple=0.0, breakeven_pct=0.0)) == 0.0
    assert exp["win_rate"] == 0.0


def test_no_resolved_rows_gives_nothing_rather_than_zero() -> None:
    """An empty dict is falsy so the report omits the block; a zero would
    print as a measured expectancy of nothing."""
    assert calibrate.expectancy([_row(r_multiple="")]) == {}


# --- the two break-even bars --------------------------------------------

def test_the_realised_bar_and_the_money_column_agree_in_sign() -> None:
    """THE POINT OF `real`. `needed` assumes every non-winner loses the
    whole stop, so a band could read edge -17.4% beside +0.132 R - the
    report calling one column authoritative and disclaiming it three
    lines later. Measured against the realised bar the two cannot
    disagree, and this sweeps randomised bands to say so.
    """
    random.seed(11)
    checked = 0
    for _ in range(300):
        rows = []
        for _ in range(random.randint(8, 40)):
            if random.random() < random.uniform(0.05, 0.7):
                rows.append(_row(r_multiple=config.SCAN_REWARD_RISK,
                                 outcome="TARGET"))
            else:
                rows.append(_row(r_multiple=random.uniform(-1.0, 0.9),
                                 outcome="CLOSE"))
        entry = calibrate.summarise(rows)[(0.6, 0.8)]
        mean_net = entry["net_r"] / entry["n"]
        edge = entry["hits"] / entry["n"] - calibrate.realised_required(entry)
        if abs(mean_net) < 1e-9:
            continue
        checked += 1
        assert (mean_net > 0) == (edge > 0), (mean_net, edge)
    assert checked > 250, checked


def test_a_band_with_no_losers_falls_back_to_the_full_loss_bar() -> None:
    """Nothing to measure means the assumption is all there is, and it is
    also the only case where the two bars agree."""
    rows = [_row(r_multiple=2.0, outcome="TARGET") for _ in range(5)]
    entry = calibrate.summarise(rows)[(0.6, 0.8)]
    assert calibrate.realised_required(entry) == pytest.approx(
        entry["required"] / entry["n"])


# --- one gate for the whole file ----------------------------------------

def test_an_unrateable_row_never_dilutes_a_band() -> None:
    """THE REGRESSION TEST for the smaller copy of the original bug.
    Adding `or 0.0` into the total while counting the row in the
    denominator halved `needed`: 20 rateable rows read 37.4%, and the
    same 20 beside 20 with a blank stop_pct read 18.7%.
    """
    good = [_row() for _ in range(20)]
    blank = [_row(stop_pct="") for _ in range(20)]
    alone = calibrate.summarise(good)[(0.6, 0.8)]
    mixed = calibrate.summarise(good + blank)[(0.6, 0.8)]
    assert mixed["n"] == alone["n"] == 20
    assert mixed["required"] / mixed["n"] == pytest.approx(
        alone["required"] / alone["n"])


def test_rateable_refuses_what_the_report_cannot_describe() -> None:
    assert calibrate.rateable(_row())
    assert not calibrate.rateable(_row(score=""))
    assert not calibrate.rateable(_row(r_multiple=""))
    assert not calibrate.rateable(_row(stop_pct=0.0))


# --- who the numbers are about ------------------------------------------

def test_disposition_tells_blocked_from_unknown() -> None:
    """A row predating the column is not a blocked row. The live
    outcomes.csv has no `taken` at all, so folding absence into False
    would report an empty taken population as a measured zero."""
    assert calibrate.disposition(_row(taken="True")) == "taken"
    assert calibrate.disposition(_row(taken="False")) == "blocked"
    row = _row()
    del row["taken"]
    assert calibrate.disposition(row) == "unknown"


# --- the readout ---------------------------------------------------------

def _live(n, **over):
    return [_row(**over) for _ in range(n)]


def test_below_the_floor_no_expectancy_is_printed() -> None:
    out = calibrate.report(_live(5, r_multiple=2.0, outcome="TARGET"))
    assert "NOT ENOUGH TO CALIBRATE" in out
    assert "PER TRADE" not in out


def test_the_floor_counts_usable_rows_not_rows() -> None:
    """THE REGRESSION TEST. The gate read len(live) while the statistic
    was computed over the rows that parsed, so 37 unresolved rows beside
    3 resolved ones printed a per-trade rupee figure from n=3.
    """
    rows = _live(37, r_multiple="") + _live(3, r_multiple=2.0,
                                           outcome="TARGET")
    out = calibrate.report(rows)
    assert "NOT ENOUGH TO CALIBRATE" in out
    assert "PER TRADE" not in out
    assert "37 of the 40 live rows" in out, out


def test_below_the_floor_the_conversion_is_still_stated() -> None:
    """What a trade risks is a config constant and knowable today; what
    the average trade returns is a statistic and is not."""
    out = calibrate.report(_live(5))
    assert "Each trade risks Rs 1,000" in out
    assert "including charges" in out
    assert "withheld" in out


def test_past_the_floor_the_expectancy_block_appears() -> None:
    rows = _live(20, r_multiple=2.0, outcome="TARGET") + \
        _live(20, r_multiple=-1.0, outcome="STOP")
    out = calibrate.report(rows)
    assert "EXPECTANCY over 40 live setups" in out
    assert "PER TRADE" in out
    assert "average winner" in out and "average loser" in out


def test_the_needed_column_is_no_longer_zero() -> None:
    """THE REGRESSION TEST for the report itself. `needed` printed 0.0%
    for every band, which made `edge` identical to the hit rate. Asserted
    on the parsed value rather than on a substring, because the old
    substring test was coupled to column widths this change moved.
    """
    rows = _live(40, r_multiple=2.0, outcome="TARGET")
    entry = calibrate.summarise(rows)[(0.6, 0.8)]
    assert entry["required"] / entry["n"] == pytest.approx(
        (1.0 + COST_R) / (config.SCAN_REWARD_RISK + 1.0))
    assert "needed" in calibrate.report(rows)


def test_a_negative_expectancy_is_named_as_such() -> None:
    """The one reading that should stop the strategy being traded."""
    out = calibrate.report(_live(40, r_multiple=-1.0, outcome="STOP"))
    assert "NEGATIVE" in out
    assert "loses it faster" in out


def test_a_positive_expectancy_is_not_flagged() -> None:
    out = calibrate.report(_live(40, r_multiple=2.0, outcome="TARGET"))
    assert "NEGATIVE" not in out


def test_the_taken_population_is_reported_separately() -> None:
    """The resolver keeps blocked setups on purpose as the counterfactual,
    and only 5.4% of the current log was taken - so an unqualified rupee
    headline describes a population the gates refused."""
    rows = _live(40, r_multiple=2.0, outcome="TARGET", taken="True")
    out = calibrate.report(rows)
    assert "EXPECTANCY over 40 TAKEN live setups" in out
    assert "strategy's own expectancy" in out


def test_too_few_taken_rows_says_so_rather_than_implying_the_headline() -> None:
    rows = _live(40, r_multiple=2.0, outcome="TARGET", taken="False")
    out = calibrate.report(rows)
    assert "NO EXPECTANCY FOR THE TAKEN ROWS: 0" in out
    assert "gates REFUSED" in out


def test_the_population_split_is_stated_before_any_number() -> None:
    rows = _live(30, taken="True", r_multiple=2.0, outcome="TARGET") + \
        _live(10, taken="False", r_multiple=-1.0, outcome="STOP")
    out = calibrate.report(rows)
    assert "30 were TAKEN by the gates and 10 were blocked" in out


def test_excluded_rows_are_counted_out_loud() -> None:
    """Silently dropping them is how a ragged population hides."""
    rows = _live(40, r_multiple=2.0, outcome="TARGET") + \
        _live(3, stop_pct=0.0)
    out = calibrate.report(rows)
    assert "3 live rows could not be costed" in out


def test_replayed_rows_never_reach_the_expectancy() -> None:
    """A replay is a reconstruction. Pooling it with live rows is how a
    first track record came out 100% replayed and was read as real."""
    rows = _live(40, r_multiple=-1.0, outcome="STOP") + \
        [_row(r_multiple=2.0, outcome="TARGET", replayed="True")
         for _ in range(200)]
    out = calibrate.report(rows)
    assert "EXPECTANCY over 40 live setups" in out
    assert "NEGATIVE" in out, "the replayed winners leaked into the mean"
