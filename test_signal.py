"""Offline unit tests for the signal side: news archive and the liquidity gate.

The existing smoke_test.py exercises the whole pipeline but needs the network.
These run in milliseconds with no network, so they can gate every commit.

Runs under pytest, or standalone with `python test_signal.py`.
"""
from __future__ import annotations

import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import config
import decision
import news_archive
from intraday import IntradaySnapshot
from models import NewsItem
from profiles import get_profile

UTC_NOW = datetime.now(timezone.utc)


def _raises(exc_type: type[BaseException], func: Callable[..., object],
            *args: object, **kwargs: object) -> None:
    """Assert that calling func raises exc_type."""
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} from {func.__name__}")


def _item(idx: int, when: "datetime | None" = None) -> NewsItem:
    """Build one synthetic NewsItem."""
    return NewsItem(
        title=f"Headline {idx}",
        summary=f"Summary body {idx}",
        link=f"https://example.com/story-{idx}",
        source="Test Feed",
        published=when or (UTC_NOW - timedelta(hours=idx)),
    )


def _snapshot(ticker: str, price: float, volume: float,
              day_change: float = 1.0, last_hour: float = 0.5) -> IntradaySnapshot:
    """Build one synthetic intraday snapshot."""
    return IntradaySnapshot(
        ticker=ticker, last_price=price, asof=UTC_NOW,
        day_change_pct=day_change, last_hour_change_pct=last_hour,
        range_position=0.8, avg_volume_20d=volume,
    )


# --- news archive --------------------------------------------------------

def test_archive_writes_and_reports_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        items = [_item(i) for i in range(1, 6)]
        assert news_archive.archive_news(items, date(2026, 9, 4), tmp) == 5
        assert news_archive.archive_path(date(2026, 9, 4), tmp).exists()


def test_archive_is_idempotent_across_reruns() -> None:
    # Running the signal several times a day must not duplicate headlines.
    with tempfile.TemporaryDirectory() as tmp:
        items = [_item(i) for i in range(1, 4)]
        assert news_archive.archive_news(items, date(2026, 9, 4), tmp) == 3
        assert news_archive.archive_news(items, date(2026, 9, 4), tmp) == 0
        extra = items + [_item(99)]
        assert news_archive.archive_news(extra, date(2026, 9, 4), tmp) == 1
        assert len(news_archive.load_archived_news(date(2026, 9, 4), tmp)) == 4


def test_archive_round_trips_every_field() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        published = datetime(2026, 9, 4, 9, 30, tzinfo=timezone.utc)
        original = NewsItem(title="Metal stocks rally", summary="Steel demand up",
                            link="https://example.com/metal", source="ET Markets",
                            published=published)
        news_archive.archive_news([original], date(2026, 9, 4), tmp)
        back = news_archive.load_archived_news(date(2026, 9, 4), tmp)
    assert len(back) == 1
    assert back[0].title == original.title
    assert back[0].summary == original.summary
    assert back[0].link == original.link
    assert back[0].source == original.source
    assert back[0].published == published


def test_archive_replays_into_the_analyzer() -> None:
    # This is the whole point of the archive: a past day must be scoreable.
    import analyzer
    with tempfile.TemporaryDirectory() as tmp:
        items = [
            NewsItem(title="Metal stocks surge on strong steel demand",
                     summary="Tata Steel and JSW rally as prices firm.",
                     link=f"https://example.com/metal-{i}", source="Test",
                     published=UTC_NOW - timedelta(hours=1))
            for i in range(4)
        ]
        news_archive.archive_news(items, date(2026, 9, 4), tmp)
        replayed = news_archive.load_archived_news(date(2026, 9, 4), tmp)
        scores = analyzer.analyze(replayed, {}, profile=get_profile("IN"))
    by_sector = {s.sector: s for s in scores}
    assert by_sector["Metal"].news_count > 0, "replayed archive must score sectors"


def test_load_returns_empty_for_a_missing_day() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert news_archive.load_archived_news(date(2026, 1, 1), tmp) == []
        assert news_archive.archived_days(tmp) == []


def test_archive_skips_corrupt_rows_and_keeps_good_ones() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = news_archive.archive_path(date(2026, 9, 4), tmp)
        path.parent.mkdir(parents=True, exist_ok=True)
        good = ('{"fetched_at":"x","title":"T","summary":"S",'
                '"link":"https://e.com/1","source":"F",'
                '"published":"2026-09-04T09:00:00+00:00"}')
        path.write_text("\n".join(["not json at all", good, "", "{}"]) + "\n",
                        encoding="utf-8")
        items = news_archive.load_archived_news(date(2026, 9, 4), tmp)
    assert len(items) == 1
    assert items[0].title == "T"


def test_empty_fetch_archives_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert news_archive.archive_news([], date(2026, 9, 4), tmp) == 0
        assert not news_archive.archive_path(date(2026, 9, 4), tmp).exists()


def test_archived_days_lists_in_order_and_ignores_junk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for day in (date(2026, 9, 3), date(2026, 9, 1), date(2026, 9, 2)):
            news_archive.archive_news([_item(1)], day, tmp)
        (Path(tmp) / "notadate.jsonl").write_text("{}\n", encoding="utf-8")
        assert news_archive.archived_days(tmp) == [
            date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]


# --- liquidity gate ------------------------------------------------------

def _decide_with(snapshots: dict) -> dict:
    """Run the engine offline over synthetic snapshots; key results by sector."""
    signals = decision.decide([], {}, snapshots, get_profile("IN"))
    return {s.sector: s for s in signals}


def test_liquidity_gate_uses_rupee_turnover_not_unit_count() -> None:
    # INFRABEES trades near 958/unit. Under the old 50,000-UNIT floor it needed
    # 4.79 crore of turnover to qualify, while an 11-rupee ETF needed 5.7 lakh:
    # an 84x inconsistency. Turnover is comparable across price levels.
    # The ticker is read from the profile: when it was hardcoded, retargeting
    # Infrastructure left the snapshot unmatched, so the sector got no
    # snapshot at all and illiquid defaulted to False. The assertion below
    # still passed while testing nothing.
    ticker = get_profile("IN").trade_etfs["Infrastructure"]
    dear_but_liquid = _snapshot(ticker, price=958.0, volume=9_000.0)
    turnover = 958.0 * 9_000.0
    assert turnover > config.SIGNAL_MIN_AVG_TURNOVER
    assert dear_but_liquid.avg_volume_20d < 50_000, "would fail the old unit gate"
    by_sector = _decide_with({ticker: dear_but_liquid})
    infra = by_sector["Infrastructure"]
    assert infra.intraday is not None, "snapshot must reach the gate to test it"
    assert infra.illiquid is False


def test_genuinely_thin_etf_is_still_illiquid() -> None:
    # Read from the profile, not written as a literal. Retargeting a
    # sector's ETF once stranded literals like this and the gate under test
    # silently stopped being exercised at all.
    energy = get_profile("IN").trade_etfs["Energy"]
    thin = _snapshot(energy, price=11.40, volume=20_000.0)
    assert 11.40 * 20_000.0 < config.SIGNAL_MIN_AVG_TURNOVER
    by_sector = _decide_with({energy: thin})
    assert by_sector["Energy"].illiquid is True


def test_illiquid_blocks_a_buy_that_would_otherwise_pass() -> None:
    # Same sector, same price action; only turnover differs.
    metal = get_profile("IN").trade_etfs["Metal"]
    liquid = _snapshot(metal, price=13.40, volume=8_000_000.0)
    thin = _snapshot(metal, price=13.40, volume=1_000.0)
    assert _decide_with({metal: liquid})["Metal"].illiquid is False
    assert _decide_with({metal: thin})["Metal"].illiquid is True


def test_liquidity_reason_is_reported_in_rupees() -> None:
    metal = get_profile("IN").trade_etfs["Metal"]
    by_sector = _decide_with(
        {metal: _snapshot(metal, 13.40, 8_000_000.0)})
    reasons = " ".join(by_sector["Metal"].reasons)
    assert "turnover" in reasons
    assert "107,200,000" in reasons, reasons


def test_explanation_mentions_turnover_not_units() -> None:
    metal = get_profile("IN").trade_etfs["Metal"]
    by_sector = _decide_with(
        {metal: _snapshot(metal, 13.40, 1_000.0)})
    text = decision.explain(by_sector["Metal"])
    assert "rupees a day" in text, text
    assert "units a day" not in text, text


def test_a_sector_with_no_snapshot_is_never_a_buy() -> None:
    by_sector = _decide_with({})
    assert all(s.action == decision.ACTION_NO_BUY for s in by_sector.values())
    profile = get_profile("IN")
    untradeable = sorted(set(profile.sectors) - set(profile.trade_etfs))
    assert untradeable, "profile must still exercise the no-ETF branch"
    for sector in untradeable:
        assert by_sector[sector].etf is None


# --- momentum gate must fail closed ------------------------------------

def _gates(momentum_score: float, momentum_weight: float) -> tuple[bool, str]:
    """Run the buy gates with everything but momentum comfortably passing."""
    from decision import _buy_gates
    snap = _snapshot("X.NS", price=100.0, volume=5_000_000.0)
    ok, reasons = _buy_gates("X.NS", snap, 0.5, 5, config.SIGNAL_MIN_ARTICLES,
                             momentum_score, momentum_weight, False)
    return ok, next(r for r in reasons if "momentum" in r)


def test_absent_momentum_fails_the_gate_rather_than_passing() -> None:
    # The bug: a missing score arrived as 0.0, and 0.0 >= -0.2, so a yfinance
    # outage made the falling-knife check PASS. A blocking gate must fail closed.
    ok, reason = _gates(0.0, 0.0)
    assert ok is False, "no momentum data must not clear the gate"
    assert "unavailable" in reason, reason


def test_momentum_on_too_little_window_weight_fails() -> None:
    # 9 of 12 Nifty indices were silently scoring on the 63d window alone,
    # which is 20% of the configured weight.
    ok, reason = _gates(0.5, 0.2)
    assert ok is False, "a fifth of the windows is not a measured trend"
    assert "20%" in reason and "below" in reason, reason


def test_full_window_weight_gates_on_the_score_itself() -> None:
    assert _gates(0.5, 1.0)[0] is True
    assert _gates(-0.5, 1.0)[0] is False, "a real falling knife must still fail"
    ok, reason = _gates(0.0, 1.0)
    assert ok is True
    assert "100% of window weight" in reason, reason


def test_threshold_boundary_is_inclusive() -> None:
    assert _gates(0.0, config.SIGNAL_MIN_MOMENTUM_WEIGHT)[0] is True
    assert _gates(0.0, config.SIGNAL_MIN_MOMENTUM_WEIGHT - 0.01)[0] is False
    assert _gates(config.SIGNAL_MIN_MOMENTUM, 1.0)[0] is True


def test_decide_reports_the_window_weight_it_used() -> None:
    from models import SectorMomentum
    full = SectorMomentum(sector="Metal", etf="METALIETF",
                          returns={"5d": 1.0, "21d": 2.0, "63d": 3.0}, score=0.4)
    thin = SectorMomentum(sector="Metal", etf="METALIETF",
                          returns={"63d": 3.0}, score=0.4)
    snaps = {"METALIETF": _snapshot("METALIETF", 13.40, 8_000_000.0)}
    for mom, expected in ((full, 1.0), (thin, 0.2)):
        signals = decision.decide([], {"Metal": mom}, snaps, get_profile("IN"))
        metal = next(s for s in signals if s.sector == "Metal")
        assert abs(metal.momentum_weight - expected) < 1e-9, metal.momentum_weight
    # And a sector with no momentum entry at all reports zero weight.
    signals = decision.decide([], {}, snaps, get_profile("IN"))
    metal = next(s for s in signals if s.sector == "Metal")
    assert metal.momentum_weight == 0.0
    assert metal.action == decision.ACTION_NO_BUY


def test_declining_diagnostic_needs_reliable_momentum_too() -> None:
    from models import SectorMomentum
    bearish = SectorMomentum(sector="Metal", etf="METALIETF",
                             returns={"63d": -9.0}, score=-0.6)
    snaps = {"METALIETF": _snapshot("METALIETF", 13.40, 8_000_000.0,
                                       day_change=-1.5, last_hour=-0.5)}
    signals = decision.decide([], {"Metal": bearish}, snaps, get_profile("IN"))
    metal = next(s for s in signals if s.sector == "Metal")
    # Only 20% window weight, so "actively declining" is not established.
    assert decision.is_declining(metal) is False


def test_declining_needs_a_real_downtrend_not_merely_a_weak_uptrend() -> None:
    # SIGNAL_MIN_MOMENTUM is negative, so `<= -SIGNAL_MIN_MOMENTUM` reads as
    # `<= +0.2` and admitted mild uptrends as "actively declining". These gates
    # were originally SELL gates, where that leniency was deliberate.
    from decision import TradeSignal
    snap = _snapshot("X.NS", 100.0, 5_000_000.0, day_change=-1.5, last_hour=-0.5)

    def declining(momentum: float) -> bool:
        signal = TradeSignal("Metal", "METALIETF", decision.ACTION_NO_BUY,
                             0.0, -0.5, 5, config.SIGNAL_MIN_ARTICLES,
                             momentum, 1.0, snap, False, [])
        return decision.is_declining(signal)

    assert declining(0.15) is False, "a rising sector is not under pressure"
    assert declining(0.0) is False, "a flat sector is not under pressure"
    assert declining(-0.1) is False, "inside the floor is not under pressure"
    assert declining(config.SIGNAL_MIN_MOMENTUM) is True, "the floor itself counts"
    assert declining(-0.5) is True, "a real downtrend must register"


# --- news classification -------------------------------------------------

def _classify(title: str, summary: str = "") -> list[str]:
    """Classify one synthetic headline through the real analyzer path."""
    import analyzer
    prof = get_profile("IN")
    item = NewsItem(title=title, summary=summary, link="https://e.com/x",
                    source="Test", published=UTC_NOW)
    return analyzer.classify_items([item], prof)[0]


def test_the_india_profile_has_no_duplicated_keywords() -> None:
    import analyzer
    # `rbi`, `repo rate` and `gross npa` were each claimed by two sectors,
    # which tied every RBI story and filed it under both: ~130 collisions in
    # three days. They now belong to Banks alone.
    assert analyzer._shared_keywords(get_profile("IN")) == frozenset()


def test_the_shared_keyword_guard_still_works_on_a_bad_profile() -> None:
    import analyzer
    import dataclasses
    from models import SectorDef
    prof = get_profile("IN")
    clashing = dataclasses.replace(prof, key="TEST-CLASH", sectors={
        "Alpha": SectorDef(name="Alpha", etf="A.NS", keywords=["shared", "alpha only"]),
        "Beta": SectorDef(name="Beta", etf="B.NS", keywords=["shared", "beta only"]),
    })
    assert analyzer._shared_keywords(clashing) == frozenset({"shared"})
    pats = analyzer._sector_patterns(clashing)
    item = NewsItem(title="shared term appears", summary="", link="x",
                    source="t", published=UTC_NOW)
    assert analyzer._sector_strengths(item, pats) == {}, "shared alone must not assign"


def test_rbi_policy_stories_now_classify_to_banks() -> None:
    # India's most market-moving banking event must not land nowhere, and must
    # not depend on whether the headline says "repo rate" or "rate cut".
    for headline in ("RBI keeps repo rate unchanged at policy review",
                     "RBI cuts repo rate by 25 bps",
                     "RBI monetary policy: repo rate held"):
        assert _classify(headline) == ["Banks"], headline


def test_a_shared_superstring_no_longer_swallows_an_exclusive_substring() -> None:
    # Banks declared both `npa` and `gross npa`; the shared superstring
    # consumed the exclusive substring, so the story classified nowhere.
    assert _classify("Gross NPA ratio declines sharply in Q2") == ["Banks"]


def test_compound_company_names_are_reachable() -> None:
    # Alternation is leftmost-match, so "icici" hid "icici bank" and 36 others.
    assert _classify("ICICI Bank profit rises") == ["Banks"]
    assert _classify("Reliance Industries refining margins improve") == ["Energy"]
    assert _classify("State Bank of India cuts lending rate") == ["PSU Banks"]


def test_a_multi_sector_wrap_with_no_headline_match_is_dropped() -> None:
    # Three sectors named only in the summary, none in the headline: a market
    # wrap. One such item pushed +0.97 into the three sectors it called losers.
    got = _classify("Sensex Tanks 500 Points",
                    "Nifty IT, Nifty Metal, Nifty Realty and Nifty FMCG led the losses")
    assert got == [], got


def test_a_bank_named_in_the_title_goes_to_banks_alone() -> None:
    got = _classify("Axis Bank Q2 profit rises on strong credit growth")
    assert got == ["Banks"], got


def test_an_nbfc_story_goes_to_financial_services_alone() -> None:
    got = _classify("Bajaj Finance posts record quarter as gold loan book grows")
    assert got == ["Financial Services"], got


def test_a_market_roundup_naming_everything_is_discarded() -> None:
    # The exact failure case: this shape matched 10 of 12 sectors and voted
    # its headline sentiment into all of them, every single day.
    roundup = ("Stocks to watch, September 7: Tata Motors, oil-linked stocks, "
               "SBI, IFCI, Lupin, NMDC, HUL, Infosys, DLF, Tata Steel")
    got = _classify(roundup)
    assert got == [], f"a roundup must carry no sector signal, got {got}"


def test_title_outweighs_summary_when_narrowing() -> None:
    import config
    # Metal named in the title, IT only in the summary: the title wins.
    got = _classify("Tata Steel raises output guidance",
                    "Infosys was mentioned elsewhere in the note.")
    assert got == ["Metal"], got
    assert config.TITLE_MATCH_WEIGHT > 1


def test_a_single_summary_mention_still_classifies() -> None:
    # MIN_MATCH_STRENGTH is 1: the sweep showed a higher bar cost 98 of 359
    # usable articles while cutting collisions only 9 -> 8.
    got = _classify("Quarterly earnings wrap", "Cipla reported higher revenue.")
    assert got == ["Pharma"], got


# --- per-sector article floor --------------------------------------------

def test_article_floor_falls_back_to_the_flat_minimum() -> None:
    assert news_archive.required_articles("Banks", {}) == config.SIGNAL_MIN_ARTICLES
    assert news_archive.required_articles("Nope", {"Banks": 40.0}) == \
        config.SIGNAL_MIN_ARTICLES


def test_article_floor_scales_with_a_sectors_own_coverage() -> None:
    baselines = {"Banks": 26.0, "Pharma": 2.7}
    # Banks draws ~26 a day, so 3 is noise for it and the floor must rise.
    assert news_archive.required_articles("Banks", baselines) == 13
    # Pharma draws under 3 a day, so it keeps the flat minimum.
    assert news_archive.required_articles("Pharma", baselines) == \
        config.SIGNAL_MIN_ARTICLES


def test_baselines_stay_empty_until_the_archive_is_deep_enough() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prof = get_profile("IN")
        for day in (date(2026, 9, 1), date(2026, 9, 2)):
            news_archive.archive_news(
                [NewsItem(title="Axis Bank profit rises", summary="",
                          link=f"https://e.com/{day}", source="T",
                          published=UTC_NOW)], day, tmp)
        assert news_archive.sector_article_baselines(prof, tmp, min_days=5) == {}
        mature = news_archive.sector_article_baselines(prof, tmp, min_days=2)
        assert mature, "with min_days met, baselines must be computed"
        # One article a day, converted to the 12h window the gate counts over.
        window = config.SIGNAL_NEWS_HOURS / 24.0
        assert mature["Banks"] == window, mature["Banks"]


def test_baseline_median_handles_an_even_number_of_days() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prof = get_profile("IN")
        for day, count in ((date(2026, 9, 1), 1), (date(2026, 9, 2), 3)):
            news_archive.archive_news(
                [NewsItem(title="Axis Bank profit rises", summary="",
                          link=f"https://e.com/{day}-{i}", source="T",
                          published=UTC_NOW) for i in range(count)], day, tmp)
        got = news_archive.sector_article_baselines(prof, tmp, min_days=2)
    window = config.SIGNAL_NEWS_HOURS / 24.0
    assert got["Banks"] == 2.0 * window, got["Banks"]


def test_classification_holds_through_the_public_analyze_path() -> None:
    # The seam that matters: analyze() is what the engine consumes. Asserting
    # only on the private helpers let the whole classifier be reverted with the
    # suite still green.
    import analyzer
    prof = get_profile("IN")
    roundup = [
        NewsItem(title=("Stocks to watch: Tata Motors, SBI, Lupin, NMDC, HUL, "
                        "Infosys, DLF, Tata Steel soar"),
                 summary="Broad market wrap.", link=f"https://e.com/wrap-{i}",
                 source="T", published=UTC_NOW)
        for i in range(6)
    ]
    focused = [
        NewsItem(title="Cipla wins US approval for new generic",
                 summary="Pharma major gains.", link=f"https://e.com/ph-{i}",
                 source="T", published=UTC_NOW)
        for i in range(3)
    ]
    scores = {s.sector: s for s in analyzer.analyze(roundup + focused, {},
                                                    profile=prof)}
    assert scores["Pharma"].news_count == 3, scores["Pharma"].news_count
    for sector in ("Auto", "Metal", "FMCG", "IT", "Realty", "Banks"):
        assert scores[sector].news_count == 0, (
            f"{sector} took {scores[sector].news_count} from a roundup")


# --- run persistence (CLI and dashboard share this) ----------------------

def test_persist_run_writes_all_three_artifacts() -> None:
    # The dashboard button recorded nothing at all until this path was shared,
    # and the inline copy it was first given shadowed a caller's variable.
    import csv as _csv
    import json as _json
    import daily_signal
    from decision import TradeSignal

    snap = _snapshot("METALIETF", 13.40, 8_000_000.0)
    buy = TradeSignal("Metal", "METALIETF", decision.ACTION_BUY, 0.5, 0.4, 6,
                      config.SIGNAL_MIN_ARTICLES, 0.3, 1.0, snap, False, [])
    skip = TradeSignal("IT", "ITETF", decision.ACTION_NO_BUY, -0.1, -0.2, 4,
                       config.SIGNAL_MIN_ARTICLES, 0.1, 1.0, snap, False, [])
    items = [_item(i) for i in range(3)]
    now = datetime(2026, 9, 8, 15, 15, tzinfo=timezone.utc)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = (config.SIGNALS_CSV, config.LAST_SIGNAL_JSON,
                    config.NEWS_ARCHIVE_DIR)
        config.SIGNALS_CSV = root / "signals.csv"
        config.LAST_SIGNAL_JSON = root / "last_signal.json"
        config.NEWS_ARCHIVE_DIR = root / "news_archive"
        try:
            archived = daily_signal.persist_run([buy, skip], "IN", now, items)
            assert archived == 3, archived
            rows = list(_csv.DictReader(
                config.SIGNALS_CSV.open(encoding="utf-8")))
            assert len(rows) == 2, rows
            assert {r["sector"] for r in rows} == {"Metal", "IT"}
            top = [r for r in rows if r["top_pick"] == "True"]
            assert len(top) == 1 and top[0]["sector"] == "Metal", rows
            payload = _json.loads(
                config.LAST_SIGNAL_JSON.read_text(encoding="utf-8"))
            assert payload["market"] == "IN"
            assert payload["top_pick"]["etf"] == "METALIETF"
            assert payload["top_pick"]["explanation"], "prose must be stored"
            assert len(news_archive.archived_days(config.NEWS_ARCHIVE_DIR)) == 1
        finally:
            (config.SIGNALS_CSV, config.LAST_SIGNAL_JSON,
             config.NEWS_ARCHIVE_DIR) = original


def test_persist_run_returns_zero_when_nothing_new_to_archive() -> None:
    import daily_signal
    from decision import TradeSignal
    snap = _snapshot("METALIETF", 13.40, 8_000_000.0)
    sig = TradeSignal("Metal", "METALIETF", decision.ACTION_NO_BUY, 0.0, 0.0,
                      1, config.SIGNAL_MIN_ARTICLES, 0.0, 1.0, snap, False, [])
    now = datetime(2026, 9, 8, 15, 15, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = (config.SIGNALS_CSV, config.LAST_SIGNAL_JSON,
                    config.NEWS_ARCHIVE_DIR)
        config.SIGNALS_CSV = root / "signals.csv"
        config.LAST_SIGNAL_JSON = root / "last_signal.json"
        config.NEWS_ARCHIVE_DIR = root / "news_archive"
        try:
            assert daily_signal.persist_run([sig], "IN", now, []) == 0
            assert config.SIGNALS_CSV.exists(), "the run must still be recorded"
        finally:
            (config.SIGNALS_CSV, config.LAST_SIGNAL_JSON,
             config.NEWS_ARCHIVE_DIR) = original


def _main() -> int:
    """Run every test_* function in this module and report the tally."""
    tests = {name: obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)}
    failures = 0
    for name, func in tests.items():
        try:
            func()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc or '(bare assert, no detail)'}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
