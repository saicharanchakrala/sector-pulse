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
    dear_but_liquid = _snapshot("INFRABEES.NS", price=958.0, volume=9_000.0)
    turnover = 958.0 * 9_000.0
    assert turnover > config.SIGNAL_MIN_AVG_TURNOVER
    assert dear_but_liquid.avg_volume_20d < 50_000, "would fail the old unit gate"
    by_sector = _decide_with({"INFRABEES.NS": dear_but_liquid})
    assert by_sector["Infrastructure"].illiquid is False


def test_genuinely_thin_etf_is_still_illiquid() -> None:
    thin = _snapshot("OILIETF.NS", price=11.40, volume=20_000.0)
    assert 11.40 * 20_000.0 < config.SIGNAL_MIN_AVG_TURNOVER
    by_sector = _decide_with({"OILIETF.NS": thin})
    assert by_sector["Energy"].illiquid is True


def test_illiquid_blocks_a_buy_that_would_otherwise_pass() -> None:
    # Same sector, same price action; only turnover differs.
    liquid = _snapshot("METALIETF.NS", price=13.40, volume=8_000_000.0)
    thin = _snapshot("METALIETF.NS", price=13.40, volume=1_000.0)
    assert _decide_with({"METALIETF.NS": liquid})["Metal"].illiquid is False
    assert _decide_with({"METALIETF.NS": thin})["Metal"].illiquid is True


def test_liquidity_reason_is_reported_in_rupees() -> None:
    by_sector = _decide_with(
        {"METALIETF.NS": _snapshot("METALIETF.NS", 13.40, 8_000_000.0)})
    reasons = " ".join(by_sector["Metal"].reasons)
    assert "turnover" in reasons
    assert "107,200,000" in reasons, reasons


def test_explanation_mentions_turnover_not_units() -> None:
    by_sector = _decide_with(
        {"METALIETF.NS": _snapshot("METALIETF.NS", 13.40, 1_000.0)})
    text = decision.explain(by_sector["Metal"])
    assert "rupees a day" in text, text
    assert "units a day" not in text, text


def test_a_sector_with_no_snapshot_is_never_a_buy() -> None:
    by_sector = _decide_with({})
    assert all(s.action == decision.ACTION_NO_BUY for s in by_sector.values())
    assert by_sector["Realty"].etf is None


# --- momentum gate must fail closed ------------------------------------

def _gates(momentum_score: float, momentum_weight: float) -> tuple[bool, str]:
    """Run the buy gates with everything but momentum comfortably passing."""
    from decision import _buy_gates
    snap = _snapshot("X.NS", price=100.0, volume=5_000_000.0)
    ok, reasons = _buy_gates("X.NS", snap, 0.5, 5, momentum_score,
                             momentum_weight, False)
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
    full = SectorMomentum(sector="Metal", etf="METALIETF.NS",
                          returns={"5d": 1.0, "21d": 2.0, "63d": 3.0}, score=0.4)
    thin = SectorMomentum(sector="Metal", etf="METALIETF.NS",
                          returns={"63d": 3.0}, score=0.4)
    snaps = {"METALIETF.NS": _snapshot("METALIETF.NS", 13.40, 8_000_000.0)}
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
    bearish = SectorMomentum(sector="Metal", etf="METALIETF.NS",
                             returns={"63d": -9.0}, score=-0.6)
    snaps = {"METALIETF.NS": _snapshot("METALIETF.NS", 13.40, 8_000_000.0,
                                       day_change=-1.5, last_hour=-0.5)}
    signals = decision.decide([], {"Metal": bearish}, snaps, get_profile("IN"))
    metal = next(s for s in signals if s.sector == "Metal")
    # Only 20% window weight, so "actively declining" is not established.
    assert decision.is_declining(metal) is False


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
