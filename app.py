"""Sector Pulse - Streamlit dashboard for both signal systems.

Two tabs, two unrelated systems:

  Positional signal  blends RSS news sentiment with sector index/ETF momentum
                     into a composite score held for weeks.
  Scan               screens NSE F&O names across FOUR holding periods:
                     intraday (3-minute bars, session-time gated), then
                     short, mid and long term (daily bars, no session
                     gate). Each prices the trade - round-trip cost and
                     whether the plausible move covers it - and ranks the
                     top 20. It is a cost and risk screen, NOT a forecast:
                     measured AUC 0.5121 against 0.4990 for the same model
                     on shuffled labels, and at the daily horizons the
                     selection was worse than equal-weighting at all three.

They share no thresholds and no data. Educational tool - not financial advice.
"""
from __future__ import annotations

import logging

import json
import re
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


import analyzer
import claude_insights
import bar_store
import config
import decision
import daily_signal
import indicators
import intraday
import market_data
import news_fetcher
import object_store
import options_chain
import position_watch
import premarket
import scan_data
import scan_publish
import instruments
import option_history
import filings
import glossary
import horizons
import instrument_report
import instrument_search
import setups
import sound
import trade_costs
from levels import LONG
from models import NewsItem, ScoredNewsItem, SectorMomentum, SectorScore
from profiles import PROFILES, MarketProfile, get_profile

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sector Pulse", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner=False)
def warm_object_store() -> bool:
    """Build the S3 client once, in a thread, before a render needs it.

    Measured 2026-09-17 from India against ap-southeast-2: the first read
    of the published scan cost 7.65 seconds and every later one 0.29, so
    almost all of it is boto3 import, credential resolution and the TLS
    handshake. cache_resource runs this once per process rather than once
    per session, and the thread keeps even that off the first render.

    CALLED FROM THE RENDER, NOT AT IMPORT. Importing this module at import
    time started a real boto3 client against the real bucket during test
    COLLECTION, before conftest's autouse fixture had cleared
    SECTOR_PULSE_S3_BUCKET, and raced object_store.reset().

    Failures are the warm-up's problem, not the page's - object_store.warm
    swallows them and the real call reports properly.
    """
    import threading

    threading.Thread(target=object_store.warm, name="warm-object-store",
                     daemon=True).start()
    return True


IST_ZONE = ZoneInfo("Asia/Kolkata")

GREEN = "#16a34a"
RED = "#dc2626"


# --- Cached data loaders -----------------------------------------------------

@st.cache_data(ttl=config.CACHE_TTL_SECONDS)
def load_news(max_age_hours: int, profile_key: str) -> list[NewsItem]:
    """Fetch and cache recent news items for one market profile's feeds."""
    return news_fetcher.fetch_all_news(
        max_age_hours=max_age_hours, profile=get_profile(profile_key)
    )


@st.cache_data(ttl=config.CACHE_TTL_SECONDS)
def load_momentum(profile_key: str) -> dict[str, SectorMomentum]:
    """Fetch and cache sector index/ETF momentum from Kite daily bars.

    Keyed on the profile key alone, but the result depends on the profile's
    trade_etfs. Editing a profile while the dashboard is running keeps
    serving the old tickers until the TTL lapses, so use "Refresh data".
    """
    return market_data.get_sector_momentum(profile=get_profile(profile_key))


# --- Small formatting helpers ------------------------------------------------

def esc_markdown(text: str) -> str:
    """Escape markdown control characters in untrusted feed text."""
    return re.sub(r"([\\`*_{}\[\]()#+!~|>])", r"\\\1", text)


def esc_link(url: str) -> str:
    """Percent-encode parentheses so a URL survives a markdown link."""
    return url.replace("(", "%28").replace(")", "%29")


def sentiment_emoji(sentiment: float) -> str:
    """Map a -1..1 sentiment score to a traffic-light emoji."""
    if sentiment > 0.15:
        return "🟢"
    if sentiment < -0.15:
        return "🔴"
    return "⚪"


def age_hours(published: datetime) -> float:
    """Hours elapsed since a timezone-aware published timestamp."""
    return max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 3600)


def market_mood(scores: list[SectorScore]) -> float:
    """Weighted mean sentiment over the unique top headlines available."""
    unique: dict[str, ScoredNewsItem] = {}
    for score in scores:
        for scored in score.top_items:
            unique.setdefault(scored.item.link, scored)
    total_weight = sum(s.weight for s in unique.values())
    if total_weight <= 0:
        return 0.0
    return sum(s.weight * s.sentiment for s in unique.values()) / total_weight


def momentum_text(momentum: SectorMomentum | None) -> str:
    """One-line momentum summary for a sector, or 'n/a' when missing."""
    if momentum is None or not momentum.returns:
        return "**Momentum:** n/a"
    windows = " | ".join(f"{w} {r:+.1f}%" for w, r in momentum.returns.items())
    return f"**Momentum ({momentum.etf}):** {windows} - score {momentum.score:+.2f}"


def headline_line(scored: ScoredNewsItem) -> str:
    """Markdown bullet for one scored headline with sentiment, source and age."""
    item = scored.item
    return (
        f"{sentiment_emoji(scored.sentiment)} "
        f"[{esc_markdown(item.title)}]({esc_link(item.link)}) - "
        f"{item.source}, {age_hours(item.published):.0f}h ago"
    )


def build_rationale(score: SectorScore) -> str:
    """One-line rationale (plus strongest headline) for an investment card."""
    bits = [
        f"composite {score.composite:+.2f}",
        f"news {score.news_score:+.2f} over {score.news_count} articles",
    ]
    if score.momentum is not None and score.momentum.returns:
        windows = ", ".join(
            f"{w} {r:+.1f}%" for w, r in score.momentum.returns.items()
        )
        bits.append(f"momentum {windows}")
    rationale = " | ".join(bits)
    if score.top_items:
        headline = esc_markdown(score.top_items[0].item.title)
        rationale += f'\n\nStrongest headline: "{headline}"'
    return rationale


# --- Chart and table builders ------------------------------------------------

def build_score_chart(scores: list[SectorScore]) -> go.Figure:
    """Horizontal bar chart of composite score per sector, best at the top."""
    ordered = list(reversed(scores))  # plotly draws the first bar at the bottom
    values = [s.composite for s in ordered]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[s.sector for s in ordered],
            orientation="h",
            marker_color=[GREEN if v >= 0 else RED for v in values],
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        showlegend=False,
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Composite score (news sentiment + momentum)",
        yaxis_title=None,
    )
    return fig


def build_ranking_frame(scores: list[SectorScore]) -> pd.DataFrame:
    """Full ranking table; missing momentum windows are left blank."""
    rows = []
    for score in scores:
        returns = score.momentum.returns if score.momentum is not None else {}
        row: dict[str, object] = {
            "Sector": score.sector,
            "ETF": score.etf,
            "Composite": round(score.composite, 2),
            "News score": round(score.news_score, 2),
            "Articles": score.news_count,
            "Buzz %": round(score.buzz * 100, 2),
        }
        for window in config.MOMENTUM_WINDOWS:
            value = returns.get(window)
            row[f"{window} %"] = round(value, 2) if value is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


# --- Page sections -----------------------------------------------------------

def render_sidebar() -> tuple[str, float, int]:
    """Sidebar controls; returns (profile_key, news_weight, max_age_hours)."""
    with st.sidebar:
        profile_keys = list(PROFILES)
        default_key = (
            config.DEFAULT_MARKET if config.DEFAULT_MARKET in PROFILES
            else next(iter(PROFILES))
        )
        profile_key = st.selectbox(
            "Market",
            profile_keys,
            index=profile_keys.index(default_key),
            format_func=lambda key: f"{key} - {PROFILES[key].label}",
        )
        if st.button("Refresh data"):
            st.cache_data.clear()
            # The scan's bar assembly moved to cache_resource, which
            # cache_data.clear() does not touch - so without this the
            # button stopped refreshing the one cache most worth
            # refreshing.
            st.cache_resource.clear()
            st.rerun()
        news_weight = st.slider(
            "News vs momentum weight", 0.0, 1.0, config.DEFAULT_NEWS_WEIGHT, 0.05
        )
        max_age_hours = st.slider(
            "News lookback (hours)", 12, 72, config.NEWS_MAX_AGE_HOURS, 6
        )
        with st.expander("About"):
            st.markdown(
                "Sector Pulse pulls recent finance and news headlines from the "
                "selected market profile's RSS feeds and classifies each article "
                "into that market's sectors via keyword matching. A finance-tuned "
                "VADER model scores headline sentiment, recency-weighted so fresh "
                "news counts more. Each sector's index or ETF momentum is measured "
                "over 5, 21 and 63 trading days from Kite daily bars. The score "
                "blends news sentiment and momentum using the weight slider "
                "above. It has no backtest behind it: the intraday rule, "
                "which was measured properly, turned out to have none."
            )
        st.warning("Educational tool - not financial advice.")
    return profile_key, news_weight, max_age_hours


def render_header_metrics(
    item_count: int,
    scores: list[SectorScore],
    momentum: dict[str, SectorMomentum],
    profile: MarketProfile,
) -> None:
    """Row of four headline metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles analyzed", item_count)
    col2.metric("Top-headline mood", f"{market_mood(scores):+.2f}")
    col3.metric("Top sector", scores[0].sector)
    col4.metric("Momentum data", f"{len(momentum)}/{len(profile.sectors)} sectors")


def render_invest_section(scores: list[SectorScore]) -> None:
    """Top-3 sector cards plus a bottom-2 underweight warning."""
    st.subheader("Where to consider investing")
    boxes = [st.success, st.info, st.info]
    for rank, (box, score) in enumerate(zip(boxes, scores[:3]), start=1):
        box(f"**{rank}. {score.sector} ({score.etf})** - {build_rationale(score)}")
    worst = list(reversed(scores[-2:]))
    avoid = ", ".join(f"{s.sector} ({s.etf}, {s.composite:+.2f})" for s in worst)
    st.warning(f"Consider avoiding / underweight: {avoid}")


def render_sector_expanders(scores: list[SectorScore]) -> None:
    """Per-sector detail expanders in ranked order."""
    st.subheader("Sector details")
    for rank, score in enumerate(scores, start=1):
        label = f"{rank}. {score.sector} ({score.etf}) - composite {score.composite:+.2f}"
        with st.expander(label):
            if score.top_items:
                for scored in score.top_items:
                    st.markdown(headline_line(scored))
            else:
                st.caption("No recent articles matched this sector.")
            st.markdown(momentum_text(score.momentum))


def signal_to_dict(signal: decision.TradeSignal) -> dict:
    """Serialize a live TradeSignal to the same plain-dict shape as last_signal.json."""
    payload = asdict(signal)
    if payload["intraday"] is not None:
        payload["intraday"]["asof"] = signal.intraday.asof.isoformat()
    payload["explanation"] = decision.explain(signal)
    return payload


def signal_frame(signal_dicts: list[dict]) -> pd.DataFrame:
    """Compact table of per-sector trade signals from plain dicts."""
    rows = []
    for sig in signal_dicts:
        snap = sig.get("intraday") or {}
        rows.append({
            "Sector": sig["sector"],
            "ETF": sig["etf"] or "-",
            "Action": sig["action"],
            "Rank": round(sig["rank_score"], 3),
            "News today": round(sig["news_today"], 2),
            "N": sig["news_count_today"],
            "Day %": round(snap["day_change_pct"], 2) if snap else None,
            "Last hr %": round(snap["last_hour_change_pct"], 2) if snap else None,
            "Momentum": round(sig["momentum"], 2),
            "Illiquid": bool(sig["illiquid"]),
        })
    return pd.DataFrame(rows)


def fmt_time(iso_timestamp: str | None) -> str:
    """Format an ISO timestamp as HH:MM, or '?' when missing/unparseable."""
    if not iso_timestamp:
        return "?"
    try:
        return datetime.fromisoformat(iso_timestamp).strftime("%H:%M")
    except ValueError:
        return "?"


def render_signal_result(top: dict | None, signal_dicts: list[dict]) -> None:
    """Render one stored or live signal result: top-pick banner plus table."""
    if top is None:
        st.info("**NO BUY today** - no sector passed every check.")
    else:
        st.success(f"**TOP SIGNAL: {top['action']} {top['etf']} "
                   f"({top['sector']})** - rank {top['rank_score']:+.3f}")
        rationale = top.get("explanation")
        if rationale:
            st.markdown(f"**Why:** {rationale}")
    show_table(signal_frame(signal_dicts))


def load_stored_signal(profile_key: str) -> "dict | None":
    """Read last_signal.json if it exists and matches the selected market."""
    path = config.LAST_SIGNAL_JSON
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("market") != profile_key:
        return None
    return data


def render_signal_section(
    items: list[NewsItem],
    momentum: dict[str, SectorMomentum],
    profile: MarketProfile,
) -> None:
    """'Today's 3:15 signal' - live preview vs saved scheduled-run signal."""
    st.subheader("Today's 3:15 signal")
    stored = load_stored_signal(profile.key)
    if st.button("Compute signal now"):
        with st.spinner("Fetching intraday data and evaluating gates..."):
            snapshots = intraday.get_intraday_snapshots(
                list(profile.trade_etfs.values())
            )
            if not snapshots:
                st.session_state["live_signal"] = {
                    "market": profile.key, "closed": True,
                }
            else:
                signals = decision.decide(items, momentum, snapshots, profile)
                pick = decision.top_pick(signals)
                now = datetime.now().astimezone()
                st.session_state["live_signal"] = {
                    "market": profile.key,
                    "closed": False,
                    "generated_at": now.isoformat(),
                    "top_pick": signal_to_dict(pick) if pick is not None else None,
                    "signals": [signal_to_dict(s) for s in signals],
                }
                # Same persistence path as the CLI. A button that shows a
                # signal and records nothing is worse than one that fails:
                # you would believe the day was captured, and the headlines
                # behind it age out of the feeds within 48h.
                archived = daily_signal.persist_run(
                    signals, profile.key, now, items, pick)
                st.caption(f"Recorded to signals.csv and last_signal.json; "
                           f"archived {archived} new headline(s).")
    live = st.session_state.get("live_signal")
    if live is not None and live.get("market") != profile.key:
        live = None
    if live is not None:
        if live.get("closed"):
            st.info("Market closed today - no live signal.")
        else:
            st.markdown(
                f"#### ⚡ Live preview - computed {fmt_time(live.get('generated_at'))} "
                f"(markets may still be open; this can change until close)"
            )
            render_signal_result(live.get("top_pick"), live.get("signals", []))
        if stored is not None:
            with st.expander(
                f"📌 Saved signal - scheduled run at "
                f"{fmt_time(stored.get('generated_at'))}"
            ):
                render_signal_result(
                    stored.get("top_pick"), stored.get("signals", [])
                )
    elif stored is not None:
        st.markdown(
            f"#### 📌 Saved signal - from scheduled run at "
            f"{stored.get('generated_at', '?')}"
        )
        st.caption("The authoritative run is the 15:15 IST scheduled task; "
                   "use 'Compute signal now' for a live preview.")
        render_signal_result(stored.get("top_pick"), stored.get("signals", []))
    else:
        st.caption("No saved signal for this market yet - run "
                   "`python daily_signal.py` or compute one now.")
    st.caption("Decision support only - unvalidated rule, possibly delayed "
               "intraday data, never financial advice, never places orders. "
               "SELL means exit/avoid/underweight - short-selling is not "
               "modelled.")


def render_ai_section(scores: list[SectorScore]) -> None:
    """Optional Claude-powered narrative analysis."""
    st.subheader("AI analysis")
    if not claude_insights.is_available():
        st.caption("Set ANTHROPIC_API_KEY to enable Claude-powered narrative analysis.")
        return
    if st.button("Generate AI analysis"):
        with st.spinner("Asking Claude for a narrative read on the data..."):
            st.session_state["ai_insights"] = claude_insights.generate_insights(scores)
        if st.session_state["ai_insights"] is None:
            st.error("AI analysis failed - check the app logs and your API key.")
    result = st.session_state.get("ai_insights")
    if result is not None:
        st.markdown(result)


# --- Instrument sync -------------------------------------------------------

def _cache_state() -> tuple["instruments.Universe | None", dict]:
    """What the local cache currently holds: instruments and option chains."""
    return instruments.load_latest(), option_history.cache_summary()


def render_cache_status() -> None:
    """Show what is cached locally, and how stale it is."""
    universe, chains = _cache_state()
    left, mid, right = st.columns(3)
    if universe is None:
        left.metric("Instruments cached", "none")
        mid.metric("F&O underlyings", "-")
    else:
        left.metric("Instruments cached", f"{len(universe.equities):,}",
                    help=f"Captured {universe.captured_at:%Y-%m-%d %H:%M}")
        mid.metric("F&O underlyings",
                   f"{len(universe.fo_indices) + len(universe.fo_stocks):,}")
    right.metric("Chain snapshots", f"{chains['snapshots']:,}",
                 help=f"{chains['megabytes']} MB on disk")
    if universe is not None:
        age = datetime.now().astimezone() - universe.captured_at
        hours = age.total_seconds() / 3600
        stale = ("captured " + f"{hours:.1f}h ago"
                 if hours >= 1 else f"captured {age.seconds // 60} min ago")
        st.caption(f"Instrument snapshot {stale}. Lot sizes and the F&O list "
                   f"change on expiry, so re-sync at least weekly.")
    if chains["snapshots"] == 0:
        st.caption("No option chains stored yet. NSE serves only a live "
                   "snapshot and keeps no archive, so any option level before "
                   "your first capture can only ever be reconstructed from the "
                   "underlying, never replayed.")
    elif chains["last"] is not None:
        st.caption(f"Chain captures span {chains['first']:%Y-%m-%d %H:%M} to "
                   f"{chains['last']:%Y-%m-%d %H:%M}.")


def sync_instruments(with_chains: bool) -> dict:
    """Fetch and cache the instrument universe, optionally every chain too."""
    status = st.status("Syncing from NSE...", expanded=True)
    result: dict = {}
    with status:
        st.write("Fetching the listed-equity master, F&O master and open "
                 "interest...")
        universe = instruments.discover()
        path = instruments.save(universe)
        result["universe"] = universe.counts()
        result["universe_path"] = path
        st.write(f"Cached {len(universe.equities):,} equities, "
                 f"{len(universe.fo_stocks):,} F&O stocks, "
                 f"{len(universe.fo_indices)} indices, "
                 f"{len(universe.fo_state):,} with live open interest.")
        for gap in universe.gaps:
            st.warning(gap)

        if with_chains:
            wanted = [i.symbol for i in universe.fo_underlyings]
            indices = {i.symbol for i in universe.fo_indices}
            bar = st.progress(0.0, text="Capturing option chains...")

            def tick(done: int, total: int, symbol: str) -> None:
                bar.progress(done / max(1, total),
                             text=f"Option chains {done}/{total} - {symbol}")

            snapshot = option_history.capture(wanted, indices, progress=tick)
            chain_path = option_history.save(snapshot)
            bar.empty()
            result["chains"] = snapshot.counts()
            result["chains_path"] = chain_path
            st.write(f"Captured {snapshot.contract_count:,} contracts across "
                     f"{len(snapshot.chains):,} underlyings.")
            if snapshot.failed:
                st.warning(f"No chain returned for {len(snapshot.failed)} "
                           f"underlying(s): "
                           f"{', '.join(snapshot.failed[:8])}"
                           + (" ..." if len(snapshot.failed) > 8 else ""))
        status.update(label="Sync complete", state="complete", expanded=False)
    return result


def render_sync_popover() -> None:
    """Cache status and the sync control, in a header popover.

    A popover rather than a tab: this is pressed once a day and does not
    earn permanent space next to the two things the page is actually for.
    """
    st.caption(
        "Everything the scanner trades is discovered from NSE, never "
        "hardcoded. This syncs that universe into the local cache so a scan "
        "can say exactly which instruments were listed when it ran."
    )
    render_cache_status()
    st.divider()
    with_chains = st.checkbox(
        "Also capture every option chain",
        value=False,
        help="About 216 paced requests, roughly two minutes. This is the only "
             "way option calls become replayable later: NSE keeps no chain "
             "history, so a premium not captured now is gone.")
    if with_chains:
        st.info(
            "Option chains are the one thing that cannot be back-filled. "
            "Equity bars stay available for ten days, so an equity signal can "
            "be replayed at any past minute. A premium that was not recorded "
            "can only be estimated from the underlying and a delta."
        )
    if st.button("Sync instruments now", type="primary"):
        result = sync_instruments(with_chains)
        st.session_state["last_sync"] = result
        st.success("Cache updated. The scan will use this snapshot.")
        st.rerun()
    last = st.session_state.get("last_sync")
    if last:
        st.write("Last sync in this session:")
        st.json({k: v for k, v in last.items() if not k.endswith("_path")})
        for key in ("universe_path", "chains_path"):
            if last.get(key):
                st.caption(f"{key.replace('_path', '')}: `{last[key]}`")


# --- Intraday scan tab -------------------------------------------------------

# None means the curated large-cap list; an int slices the F&O list, which is
# alphabetical, so those labels say "first N" rather than implying liquidity.
# None means every F&O single stock, which is the default and the whole
# tradeable intraday universe. An int caps the listed-equity sweep, which is
# slow and mostly rejected on turnover.
# Ordered so the FIRST entry is the default the selectbox lands on.
# "All listed equities" leads by request; the labels carry the cost,
# because only the F&O set is covered by the live feed - everything wider
# has to download bars for roughly 2,350 symbols at Kite's 3 requests a
# second, which is about 13 minutes.
_SCAN_SCOPES: dict[str, "int | None"] = {
    "All listed equities (~2,570 - downloads the illiquid tail)": 0,
    "All listed equities, first 300": 300,
    "F&O single stocks (fastest)": None,
}

# Scopes small enough that the live feed covers essentially all of them,
# so they are cheap enough to start without being asked. The feed streams
# every F&O underlying plus the top config.FEED_UNIVERSE_SIZE by turnover
# - about 1,000 symbols - so anything at or under that size needs at most
# a handful of downloads. Only the full universe reaches into the illiquid
# tail the feed does not carry.
# The scope a fresh session starts on. The dict is ordered widest-first
# for the reader, so the default has to be named rather than inherited
# from position - naming it is also what keeps it honest against the help
# text beside the widget.
_DEFAULT_SCAN_SCOPE = "F&O single stocks (fastest)"

_AUTOSCAN_SCOPES = frozenset({
    "All listed equities, first 300",
    "F&O single stocks (fastest)",
})

# The scope that reaches past the live feed's universe, named once so the
# gate below and the tests both refer to the same string.
_UNBOUNDED_SCAN_SCOPE = "All listed equities (~2,570 - downloads the illiquid tail)"


def offered_scopes() -> dict:
    """The scopes a user may actually pick, honouring SCAN_UI_MAY_DOWNLOAD.

    NOT the same dict as _SCAN_SCOPES, which stays the full catalogue so
    run_scan can still resolve a scope arriving from stale session state.
    Hiding the widest choice is the point: its own label said it downloads
    the illiquid tail, and on 2026-09-17 picking it took the machine down
    - 31,537 cache files, an unresponsive tab, and Kite requests competing
    with the live feed. A warning in a label is not a guard.
    """
    if getattr(config, "SCAN_UI_MAY_DOWNLOAD", False):
        return dict(_SCAN_SCOPES)
    return {name: limit for name, limit in _SCAN_SCOPES.items()
            if name != _UNBOUNDED_SCAN_SCOPE}


def resolve_scope(scope: "str | None") -> tuple:
    """(scope, limit), falling back when the asked-for scope is not allowed.

    Streamlit persists the selectbox choice under its key, so a session
    that picked the unbounded scope before the gate existed would arrive
    here with it still selected. Falling back beats honouring it.
    """
    allowed = offered_scopes()
    if scope in allowed:
        return scope, allowed[scope]
    return _DEFAULT_SCAN_SCOPE, _SCAN_SCOPES[_DEFAULT_SCAN_SCOPE]


@st.cache_resource(ttl=config.CACHE_TTL_SECONDS, max_entries=2,
                   show_spinner=False)
def load_scan_bars(tickers: tuple[str, ...],
                   target: "date | None" = None,
                   live_stamp: str = "") -> scan_data.BarSet:
    """Assemble intraday plus daily bars for a ticker tuple.

    Cached on the tickers, the replay date, and - for a live scan - the
    newest live bar's timestamp. That last key matters: without it a cached
    BarSet outlives the feed's newest bar and the scan keeps re-evaluating
    gates against whatever it first saw, which defeats the point of
    streaming. A past window is a different download, so the date belongs
    in the key too.

    max_entries IS LOAD-BEARING, because that key churns far faster than
    the TTL clears. live_bar_stamp is "{newest bar}|{total bar count}" and
    the count rises with every bar the feed writes - measured 2026-09-17,
    it moved every 20 seconds (2257, 2322, 2357, 2380, 2397, 2412 ...).
    At a 900-second TTL and no cap that is about forty-five live BarSets,
    each holding intraday AND daily frames for the whole scope, and
    cache_resource keeps the objects themselves rather than sharing them.
    At 210 F&O names it is merely wasteful; at 2,570 it is roughly 5,140
    DataFrames times forty-five, which is how a scan that downloaded too
    much also ran the machine out of memory rather than just being slow.
    Two entries is what the work needs: the current one, and the one it is
    replacing while a rerun is in flight.

    cache_RESOURCE, not cache_data, and the difference is not cosmetic.
    cache_data pickles whatever it returns so each caller gets its own
    copy; a BarSet holds two dicts of DataFrames - thousands of them once
    the scope is the whole equity universe - and pickling it raised
    UnserializableReturnValueError in production, taking the tab down
    mid-refresh. cache_resource stores the object itself.

    What that costs, stated because it is the real trade: every session
    shares ONE instance, so anything mutating it in place would corrupt
    every other reader. Nothing does - truncate() and _carry_source()
    both build new BarSets rather than editing one - and
    test_the_cached_barset_is_never_mutated_in_place pins that, because it
    is the assumption this decorator rests on.
    """
    return scan_data.fetch_bars(list(tickers), target=target)


def live_bar_stamp() -> str:
    """The newest live bar's time, as a cache key. Empty when no feed."""
    try:
        import live_bars
        state = live_bars.status()
    except Exception:
        return ""
    if not state.get("present") or not state.get("bars"):
        return ""
    latest = state.get("latest_bar")
    return f"{latest}|{state.get('bars')}" if latest is not None else ""


# Highest share of 5-minute setups that has ever resolved in the target's
# favour in this project's measurements, across every geometry tried. Any
# required win rate above it cannot be met by a rule of this kind, so the
# row is flagged rather than silently ranked alongside the others.
BEST_OBSERVED_WIN_RATE = 0.58


def show_table(frame, target=None) -> None:
    """Render a table with a plain-English tooltip on every column.

    Routed through one helper so a column cannot be explained in one table
    and left bare in another, and so improving a wording improves it
    everywhere. Columns with no glossary entry simply render untouched.
    """
    surface = target if target is not None else st
    surface.dataframe(frame, width="stretch", hide_index=True,
                      column_config=glossary.config_for(frame, st))


def watch_table(frame, key: str, target=None, symbol: "str | None" = None,
                entry: "float | None" = None, note: str = "") -> None:
    """A table whose rows can be sent to Positions, with tooltips.

    THE INTERACTION, and why it is not hover. A button that appears on the
    row under the mouse needs a per-row control; Streamlit 1.58 has no
    ButtonColumn and the dataframe is a canvas the page cannot draw into.
    So the row's own selection checkbox is the affordance: click it and the
    action appears directly below, prefilled from that row.

    `symbol` and `entry` are for tables whose rows do not carry them - the
    lookup shows one row per HORIZON for a single instrument, so both come
    from the report around the table.

    A row with no side or no levels gets no button and says why. That is a
    real answer for a NO SETUP row, and better than a button that would
    add a position nothing can be checked against.
    """
    surface = target if target is not None else st
    picked = surface.dataframe(
        frame, width="stretch", hide_index=True,
        column_config=glossary.config_for(frame, st),
        on_select="rerun", selection_mode="single-row", key=key)
    rows = list(getattr(picked, "selection", {}).get("rows", []))
    if not rows:
        return
    index = rows[0]
    if index >= len(frame):
        return
    held = position_watch.position_from_row(
        frame.iloc[index], symbol=symbol, entry=entry, note=note)
    if held is None:
        surface.caption(
            "That row has no side and levels to watch. Rows showing a "
            "direction with a stop and an exit can be added.")
        return
    if surface.button(f"Add {held.symbol} {held.side} to Positions",
                      key=f"{key}_add", type="primary",
                      icon=":material/visibility:"):
        open_watch_draft(held.symbol, held.side, held.entry, held.stop,
                         held.target, held.quantity, held.note)


def show_terms(*names: str) -> None:
    """An expander explaining the concepts behind a table."""
    wanted = [n for n in names if n in glossary.CONCEPTS]
    if not wanted:
        return
    with st.expander("What these terms mean"):
        for name in wanted:
            st.markdown(f"**{name}**")
            st.caption(glossary.CONCEPTS[name])


def scan_frame(actionable: list[setups.Setup]) -> pd.DataFrame:
    """Actionable setups as a display table, cost columns first.

    Column order is deliberate. The cost and win-rate columns are
    arithmetic and have held up under measurement; the rank score has not,
    so it sits last and is labelled as unvalidated rather than presented as
    a quality ordering.
    """
    rows = []
    for s in actionable:
        needed = s.levels.required_win_rate
        rows.append({
            "Symbol": s.symbol,
            "Side": s.direction,
            "Entry price": round(s.levels.entry, 2),
            "Stop loss at": round(s.levels.stop, 2),
            "Exit price": round(s.levels.target, 2),
            "Qty": s.levels.quantity,
            "Risk": round(s.levels.lot_risk, 0),
            "Cost": round(s.levels.cost_rupees, 0),
            "Win % needed": round(needed * 100, 1),
            "Can pay for itself?": ("no - needs more than any measured rate"
                                    if needed > BEST_OBSERVED_WIN_RATE
                                    else "maybe"),
            "Stop %": round(s.levels.stop_pct, 2),
            "Target %": round(s.levels.target_pct, 2),
            "Volume vs normal": round(s.readings.rvol or 0.0, 2),
            "Strength vs Nifty": round(s.readings.relative_strength or 0.0, 2),
            "OI chg %": (round(s.readings.oi_change_pct, 2)
                         if s.readings.oi_change_pct is not None else None),
            "Our ranking": round(s.rank_score, 3),
        })
    return pd.DataFrame(rows)


def _session_bounds() -> tuple[time, time]:
    """The first computable minute of the session, and the close.

    The lower bound is the opening-range close, not the opening bell. Before
    the range has elapsed the latest bar is one of the bars defining it, so
    the range brackets the current price by construction and no breakout can
    be represented - offering 09:15 would be offering a control that cannot
    produce a signal on any day.
    """
    open_h, open_m = config.SCAN_SESSION_OPEN
    close_h, close_m = config.SCAN_SESSION_CLOSE
    first = open_h * 60 + open_m + config.SCAN_OPENING_RANGE_MINUTES
    return time(first // 60, first % 60), time(close_h, close_m)


def render_replay_controls() -> "datetime | None":
    """Date and time pickers; returns the replay instant, or None for live."""
    earliest, latest = _session_bounds()
    today = datetime.now(IST_ZONE).date()
    mode = st.radio(
        "When", ["Live (now)", "Replay a past instant"], horizontal=True,
        help="A replay discards every bar after the chosen instant, so the "
             "scan sees only what was knowable then.")
    if mode == "Live (now)":
        return None
    left, right = st.columns(2)
    with left:
        chosen_day = st.date_input(
            "Session date",
            value=today,
            min_value=today - timedelta(days=config.SCAN_UI_REPLAY_DAYS),
            max_value=today,
            help=f"The last {config.SCAN_UI_REPLAY_DAYS} days. Weekends and "
                 f"holidays have no session and will come back empty.")
    with right:
        chosen_time = st.slider(
            "Time (IST)", min_value=earliest, max_value=latest,
            value=time(10, 0), step=timedelta(minutes=5),
            help=f"5-minute steps, matching the bar size. Starts at "
                 f"{earliest.strftime('%H:%M')} because the opening range "
                 f"has not closed before then and no breakout is computable.")
    moment = datetime.combine(chosen_day, chosen_time, tzinfo=IST_ZONE)
    if chosen_day != today:
        st.info(
            f"Replaying {chosen_day:%A %d %B} at {chosen_time:%H:%M}. Open "
            f"interest is dropped for a past session (NSE publishes no OI "
            f"history, and today's figures would be lookahead), and option "
            f"contracts cannot be shown unless a chain snapshot was stored "
            f"at the time."
        )
    return moment


def autoscan_blocked(as_of, want_options: bool,
                     scope: "str | None" = None) -> str:
    """Why the first scan must NOT run itself, or "" when it may.

    Deliberately conservative. An automatic scan costs a full sweep of the
    universe, so it only happens when the result will be both current and
    cheap: inside market hours, with the feed alive and fresh, on a live
    scan rather than a replay, and without option chains - those cost one
    NSE request per setup and are not something to trigger unasked.
    """
    if as_of is not None:
        return "a replay is a deliberate choice, not something to start for you"
    if want_options:
        return ("option contracts cost an NSE request per setup, so they "
                "are never fetched automatically")
    if scope is not None and scope not in _AUTOSCAN_SCOPES:
        # The feed streams every F&O underlying plus the ~1,000 most
        # traded, so the narrower scopes cost at most a few downloads.
        # The FULL universe still reaches ~1,580 symbols the feed does not
        # carry, at three requests a second, and starting nine minutes of
        # downloading because someone opened a tab is not a thing to do on
        # their behalf.
        return (f"{scope!r} reaches symbols the feed does not stream, so it "
                f"has to download them - minutes, not seconds. It is never "
                f"started automatically")
    if not live_bars_in_hours():
        return "the market is closed"
    state = live_feed_state()
    if not state.get("running"):
        return "the live feed is not running, so a scan would download 216 symbols"
    age = state.get("age_seconds", float("nan"))
    if not state.get("bars"):
        return "the feed has not written a completed bar yet"
    if not (age == age) or age > config.SCAN_LIVE_MAX_AGE_SECONDS:
        return "the newest live bar is too old, so the feed looks stopped"
    return ""


def autoscan_will_clear(as_of, want_options: bool,
                        scope: "str | None" = None) -> bool:
    """Whether the block on the first scan is one TIME lifts by itself.

    The first three refusals in autoscan_blocked are deliberate choices -
    a replay, option chains, a scope the feed does not stream - and no
    amount of waiting changes them. Everything after is a fact about the
    clock or the feed, and will clear on its own.

    Only the second kind is worth polling for.
    """
    if as_of is not None or want_options:
        return False
    return scope is None or scope in _AUTOSCAN_SCOPES


def autoscan_watch_fragment() -> None:
    """Re-check the autoscan gate on a timer, and wake the page when it opens.

    A fragment tick NEVER causes a full script run, so this cannot simply
    exist and let the main body re-evaluate - it has to detect the change
    and ask for the rerun explicitly. Without that, a tab opened before
    09:15 showed "No scan yet - the market is closed" indefinitely, because
    nothing re-ran the check once the market actually opened.
    """
    if st.session_state.get("scan") is not None:
        return
    # A SCAN ALREADY RUNNING IS NOT A REASON TO START ANOTHER. `scan` is
    # only set once one COMPLETES, so without this guard every scan taking
    # longer than the poll interval was restarted from the top by this
    # fragment - forever, never finishing. Observed on 2026-09-15: the
    # spinner ran for minutes with nothing left to download.
    if st.session_state.get("scan_running"):
        return
    # FIRES AT MOST ONCE. Waking the page when the gate opens is the whole
    # job; after that the main flow owns the decision. Relying on
    # `scan_running` alone was not enough - a rerun tears the script down
    # and the flag goes with it, so a scan slower than the poll interval
    # could still be restarted in the gap. `scan_autotried` is set by the
    # main flow the moment it accepts the automatic scan and is never
    # cleared, so it survives that gap.
    if st.session_state.get("scan_autotried"):
        return
    watching = st.session_state.get("autoscan_watch") or {}
    if autoscan_blocked(watching.get("as_of"), watching.get("want_options"),
                        watching.get("scope")):
        return
    # Claimed HERE, not left to the main flow after the rerun: between the
    # rerun and the main flow reaching its own assignment there is a window
    # in which another tick would fire again.
    st.session_state["scan_autotried"] = True
    st.rerun(scope="app")


def render_scan_refresh_controls(replaying: bool,
                                 want_options: bool) -> "int | None":
    """The auto-refresh toggle for the intraday scan, or None when idle.

    The toggle is rendered OUTSIDE the refreshing fragment on purpose: a
    control that redraws itself on every tick is hard to interact with.
    """
    enabled = st.checkbox(
        "Keep this scan up to date while the feed is running",
        value=False, key="scan_auto_refresh",
        help="Re-runs the scan on a timer, so entry, stop and the gates "
             "track the live bars instead of freezing at whenever you last "
             "pressed Run. Off by default because each refresh re-scores "
             "every symbol in scope - it is a real scan, not a repaint.")
    requested = config.SCAN_AUTO_REFRESH_SECONDS
    if enabled:
        requested = st.select_slider(
            "Refresh every",
            options=list(config.SCAN_AUTO_REFRESH_CHOICES),
            value=st.session_state.get("scan_auto_every",
                                       config.SCAN_AUTO_REFRESH_SECONDS),
            format_func=lambda s: f"{s}s" if s < 60 else f"{s // 60} min",
            key="scan_auto_every")
    feed = live_feed_state()
    # `enabled` is PASSED rather than short-circuited here, so the rule
    # lives in exactly one place and production exercises the same branch
    # the tests do. Returning early instead left the function's own
    # `enabled` check dead outside the test suite.
    interval, note = scan_data.auto_refresh_interval(
        enabled=bool(enabled), requested=int(requested), replaying=replaying,
        want_options=want_options,
        in_trading_hours=live_bars_in_hours(),
        feed_running=bool(feed.get("running")),
        feed_age=feed.get("age_seconds", float("nan")),
        last_scan_seconds=float(st.session_state.get("scan_seconds", 0.0)))
    if note:
        st.caption(f"Auto-refresh {'idle' if interval is None else 'adjusted'}"
                   f" - {note}.")
    return interval


def live_bars_in_hours() -> bool:
    """Whether the clock is inside NSE trading hours right now."""
    try:
        import live_bars
        now = datetime.now(IST_ZONE)
        return live_bars.in_session(live_bars.bucket_start(now))
    except Exception:
        return False


def refresh_scan_fragment() -> None:
    """Re-run the stored scan and render it. The auto-refresh body.

    Inputs come from session state rather than from arguments because a
    fragment on a timer is re-invoked by Streamlit, not by this script -
    reading them back each tick is the only way to be sure they are the
    ones currently on screen.
    """
    import time

    params = st.session_state.get("scan_params")
    if params is None:
        st.info("Run a scan first; the timer refreshes an existing one.")
        return
    # A fragment body also runs INLINE on the script run that declares it,
    # so without this the first "refresh" fired the instant the box was
    # ticked - which on a 20-second scan reads as the page hanging. The
    # already-rendered results stand until the first real tick.
    if not st.session_state.get("scan_auto_armed"):
        st.session_state["scan_auto_armed"] = True
        render_scan_results(*st.session_state["scan"], with_terms=False)
        return
    started = time.monotonic()
    with st.spinner("refreshing from the live feed..."):
        ranked, bars, benchmark, now, past = run_scan(params["scope"], None)
    st.session_state["scan_seconds"] = time.monotonic() - started
    st.session_state["scan"] = (ranked, bars, benchmark, now,
                                params["want_options"], past)
    st.caption(f"Auto-refreshed at {now:%H:%M:%S} IST in "
               f"{st.session_state['scan_seconds']:.0f}s.")
    render_scan_results(ranked, bars, benchmark, now,
                        params["want_options"], past, with_terms=False)
    # F6: the interval is fixed when the fragment is wrapped, and a
    # fragment rerun does not re-execute the controls - so a scan that has
    # slowed down cannot raise its own floor. A full rerun re-evaluates
    # the rule with the duration just measured.
    taken = st.session_state["scan_seconds"]
    if taken * 2 > float(st.session_state.get("scan_auto_every",
                                              config.SCAN_AUTO_REFRESH_SECONDS)):
        st.session_state.pop("scan_auto_armed", None)
        st.rerun()


@st.cache_data(ttl=1800, show_spinner=False)
def _watchlist(stamp: str):
    """The PUBLISHED pre-open list. Never builds one.

    Building it costs 19.4 seconds - 3.98M rows across 2,520 symbols, then
    a per-symbol loop - and Streamlit re-executes this whole file on every
    interaction. So the page reads what run_nightly.cmd wrote and offers
    the slow path as a button rather than making every visitor pay for it.

    `stamp` is the published file's own timestamp, used as a cache key so
    the memo invalidates exactly when the file is rewritten.
    """
    return premarket.load_published()


def render_premarket_panel() -> None:
    """What is computable before an opening range exists.

    WHY IT IS HERE. choose_direction needs an opening-range break, so the
    scan below can say nothing until about 09:30 - the tab was honestly
    empty from 08:00 until fifteen minutes after the bell. This fills that
    window with the only thing settled daily bars can support.

    Expanded while the scan cannot answer and collapsed once it can, so it
    does not push the live table down the page during the session.
    """
    now = datetime.now(IST_ZONE)
    try:
        folded = premarket.store_written_at()
        through = premarket.settled_through(now, written_at=folded)
    except Exception as exc:
        st.caption(f"Pre-open watchlist unavailable ({exc}).")
        return

    open_at = now.replace(hour=config.SCAN_SESSION_OPEN[0],
                          minute=config.SCAN_SESSION_OPEN[1],
                          second=0, microsecond=0)
    close_at = now.replace(hour=config.SCAN_SESSION_CLOSE[0],
                           minute=config.SCAN_SESSION_CLOSE[1],
                           second=0, microsecond=0)
    range_ends = open_at + timedelta(
        minutes=config.SCAN_OPENING_RANGE_MINUTES)
    scan_can_answer = range_ends <= now <= close_at

    with st.expander("Pre-open watchlist - what settled bars can tell you",
                     expanded=not scan_can_answer):
        stamp = f"Daily store folded {folded:%d %b %H:%M}. " if folded else ""
        # ON OR BEFORE, not "up to and including". 2026-09-14 was a
        # holiday and the 12th and 13th a weekend, so the cutoff read
        # 14 September while the newest bar in it was the 11th. Stating a
        # date the data does not reach is how a stale list passes for a
        # fresh one.
        st.caption(f"Every figure describes sessions **on or before "
                   f"{through}** - the newest may be earlier if that was "
                   f"a holiday. {stamp}Needs no live feed and no scan.")
        # SAID FIRST AND PLAINLY. Ranked on how far a name usually moves,
        # which is the effect that replicated out of sample here - 2.03x
        # in training, 1.22x in holdout. Direction did not: 0.5121 AUC
        # against a 0.4990 null. So nothing here is a view on which way
        # anything goes, and a caption that implied otherwise would be
        # asserting the thing the measurements rejected.
        st.info("Ranked by **how far a name usually moves**, not by which "
                "way it might go. A wide range is as likely to go against "
                "you as for you. This is what to watch, not what to buy.")
        try:
            mtime = (premarket.PUBLISHED.stat().st_mtime
                     if premarket.PUBLISHED.exists() else 0)
            published = _watchlist(str(mtime))
        except Exception as exc:
            st.warning(f"Could not read the watchlist ({exc}).")
            return

        if published is None:
            # NOT built on demand. That is twenty seconds with the page
            # frozen, and it would happen on every cold cache - which is
            # every restart and every first load of a morning.
            st.caption("No watchlist published yet. run_nightly.cmd writes "
                       "one after the close; it takes about 20 seconds.")
            if st.button("Build it now (~20s)"):
                with st.spinner("Building from the daily store..."):
                    premarket.publish()
                st.rerun()
            return

        table, built_at = published
        limit = st.slider("How many names", 10, 100, 25, step=5,
                          key="premarket_limit")
        st.caption(f"Published {built_at:%d %b %H:%M}. Nothing in it "
                   f"changes intraday - every input is a settled session.")
        st.dataframe(table.head(limit), use_container_width=True,
                     hide_index=True)
        st.caption(
            "**atr_pct** is the average daily range as a share of price. "
            "**expected_low/high** is one ATR either side of the last "
            "close: a stop inside that band sits inside ordinary movement "
            "and gets hit by noise rather than by being wrong. "
            "**cost_in_atr** is the round-trip breakeven as a share of a "
            "typical day - above about 0.15 the charges eat most of what "
            "an ordinary session offers. **spike_ratio** above ~2 means "
            "the turnover rests on a few big days rather than steady "
            "trade. **filing_18h** is context only: a material filing "
            "measured 29.6% here against a 30.6% control, so it is not a "
            "reason to take a trade.")


def render_published_scan() -> bool:
    """Show the scan the feed published. True when it rendered one.

    WHY READ RATHER THAN COMPUTE. Measured 2026-09-16, computing this here
    cost 48 seconds for the 216 F&O underlyings - and only 5.8 of those
    were the scan. The rest was fetching bars the feed already held in
    memory: 5.6s counting rows in a 5 MB object, 7.6s looking up tokens,
    29.4s reassembling bars over the network. The feed now does the scan
    where the bars are and publishes a 56 KB table; this reads it.

    IT REFUSES A STALE ONE. A table older than one bar describes a bar
    that has since closed, so its levels were computed against a price the
    market has left behind. Past that age this returns False and the
    caller falls back to computing, which is slow but current - the same
    trade the live-bar staleness gate already makes.
    """
    try:
        found = scan_publish.load()
    except Exception as exc:
        st.caption(f"Published scan unavailable ({exc}).")
        return False
    if found is None:
        return False
    table, age = found
    if table.empty:
        st.info("The feed scanned and found nothing that cleared its gates.")
        return True
    if age == age and age > config.SCAN_PUBLISHED_MAX_AGE_SECONDS:
        st.warning(
            f"The published scan is {age:.0f}s old, past the "
            f"{config.SCAN_PUBLISHED_MAX_AGE_SECONDS}s limit - the feed has "
            f"most likely stopped. Run a scan below to compute a fresh one.",
            icon=":material/schedule:")
        return False

    actionable = int(table["actionable"].sum()) if "actionable" in table else 0
    directional = int((table["direction"] != setups.NO_SETUP).sum()) \
        if "direction" in table else 0
    left, mid, right = st.columns(3)
    left.metric("Symbols scanned", f"{len(table):,}")
    mid.metric("Got a direction", f"{directional:,}")
    right.metric("Cleared every gate", f"{actionable:,}")
    st.caption(f"Computed by the feed {age:.0f}s ago, where the bars "
               f"already are. Nothing here was calculated in this page.")

    # Cleared trades first, then the ones that got a direction and were
    # blocked - the second group is the useful half, since the gates are
    # the part of this scanner the measurements support.
    show = table.sort_values(["actionable", "score"], ascending=[False, False])
    columns = [c for c in ("symbol", "direction", "actionable", "score",
                           "entry", "stop", "target", "quantity",
                           "required_win_rate", "breakeven_pct", "rvol",
                           "relative_strength", "turnover_20d")
               if c in show.columns]
    st.dataframe(show[columns], width="stretch", hide_index=True)

    with st.expander("Why each one passed or was blocked"):
        reasons = show[["symbol", "direction", "reasons"]] \
            if "reasons" in show.columns else show[["symbol"]]
        st.dataframe(reasons, width="stretch", hide_index=True)

    render_approaching(table)
    return True


def render_approaching(table) -> None:
    """Names near a trigger they have NOT broken yet.

    WHY THIS PANEL EXISTS. Every gate above reports on a break that has
    already happened, because choose_direction requires the opening range
    to be broken before a symbol has a direction at all. So the table
    answers "what has moved" and can never answer "what might". Measured
    2026-09-18, the setups that cleared had already moved 3.16% at the
    median while asking for 3.10% more.

    This is the same scan read one step earlier: the level, the distance
    to it in units of the symbol's own volatility, and the gates that
    would still block it on arrival. It is NOT a prediction that the
    break comes, and most will not - being early costs precision, and
    saying so is the honest way to offer it.
    """
    # BOTH columns, because both are used below. Guarding only on
    # approach_atr left sort_values and the readiness count to raise a
    # KeyError - and this runs inside render_published_scan, which has no
    # error boundary, so that would have taken the metrics, the main table
    # and the reasons expander down with it. The feed and the UI deploy
    # separately, so a table written at one revision and read at another
    # is a normal state rather than a corner case.
    required = {"approach_atr", "approach_ready"}
    if not required <= set(table.columns):
        return
    near = table[table["approach_atr"].notna()].copy()
    if near.empty:
        return
    near["approach_ready"] = near["approach_ready"].fillna(False).astype(bool)
    near = near.sort_values(["approach_ready", "approach_atr"],
                            ascending=[False, True])
    ready = int(near["approach_ready"].sum())

    # A STATIC LABEL. Streamlit takes an expander's identity from its
    # label, so embedding the counts made every 20-second fragment tick a
    # different element - which remounts it and collapses the panel under
    # anyone who had opened it. The counts belong in the body.
    with st.expander("Approaching a trigger - names that have not broken yet"):
        st.caption(
            f"**{len(near):,} names**, of which **{ready}** have every "
            f"other gate already passing. Inside the opening range, within "
            f"{config.SCAN_APPROACH_MAX_ATR:g} ATR of the edge whose break "
            "would land them on the agreeing side of VWAP. "
            "`approach_ready` means every gate the scanner would run - "
            "including cost, win rate and room to move, judged on levels "
            "built at the trigger - already passes, so only the break is "
            "missing. It is NOT a forecast that the break happens, and "
            "most will not."
        )
        columns = [c for c in ("symbol", "approach_side", "approach_ready",
                               "approach_atr", "last", "approach_trigger",
                               "approach_distance", "rvol",
                               "relative_strength", "day_change_pct",
                               "approach_blockers")
                   if c in near.columns]
        st.dataframe(near[columns], width="stretch", hide_index=True)


def published_scan_fragment() -> None:
    """Re-read and redraw the feed's published scan. The timer body.

    WHY A FRAGMENT AND NOT THE EXISTING AUTO-REFRESH. That one
    (refresh_scan_fragment) re-RUNS a local scan from session state and
    says so - "Run a scan first; the timer refreshes an existing one". It
    never touched the published table, so the panel most people actually
    read only updated when the page happened to re-run for some other
    reason: the count and the "computed Ns ago" caption sat frozen while
    the feed published a new table every half minute.

    Cheap enough to do on a timer because it is a READ. scan_publish.load
    HEADs the object first and reuses the parsed table when the ETag has
    not moved, so a tick with nothing new costs one small request rather
    than 244 KB and a parquet parse.

    The verdict goes through session state because a fragment cannot
    return a value to the script that declared it. The body also runs
    inline on that declaring run, so the caller reads a fresh answer
    rather than a stale one.
    """
    st.session_state["published_scan_shown"] = render_published_scan()


def published_poll_interval() -> "int | None":
    """Seconds between re-reads, or None when there is nothing to poll.

    Outside trading hours the feed is scaled to zero and the object cannot
    change, so a timer would be pure waste. A page opened before the open
    therefore starts untimed; autoscan_watch_fragment is what wakes it
    once the session begins.
    """
    if not object_store.enabled():
        return None
    if not live_bars_in_hours():
        return None
    return max(5, int(getattr(config, "PUBLISHED_SCAN_POLL_SECONDS", 20)))


def render_scan_controls() -> tuple[str, bool, "datetime | None", bool]:
    """Scan inputs; returns (scope, want_options, as_of, run).

    CAPITAL AND RISK ARE NOT INPUTS ANY MORE. They only ever fed position
    sizing - how many shares fit - and never changed which setups appeared
    or how they ranked. Two number boxes at the top of the tab implied
    otherwise, and they came before the one control that does change the
    answer. Sizing now reads config.SCAN_CAPITAL and
    config.SCAN_RISK_PCT_PER_TRADE, which is what setups.evaluate already
    defaulted to when nothing was passed.
    """
    left, _ = st.columns([1, 2])
    with left:
        # index= AND key=, both deliberately. Without index the selectbox
        # defaults to item 0 - the ~2,570 full universe, nine minutes of
        # downloading - while the help text below claimed the default was
        # the F&O list. Without key the choice is lost on every Streamlit
        # restart, which silently put a live session back on the slow
        # scope at 09:06 on 2026-09-15.
        choices = list(offered_scopes())
        picked = st.selectbox(
            "Which stocks to look at", choices,
            index=choices.index(_DEFAULT_SCAN_SCOPE),
            key="scan_scope",
            help="The default is the ~210 large, heavily traded names that "
                 "have futures and options. The wider choices include "
                 "smaller stocks and take much longer.")
        # RESOLVED HERE, AND SAID OUT LOUD. Streamlit's selectbox returns a
        # persisted choice unchanged even when it is no longer among the
        # options, so a session that picked the unbounded scope before it
        # was withdrawn still arrives with it. run_scan would substitute
        # the default silently, and the spinner would go on promising a
        # sweep of ~2,570 names while 210 were scanned.
        scope, _ = resolve_scope(picked)
        if scope != picked:
            st.info(f"'{picked}' is no longer offered, so this scans "
                    f"{scope} instead. It fetched bars for about 2,570 "
                    f"symbols inside this page, which is what made the app "
                    f"unusable on 17 Sep. Set SCAN_UI_MAY_DOWNLOAD to "
                    f"bring it back.", icon=":material/info:")
    as_of = render_replay_controls()
    want_options = st.checkbox(
        "Also pick an option contract for each setup",
        help="Adds one NSE chain request per setup, so it is slower. Live "
             "scans only: a past session has no chain to read.")
    return (scope, want_options, as_of, st.button("Run intraday scan"))


def run_scan(scope: str, as_of: "datetime | None" = None):
    """Fetch bars and evaluate every symbol in scope, live or replayed.

    Sizing comes from config rather than from arguments: it affects only
    the share count on a setup, never whether the setup exists.
    """
    discovered = instruments.load_latest()
    if discovered is None:
        discovered = instruments.discover()
        instruments.save(discovered)
    scope, limit = resolve_scope(scope)
    if limit is None:
        equity = [inst.symbol for inst in discovered.fo_stocks]
    else:
        equity = [inst.symbol for inst in discovered.equities]
        if limit:
            equity = equity[:limit]
    tickers = tuple([instruments.to_ticker(sym) for sym in equity]
                    + [config.SCAN_BENCHMARK])

    actual_now = datetime.now(IST_ZONE)
    now = actual_now if as_of is None else as_of
    # ANY earlier instant is a replay, not merely an earlier date. Comparing
    # dates let a replay of 10:00 today pass straight through, and an audit
    # found the resulting scan carrying open interest captured at 20:36 and
    # an option chain quoting that day's closing prices, both presented as
    # 10:00 readings.
    replaying = as_of is not None and as_of < actual_now
    other_day = now.date() != actual_now.date()
    # The live stamp forces a fresh assemble whenever the feed has written a
    # newer bar; on a replay it is empty, so the cache behaves as before.
    bars = load_scan_bars(tickers, target=now.date() if other_day else None,
                          live_stamp="" if other_day else live_bar_stamp())
    if as_of is not None:
        bars = scan_data.truncate(bars, now)
    benchmark = scan_data.benchmark_change_pct(bars)
    # The instrument snapshot may only inform a scan of an instant at or
    # after its own capture time. NSE publishes no open-interest history, so
    # a later snapshot cannot be backfilled onto an earlier moment.
    states = (discovered.fo_state
              if discovered.captured_at <= now else {})

    evaluated = []
    for symbol in equity:
        ticker = instruments.to_ticker(symbol)
        frame = bars.intraday.get(ticker)
        if frame is None:
            continue
        reading = setups.measure(symbol, ticker, frame, bars.daily.get(ticker),
                                 benchmark, now, fo_state=states.get(symbol))
        if reading is not None:
            evaluated.append(setups.evaluate(
                reading, capital=config.SCAN_CAPITAL,
                risk_pct=config.SCAN_RISK_PCT_PER_TRADE))
    return setups.rank(evaluated), bars, benchmark, now, replaying


def render_scan_option_picks(actionable: list[setups.Setup]) -> None:
    """Show the tradeable contract, if any, for each actionable setup."""
    st.subheader("Option contracts")
    st.caption(
        "Direction comes from the equity setup above. All that is assessed "
        "here is whether a contract is liquid and cheap enough to express it. "
        "Buying only - selling naked options is never proposed."
    )
    discovered = instruments.load_latest()
    lot_sizes = ({inst.symbol: inst.lot_size or 0
                  for inst in discovered.fo_underlyings} if discovered else {})
    index_symbols = ({inst.symbol for inst in discovered.fo_indices}
                     if discovered else set())
    session = options_chain.open_session()
    for setup in actionable:
        lot_size = lot_sizes.get(setup.symbol, 0)
        with st.expander(f"{setup.symbol} {setup.direction}"):
            if lot_size <= 0:
                st.info("No lot size known, so option costs cannot be computed.")
                continue
            contracts, spot, expiry = options_chain.fetch_chain(
                setup.symbol, setup.symbol in index_symbols, session=session)
            if not contracts or spot is None:
                st.info("No option chain came back for this underlying.")
                continue
            metrics = options_chain.summarise(setup.symbol, contracts, spot,
                                              expiry)
            if metrics is not None:
                pcr = metrics.put_call_ratio
                st.write(
                    f"Expiry {expiry} | spot {spot:,.2f} | lot {lot_size} | "
                    f"ATM {metrics.atm_strike:,.0f} | IV call "
                    f"{metrics.atm_call_iv:.1f} / put {metrics.atm_put_iv:.1f}"
                    + (f" | PCR {pcr:.2f}" if pcr is not None else "")
                )
            contract, reasons = options_chain.pick_contract(
                contracts, spot, setup.direction, lot_size)
            if contract is None:
                st.warning(f"No tradeable contract: {reasons[0]}")
            else:
                side = "CALL" if setup.direction == LONG else "PUT"
                exact = options_chain.kite_tradingsymbol(
                    setup.symbol, expiry, contract.strike, side)
                name = exact or (f"{setup.symbol} {expiry} "
                                 f"{contract.strike:,.0f} {side}")
                outlay = contract.mid * lot_size
                st.success(f"BUY  **{name}**")
                if exact is None:
                    st.warning(
                        "Could not confirm this contract in Kite's "
                        "instrument master, so the name above is assembled "
                        "rather than verified. Check it before acting."
                    )
                st.write(
                    f"{setup.symbol} | expiry {expiry} | strike "
                    f"{contract.strike:,.0f} | {side} | lot {lot_size:,}"
                )
                st.write(
                    f"mid {contract.mid:.2f} (bid {contract.bid:.2f} / ask "
                    f"{contract.ask:.2f}) -> **Rs {outlay:,.0f} per lot**"
                )
                # The cost arithmetic, which is the part of this tool that
                # held up under measurement. Shown here rather than left
                # for the user to work out.
                if contract.mid > 0 and lot_size > 0:
                    breakeven = trade_costs.options_breakeven_pct(
                        contract.mid, 1, lot_size)
                    recover = contract.mid * (1 + breakeven / 100.0)
                    needed = (contract.strike + recover
                              if side == "CALL"
                              else contract.strike - recover)
                    move = ((needed / spot - 1.0) * 100.0
                            if spot else float("nan"))
                    st.write(
                        f"round-trip charges {breakeven:.2f}% of premium | "
                        f"break-even at expiry needs spot "
                        f"{needed:,.2f} ({move:+.2f}% from {spot:,.2f})"
                    )
                    if breakeven > 3.0:
                        st.error(
                            f"Charges are {breakeven:.1f}% of the premium on "
                            f"this position. The flat per-order fee dominates "
                            f"small premiums, and a move that large is not "
                            f"recoverable by being right about direction."
                        )
            for reason in reasons:
                st.caption(f"- {reason}")


def render_recent_filings(symbol: str, as_of=None) -> None:
    """Any exchange filing on this symbol in the last half hour.

    CONTEXT, NOT A SIGNAL, and the caption says so in as many words. The
    measurement is unambiguous: trades taken after a material filing
    resolved at 29.6% against 30.6% for trades with no filing, so this
    must never read as a reason to take the trade. What DOES replicate out
    of sample is that filings raise volatility - which makes the stop
    tighter and the plausible move larger than the fourteen-session ATR
    behind them assumes, and that is a reason for a human to be careful.

    Silence here is only meaningful if the store is current, so a stale
    store says so rather than implying there is no news.
    """
    try:
        note = filings.note_for(symbol, now=as_of)
        age = filings.store_age()
    except Exception as exc:
        logger.warning("Filings lookup failed for %s: %s", symbol, exc)
        return
    if age is None:
        return
    newest, behind = age
    if behind > 1:
        st.caption(
            f":material/history: Filings shown are only as current as the "
            f"store, whose newest record is {newest} - {behind} days back. "
            f"Rebuild with `python fetch_announcements.py` before reading "
            f"silence here as 'no news'.")
        return
    if note:
        st.caption(f":material/campaign: {note}")


def render_scan_results(ranked: list[setups.Setup], bars: scan_data.BarSet,
                        benchmark, now: datetime, want_options: bool,
                        replaying: bool = False,
                        with_terms: bool = True) -> None:
    """Render one scan's headline counts, table, gate detail and options."""
    actionable = [s for s in ranked if s.actionable]
    directional = [s for s in ranked
                   if s.direction != setups.NO_SETUP and not s.actionable]
    left, mid, right = st.columns(3)
    left.metric("Symbols with bars", f"{bars.covered}/{bars.requested}")
    mid.metric("Got a direction", f"{len(actionable) + len(directional)}")
    right.metric("Cleared every gate", f"{len(actionable)}")
    nifty = (f" | Nifty {benchmark:+.2f}% today" if benchmark is not None
             else " | today's Nifty move is unavailable, so the "
                  "compare-to-the-market check refuses to pass")
    stamp = (now.strftime("%Y-%m-%d %H:%M:%S") if replaying
             else now.strftime("%H:%M:%S"))
    origin = getattr(bars, "source", "download")
    streaming = getattr(bars, "live_symbols", 0)
    detail = ""
    if origin == "live feed":
        age = getattr(bars, "live_age_seconds", float("nan"))
        detail = f" ({streaming}/{bars.requested} streaming"
        detail += f", newest {age:.0f}s old)" if age == age else ")"
    st.caption(
        f"{'Replayed as of' if replaying else 'Scanned'} {stamp} IST{nifty} | "
        f"{indicators.minutes_left_in_session(now)} min left in the session | "
        f"bars from **{origin}**{detail}"
    )
    if not replaying:
        # Three distinct states, where there used to be two. The middle one
        # is the dangerous one: bars that came through the live path but
        # where most symbols were served from cache or the startup seed, so
        # they are as stale as a download while looking live.
        thin = (origin == "live feed" and bars.requested
                and streaming < config.SCAN_LIVE_MIN_COVERAGE * bars.requested)
        if origin != "live feed":
            st.warning(
                "These bars were DOWNLOADED, not streamed. The newest candle "
                "can be several minutes old, so entry and stop are computed "
                "against a price that has already moved. Start the live feed "
                "above to remove both the wait and the staleness."
            )
        elif thin:
            st.warning(
                f"Only {streaming} of {bars.requested} symbols are actually "
                "streaming. The rest came from cached history or the startup "
                "backfill, so their newest candle is as old as a download's. "
                "Check the live feed is subscribed to this universe."
            )

    if not actionable:
        premature = [s for s in ranked if not s.readings.range_closed]
        if not bars.intraday or bars.covered == 0:
            # A total download failure must not read as a quiet market.
            st.error(
                f"No bars came back at all ({bars.covered}/{bars.requested}). "
                f"This is a data failure, not an absence of setups - the "
                f"source may be rate limiting or the market may have been "
                f"closed on the chosen date. Re-run before drawing any "
                f"conclusion."
            )
        elif premature and len(premature) == len(ranked):
            open_h, open_m = config.SCAN_SESSION_OPEN
            closes_at = open_h * 60 + open_m + config.SCAN_OPENING_RANGE_MINUTES
            st.warning(
                f"**No signal is computable yet.** The opening range covers "
                f"the first {config.SCAN_OPENING_RANGE_MINUTES} minutes and "
                f"does not close until "
                f"{closes_at // 60:02d}:{closes_at % 60:02d}. Until then the "
                f"latest bar is one of the bars defining the range, so the "
                f"range brackets the current price by construction and no "
                f"breakout can be represented. This is arithmetic, not a "
                f"reading of the market: the same empty result would appear "
                f"on the sharpest gap-up morning on record."
            )
        else:
            st.info(
                "Nothing passed every check. On most days that is the "
                "expected answer, not a fault. To get a direction at all, "
                "the price has to be above its average price for the day "
                "AND have broken out of the range it set in the first 15 "
                "minutes. Usually it has done one but not the other."
            )
    else:
        frame = scan_frame(actionable)
        # Selectable, like every other table in the app: the row already
        # holds every number the watch needs, and retyping them into a
        # form was tedious and a chance to fumble a digit into a watch
        # that then reports nonsense.
        watch_table(frame, "scan_pick", note="from the intraday scan")
        show_terms("How a stop and target are set",
                   "Why charges matter so much",
                   "What 'volume vs normal' tells you")
        # Each setup is sized to a fraction of capital on its own. Taken
        # together they are not, and nothing said so: sixteen setups on one
        # session came to 16x the assumed capital in notional.
        total_risk = sum(s.levels.lot_risk for s in actionable)
        total_notional = sum(s.levels.entry * s.levels.quantity
                             for s in actionable)
        st.warning(
            f"**Aggregate exposure:** taking all {len(actionable)} risks "
            f"{total_risk:,.0f} rupees across {total_notional:,.0f} of "
            f"notional. Each row is sized independently against its own "
            f"stop, so the per-trade cap does not bound the total."
        )
        best = actionable[0]
        st.markdown(f"**Top setup: {best.symbol} {best.direction}**")
        st.write(setups.explain(best))

    shown = actionable or directional[:5]
    if shown:
        st.subheader("Gate detail" if actionable
                     else "Blocked - these are NOT suggestions")
        if not actionable:
            # WHY THIS WARNING EXISTS. A blocked setup used to render as
            # ten green PASS lines with one FAIL among them, above a full
            # set of levels - and when nothing is actionable this panel is
            # the only thing on screen. Measured on HDFCLIFE at 10:00 on
            # 2026-09-10: SHORT, entry 511.20, stop 516.00, target 501.60,
            # every gate passing except "relative strength +3.06pp, needs
            # to underperform". It read as a trade ticket and was taken as
            # one. The levels below are what the setup WOULD have used had
            # it passed, which is not the same thing as a plan.
            st.warning(
                "Nothing cleared the gates, so these are the closest "
                "misses. Each one names the check that stopped it. The "
                "levels shown are what it WOULD have used had it passed - "
                "they are arithmetic, not a recommendation.",
                icon=":material/block:")
        for setup in shown[:8]:
            failed = [r for r in setup.reasons if "[FAIL]" in r]
            if setup.actionable:
                title = (f"{setup.symbol} - {setup.direction} "
                         f"(score {setup.rank_score:.3f})")
            else:
                # The blocking reason belongs in the TITLE. Buried among
                # eleven lines it was invisible.
                why = (failed[0].split(" [")[0] if failed
                       else "no directional agreement")
                title = f"{setup.symbol} - BLOCKED: {why}"
            with st.expander(title):
                # Failures first for a blocked setup, so the reason is not
                # read after the levels that look like a plan.
                ordered = (setup.reasons if setup.actionable
                           else failed + [r for r in setup.reasons
                                          if "[FAIL]" not in r])
                for reason in ordered:
                    st.caption(reason)
                render_recent_filings(setup.symbol, now)

    if want_options and actionable:
        if replaying:
            st.warning(
                "**Option contracts cannot be shown for a replayed instant.** "
                "NSE serves only a live chain and publishes no archive, so a "
                "chain fetched now describes now. Pinning it onto an earlier "
                "signal changes the strike, the premium and the spread: an "
                "audit of a 10:00 replay found every printed spot equal to "
                "that day's close, ten for ten. Capture chains with the "
                "Instrument sync tab and replays after the first capture can "
                "read a real one."
            )
        else:
            render_scan_option_picks(actionable[:5])


def _forecast_feature_count() -> "int | str":
    """How many features the research module actually uses today.

    Read from the module rather than typed here, so this caption cannot
    drift the way the one above it did.
    """
    try:
        import forecast_intraday
        return len(forecast_intraday.FEATURES)
    except Exception:
        return "an unknown number of"


def _forecast_permutations() -> "int | str":
    try:
        import forecast_intraday
        return forecast_intraday.PERMUTATIONS
    except Exception:
        return "an unknown number of"


def live_feed_state() -> dict:
    """What the live bar feed is doing, for the status panel."""
    try:
        import live_bars
        import live_feed
    except Exception as exc:
        return {"available": False, "why": str(exc)}
    state = dict(live_bars.status())
    # A storage fault reuses the panel's existing "unavailable, and here is
    # why" shape rather than rendering as a feed that has not started.
    if state.get("error"):
        return {"available": False, "why": state["error"]}
    state["available"] = True

    # A REMOTE FEED HAS NO LOCAL PID. live_feed.take_lock() deliberately
    # writes no lock file on Fargate - a lock guards a shared disk, and a
    # container's disk is neither shared nor durable, so ECS's
    # desiredCount 1 is the singleton guarantee instead. The consequence
    # was this panel: it derived "running" purely from that lock file, so
    # with the feed on ECS it read "stopped" forever while 14,242 bars
    # across 2,489 symbols arrived in S3 behind it.
    #
    # The honest signal for a feed you cannot see the process of is
    # whether its BARS are arriving. Same threshold the scan itself uses
    # to decide a file is too stale to trade on, so the panel and the
    # scanner cannot disagree about whether the feed is alive.
    import object_store

    age = state.get("age_seconds")
    fresh = (isinstance(age, (int, float)) and age == age
             and age <= config.SCAN_LIVE_MAX_AGE_SECONDS)
    if object_store.enabled():
        state["remote"] = object_store.describe()
        state["pid"] = None
        state["running"] = bool(state.get("bars")) and fresh
        return state

    state["remote"] = ""
    lock = live_feed.LOCK
    holder = None
    if lock.exists():
        try:
            holder = int(lock.read_text(encoding="utf-8").strip())
        except Exception:
            holder = None
    state["pid"] = holder
    state["running"] = bool(holder and live_feed._alive(holder))
    return state


def start_live_feed() -> str:
    """Spawn the feed DETACHED so it outlives Streamlit's script reruns.

    Streamlit re-executes this file on every interaction, so anything owned
    by the script dies with the session. A detached child keeps its socket
    and its output file; live_feed's PID lock stops a second one starting,
    which matters because Kite allows only three sockets per API key.
    """
    import subprocess
    import sys

    flags = 0
    for name in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS",
                 "CREATE_NO_WINDOW"):
        flags |= getattr(subprocess, name, 0)
    try:
        subprocess.Popen(
            [sys.executable, "-u", "-m", "live_feed", "--flush-every", "15"],
            cwd=str(config.PROJECT_ROOT), creationflags=flags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL)
    except Exception as exc:
        return f"could not start: {exc}"
    return ("started - it prewarms prior sessions first, which takes a few "
            "minutes, then streams")


def render_live_feed_panel() -> None:
    """Feed status, freshness, and a start control."""
    state = live_feed_state()
    if not state.get("available"):
        st.caption(f"Live bars unavailable: {state.get('why', 'unknown')}")
        return
    with st.expander("Live tick feed", expanded=not state.get("running")):
        st.caption(
            f"A separate process streams ticks and builds "
            f"{config.SCAN_BAR_INTERVAL} bars - the same size the historical "
            f"fetch asks for, or the two halves of a session would not join "
            f"- so a scan reads them instead of downloading every symbol at "
            f"Kite's 3 requests a second. Prior sessions come from a cached "
            f"window ending yesterday; only today comes from the stream."
        )
        if state.get("remote"):
            st.caption(f"Reading bars published by the container at "
                       f"**{state['remote']}** - there is no local process "
                       f"to start or stop.")
        left, mid, right = st.columns(3)
        left.metric("Feed", "running" if state.get("running") else "stopped")
        mid.metric("Instruments", state.get("instruments", 0))
        age = state.get("age_seconds")
        right.metric("Newest bar",
                     f"{age:.0f}s ago" if isinstance(age, float) and age == age
                     else "none yet")
        if (state.get("remote") and state.get("bars")
                and not state.get("running")):
            st.warning(
                f"Bars are present but the newest is "
                f"{state.get('age_seconds', 0):.0f}s old, past the "
                f"{config.SCAN_LIVE_MAX_AGE_SECONDS}s limit - so the "
                f"container has most likely stopped. Check it with: "
                f"aws ecs describe-services --cluster avsp-cluster "
                f"--services zone-pulse --profile innomesh-dev")
        if state.get("running") and not state.get("bars"):
            st.info(
                "Streaming, but no completed bar yet. Bars are stamped at "
                "the start of their five-minute bucket and only written once "
                "the bucket closes, so the first one appears at the next "
                "five-minute boundary."
            )
        if state.get("partial"):
            st.caption(
                f"{state['partial']} bar(s) marked partial - the feed joined "
                f"part-way through their bucket, so their volume is measured "
                f"from an unknown baseline and the scan skips them."
            )
        # Only for a LOCAL feed. A remote one has no PID and nothing on
        # this machine to taskkill, so the line rendered as the frankly
        # useless "PID None - stop it from the terminal that owns it".
        if state.get("running") and not state.get("remote"):
            st.caption(f"PID {state.get('pid')} - stop it from the terminal "
                       f"that owns it, or with taskkill.")


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, max_entries=4,
               show_spinner=False)
def load_horizon_picks(symbols: tuple[str, ...], top: int,
                       bucket: int = 0, anchor_live: bool = False) -> dict:
    """Ranked assessments per daily horizon, cached on the universe.

    `bucket` is the clock floored to HORIZON_ANCHOR_SECONDS and exists
    solely to be part of the cache key. The history behind these tables
    changes once a session, but the PRICE they are anchored on changes all
    day, so caching on (symbols, top) alone would have served the first
    minute's levels until the TTL expired. A bucket in the key re-runs the
    sweep exactly as often as the anchor can move.

    Which is exactly why max_entries is set: a bucket a minute against a
    900-second TTL is fifteen live results, and cache_data PICKLES, so
    they are fifteen independent copies rather than fifteen references.
    Four covers the live bucket, the previous one, and the two scope sizes
    a session realistically switches between.
    """
    live = horizons.live_prices_for(list(symbols)) if anchor_live else {}
    picks = horizons.assess_universe(list(symbols), top=top, live=live)
    return {"picks": picks, "live_count": len(live)}


HORIZON_ANCHOR_SECONDS = 60


def horizon_anchor() -> tuple:
    """(should we anchor on live prices, cache bucket, why).

    Live only during trading hours and only with a usable session. Outside
    both, the last daily close is not a degraded anchor - it is the
    correct one, because it is where the instrument actually last traded.
    """
    import time

    bucket = int(time.time() // HORIZON_ANCHOR_SECONDS)
    if not live_bars_in_hours():
        return False, 0, ""
    try:
        import market_source
        if not market_source.session_available():
            return False, 0, ("no Kite session, so these are anchored on the "
                              "last daily close and may be a day stale")
    except Exception:
        return False, 0, "no market source, so anchored on the last close"
    return True, bucket, ""


def render_horizon_tables(symbols: list, top: int = 20) -> None:
    """Short, mid and long horizon tables, each filled as it completes.

    Placeholders are written per horizon rather than after all three, so
    the first table is readable while the rest are still being scored.
    """
    st.subheader("Longer horizons")
    slots = {}
    for name in ("short", "mid", "long"):
        sessions = horizons.HORIZONS[name]["sessions"]
        st.markdown(f"**{name.title()} term** - about {sessions} sessions, "
                    f"round trip {horizons.HORIZONS[name]['cost_pct']:.3f}%")
        slots[name] = st.empty()
        slots[name].info("scoring...")
    anchor_live, bucket, why = horizon_anchor()
    result = load_horizon_picks(tuple(symbols), top, bucket, anchor_live)
    picks = result.get("picks") or {}
    # THE AGE OF THE DAILY STORE, on screen rather than in a log. Trend,
    # drawdown and position-in-range come from it, two of them are ranking
    # columns, and the caption below says they "come from completed daily
    # bars" - which was true and yet misleading while the store sat two
    # sessions behind.
    newest, missing = horizons.daily_store_lag()
    if missing is not None and missing > horizons.DAILY_MAX_STALE_SESSIONS:
        st.warning(
            f"**The daily bars behind these tables stop at {newest}, "
            f"{missing} completed session(s) ago.** Trend, drawdown and "
            f"position in range describe that date, not today - and two of "
            f"them are ranking columns, so the ORDER below is that old too. "
            f"Rebuild with `python -m bar_store --intervals day`.",
            icon=":material/update:")
    if anchor_live and result.get("live_count"):
        st.caption(
            f"Prices, stops and exits are anchored on the LIVE price for "
            f"{result['live_count']:,} symbols, refreshed about every "
            f"{HORIZON_ANCHOR_SECONDS}s. The trend and volatility columns "
            f"still come from completed daily bars, which is why the "
            f"ordering barely moves during a session."
        )
    else:
        st.caption(
            f"{why or 'Anchored on the last daily close.'} Stops and exits "
            f"are computed from that price, so during a session they can sit "
            f"a full day's move away from where the stock is now."
        )
    if not picks:
        for slot in slots.values():
            slot.error(
                "No consolidated daily bars. Build the store first: "
                "`.venv\\Scripts\\python -m bar_store`"
            )
        return
    for name, slot in slots.items():
        rows = picks.get(name) or []
        if not rows:
            slot.info("nothing assessable at this horizon")
            continue
        frame = horizons.to_frame(rows)
        # .container() rather than the bare placeholder: the action button
        # renders below the table and st.empty holds one element.
        watch_table(frame, f"horizon_pick_{name}",
                    target=slot.container(),
                    note=f"from the {name} horizon table")


@st.cache_data(ttl=60, show_spinner=False)
def analyse_instrument(symbol: str, with_intraday: bool) -> dict:
    """One instrument's report, as a plain dict so Streamlit can cache it.

    Cached briefly rather than not at all: retyping the same symbol should
    not refetch, but a 60-second ceiling keeps an intraday verdict from
    going stale on screen.
    """
    report = instrument_report.analyse(symbol, include_intraday=with_intraday)
    return {
        "found": report.found, "symbol": report.symbol,
        "note": report.note, "price": report.price, "in_fo": report.in_fo,
        "lot_size": report.lot_size, "buys": report.buys,
        "as_of": report.as_of, "live_anchored": report.live_anchored,
        "table": instrument_report.summary_frame(report),
        "pivots": (None if report.pivots is None else
                   {"S3": report.pivots.s3, "S2": report.pivots.s2,
                    "S1": report.pivots.s1, "Pivot": report.pivots.pivot,
                    "R1": report.pivots.r1, "R2": report.pivots.r2,
                    "R3": report.pivots.r3}),
        "detail": {name: {"verdict": v.verdict, "blocker": v.blocker,
                          "reasons": v.reasons}
                   for name, v in report.verdicts.items()},
    }


@st.cache_data(ttl=3600, show_spinner=False)
def search_instruments(text: str, limit: int = 25) -> list:
    """Typeahead matches, cached so a keystroke does not rebuild anything.

    Returned as plain tuples rather than Match objects: Streamlit pickles
    whatever a cached function returns, and a tuple cannot go stale against
    a code change to the dataclass.
    """
    return [(m.symbol, m.label, m.kind, m.underlying,
             m.expiry, m.strike, m.right, m.lot_size)
            for m in instrument_search.search(text, limit=limit)]


def render_pivot_levels(levels: "dict | None", price: float) -> None:
    """The previous session's pivot ladder, purely as reference levels.

    Shown because they are on every broker's screen and cost nothing to
    display. NOT used for anything: placing stops on them was built and
    measured, and conditioning on the session the pivot stop is hit if
    anything MORE often (odds ratio 1.100, p 0.616 over 8,687 sessions).
    See pivot_measurement. Levels to look at, not to trade against.
    """
    if not levels:
        return
    with st.expander("Yesterday's pivot levels"):
        st.caption(
            "Arithmetic from yesterday's high, low and close - fixed before "
            "today's open, and the same numbers every broker shows. We "
            "tested whether price actually turns at them and could not "
            "show that it does, so treat them as landmarks rather than as "
            "signals."
        )
        rows = [{"Level": name, "Price": round(value, 2),
                 "vs now": f"{(value / price - 1.0) * 100:+.2f}%"}
                for name, value in levels.items()]
        # Resistances first, descending, so the ladder reads like a chart.
        frame = pd.DataFrame(rows).sort_values("Price", ascending=False)
        # show_table, not st.dataframe: every table in this app gets its
        # tooltips from one helper, and bypassing it is how a column ends
        # up explained in one place and bare in another.
        show_table(frame)
        show_terms("What the pivot levels are")


def render_derivative_panel(symbol: str, kind: str, underlying: str,
                            expiry, strike: float, right: str,
                            lot_size: int) -> None:
    """What is known about a contract, and what is honestly not.

    A derivative reaches here rather than the horizon assessment because
    the consolidated store holds cash daily bars only, and the assessment
    also assumes a flat delivery round trip (horizons._DELIVERY_COST_PCT)
    and a holding period bounded
    by nothing but the horizon. Neither holds for a contract with an
    expiry, so producing a number would be worse than declining to.
    """
    side = {"CE": "call", "PE": "put"}.get(right, "")
    title = f"**{symbol}**  |  {underlying} {kind}"
    if side:
        title += f", {side} at {strike:,.0f}"
    st.markdown(title)
    left, mid, right_col = st.columns(3)
    left.metric("Lot size", f"{lot_size:,}")
    if expiry is not None:
        days = (expiry - datetime.now(IST_ZONE).date()).days
        mid.metric("Expires", f"{expiry:%d %b %Y}")
        right_col.metric("Days left", days)
        if days < 0:
            st.error("This contract has already expired.")
        elif days <= 7:
            st.warning(
                f"{days} day(s) to expiry. Time decay dominates an option "
                f"this close to expiry, and the horizon buckets here start "
                f"at 10 sessions - none of them fits."
            )
    render_contract_assessment(symbol, kind, underlying)
    if underlying and st.button(f"Analyse {underlying} instead",
                                key=f"jump_{symbol}"):
        st.session_state["lookup_last"] = underlying
        st.session_state["lookup_symbol"] = underlying
        # The picker has to be cleared as well. This contract's own label
        # is still among the matches for its underlying, so leaving the
        # selection in place made Streamlit restore it and render this
        # same panel again - the escape hatch never escaped.
        st.session_state.pop("lookup_pick", None)
        st.rerun()
    if kind == instrument_search.KIND_OPTION:
        render_option_cost_arithmetic(lot_size)


def render_contract_assessment(symbol: str, kind: str,
                               underlying: str) -> None:
    """A horizon assessment for a FUTURE, or an honest refusal for an option.

    A future gets the real thing: the underlying's daily readings, this
    contract's live price for the levels, the futures cost stack, and only
    the horizons that fit before expiry.

    An option does not, and the reason is a modelling limit rather than
    missing plumbing. A premium moves with the underlying only through
    delta and decays with time whatever the underlying does, so the
    "plausible move" of a premium is not something this project can
    compute. Inventing one would contradict everything else measured here.
    What an option gets instead is the underlying's assessment, labelled as
    the underlying's, plus the premium arithmetic - which is the half that
    survived measurement.
    """
    if kind == instrument_search.KIND_OPTION:
        st.info(
            f"**No horizon verdict on an option contract, and that is a "
            f"modelling limit rather than a verdict.** A premium tracks "
            f"{underlying} only partly, through delta, and decays with time "
            f"whatever {underlying} does - so a plausible move for the "
            f"premium is not something this tool can compute honestly. "
            f"Below is {underlying}'s own assessment, which is a statement "
            f"about {underlying} and not about this contract, and the "
            f"premium cost arithmetic, which is exact."
        )
        if underlying:
            report = analyse_instrument(underlying, False)
            if report.get("found"):
                table = report.get("table")
                if table is not None and not table.empty:
                    st.markdown(f"**{underlying}** "
                                f"{report['price']:,.2f}"
                                + ("  (live)" if report.get("live_anchored")
                                   else "  (last close)"))
                    watch_table(table, f"under_pick_{underlying}",
                                symbol=underlying,
                                entry=report.get("price"),
                                note=f"{underlying}, via an option lookup")
                    st.caption(
                        f"Every level above is a level on {underlying} "
                        f"itself. None of them is a level on this option, "
                        f"and adding one watches {underlying}.")
        return
    match = instrument_search.find(symbol)
    if match is None:
        st.caption("Contract not found in the instrument master.")
        return
    days = match.days_to_expiry
    if not horizons.reachable(days):
        st.warning(
            f"No horizon fits before expiry: {days} calendar day(s) left, "
            f"and the shortest horizon here is "
            f"{horizons.HORIZONS['short']['sessions']} sessions. A verdict "
            f"for a holding period the contract cannot survive would be an "
            f"answer to an impossible question."
        )
        return
    with st.spinner(f"assessing {symbol}..."):
        verdicts = contract_verdicts(symbol)
    if not verdicts:
        st.caption("Not enough underlying history to assess this contract.")
        return
    rows = []
    for name, detail in verdicts.items():
        rows.append({
            "Horizon": name,
            "Held": f"{horizons.HORIZONS[name]['sessions']} session(s)",
            "Verdict": detail["verdict"],
            "View": detail["view"],
            "Stop loss at": detail["stop"],
            "Exit price": detail["exit"],
            "Round trip": f"{detail['cost']:.3f}%",
            "Move vs fees": detail["covers"],
            "Blocked by": detail["blocker"],
        })
    priced = [d["price"] for d in verdicts.values() if d.get("price")]
    watch_table(pd.DataFrame(rows), f"contract_pick_{symbol}",
                symbol=symbol, entry=(priced[0] if priced else None),
                note=f"{symbol} futures")
    st.caption(
        f"Readings from **{underlying}**'s daily history - a contract's own "
        f"history is weeks long and breaks across rolls. Levels from **this "
        f"contract's** live price. Round trip is the futures stack on "
        f"notional, not the "
        f"{horizons.HORIZONS['short']['cost_pct']:.2f}% a cash delivery pays."
    )
    for name, detail in verdicts.items():
        with st.expander(f"{name} - {detail['verdict']}"):
            for line in detail["reasons"]:
                st.caption(f"- {line}")


@st.cache_data(ttl=60, show_spinner=False)
def contract_verdicts(symbol: str) -> dict:
    """Futures verdicts per horizon, as plain dicts so they cache."""
    match = instrument_search.find(symbol)
    if match is None:
        return {}
    frames = bar_store.load("day", symbols=[match.underlying, "NIFTY 50"])
    frame = frames.get(match.underlying)
    if frame is None or frame.empty:
        return {}
    live = horizons.live_prices_for([symbol]).get(symbol)
    verdicts = instrument_report.assess_contract(
        match, frame, frames.get("NIFTY 50"), live)
    out = {}
    for name, verdict in verdicts.items():
        a = verdict.assessment
        out[name] = {
            "verdict": verdict.verdict, "view": a.direction,
            # The contract's own price, carried so a row from this table
            # can be watched: the table shows a stop and an exit but no
            # price, and a watch with no entry is unwatchable.
            "price": round(a.price, 2),
            "stop": round(a.stop_price, 2), "exit": round(a.target_price, 2),
            "cost": a.cost_pct, "covers": f"{a.cost_multiple:.0f}x",
            "blocker": (verdict.blocker.split(" [")[0]
                        if verdict.blocker else ""),
            "reasons": list(verdict.reasons),
        }
    return out


def render_fno_browser(underlying: str) -> None:
    """Reach a contract by expiry and strike rather than from a flat list.

    A strike ladder is dozens of contracts - 39 strikes on one RELIANCE
    expiry - so a list of chips is the wrong control no matter how long it
    is allowed to be.
    """
    futures = [m for m in instrument_search.search(
        underlying, limit=40, kinds=(instrument_search.KIND_FUTURE,))
        if m.underlying == underlying]
    expiries = instrument_search.expiries_for(underlying)
    if not futures and not expiries:
        return
    with st.expander(f"F&O contracts on {underlying}"):
        if futures:
            st.markdown("**Futures**")
            picked = st.pills(
                "expiry", [f.symbol for f in futures],
                selection_mode="single", key=f"fut_{underlying}",
                label_visibility="collapsed")
            if picked:
                render_contract_assessment(
                    picked, instrument_search.KIND_FUTURE, underlying)
        if not expiries:
            return
        st.markdown("**Options**")
        left, mid, right = st.columns([2, 2, 1])
        expiry = left.selectbox(
            "Expiry", expiries, key=f"opt_exp_{underlying}",
            format_func=lambda d: f"{d:%d %b %Y}")
        strikes = instrument_search.strikes_for(underlying, expiry)
        if not strikes:
            st.caption("no strikes listed for that expiry")
            return
        strike = mid.selectbox(
            f"Strike ({len(strikes)} listed)", strikes,
            index=len(strikes) // 2, key=f"opt_strike_{underlying}",
            format_func=lambda v: f"{v:,.0f}")
        side = right.segmented_control(
            "Side", ["CE", "PE"], default="CE",
            key=f"opt_side_{underlying}")
        contract = instrument_search.contract(underlying, expiry, strike,
                                              side or "CE")
        if contract is None:
            st.caption("no contract at that combination")
            return
        st.caption(contract.label)
        render_option_cost_arithmetic(contract.lot_size)


def render_option_cost_arithmetic(lot_size: int) -> None:
    """The one part of an option that IS measurable: what it costs to trade.

    Charged on premium turnover, with brokerage a FLAT 20 rupees an order.
    That is why a cheap option is expensive: the flat fee is a percentage
    of premium that grows as the premium shrinks, and on a small position
    it can dwarf every exchange charge put together.
    """
    with st.expander("What a round trip on this contract costs", expanded=True):
        st.caption(
            "This is arithmetic, not a forecast, and it is the half of this "
            "tool that held up under measurement. Enter the premium you "
            "would pay and it says how far the option has to move before "
            "you keep anything."
        )
        cols = st.columns(2)
        premium = cols[0].number_input(
            "Premium per share (Rs)", min_value=0.05, value=50.0, step=1.0,
            help="The quoted option price, per share - not per lot.")
        lots = cols[1].number_input("Lots", min_value=1, value=1, step=1)
        breakeven = trade_costs.options_breakeven_pct(
            float(premium), int(lots), int(lot_size or 1))
        outlay = float(premium) * int(lots) * int(lot_size or 1)
        charges = trade_costs.options_cost(
            float(premium), float(premium), int(lots),
            int(lot_size or 1)).total
        one, two, three = st.columns(3)
        one.metric("Premium outlay", f"Rs {outlay:,.0f}")
        two.metric("Round-trip charges", f"Rs {charges:,.0f}")
        three.metric("Breakeven move", f"{breakeven:.2f}%",
                     help="How far the premium must rise before the trade "
                          "makes anything after charges.")
        if breakeven >= 5.0:
            st.warning(
                f"Charges are {breakeven:.1f}% of the premium. Being right "
                f"about direction is not enough at that level - the move "
                f"has to clear this before anything reaches you."
            )


def scanner_view(symbol: str) -> tuple:
    """(direction, actionable, blocker) for one symbol from the live scan.

    Read from the scan already in session state rather than re-scanning:
    the watch must be cheap enough to render on every rerun, and a scan
    that is 60 seconds old is the same scan the user is looking at.
    Returns (None, None, "") when that symbol was not in the last scan,
    which leaves the watch reporting the level checks only.
    """
    stored = st.session_state.get("scan")
    if not stored:
        return None, None, ""
    ranked = stored[0] or []
    for setup in ranked:
        if setup.symbol != symbol:
            continue
        # ENTRY-ONLY GATES ARE IGNORED FOR A POSITION ALREADY HELD, and
        # the room-to-move gate is the reason this distinction now exists.
        # It refuses a setup with less plausible move left than the day has
        # already spent, which is the right question when deciding whether
        # to ENTER and the wrong one once you are in: a position that is
        # working has moved further from the previous close and has fewer
        # bars left, so its ratio falls monotonically as it wins. Left
        # unfiltered the watch would headline "the setup no longer clears
        # the gates" on precisely the trades going well, and a tab whose
        # purpose is to break silence cannot afford to cry wolf.
        failed = [r for r in setup.reasons
                  if "[FAIL]" in r and "[ENTRY]" not in r]
        blocker = failed[0] if failed else ""
        # `actionable` is recomputed from the surviving failures for the
        # same reason: setup.actionable already folded the entry gate in.
        holds_up = not failed
        direction = (setup.direction
                     if setup.direction in (position_watch.LONG,
                                            position_watch.SHORT) else None)
        return direction, holds_up, blocker
    return None, None, ""


def watch_statuses(positions: list) -> tuple:
    """(ranked statuses, note) for everything being watched, priced live.

    The scanner's own view comes from the scan already in session state
    rather than a fresh scan: this runs every few seconds and must stay
    cheap, and a 60-second-old scan is the same scan on screen.
    """
    note = ""
    live = {}
    if positions:
        live = horizons.live_prices_for([p.symbol for p in positions])
        # DERIVED FROM THE RESULT, not from an exception: live_prices_for
        # catches everything and returns {}, so a try/except here could
        # never fire and a dead feed showed no warning at all - every
        # position simply read NO LIVE PRICE, which is neither an alert
        # nor a warning. Silence in the alerting system is the one failure
        # this whole tab exists to remove.
        if not live:
            note = ("**No live prices came back, so nothing below has been "
                    "checked.** That is not the same as nothing being "
                    "wrong. The market may be closed; during a session it "
                    "means the Kite session has expired or the feed is "
                    "down, and the watch cannot alert you until it is back.")
    out = []
    for held in positions:
        direction, actionable, blocker = scanner_view(held.symbol)
        out.append(position_watch.assess(
            held, live_price=live.get(held.symbol),
            current_direction=direction, actionable=actionable,
            blocker=blocker))
    return position_watch.rank(out), note


# --- adding one, from a row or from the lookup ---------------------------

def open_watch_draft(symbol: str, side: str, entry: float, stop: float,
                     target: float, quantity: int = 0, note: str = "") -> None:
    """Stash prefilled levels and rerun the whole app so the modal opens.

    scope="app" rather than the default: the scan table lives inside an
    auto-refreshing fragment, and a dialog opened from inside a fragment
    is a corner of Streamlit not worth betting an alert on. Rerunning the
    app puts the dialog call at the top level where it plainly works.
    """
    # THE WIDGETS MUST BE CLEARED FIRST. number_input, text_input and
    # selectbox take their identity from the KEY alone in 1.58, so `value=`
    # is only an initial default: once draft_entry holds state, a second
    # open of the modal shows the FIRST row's numbers and quietly ignores
    # the prefill. Selecting row B and pressing its button would then add
    # row A. Popping the keys makes each open a fresh form.
    for stale in ("draft_symbol", "draft_side", "draft_qty", "draft_entry",
                  "draft_stop", "draft_target", "draft_note"):
        st.session_state.pop(stale, None)
    st.session_state["watch_draft"] = {
        "symbol": symbol, "side": side, "entry": float(entry),
        "stop": float(stop), "target": float(target),
        "quantity": int(quantity or 0), "note": note,
    }
    # An app-scope rerun, so the dialog is called from the top level. A
    # dialog opened from inside the auto-refreshing scan fragment is a
    # corner of Streamlit not worth betting an alert on.
    st.rerun(scope="app")


def forget_watch_draft() -> None:
    """Drop the prefilled draft. Called when the modal is dismissed.

    Without this, Escape or the X closed the dialog with no rerun, the
    draft survived in session state, and the modal reappeared on the next
    full rerun - with the previous row's numbers still in its widgets.
    """
    st.session_state.pop("watch_draft", None)


@st.dialog("Add to Positions", on_dismiss=forget_watch_draft)
def watch_dialog() -> None:
    """Confirm or correct the suggested levels, then watch them.

    Everything is prefilled from the row that opened it, and everything is
    editable, because the row cannot know one crucial number: what you
    actually filled at. Its entry is the last traded price when the scan
    ran.
    """
    draft = st.session_state.get("watch_draft") or {}
    sides = [position_watch.LONG, position_watch.SHORT]
    st.caption(
        "Prefilled from the row. Correct anything that does not match what "
        "you actually did - the fill price especially, since the row's "
        "entry is the last traded price when the scan ran, not your fill."
    )
    one, two, three = st.columns([2, 1, 1])
    symbol = one.text_input("Symbol", value=draft.get("symbol", ""),
                            key="draft_symbol")
    side = two.selectbox(
        "Side", sides, key="draft_side",
        index=sides.index(draft["side"]) if draft.get("side") in sides else 0)
    quantity = three.number_input(
        "Quantity", min_value=0, step=1, key="draft_qty",
        value=int(draft.get("quantity", 0)),
        help="Optional - only used to turn a percentage into rupees.")
    four, five, six = st.columns(3)
    entry = four.number_input("Filled at", min_value=0.0, step=0.05,
                              format="%.2f", key="draft_entry",
                              value=float(draft.get("entry", 0.0)))
    stop = five.number_input("Stop", min_value=0.0, step=0.05,
                             format="%.2f", key="draft_stop",
                             value=float(draft.get("stop", 0.0)))
    target = six.number_input("Exit / target", min_value=0.0, step=0.05,
                              format="%.2f", key="draft_target",
                              value=float(draft.get("target", 0.0)))
    note = st.text_input("Note", value=draft.get("note", ""),
                         key="draft_note",
                         placeholder="why you took it, in your own words")
    add, cancel = st.columns(2)
    if add.button("Watch it", type="primary", width="stretch"):
        held = position_watch.Position(
            symbol=symbol, side=side, entry=float(entry), stop=float(stop),
            target=float(target), quantity=int(quantity), note=note)
        if not held.symbol:
            st.warning("A symbol is needed.")
        elif not held.is_valid:
            st.error(
                f"Those levels cannot describe a {held.side}. For a "
                f"{position_watch.LONG} the stop must sit below the entry "
                f"and the target above it; for a {position_watch.SHORT} the "
                f"other way round. Nothing was added.")
        else:
            position_watch.add(held)
            st.session_state.pop("watch_draft", None)
            st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.session_state.pop("watch_draft", None)
        st.rerun()


# --- the alert ------------------------------------------------------------

def announce(statuses: list) -> bool:
    """Chime any alert not already announced. True if one fired.

    The dedupe rule lives in position_watch.alerts_to_announce, tested
    there: one alert per position and STATE, and the key forgotten as soon
    as the state clears so a recurrence is not swallowed.
    """
    said = st.session_state.get("announced", set())
    fresh, said = position_watch.alerts_to_announce(statuses, said)
    st.session_state["announced"] = said
    if not fresh:
        return False
    # A NONCE PER ALERT, and not for decoration. st.audio derives the
    # element id from a hash of the audio content and the frontend refuses
    # to autoplay an id it has already seen, so with identical bytes the
    # first stop of a session sounded and every one after it was silent.
    nonce = int(st.session_state.get("alert_nonce", 0)) + 1
    st.session_state["alert_nonce"] = nonce
    for status in fresh:
        st.toast(f"{status.position.symbol} {status.position.side} - "
                 f"{status.state}", icon=":material/campaign:")
    # One chime for the batch, chosen by the worst state in it: four
    # overlapping tones is noise, and alerts_to_announce returns them
    # worst first.
    chime = sound.chime_for(fresh[0].state, nonce)
    if chime:
        st.audio(chime, format="audio/wav", autoplay=True)
    phrases = [s.spoken for s in fresh]
    # Kept, and now the ONLY place the sentence survives: this panel is
    # redrawn every twenty seconds, so the toast scrolls into history
    # within one tick. The chime says something happened; this says what.
    st.session_state["standing_alert"] = {
        "phrases": phrases,
        "worst": fresh[0].state,
        "at": datetime.now(IST_ZONE).strftime("%H:%M:%S"),
    }
    return True


def render_standing_alert() -> None:
    """The last alert, until it is acknowledged.

    The chime is not guaranteed to arrive - a page nobody has clicked can
    have its audio suppressed - and the fragment redraws over the toast
    within one tick. So the sentence stays on screen until acknowledged,
    which is the channel that cannot be suppressed or scrolled away.
    """
    standing = st.session_state.get("standing_alert")
    if not standing:
        return
    left, right = st.columns([4, 1], vertical_alignment="center")
    left.warning(f"**{standing['at']}** - {' '.join(standing['phrases'])}",
                 icon=":material/notifications_active:")
    with right:
        if st.button("Acknowledge", key="ack_alert", width="stretch"):
            st.session_state.pop("standing_alert", None)
            st.rerun()


# --- the tab --------------------------------------------------------------

def watch_frame(statuses: list) -> pd.DataFrame:
    """Watched positions as one table, worst first.

    The column names deliberately differ from the scan's, and so do their
    tooltips. The scan's are written about the scanner's own proposals -
    "Sell here if it goes against you", which is advice this project does
    not give, and "Set at twice the distance to the stop", which is false
    of a number the user typed. Same table shape, different meaning, so
    different names and their own entries in the glossary.
    """
    rows = []
    for status in statuses:
        held = status.position
        rows.append({
            "Symbol": held.symbol,
            "Held": held.side,
            "State": status.state,
            # `x == x` rather than truthiness: NaN is truthy and 0.0 is
            # not, so `if status.price` had it exactly backwards.
            "Your fill": round(held.entry, 2),
            "Price now": (round(status.price, 2)
                          if status.price == status.price else None),
            "Move %": (round(status.move_pct, 2)
                       if status.move_pct == status.move_pct else None),
            "Your stop": round(held.stop, 2),
            "Your exit": round(held.target, 2),
            "Shares": held.quantity or None,
            "Why you took it": held.note,
        })
    return pd.DataFrame(rows)


def render_watched_position(status) -> None:
    """One position: its state, the checks behind it, and its two edits."""
    held = status.position
    title = (f"**{held.symbol} {held.side}** from {held.entry:,.2f} - "
             f"{status.state}")
    body = f"{title}\n\n{status.headline}. {status.detail}"
    if status.state == position_watch.STOP_BREACHED:
        st.error(body, icon=":material/warning:")
    elif status.state == position_watch.TARGET_REACHED:
        st.success(body, icon=":material/flag:")
    elif status.state == position_watch.NEAR_TARGET:
        st.success(body, icon=":material/trending_up:")
    elif status.needs_attention:
        st.warning(body, icon=":material/change_circle:")
    else:
        st.info(body)
    for line in status.lines:
        st.caption(f"- {line}")
    edit, drop, _ = st.columns([1, 1, 3])
    if edit.button("Edit levels", key=f"edit_{held.symbol}_{held.side}",
                   width="stretch"):
        open_watch_draft(held.symbol, held.side, held.entry, held.stop,
                         held.target, held.quantity, held.note)
    if drop.button("Stop watching", key=f"unwatch_{held.symbol}_{held.side}",
                   width="stretch"):
        position_watch.remove(held.symbol, held.side)
        st.rerun()


def positions_panel(speak: bool) -> None:
    """The re-priced half of the tab, re-run on its own timer.

    Everything that must stay CURRENT lives in here and nothing else
    does. The manual form used to be here too and re-rendered under the
    user's fingers every twenty seconds.
    """
    positions = position_watch.load()
    if not positions:
        st.session_state["watch_attention"] = False
        st.info(
            "Nothing is being watched. Tick a row in any table - the "
            "intraday scan, the short, mid or long horizon tables, or an "
            "instrument you looked up - and press the button that appears. "
            "The levels arrive prefilled and editable."
        )
        return
    statuses, note = watch_statuses(positions)
    trading = live_bars_in_hours()
    # A feed that cannot price anything IS something needing attention,
    # during a session. Outside one it is just the market being shut.
    st.session_state["watch_attention"] = (
        any(s.needs_attention for s in statuses) or bool(note and trading))
    if note:
        (st.warning if trading else st.info)(note)
    if not (speak and announce(statuses)):
        render_standing_alert()
    show_table(watch_frame(statuses))
    st.caption(
        "Every number in that table is YOURS - the fill, the stop and the "
        "exit you recorded. Nothing in it is a suggestion."
    )
    for status in statuses:
        render_watched_position(status)
    stamp = datetime.now(IST_ZONE).strftime("%H:%M:%S")
    st.caption(
        f"Checked at {stamp} IST"
        + (f", again in {config.POSITION_WATCH_REFRESH_SECONDS}s."
           if trading else
           ". The market is closed, so this is the last traded state."))


def render_positions_tab() -> None:
    """Positions you hold, checked against the scanner's own gates.

    The scanner is stateless: it describes every symbol now and forgets.
    This is the one place that knows what you already did, which is the
    gap that cost real money on HDFCLIFE - the invalidation was on the
    page and nowhere near where someone holding the position would look.
    """
    st.caption(
        "What you already hold, re-priced against your own stop and exit "
        "and against the gates that produced it. It reports; it never says "
        "exit, hold or add - that is a decision about your money."
    )
    left, right = st.columns([1, 2], vertical_alignment="center")
    speak = left.toggle(
        "Chime on alerts", value=True, key="watch_speak",
        help="A chime when a stop or target is hit, a target is neared, or "
             "the scan flips against you - falling for a stop, rising for a "
             "target. Each alert sounds once, not on every refresh. The "
             "sentence itself stays on screen until you acknowledge it.")
    if right.button("Test the sound", key="watch_sound_test"):
        # Its own nonce, for the same reason a real alert has one: the
        # frontend will not autoplay an audio id it has already seen, so a
        # second press was silent while the sentence still claimed the
        # sound worked. It also stops a test burning the id that a real
        # TARGET REACHED would need later in the session.
        nonce = int(st.session_state.get("alert_nonce", 0)) + 1
        st.session_state["alert_nonce"] = nonce
        st.audio(sound.chime_for(position_watch.TARGET_REACHED, nonce),
                 format="audio/wav", autoplay=True)
    # UNCONDITIONAL, and that is the fix for the worst bug review found.
    # This used to pass run_every=None unless the market was open AND
    # something was already watched - both evaluated on a full app run.
    # Fragment ticks never cause a full run, so a page opened at 09:00, or
    # before the first position was added, got no timer for the rest of
    # the session and never alerted at all. Whether the market is open is
    # now decided INSIDE the fragment, on every tick, where it can change.
    st.fragment(lambda: positions_panel(speak),
                run_every=config.POSITION_WATCH_REFRESH_SECONDS)()
    # Outside the fragment on purpose: a form that re-renders itself every
    # twenty seconds is a form you cannot finish filling in.
    with st.expander("Add one by hand"):
        st.caption(
            "Only needed when there is no row to work from - a position "
            "taken before this session, say."
        )
        render_watch_form()


def render_watch_form() -> None:
    """Record a position to watch. Levels are typed, never inferred.

    Deliberately manual: the app cannot know what you filled at, and
    guessing an entry from the last scan would attach real money to a
    number nobody agreed to.
    """
    with st.form("watch_add", clear_on_submit=True):
        st.caption("Add a position - use the price you actually filled at.")
        one, two, three = st.columns([2, 1, 1])
        symbol = one.text_input("Symbol", placeholder="HDFCLIFE")
        side = two.selectbox("Side", [position_watch.LONG,
                                      position_watch.SHORT])
        quantity = three.number_input("Quantity", min_value=0, step=1,
                                      value=0,
                                      help="Optional - only used to turn a "
                                           "percentage into rupees.")
        four, five, six = st.columns(3)
        entry = four.number_input("Filled at", min_value=0.0, step=0.05,
                                  format="%.2f")
        stop = five.number_input("Stop", min_value=0.0, step=0.05,
                                 format="%.2f")
        target = six.number_input("Target", min_value=0.0, step=0.05,
                                  format="%.2f")
        if st.form_submit_button("Watch this", type="primary"):
            held = position_watch.Position(
                symbol=symbol, side=side, entry=float(entry),
                stop=float(stop), target=float(target),
                quantity=int(quantity))
            if not held.symbol:
                st.warning("A symbol is needed.")
            elif not held.is_valid:
                # Refused rather than stored: a long whose stop sits above
                # its entry would report a breach from the first tick.
                st.error(
                    f"Those levels cannot describe a {held.side}. For a "
                    f"{position_watch.LONG} the stop must sit below the "
                    f"entry and the target above it; for a "
                    f"{position_watch.SHORT} the other way round."
                )
            else:
                position_watch.add(held)
                st.success(f"Watching {held.symbol} {held.side}.")
                st.rerun()


def render_instrument_search() -> None:
    """Search any instrument and show its verdict at every horizon.

    Rendered ABOVE the tabs. It answers "what about this one stock",
    which is neither an intraday scan nor an end-of-day sector signal, so
    it does not belong inside either tab.
    """
    left, middle, right = st.columns([4, 1, 1], vertical_alignment="bottom")
    typed = left.text_input(
        "Look up any stock, future or option",
        value="", placeholder="RELIANCE, RELIANCE26SEPFUT, RELIANCE26SEP1400CE",
        key="lookup_symbol",
        help="Type any part of a symbol. Options are held back until four "
             "characters, because two letters match tens of thousands of "
             "strikes and would bury the stock you were after.")
    with_intraday = middle.checkbox("Intraday too", value=False,
                                    help="Needs 5-minute bars for today, so "
                                         "it is slower and only meaningful "
                                         "during or just after a session.")
    # An explicit button as well as Enter. A text_input alone commits only
    # on Enter or blur, which is easy to miss and impossible to drive from
    # anything but a keyboard.
    pressed = right.button("Analyse", key="lookup_go",
                           width="stretch")
    text = (typed or "").strip()
    query, picked = "", None
    if text:
        with st.spinner("searching..."):
            # 60, not 25. A strike ladder is dozens of contracts and the
            # old cap silently hid most of them - though the real fix for
            # options is the F&O browser below, not a longer list.
            matches = search_instruments(text.upper(), 60)
        if not matches:
            # NOT a dead end. The horizon assessment reads the daily store
            # and needs no instrument master at all, so a symbol the
            # typeahead cannot offer is still analysable. Returning here
            # meant one failed download of Kite's CSV made every symbol
            # un-analysable while the UI blamed the user's spelling.
            if instrument_search.catalogue():
                st.caption(
                    f"No instrument matches {text.upper()!r}. Press Analyse "
                    f"to try it against the daily store anyway.")
            else:
                st.warning(
                    "The instrument list could not be downloaded, so the "
                    "typeahead is empty. Analysis still works - it reads "
                    "the daily store, not the instrument list - so type the "
                    "exact symbol and press Analyse."
                )
            if pressed:
                query = text.upper()
        elif len(matches) == 1:
            # An unambiguous hit needs no picker.
            picked = matches[0]
        else:
            # PILLS rather than a selectbox. Streamlit draws a selectbox in
            # a muted grey that reads as disabled, and it took a full row
            # for what is a short list of choices. Pills are chips, and
            # nothing being selected is the natural "not chosen yet" state
            # so no sentinel option is needed - which also means typing
            # cannot silently analyse whatever ranks first.
            # Options are reached through the F&O browser rather than
            # these chips, so they do not get to crowd out the stock and
            # its futures - which is what a 12-pill window did when 79
            # contracts matched.
            head = [r for r in matches if r[2] != instrument_search.KIND_OPTION]
            shown = (head or matches)[:12]
            chosen = st.pills(
                f"{len(matches)} match(es) - pick one",
                [row[0] for row in shown], selection_mode="single",
                key="lookup_pick",
                help="Ordered stock first, then futures by expiry, then "
                     "option strikes - the underlying is usually what you "
                     "want.")
            if len(matches) > len(shown):
                st.caption(f"showing the closest {len(shown)} of "
                           f"{len(matches)} - type more to narrow it")
            if chosen:
                picked = next(r for r in shown if r[0] == chosen)
                st.caption(picked[1])
    if picked is not None:
        st.session_state["lookup_last"] = picked[0]
        query = picked[0]
        if picked[2] in (instrument_search.KIND_FUTURE,
                         instrument_search.KIND_OPTION):
            render_derivative_panel(picked[0], picked[2], picked[3],
                                    picked[4], picked[5], picked[6],
                                    picked[7])
            return
    query = query or st.session_state.get("lookup_last", "")
    if not query:
        st.caption("e.g. RELIANCE for a stock, RELIANCE26SEPFUT for a "
                   "future, RELIANCE26SEP1400CE for an option")
        return
    with st.spinner(f"analysing {query}..."):
        report = analyse_instrument(query, with_intraday)
    if not report["found"]:
        st.error(f"**{report['symbol']}** - {report['note']}")
        return

    header = f"**{report['symbol']}**  {report['price']:,.2f}"
    header += "  (live)" if report.get("live_anchored") else "  (last close)"
    header += (f"  |  F&O, lot {report['lot_size']:,}" if report["in_fo"]
               else "  |  cash only, no derivatives")
    if report["as_of"] is not None:
        header += f"  |  latest daily bar {report['as_of']:%Y-%m-%d}"
    st.markdown(header)
    if not report.get("live_anchored"):
        # Worth saying loudly. MOLBIO closed at 1,253.00 and opened the
        # next morning on its way to 1,509.40 - a lookup anchored on the
        # close was 20% away from the price you would actually pay, and
        # every stop and exit below it was wrong by the same margin.
        st.warning(
            "This is the last DAILY CLOSE, not a live price - the market "
            "may be closed, or there is no Kite session. Every stop and "
            "exit below is derived from it, so during a session they can "
            "sit a full day's move away from where the stock is now."
        )

    buys = report["buys"]
    if buys:
        st.success(f"BUY at: {', '.join(buys)}")
    else:
        st.warning("NO BUY at any horizon")
    st.caption(
        "A BUY means every gate passed and the plausible move covers the "
        "round trip at least 3x. It is not a forecast - four horizons were "
        "measured on this data and none showed predictive skill, so read it "
        "as 'nothing measurable rules this out'."
    )
    table = report["table"]
    if table is not None and not table.empty:
        # One row per horizon, so the symbol and the price come from the
        # report rather than from the row. The intraday row has no daily
        # levels and is refused with a reason, which is the honest answer.
        watch_table(table, f"lookup_pick_{report['symbol']}",
                    symbol=report["symbol"], entry=report.get("price"),
                    note="from the instrument lookup")
        show_terms("What the horizons mean", "Why charges matter so much",
                   "Why this tool will not predict for you")
    render_pivot_levels(report.get("pivots"), report["price"])
    if report.get("in_fo"):
        # Only for names that actually have derivatives - the browser is
        # empty and confusing otherwise.
        render_fno_browser(report["symbol"])
    for name, detail in report["detail"].items():
        label = (f"{name} - {detail['verdict']}"
                 + (f" ({detail['blocker'].split(' [')[0]})"
                    if detail["blocker"] else ""))
        with st.expander(label):
            for reason in detail["reasons"]:
                st.caption(f"- {reason}")


def render_scan_tab() -> None:
    """Intraday scanner tab: controls, then the last scan's results."""
    warm_object_store()
    render_live_feed_panel()
    render_premarket_panel()
    # READ FIRST. When the feed has published a fresh scan the page shows
    # it without computing anything. The controls below stay available so
    # a deliberate local run is still one click away - the fallback is not
    # removed, only demoted, because a dead feed must not mean a dead UI.
    #
    # ON A TIMER, because the feed republishes every half minute and this
    # panel used to update only when something else re-ran the page.
    st.fragment(published_scan_fragment,
                run_every=published_poll_interval())()
    published = bool(st.session_state.get("published_scan_shown"))
    scope, want_options, as_of, run = render_scan_controls()
    # AUTO-RUN THE FIRST SCAN when the session is live and the feed is
    # carrying it. Requiring a button press meant the table was simply
    # absent for anyone who opened the tab during market hours - and a
    # Streamlit restart clears session_state, so a scan run before the
    # restart is gone too. Only the FIRST scan is automatic; refreshing
    # after that stays opt-in, because each one re-scores every symbol.
    auto = False
    if published:
        # The feed's table is already on screen. Auto-running a local scan
        # here would spend 48 seconds recomputing what the reader is
        # looking at, on a machine that has other things to do.
        pass
    elif not run and st.session_state.get("scan") is None:
        why_not = autoscan_blocked(as_of, want_options, scope)
        if why_not:
            if autoscan_will_clear(as_of, want_options, scope):
                st.session_state["autoscan_watch"] = {
                    "as_of": as_of, "want_options": want_options,
                    "scope": scope}
                st.caption(
                    f"No scan yet - {why_not}. Watching, and it will run "
                    f"itself as soon as that clears. Press Run intraday "
                    f"scan to do it now.")
                st.fragment(autoscan_watch_fragment,
                            run_every=config.AUTOSCAN_RECHECK_SECONDS)()
            else:
                st.caption(f"No scan yet - {why_not}. Press Run intraday "
                           f"scan to do it anyway.")
        elif not st.session_state.get("scan_autotried"):
            st.session_state["scan_autotried"] = True
            auto = run = True
    if run:
        import time

        # The label described HOW THE SCAN WAS STARTED, not where its bars
        # come from, so a manual scan reading entirely from the feed still
        # announced "Downloading" - and a full-universe scan that really
        # was downloading 1,568 symbols said the same thing, giving no way
        # to tell seconds from minutes. It now describes the WORK.
        if as_of is not None:
            label = f"Replaying {as_of:%Y-%m-%d %H:%M} IST..."
        elif scope in _AUTOSCAN_SCOPES and live_feed_state().get("running"):
            label = "Scanning from the live feed..."
        elif scope in _AUTOSCAN_SCOPES:
            label = ("Feed not running, so downloading these bars once - "
                     "seconds, not minutes...")
        else:
            label = (f"{scope} reaches symbols the feed does not stream, so "
                     f"they download at three a second. This takes MINUTES. "
                     f"Switch to '{_DEFAULT_SCAN_SCOPE}' if that was not "
                     f"intended.")
        started = time.monotonic()
        # Claimed BEFORE the spinner and released in the finally below, so
        # the autoscan poll can tell "no scan yet" from "a scan is running".
        st.session_state["scan_running"] = True
        try:
            with st.spinner(label):
                ranked, bars, benchmark, now, past = run_scan(scope, as_of)
        except Exception as exc:
            # An automatic scan must not take the tab down. A manual one
            # still raises, because the user asked for it and wants to see
            # why it failed.
            if not auto:
                raise
            # No logger in this module, and st.warning IS the right
            # channel here anyway - the person who needs to know is
            # looking at the page.
            st.warning(f"The automatic scan could not complete ({exc}). "
                       f"Press Run intraday scan to retry.")
            ranked = None
        finally:
            st.session_state.pop("scan_running", None)
        if ranked is not None:
            # Measured so the auto-refresh cannot be set faster than a scan
            # actually takes.
            st.session_state["scan_seconds"] = time.monotonic() - started
            st.session_state["scan"] = (ranked, bars, benchmark, now,
                                        want_options, past)
            # The inputs are kept beside the results so a timed refresh can
            # reproduce the same scan without the widgets being on screen.
            st.session_state["scan_params"] = {
                "scope": scope, "want_options": want_options}
    stored = st.session_state.get("scan")
    if stored is not None:
        every = render_scan_refresh_controls(replaying=bool(stored[5]),
                                             want_options=stored[4])
        if every:
            # Wrapped per script run rather than decorated once, because the
            # interval is a user choice and `run_every` is fixed at
            # decoration time.
            st.fragment(refresh_scan_fragment, run_every=every)()
        else:
            st.session_state.pop("scan_auto_armed", None)
            render_scan_results(*stored)

    st.divider()
    discovered = instruments.load_latest()
    if discovered is None:
        st.info("Sync the instruments to assess the longer horizons.")
        return
    render_horizon_tables(sorted({i.symbol for i in discovered.fo_stocks}))


def render_positional_tab(profile_key: str, news_weight: float,
                          max_age_hours: int) -> None:
    """The end-of-day sector signal and ranking.

    Returns early instead of calling st.stop() on a data outage: st.stop()
    aborts the whole script, which would blank the intraday tab too even
    though it needs neither news nor this profile.
    """
    profile = get_profile(profile_key)
    st.caption(
        f"{profile.label} - news sentiment + index/ETF momentum ({profile.currency}), "
        f"blended into a live ranking of {len(profile.sectors)} market sectors."
    )
    with st.spinner("Fetching news and market data..."):
        items = load_news(max_age_hours, profile_key)
        momentum = load_momentum(profile_key)

    if not items and not momentum:
        st.warning(
            "No news items and no momentum data could be fetched. Check your "
            "connection, widen the news lookback, or hit 'Refresh data' in "
            "the sidebar."
        )
        return

    if not items:
        st.warning(
            "No recent news items could be fetched. Check your connection, widen "
            "the news lookback, or hit 'Refresh data' in the sidebar."
        )
        st.info("News data unavailable - rankings reflect momentum only.")

    if not momentum:
        st.info("Momentum data unavailable - rankings reflect news sentiment "
                "only. The usual cause is a missing or expired Kite session; "
                "it expires around 6am, so run `python -m kite_login`.")

    scores = analyzer.analyze(items, momentum, news_weight=news_weight,
                              profile=profile)
    render_header_metrics(len(items), scores, momentum, profile)
    st.plotly_chart(build_score_chart(scores), width="stretch")
    render_invest_section(scores)
    render_signal_section(items, momentum, profile)

    st.subheader("Full ranking")
    show_table(build_ranking_frame(scores))

    render_sector_expanders(scores)
    render_ai_section(scores)


# --- Top-level flow ----------------------------------------------------------

profile_key, news_weight, max_age_hours = render_sidebar()

# The header carries the one control that used to own a whole tab. A
# popover keeps the cache status and the sync button one click away
# without giving a once-a-day action permanent space.
_title, _sync = st.columns([5, 1], vertical_alignment="center")
_title.title("📈 Sector Pulse")
with _sync.popover("Instrument sync", width="stretch"):
    render_sync_popover()

# Above the tabs on purpose: "what about this one stock" is neither an
# intraday scan nor an end-of-day sector signal, so it does not belong
# inside either of them.
render_instrument_search()
st.divider()

# Opened here, at the top level, rather than from inside the scan
# fragment that sets the draft - see open_watch_draft.
if st.session_state.get("watch_draft"):
    watch_dialog()

# The count comes from the watch file, which is exact. The warning mark
# comes from the last completed check AND ONLY UPDATES ON A FULL APP RUN -
# the watch re-checks itself in a fragment, and fragment ticks do not
# re-render tab labels. So it can lag by a long time. The chime, the
# toast and the standing alert inside the tab are the alerts; this is a
# signpost for when you next interact with the page.
_held = position_watch.load()
_mark = "⚠️ " if st.session_state.get("watch_attention") else ""
tab_intraday, tab_positional, tab_positions = st.tabs(
    ["Scan: intraday / short / mid / long",
     "Positional signal (end of day)",
     f"{_mark}Positions ({len(_held)})"])

# The scan leads. It is the tab with live data behind it, and the one the
# search above most often sends people to.
with tab_intraday:
    render_scan_tab()

with tab_positional:
    render_positional_tab(profile_key, news_weight, max_age_hours)

# Streamlit renders every tab's children whether or not it is on top, so
# the watch keeps re-pricing and keeps alerting while you are looking at
# the scan. That is the whole point of it.
with tab_positions:
    render_positions_tab()

