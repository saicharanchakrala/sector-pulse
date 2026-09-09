"""Sector Pulse - Streamlit dashboard for both signal systems.

Two tabs, two unrelated systems:

  Positional signal  blends RSS news sentiment with sector index/ETF momentum
                     into a composite score held for weeks.
  Scan               screens NSE F&O names across FOUR holding periods:
                     intraday (5-minute bars, session-time gated), then
                     short, mid and long term (daily bars, no session
                     gate). Each prices the trade - round-trip cost and
                     whether the plausible move covers it - and ranks the
                     top 20. It is a cost and risk screen, NOT a forecast:
                     measured AUC 0.5205 against 0.5165 for the same model
                     on shuffled labels, and at the daily horizons the
                     selection was worse than equal-weighting.

They share no thresholds and no data. Educational tool - not financial advice.
"""
from __future__ import annotations

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
import config
import decision
import daily_signal
import indicators
import intraday
import market_data
import news_fetcher
import options_chain
import scan_data
import instruments
import option_history
import glossary
import horizons
import instrument_report
import setups
import trade_costs
from levels import LONG
from models import NewsItem, ScoredNewsItem, SectorMomentum, SectorScore
from profiles import PROFILES, MarketProfile, get_profile

st.set_page_config(page_title="Sector Pulse", page_icon="📈", layout="wide")

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


def render_sync_tab() -> None:
    """Cache status plus the sync button."""
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
        st.success("Cache updated. The scan tab will use this snapshot.")
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
_SCAN_SCOPES: dict[str, "int | None"] = {
    "F&O single stocks (default)": None,
    "All listed equities, first 300": 300,
    "All listed equities (very slow)": 0,
}


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
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
            "Entry": round(s.levels.entry, 2),
            "Stop": round(s.levels.stop, 2),
            "Target": round(s.levels.target, 2),
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


def render_scan_controls() -> tuple[float, float, str, bool, "datetime | None", bool]:
    """Scan inputs; returns (capital, risk_pct, scope, want_options, as_of, run)."""
    left, mid, right = st.columns(3)
    with left:
        capital = st.number_input(
            "Money available today", min_value=1_000.0,
            value=config.SCAN_CAPITAL, step=10_000.0,
            help="How much you have to trade with. Used only to work out "
                 "how many shares would fit - it is not a suggestion to "
                 "use all of it.")
    with mid:
        risk_pct = st.number_input(
            "Most you would lose per trade (%)", min_value=0.1,
            max_value=10.0, value=config.SCAN_RISK_PCT_PER_TRADE, step=0.25,
            help="If the trade goes wrong and you exit at the stop, this "
                 "is the share of your money you lose. 1% of Rs 1,00,000 "
                 "is Rs 1,000. It decides how many shares fit.")
    with right:
        scope = st.selectbox(
            "Which stocks to look at", list(_SCAN_SCOPES),
            help="The default is the ~210 large, heavily traded names that "
                 "have futures and options. The wider choices include "
                 "smaller stocks and take much longer.")
    as_of = render_replay_controls()
    want_options = st.checkbox(
        "Also pick an option contract for each setup",
        help="Adds one NSE chain request per setup, so it is slower. Live "
             "scans only: a past session has no chain to read.")
    return (capital, risk_pct, scope, want_options, as_of,
            st.button("Run intraday scan"))


def run_scan(capital: float, risk_pct: float, scope: str,
             as_of: "datetime | None" = None):
    """Fetch bars and evaluate every symbol in scope, live or replayed."""
    discovered = instruments.load_latest()
    if discovered is None:
        discovered = instruments.discover()
        instruments.save(discovered)
    limit = _SCAN_SCOPES[scope]
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
            evaluated.append(setups.evaluate(reading, capital=capital,
                                             risk_pct=risk_pct))
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


def render_scan_results(ranked: list[setups.Setup], bars: scan_data.BarSet,
                        benchmark, now: datetime, want_options: bool,
                        replaying: bool = False) -> None:
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
    st.caption(
        f"{'Replayed as of' if replaying else 'Scanned'} {stamp} IST{nifty} | "
        f"{indicators.minutes_left_in_session(now)} min left in the session | "
        f"bars from **{origin}**"
    )
    if not replaying and origin != "live feed":
        st.warning(
            "These bars were DOWNLOADED, not streamed. The newest candle can "
            "be several minutes old, so entry and stop are computed against "
            "a price that has already moved. Start the live feed above to "
            "remove both the wait and the staleness."
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
        show_table(scan_frame(actionable))
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
        st.subheader("Gate detail")
        for setup in shown[:8]:
            suffix = (f" (score {setup.rank_score:.3f})" if setup.actionable
                      else " - not actionable")
            with st.expander(f"{setup.symbol} - {setup.direction}{suffix}"):
                for reason in setup.reasons:
                    st.caption(reason)

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


def live_feed_state() -> dict:
    """What the live bar feed is doing, for the status panel."""
    try:
        import live_bars
        import live_feed
    except Exception as exc:
        return {"available": False, "why": str(exc)}
    state = dict(live_bars.status())
    state["available"] = True
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
            "A separate process streams ticks and builds 5-minute bars, so a "
            "scan reads them instead of downloading 216 symbols at Kite's "
            "3 requests a second. Prior sessions come from a cached window "
            "ending yesterday; only today comes from the stream."
        )
        left, mid, right = st.columns(3)
        left.metric("Feed", "running" if state.get("running") else "stopped")
        mid.metric("Instruments", state.get("instruments", 0))
        age = state.get("age_seconds")
        right.metric("Newest bar",
                     f"{age:.0f}s ago" if isinstance(age, float) and age == age
                     else "none yet")
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
        if not state.get("running"):
            if st.button("Start live feed"):
                st.info(start_live_feed())
            st.code(".venv\\Scripts\\python -m live_feed", language="text")
        else:
            st.caption(f"PID {state.get('pid')} - stop it from the terminal "
                       f"that owns it, or with taskkill.")


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def load_horizon_picks(symbols: tuple[str, ...], top: int) -> dict:
    """Ranked assessments per daily horizon, cached on the universe.

    Cached because the inputs are daily bars from the consolidated store -
    they change once a session, not once a minute, so re-running the sweep
    on every rerun would burn two seconds for an identical answer.
    """
    return horizons.assess_universe(list(symbols), top=top)


def render_horizon_tables(symbols: list, top: int = 20) -> None:
    """Short, mid and long horizon tables, each filled as it completes.

    Placeholders are written per horizon rather than after all three, so
    the first table is readable while the rest are still being scored.
    """
    st.subheader("Longer horizons")
    st.caption(
        "Assessed from the consolidated daily store, with NO session-time "
        "gate - whether minutes remain today is irrelevant to a position "
        "held for weeks. Intraday above keeps its own gates."
    )
    st.warning(horizons.HONESTY)
    with st.expander("The exact figures, if you want them"):
        st.caption(horizons.HONESTY_DETAIL)
    slots = {}
    for name in ("short", "mid", "long"):
        sessions = horizons.HORIZONS[name]["sessions"]
        st.markdown(f"**{name.title()} term** - about {sessions} sessions, "
                    f"round trip {horizons.HORIZONS[name]['cost_pct']:.3f}%")
        slots[name] = st.empty()
        slots[name].info("scoring...")
    picks = load_horizon_picks(tuple(symbols), top)
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
        show_table(frame, target=slot)


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
        "as_of": report.as_of,
        "table": instrument_report.summary_frame(report),
        "detail": {name: {"verdict": v.verdict, "blocker": v.blocker,
                          "reasons": v.reasons}
                   for name, v in report.verdicts.items()},
    }


def render_instrument_search() -> None:
    """Search one instrument and show its verdict at every horizon."""
    st.subheader("Look up one instrument")
    st.caption(
        "Any NSE symbol, in or out of the F&O universe. Every gate is shown "
        "with the number behind it, so a NO BUY names what failed rather "
        "than just withholding."
    )
    left, middle, right = st.columns([3, 1, 1])
    typed = left.text_input("Symbol", value="", placeholder="RELIANCE",
                            key="lookup_symbol")
    with_intraday = middle.checkbox("Include intraday", value=False,
                                    help="Needs 5-minute bars for today, so "
                                         "it is slower and only meaningful "
                                         "during or just after a session.")
    # An explicit button as well as Enter. A text_input alone commits only
    # on Enter or blur, which is easy to miss and impossible to drive from
    # anything but a keyboard.
    right.markdown("&nbsp;")
    pressed = right.button("Analyse", key="lookup_go")
    query = (typed or "").strip().upper()
    if pressed and query:
        st.session_state["lookup_last"] = query
    query = query or st.session_state.get("lookup_last", "")
    if not query:
        options = instrument_report.suggest("", limit=10)
        if options:
            st.caption(f"e.g. {', '.join(options[:8])}")
        return
    matches = instrument_report.suggest(query, limit=8)
    if matches and query not in matches:
        st.caption(f"did you mean: {', '.join(matches)}")
    with st.spinner(f"analysing {query}..."):
        report = analyse_instrument(query, with_intraday)
    if not report["found"]:
        st.error(f"**{report['symbol']}** - {report['note']}")
        return

    header = f"**{report['symbol']}**  {report['price']:,.2f}"
    header += (f"  |  F&O, lot {report['lot_size']:,}" if report["in_fo"]
               else "  |  cash only, no derivatives")
    if report["as_of"] is not None:
        header += f"  |  latest daily bar {report['as_of']:%Y-%m-%d}"
    st.markdown(header)

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
        show_table(table)
        show_terms("What the horizons mean", "Why charges matter so much",
                   "Why this tool will not predict for you")
    for name, detail in report["detail"].items():
        label = (f"{name} - {detail['verdict']}"
                 + (f" ({detail['blocker'].split(' [')[0]})"
                    if detail["blocker"] else ""))
        with st.expander(label):
            for reason in detail["reasons"]:
                st.caption(f"- {reason}")


def render_scan_tab() -> None:
    """Intraday scanner tab: controls, then the last scan's results."""
    st.caption(
        "A separate system from the positional signal. It holds for hours, "
        "not weeks, so it has its own data, thresholds and cost model - no "
        "threshold is shared between the two. Read it as a screen that "
        "prices trades and rules out the ones that cannot pay, not as a "
        "ranked list of opportunities."
    )
    st.error(
        "**This does not tell you which way a price will go.** We checked: "
        "we ran the same method over 760,458 past moments, and it picked "
        "winners no better than the identical method fed deliberately "
        "scrambled answers. None of the 12 stop-and-target combinations we "
        "tried made money on data it had not seen. So please do not read "
        "the list below as a forecast."
    )
    st.info(
        "**What it is genuinely good for.** The money columns are just "
        "arithmetic, and those hold up: what a trade costs you in fees, how "
        "often it would have to work to break even, and which setups cannot "
        "pay for themselves however right you are about direction. Used to "
        "say no to trades, this saves money. Used to pick them, it does not."
    )
    with st.expander("The exact figures, if you want them"):
        st.caption(
            "Gradient-boosted model, 32 features, 760,458 samples, purged "
            "walk-forward with an embargo. Ranking accuracy (AUC) 0.5205 "
            "against 0.5165 for the same model on shuffled labels, where "
            "0.50 is a coin flip. Best geometry net -0.009 R per trade; "
            "exact permutation p 0.091 over 10 shuffles, whose floor is "
            "0.091, so it cannot show significance at all. 0 of 12 "
            "geometries profitable."
        )
    render_instrument_search()
    st.divider()
    render_live_feed_panel()
    capital, risk_pct, scope, want_options, as_of, run = render_scan_controls()
    if run:
        label = ("Downloading bars and evaluating setups..." if as_of is None
                 else f"Replaying {as_of:%Y-%m-%d %H:%M} IST...")
        with st.spinner(label):
            ranked, bars, benchmark, now, past = run_scan(
                capital, risk_pct, scope, as_of)
        st.session_state["scan"] = (ranked, bars, benchmark, now,
                                    want_options, past)
    stored = st.session_state.get("scan")
    if stored is None:
        st.info("Set your capital and risk, then hit Run intraday scan. "
                "The longer horizons below need no scan and no live feed.")
    else:
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

st.title("📈 Sector Pulse")

tab_positional, tab_intraday, tab_sync = st.tabs(
    ["Positional signal (end of day)",
     "Scan: intraday / short / mid / long",
     "Instrument sync"])

with tab_positional:
    render_positional_tab(profile_key, news_weight, max_age_hours)

with tab_intraday:
    render_scan_tab()

with tab_sync:
    render_sync_tab()

st.caption(
    "Educational tool only. Data comes from free public sources and may be delayed "
    "or incomplete. Nothing here is financial advice."
)
