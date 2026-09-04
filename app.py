"""Sector Pulse — Streamlit dashboard ranking market sectors per profile.

Blends RSS news sentiment with sector index/ETF momentum into a composite
investment-attractiveness score. Educational tool — not financial advice.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analyzer
import claude_insights
import config
import decision
import intraday
import market_data
import news_fetcher
from models import NewsItem, ScoredNewsItem, SectorMomentum, SectorScore
from profiles import PROFILES, MarketProfile, get_profile

st.set_page_config(page_title="Sector Pulse", page_icon="📈", layout="wide")

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
    """Fetch and cache sector index/ETF momentum from yfinance."""
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
    return f"**Momentum ({momentum.etf}):** {windows} — score {momentum.score:+.2f}"


def headline_line(scored: ScoredNewsItem) -> str:
    """Markdown bullet for one scored headline with sentiment, source and age."""
    item = scored.item
    return (
        f"{sentiment_emoji(scored.sentiment)} "
        f"[{esc_markdown(item.title)}]({esc_link(item.link)}) — "
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
            config.DEFAULT_MARKET if config.DEFAULT_MARKET in PROFILES else "US"
        )
        profile_key = st.selectbox(
            "Market",
            profile_keys,
            index=profile_keys.index(default_key),
            format_func=lambda key: f"{key} — {PROFILES[key].label}",
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
                "over 5, 21 and 63 trading days via yfinance. The composite score "
                "blends news sentiment and momentum using the weight slider above."
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
        box(f"**{rank}. {score.sector} ({score.etf})** — {build_rationale(score)}")
    worst = list(reversed(scores[-2:]))
    avoid = ", ".join(f"{s.sector} ({s.etf}, {s.composite:+.2f})" for s in worst)
    st.warning(f"Consider avoiding / underweight: {avoid}")


def render_sector_expanders(scores: list[SectorScore]) -> None:
    """Per-sector detail expanders in ranked order."""
    st.subheader("Sector details")
    for rank, score in enumerate(scores, start=1):
        label = f"{rank}. {score.sector} ({score.etf}) — composite {score.composite:+.2f}"
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
        st.info("**NO ACTION today** — no sector passed every gate in "
                "either direction.")
    else:
        banner = st.error if top["action"] == "SELL" else st.success
        banner(f"**TOP SIGNAL: {top['action']} {top['etf']} ({top['sector']})** "
               f"— rank {top['rank_score']:+.3f}")
    st.dataframe(signal_frame(signal_dicts), width="stretch", hide_index=True)


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
    """'Today's 3:15 signal' — live preview vs saved scheduled-run signal."""
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
                st.session_state["live_signal"] = {
                    "market": profile.key,
                    "closed": False,
                    "generated_at": datetime.now().astimezone().isoformat(),
                    "top_pick": signal_to_dict(pick) if pick is not None else None,
                    "signals": [signal_to_dict(s) for s in signals],
                }
    live = st.session_state.get("live_signal")
    if live is not None and live.get("market") != profile.key:
        live = None
    if live is not None:
        if live.get("closed"):
            st.info("Market closed today — no live signal.")
        else:
            st.markdown(
                f"#### ⚡ Live preview — computed {fmt_time(live.get('generated_at'))} "
                f"(markets may still be open; this can change until close)"
            )
            render_signal_result(live.get("top_pick"), live.get("signals", []))
        if stored is not None:
            with st.expander(
                f"📌 Saved signal — scheduled run at "
                f"{fmt_time(stored.get('generated_at'))}"
            ):
                render_signal_result(
                    stored.get("top_pick"), stored.get("signals", [])
                )
    elif stored is not None:
        st.markdown(
            f"#### 📌 Saved signal — from scheduled run at "
            f"{stored.get('generated_at', '?')}"
        )
        st.caption("The authoritative run is the 15:15 IST scheduled task; "
                   "use 'Compute signal now' for a live preview.")
        render_signal_result(stored.get("top_pick"), stored.get("signals", []))
    else:
        st.caption("No saved signal for this market yet — run "
                   "`python daily_signal.py` or compute one now.")
    st.caption("Decision support only — unvalidated rule, possibly delayed "
               "intraday data, never financial advice, never places orders. "
               "SELL means exit/avoid/underweight — short-selling is not "
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
            st.error("AI analysis failed — check the app logs and your API key.")
    result = st.session_state.get("ai_insights")
    if result is not None:
        st.markdown(result)


# --- Top-level flow ----------------------------------------------------------

profile_key, news_weight, max_age_hours = render_sidebar()
profile = get_profile(profile_key)

st.title("📈 Sector Pulse")
st.caption(
    f"{profile.label} — news sentiment + index/ETF momentum ({profile.currency}), "
    f"blended into a live ranking of {len(profile.sectors)} market sectors."
)

with st.spinner("Fetching news and market data..."):
    items = load_news(max_age_hours, profile_key)
    momentum = load_momentum(profile_key)

if not items and not momentum:
    st.warning(
        "No news items and no momentum data could be fetched. Check your "
        "connection, widen the news lookback, or hit 'Refresh data' in the sidebar."
    )
    st.stop()

if not items:
    st.warning(
        "No recent news items could be fetched. Check your connection, widen the "
        "news lookback, or hit 'Refresh data' in the sidebar."
    )
    st.info("News data unavailable — rankings reflect momentum only.")

if not momentum:
    st.info("Momentum data unavailable — rankings reflect news sentiment only.")

scores = analyzer.analyze(items, momentum, news_weight=news_weight, profile=profile)

render_header_metrics(len(items), scores, momentum, profile)
st.plotly_chart(build_score_chart(scores), width="stretch")
render_invest_section(scores)
render_signal_section(items, momentum, profile)

st.subheader("Full ranking")
st.dataframe(build_ranking_frame(scores), width="stretch", hide_index=True)

render_sector_expanders(scores)
render_ai_section(scores)

st.caption(
    "Educational tool only. Data comes from free public sources and may be delayed "
    "or incomplete. Nothing here is financial advice."
)
