"""Optional Claude-powered narrative analysis of sector rankings."""
from __future__ import annotations

import logging
import os

import config
from models import SectorScore

logger = logging.getLogger(__name__)

_TOP_SECTORS_WITH_HEADLINES = 5
_HEADLINES_PER_SECTOR = 3

_SYSTEM_PROMPT = (
    "You are a seasoned market analyst reviewing a quantitative digest that ranks "
    "the market sectors by a composite investment-attractiveness score built from "
    "recent news sentiment and sector price momentum (all scores roughly -1..1). "
    "Base your analysis ONLY on the supplied data - never invent facts, prices, "
    "events, or headlines that are not in the digest. "
    "Explain which sectors look most attractive and why, naming the specific "
    "headlines that drive the view. Flag 2-3 key risks or caveats (for example thin "
    "news coverage, divergence between news sentiment and momentum, or stale data). "
    "Use markdown with short, clearly titled sections and keep the whole response "
    "under roughly 400 words. ALWAYS end with a final line noting that this is "
    "educational analysis, not financial advice."
)


def is_available() -> bool:
    """Return True if the optional Claude insights feature can be used."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _format_ranking_line(rank: int, score: SectorScore) -> str:
    """Format one ranked sector as a compact single line of metrics."""
    momentum = f"{score.momentum.score:+.3f}" if score.momentum is not None else "n/a"
    return (
        f"{rank:2d}. {score.sector} ({score.etf}): "
        f"composite={score.composite:+.3f} "
        f"news_score={score.news_score:+.3f} "
        f"news_count={score.news_count} "
        f"momentum={momentum}"
    )


def _format_headlines(score: SectorScore) -> list[str]:
    """Format up to three top headlines for a sector, with sentiment."""
    lines = [f"{score.sector}:"]
    for scored in score.top_items[:_HEADLINES_PER_SECTOR]:
        lines.append(
            f"  - [{scored.sentiment:+.2f}] {scored.item.title} ({scored.item.source})"
        )
    if not score.top_items:
        lines.append("  - (no headlines)")
    return lines


def _build_digest(scores: list[SectorScore], max_chars: int) -> str:
    """Build a compact plain-text digest of rankings and top headlines."""
    lines = [
        "Sector Pulse digest - the ranked market sectors, ordered by composite "
        "investment-attractiveness score (news sentiment + price momentum, -1..1).",
        "",
        "RANKINGS",
    ]
    for rank, score in enumerate(scores, start=1):
        lines.append(_format_ranking_line(rank, score))
    lines.append("")
    lines.append(f"TOP HEADLINES (top {_TOP_SECTORS_WITH_HEADLINES} sectors)")
    for score in scores[:_TOP_SECTORS_WITH_HEADLINES]:
        lines.extend(_format_headlines(score))
    return "\n".join(lines)[:max_chars]


def generate_insights(scores: list[SectorScore], max_chars: int = 6000) -> str | None:
    """Ask Claude for a narrative analysis of the rankings; None on any failure."""
    if not is_available():
        logger.warning("Claude insights unavailable (no API key or anthropic package)")
        return None

    try:
        import anthropic
    except ImportError as exc:
        logger.warning("anthropic package not installed, skipping insights: %s", exc)
        return None

    digest = _build_digest(scores, max_chars)
    try:
        with anthropic.Anthropic() as client:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": digest}],
            )
    except anthropic.APIError as exc:
        logger.warning("Claude API call failed: %s", exc)
        return None
    return next((b.text for b in response.content if b.type == "text"), None)
