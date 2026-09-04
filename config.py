"""Tunable settings for Sector Pulse."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# --- Market profile ---
DEFAULT_MARKET = os.environ.get("SECTOR_PULSE_MARKET", "US")

# --- News gathering ---
NEWS_MAX_AGE_HOURS = 48          # ignore articles older than this
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SectorPulse/1.0"
MAX_ITEMS_PER_FEED = 40

# --- Scoring ---
RECENCY_HALF_LIFE_HOURS = 24     # article weight halves every N hours
MIN_ARTICLES_FULL_CONFIDENCE = 5 # news score damped below this many articles
DEFAULT_NEWS_WEIGHT = 0.5        # composite = w*news + (1-w)*momentum

# --- Momentum ---
MOMENTUM_WINDOWS = {             # trading-day window -> (weight, tanh scale in %)
    "5d": (0.5, 3.0),
    "21d": (0.3, 6.0),
    "63d": (0.2, 12.0),
}

# --- UI ---
CACHE_TTL_SECONDS = 900          # 15 min
TOP_HEADLINES_PER_SECTOR = 6

# --- Daily trade signal ---
SIGNAL_NEWS_HOURS = 12           # "today's news" = items within this window
SIGNAL_MIN_NEWS = 0.15           # gate: today-only news score floor
SIGNAL_MIN_ARTICLES = 3          # gate: today-only article count floor
SIGNAL_MIN_INTRADAY_PCT = 0.2    # gate: intraday day-change floor (%)
SIGNAL_MIN_MOMENTUM = -0.2       # gate: multi-day momentum floor (not a falling knife)
# Gate on rupee TURNOVER, not a unit count. A 50,000-unit floor was 84x
# stricter for INFRABEES (958/unit) than OILIETF (11.40/unit); turnover is
# comparable across price levels. This default is set near the median ETF's
# old effective bar and still needs calibrating against a real backtest.
SIGNAL_MIN_AVG_TURNOVER = 2_500_000   # gate: 20-session mean daily turnover
SIGNALS_CSV = PROJECT_ROOT / "signals.csv"
LAST_SIGNAL_JSON = PROJECT_ROOT / "last_signal.json"

# --- Claude insights (optional, needs ANTHROPIC_API_KEY) ---
CLAUDE_MODEL = "claude-opus-4-8"
CLAUDE_MAX_TOKENS = 4096

# --- Portfolio rebalancing (target-weight strategy) ---
MONTHLY_CONTRIBUTION = 20_000.0   # rupees deployed per scheduled contribution
CONTRIBUTION_INTERVAL_MONTHS = 1  # scheduled cadence: calendar monthly, not daily
REBALANCE_BAND_PP = 5.0           # off-cycle trigger: |drift| this many pp or more
MIN_DAYS_BETWEEN_BUYS = 7         # floor between any two buys, band breach included
MIN_ORDER_VALUE = 500.0           # skip dribble orders below this rupee value
DEFAULT_ALLOCATION_MODE = "fill"  # "fill" (waterfall) or "spread" (proportional)
QUOTE_SUFFIX = ".NS"              # yfinance suffix for bare NSE symbols
HOLDINGS_CSV = PROJECT_ROOT / "holdings.csv"
TARGETS_YAML = PROJECT_ROOT / "targets.yaml"
CONTRIBUTIONS_CSV = PROJECT_ROOT / "contributions.csv"
LAST_PLAN_JSON = PROJECT_ROOT / "last_plan.json"

# --- News archive (makes a sentiment backtest possible) ---
NEWS_ARCHIVE_DIR = PROJECT_ROOT / "news_archive"
