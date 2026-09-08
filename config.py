"""Tunable settings for Sector Pulse."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# --- Market profile ---
DEFAULT_MARKET = os.environ.get("SECTOR_PULSE_MARKET", "IN")

# --- News gathering ---
# Ignore articles older than this. Matches daily_signal's own fetch window
# (SIGNAL_NEWS_HOURS * 2 = 24), which matters now that the dashboard button
# also archives: at 48 it wrote two days of headlines into one archive file
# and inflated that day's per-sector baseline against a CLI-written day.
NEWS_MAX_AGE_HOURS = 24
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SectorPulse/1.0"
MAX_ITEMS_PER_FEED = 40

# --- Scoring ---
RECENCY_HALF_LIFE_HOURS = 24     # article weight halves every N hours
MIN_ARTICLES_FULL_CONFIDENCE = 5 # news score damped below this many articles

# --- News classification ---
# A single keyword hit anywhere used to assign a sector, so a daily
# "stocks to watch" roundup naming one company per sector voted in 10 of 12
# sectors at once, and 66% of Banks articles also matched Financial Services.
TITLE_MATCH_WEIGHT = 2        # a title keyword counts double a summary one
# Swept on a 1,052-headline sample: raising this above 1 cost 98 articles
# (34.1% -> 24.8% usable) and cut Banks/FS collisions only 9 -> 8. The
# shared-keyword exclusion and the roundup cap do the precision work.
MIN_MATCH_STRENGTH = 1        # any one distinct, sector-exclusive keyword
MAX_SECTORS_PER_ARTICLE = 3   # above this it is a roundup: carries no signal
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
# A flat article floor is not comparable across sectors: Banks averages 26
# articles a day and Pharma 2.7, so 3 is noise for one and the entire daily
# volume for the other. Once the archive has enough days, each sector's floor
# becomes a fraction of its own median daily volume, never below the flat
# minimum. This tightens the loose end rather than loosening the tight one.
SIGNAL_MIN_ARTICLES_FRACTION = 0.5
SIGNAL_BASELINE_MIN_DAYS = 5     # archive days needed before baselines apply
SIGNAL_BASELINE_MAX_DAYS = 60    # trailing window, so cost stays bounded
SIGNAL_MIN_INTRADAY_PCT = 0.2    # gate: intraday day-change floor (%)
SIGNAL_MIN_MOMENTUM = -0.2       # gate: multi-day momentum floor (not a falling knife)
# Fraction of MOMENTUM_WINDOWS weight that must actually be computable
# before the momentum gate is trusted. yfinance served only 1 daily bar
# since 2026-07-20 for 9 of 12 Nifty sector indices, silently reducing
# their score to the 63d window alone (weight 0.2). A gate measured on a
# fifth of its inputs must not be treated as having passed.
SIGNAL_MIN_MOMENTUM_WEIGHT = 0.8
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
# Floor between any two buys, off-cycle band breaches included. At 7 days a
# book far from target keeps re-breaching the band, so it deploys faster
# while the gap is widest and settles to monthly once drift falls inside
# REBALANCE_BAND_PP. Measured on a book 19.5pp under on gold: about 60,000
# a month for two months, then 20,000, totalling 319,000 over a year
# against 240,000 at a pure monthly cadence. That acceleration is
# deliberate. Note the planner has no view of available cash: it will
# print an order every 7 days whether or not the money is there.
MIN_DAYS_BETWEEN_BUYS = 7
MIN_ORDER_VALUE = 500.0           # skip dribble orders below this rupee value
DEFAULT_ALLOCATION_MODE = "fill"  # "fill" (waterfall) or "spread" (proportional)
QUOTE_SUFFIX = ".NS"              # yfinance suffix for bare NSE symbols
HOLDINGS_CSV = PROJECT_ROOT / "holdings.csv"
TARGETS_YAML = PROJECT_ROOT / "targets.yaml"
CONTRIBUTIONS_CSV = PROJECT_ROOT / "contributions.csv"
LAST_PLAN_JSON = PROJECT_ROOT / "last_plan.json"

# --- News archive (makes a sentiment backtest possible) ---
NEWS_ARCHIVE_DIR = PROJECT_ROOT / "news_archive"
