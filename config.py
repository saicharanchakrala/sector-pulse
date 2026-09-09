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
# before the momentum gate is trusted. The gap that motivated it is gone
# now that bars come from Kite, but the gate stays: yfinance served 1 daily bar
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
HOLDINGS_CSV = PROJECT_ROOT / "holdings.csv"
TARGETS_YAML = PROJECT_ROOT / "targets.yaml"
CONTRIBUTIONS_CSV = PROJECT_ROOT / "contributions.csv"
LAST_PLAN_JSON = PROJECT_ROOT / "last_plan.json"

# --- News archive (makes a sentiment backtest possible) ---
NEWS_ARCHIVE_DIR = PROJECT_ROOT / "news_archive"

# --- Intraday scanner (separate system from the positional signal) ---
# The positional signal holds for weeks; this one holds for hours. Different
# data, different costs, different failure mode - so it gets its own gates
# and never shares a threshold with the EOD path.
SCAN_BAR_INTERVAL = "5m"          # intraday bar size (Kite: "5minute")
# Ten trading SESSIONS of 5-minute bars, which market_source.calendar_days
# turns into about seventeen calendar days. That is the intraday working set
# - enough prior sessions for a relative-volume median and a gap-free ATR,
# without dragging a month of bars through every scan. The median baseline
# is noisier on a dozen prior sessions than on twenty; that is the trade for
# a baseline reflecting this week rather than last month. Raise this if you
# want the steadier median instead.
SCAN_BAR_LOOKBACK = "10d"
SCAN_DAILY_LOOKBACK = "3mo"       # daily bars for ATR and previous-session levels
# Kept for callers that still pass it, but it no longer batches anything:
# Kite is queried per instrument and paced centrally at its documented
# rate, so grouping symbols buys nothing.
SCAN_BATCH_SIZE = 40
SCAN_BENCHMARK = "NIFTY 50"       # relative-strength benchmark, as Kite names it

SCAN_OPENING_RANGE_MINUTES = 15   # opening range = first N minutes of the session
SCAN_ATR_BARS = 14                # ATR period on 5m bars, for intraday stops
# One horizon governs everything: an intraday trade is held to the close, so
# the plausible remaining move is sigma = bar ATR * sqrt(bars left) and both
# the stop and the target are fractions of that same sigma. Sizing the stop
# over one horizon while judging the target over another is incoherent - it
# rejected every setup, because a 2:1 target on a 12-bar stop needs 6.9 ATRs
# when the rest of the day offers 5.5.
SCAN_STOP_FRACTION = 0.5          # stop distance as a fraction of that sigma
# With a 0.5 fraction and a 2:1 ratio the target lands exactly on sigma, so
# reachability holds by construction and only a structural stop can break it.
SCAN_STRUCTURE_BAND = 0.4         # accept structure within +/- this of the stop
# MIS intraday leverage. Sizing is capped by capital times leverage, because
# risk-based sizing alone will happily print a 7 lakh position on 1 lakh.
SCAN_MIS_LEVERAGE = 5.0
# Reject a setup whose costs push the breakeven win rate above this. Under a
# random walk a 2:1 setup wins about 33% of the time; anything needing far
# more than that is paying the broker to gamble.
SCAN_MAX_WIN_RATE = 0.55
# Open interest rising alongside a price move is conventionally read as
# fresh positioning, and falling OI as an unwind. It is reported for every
# F&O name and contributes to the rank, but it is NOT a hard gate by
# default: the reading is a market convention with no backtest behind it,
# and promoting an unvalidated belief to a veto would silently halve the
# output. Set this True to make it binding.
SCAN_REQUIRE_OI_CONFIRMATION = False
SCAN_MIN_OI_CHANGE_PCT = 0.0      # only consulted when the above is True
# The target is defined as this multiple of the stop distance. It is not a
# rejection threshold - nothing gates on reward:risk, because the target is
# constructed from it rather than measured against it.
SCAN_REWARD_RISK = 2.0
# Today's cumulative volume against the median of the PRIOR SESSIONS
# PRESENT IN THE FRAME at the same clock time - which is however many
# SCAN_BAR_LOOKBACK supplies, currently about a dozen, not twenty. The old
# "20d median" label here was never true of the code.
SCAN_RVOL_MIN = 1.2
SCAN_MIN_TURNOVER = 50_000_000.0  # 5 crore/day: intraday needs depth, not just listing
SCAN_MIN_PRICE = 20.0             # sub-20 names move in ticks too coarse to manage
SCAN_MIN_MINUTES_LEFT = 45        # no entry without time for the target to work
SCAN_COST_MULTIPLE = 3.0          # target must clear round-trip cost this many times
# How old the newest live bar may be before the feed counts as dead.
# The arithmetic, which an earlier version of this comment got wrong by a
# whole bar: age is measured from the bar's START stamp, and a bar closes
# only when a tick from the NEXT bucket arrives. So the bar starting at S
# closes at S+300, reaches the file by S+300+flush, and remains the newest
# bar in it until its successor lands at S+600+flush. With the default
# --flush-every of 20s the peak age of a perfectly healthy feed is
# therefore 620s, not 340s. A 600s limit refused the live path for the
# first 20s of every bucket - about 7% of the session - and fell back to
# the 216-symbol download this whole path exists to avoid.
SCAN_LIVE_MAX_AGE_SECONDS = 900
# Below this share of requested symbols actually streaming, the live path is
# not meaningfully live and the scan says so rather than claiming it is.
SCAN_LIVE_MIN_COVERAGE = 0.5

SCAN_SESSION_OPEN = (9, 15)       # NSE equity session, IST
SCAN_SESSION_CLOSE = (15, 30)

# Zerodha intraday (MIS) equity charges. Rates change: verify against
# zerodha.com/charges before trusting the breakeven figures downstream.
COST_EQ_BROKERAGE_PCT = 0.0003    # 0.03% per executed order...
COST_EQ_BROKERAGE_CAP = 20.0      # ...capped at 20 rupees per order
COST_EQ_STT_SELL_PCT = 0.00025    # 0.025%, sell leg only (intraday equity)
COST_EQ_TXN_PCT = 0.0000297       # NSE transaction charge, both legs
COST_EQ_STAMP_BUY_PCT = 0.00003   # 0.003%, buy leg only
COST_SEBI_PCT = 0.000001          # 10 rupees per crore, both legs
COST_GST_PCT = 0.18               # on brokerage + transaction + SEBI

# Zerodha options charges, levied on premium turnover rather than contract value.
COST_OPT_BROKERAGE_FLAT = 20.0    # per executed order
COST_OPT_STT_SELL_PCT = 0.001     # 0.1% of premium, sell leg only
COST_OPT_TXN_PCT = 0.0003503      # NSE options transaction charge
COST_OPT_STAMP_BUY_PCT = 0.00003  # 0.003%, buy leg only

# Option-contract quality gates. These say whether a contract is cheap and
# liquid enough to trade - never that its direction is right.
OPT_MAX_SPREAD_PCT = 2.0          # bid-ask as % of mid
OPT_MIN_OPEN_INTEREST = 500       # in contracts (lots)
OPT_MIN_VOLUME = 100
OPT_STRIKES_EITHER_SIDE = 5       # ATM window to report

# Position sizing. Risk per trade drives quantity, so the stop distance
# decides the size rather than the other way round: a wider stop buys fewer
# shares for the same rupees at risk.
SCAN_CAPITAL = 100_000.0          # intraday capital assumed by the sizer
SCAN_RISK_PCT_PER_TRADE = 1.0     # percent of capital risked between entry and stop

# This was 57 because yfinance stopped serving 5-minute bars near day 60.
# Kite serves them for years, so that cliff is gone and the cap now rests on
# two limits that are real but softer, and worth knowing before trusting an
# old replay:
#
#   * The instrument universe is a single latest snapshot, by choice - no
#     history is kept. So a replay reconstructs a past session using TODAY'S
#     F&O list. The further back you go, the more that list has drifted, and
#     the drift is survivorship-shaped: names added since are scanned on
#     dates they were not yet tradeable, and names dropped are missing.
#   * Expired option contracts leave the instrument master and take their
#     tokens with them, so no past chain is recoverable except from a
#     snapshot recorded at the time (see option_history).
#
# A year keeps the equity replay useful - which is most of why Kite was
# worth the migration - while staying inside one instrument cycle. Lower it
# if the universe drift matters more to you than the reach.
SCAN_MAX_REPLAY_DAYS = 365
# How far back the dashboard's date picker reaches. Deliberately shorter
# than the CLI cap: a few sessions is what anyone reviews by hand, and the
# picker stays honest about weekends by reporting an empty session rather
# than trying to guess the exchange calendar.
SCAN_UI_REPLAY_DAYS = 3
# Calendar days of history to pull *before* the replay date, so relative
# volume has prior sessions to compare against and ATR has enough ranges.
SCAN_REPLAY_LOOKBACK_DAYS = 12

SCAN_UNIVERSE_CSV = PROJECT_ROOT / "scan_universe.csv"
FO_MKTLOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
NSE_BASE_URL = "https://www.nseindia.com"
NSE_TIMEOUT_SECONDS = 20
SCAN_LOG_CSV = PROJECT_ROOT / "scan_log.csv"

# --- Instrument discovery (no hardcoded universe) ---
# Every tradeable name is discovered from NSE at run time. Hardcoding a list
# is how the scanner ended up missing NIFTYFPI: the F&O master file marks its
# own index/stock boundary with a section row, so the split is derivable and
# a hand-maintained list is strictly worse.
NSE_EQUITY_LIST_URL = ("https://nsearchives.nseindia.com/content/equities/"
                       "EQUITY_L.csv")
NSE_OI_SPURTS_PATH = "/api/live-analysis-oi-spurts-underlyings"
NSE_LIVE_DERIVATIVES_PATH = "/api/liveEquity-derivatives"
NSE_CONTRACT_INFO_PATH = "/api/option-chain-contract-info"
# Index futures and a top-20 stock-futures watch are the only per-contract
# futures feeds still answering. There is no bulk stock-futures file: the
# derivatives bhavcopy 404s on every published URL pattern and
# /api/quote-derivative has been withdrawn. Per-underlying futures turnover
# and open interest from the OI snapshot is what futures data is available.
NSE_FUTURES_INDEX_KEYS = ("nse50_fut", "stock_fut")
INSTRUMENT_DIR = PROJECT_ROOT / "instruments"
# One current file, not a history. The universe is a description of what is
# listed right now, and an old copy of it has no use: a scan wants today's
# lot sizes and today's F&O list. Chain snapshots are the opposite case and
# do keep history, because a premium that was not recorded is unrecoverable.
INSTRUMENT_FILE = INSTRUMENT_DIR / "universe.json"
NSE_REQUEST_PAUSE_SECONDS = 0.15   # pacing between per-underlying requests
CALL_ALIASES_CSV = PROJECT_ROOT / "call_aliases.csv"
CALLS_CSV = PROJECT_ROOT / "calls.csv"
OPTION_SNAPSHOT_DIR = PROJECT_ROOT / "option_snapshots"

# --- Zerodha Kite Connect ---
# Credentials are read from the environment and never stored in this repo.
# The instrument master at kite_instruments.py needs none of this; only
# quotes, historical candles and the tick stream require a session.
KITE_API_KEY_ENV = "KITE_API_KEY"
KITE_API_SECRET_ENV = "KITE_API_SECRET"
KITE_ACCESS_TOKEN_ENV = "KITE_ACCESS_TOKEN"
# The access token is written here after a login so a scan does not need one
# per run. It is a session credential: gitignored, and short-lived by design
# (Kite expires it around 6am the next day, with no non-interactive refresh
# for retail apps, so a login is a daily manual step).
KITE_TOKEN_FILE = PROJECT_ROOT / ".kite_session.json"
KITE_API_BASE = "https://api.kite.trade"
KITE_LOGIN_BASE = "https://kite.zerodha.com/connect/login"
KITE_API_VERSION = "3"
KITE_TIMEOUT_SECONDS = 30
# Kite caps one historical request at 60 days for minute data, so a longer
# span is paged. This is the reason to want Kite at all right now: minute
# candles reach back years, where yfinance stops at 8 days.
# Kite's per-request span depends on the candle size. Chunking by the wrong
# one silently truncates a range instead of erroring, so these are explicit.
KITE_HISTORICAL_MAX_DAYS = {
    "minute": 60, "3minute": 100, "5minute": 100, "10minute": 100,
    "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000,
}
KITE_HISTORICAL_DEFAULT_SPAN = 60      # used for any interval not listed
# Kite documents 3 requests/second for historical data. Pacing at 3/s keeps
# a 1,500-request sweep inside the limit instead of collecting 429s.
KITE_HISTORICAL_RATE_PER_SEC = 3.0
# /quote is documented at 1 request/second, stricter than historical. It is a
# separate budget with a separate clock in kite_client: pacing both endpoint
# classes off one timestamp would let a quote sweep spend the historical
# allowance, and pacing them together at the stricter rate would throw away
# two thirds of the historical one.
KITE_QUOTE_RATE_PER_SEC = 1.0
