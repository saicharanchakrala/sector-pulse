# Sector Pulse - Build Specification

A local Streamlit dashboard that gathers finance + major world news from free RSS
feeds, measures sector index/ETF price momentum via yfinance, scores a market
profile's sectors on a composite of news sentiment and momentum, and ranks where to
consider investing. Markets are config-driven **profiles** (US and IN today) - each
profile bundles its feeds, sector definitions, and sentiment-lexicon additions.
**Educational tool - not financial advice.** The UI must say so prominently.

No API keys are required to run the core app. `ANTHROPIC_API_KEY` optionally enables an
AI narrative-analysis section.

## File map and ownership

| File | Responsibility | Owner |
|---|---|---|
| `models.py` | Shared dataclasses (pre-written - do not modify) | core |
| `config.py` | Tunables + `DEFAULT_MARKET` | core |
| `profiles/__init__.py` | `MarketProfile` dataclass, `PROFILES` registry, `get_profile()` | core |
| `profiles/us.py` | US profile: SPDR ETF sectors + US/world feeds | news agent |
| `profiles/india.py` | IN profile: Nifty sectoral indices + Indian press feeds | news agent |
| `news_fetcher.py` | Concurrent RSS fetching, parsing, dedupe | news agent |
| `analyzer.py` | Classification, sentiment, composite scoring | analysis agent |
| `market_data.py` | yfinance sector ETF momentum | market agent |
| `claude_insights.py` | Optional Claude narrative analysis | insights agent |
| `intraday.py` | yfinance intraday ETF snapshots | market agent |
| `decision.py` | Daily BUY / DON'T BUY decision engine | analysis agent |
| `daily_signal.py` | End-of-day signal CLI (report + signals.csv + last_signal.json) | core |
| `news_archive.py` | Dated JSONL headline archive + replay loader | core |
| `app.py` | Streamlit dashboard | ui agent |
| `README.md` | Docs + quickstart | scaffold agent |
| `smoke_test.py` | End-to-end smoke test | verify agent |

## Data flow

```
profile.feeds ──> news_fetcher.fetch_all_news(profile=...) ──> list[NewsItem] ─┐
                                                                               ├─> analyzer.analyze(..., profile=...) ──> list[SectorScore] ──> app.py
profile.sectors ──> market_data.get_sector_momentum(profile=...) ──> dict ─────┘                              │
                                                                                  claude_insights.generate_insights() (optional)
```

## Market profiles

### profiles/__init__.py
```python
@dataclass(frozen=True)
class MarketProfile:
    key: str                       # "US" | "IN"
    label: str
    currency: str
    feeds: list[dict]              # {"name", "url", "category"} - see news feeds below
    sectors: dict[str, SectorDef]
    lexicon: dict[str, float]      # profile-specific VADER unigram additions
    phrases: dict[str, float]      # multi-word phrase sentiment boosts (-4..4)
    trade_etfs: dict[str, str]     # sector name -> tradeable exchange ticker;
                                   # sectors absent here have no tradeable ETF

PROFILES: dict[str, MarketProfile]
def get_profile(key: str | None = None) -> MarketProfile
```
`get_profile(None)` resolves to `config.DEFAULT_MARKET`
(env `SECTOR_PULSE_MARKET`, default `"IN"`); an unknown key logs a warning and
falls back to the configured default, then to the registry's first entry.

### Sector universes

- **US** (`profiles/us.py`, 11 GICS sectors): Technology/XLK, Financials/XLF,
  Energy/XLE, Healthcare/XLV, Industrials/XLI, Consumer Discretionary/XLY,
  Consumer Staples/XLP, Utilities/XLU, Materials/XLB, Real Estate/XLRE,
  Communication Services/XLC.
- **IN** (`profiles/india.py`, 12 Nifty sectoral indices): Banks/^NSEBANK,
  Financial Services/NIFTY_FIN_SERVICE.NS, IT/^CNXIT, Pharma/^CNXPHARMA,
  Auto/^CNXAUTO, FMCG/^CNXFMCG, Metal/^CNXMETAL, Energy/^CNXENERGY,
  Realty/^CNXREALTY, Infrastructure/^CNXINFRA, PSU Banks/^CNXPSUBANK,
  Media/^CNXMEDIA. Feeds are the major Indian financial press plus three global
  feeds (category `"world"`) because Indian markets move on global cues.

Each `profile.feeds` entry is `{"name": str, "url": str, "category": "business" |
"world" | "markets"}` - free, live-verified RSS/Atom feeds. Each
`SectorDef.keywords` is a rich lowercase list (25–60 entries: companies, industry
terms, regulators, commodities, themes).

## Module contracts

### news_fetcher.py
```python
def fetch_all_news(
    max_age_hours: int = config.NEWS_MAX_AGE_HOURS,
    profile: MarketProfile | None = None,
) -> list[NewsItem]
```
- `profile=None` resolves via `get_profile()`.
- Fetch every feed in `profile.feeds` concurrently (`ThreadPoolExecutor`, max_workers=8).
- Use `requests.get` with `timeout=config.REQUEST_TIMEOUT_SECONDS` and
  `headers={"User-Agent": config.USER_AGENT}`, then `feedparser.parse(response.content)`.
  (Do not let feedparser fetch URLs itself - it has no timeout.)
- `published` parsing: prefer `entry.published_parsed` then `entry.updated_parsed`
  (convert via `calendar.timegm` → `datetime.fromtimestamp(ts, tz=timezone.utc)`);
  fall back to `datetime.now(timezone.utc)` if absent. All datetimes tz-aware UTC.
- Strip HTML tags from summaries (regex `<[^>]+>` is acceptable); collapse whitespace.
- Drop items older than `max_age_hours`; cap each feed at `config.MAX_ITEMS_PER_FEED`.
- Dedupe across feeds by exact `link` AND by normalized title (lowercase, alphanumeric
  + spaces only) - keep the first (newest) occurrence.
- A failing feed logs `logger.warning("...: %s", exc)` and contributes nothing. The
  function never raises and returns whatever succeeded, sorted newest first.

### market_data.py
```python
def get_sector_momentum(
    profile: MarketProfile | None = None,
) -> dict[str, SectorMomentum]   # key = sector name
```
- `profile=None` resolves via `get_profile()`. The ticker per sector is
  `profile.trade_etfs.get(sector)` when present, else `sectors[s].etf` - the
  tradeable ETF is both what you buy and far better served by yfinance.
- One batched `yf.download([all profile tickers], period="6mo", interval="1d",
  auto_adjust=True, progress=False)` call; read Close prices per ticker.
- For each window in `config.MOMENTUM_WINDOWS` (`"5d": (weight, scale_pct)`, ...):
  pct return over the trailing N trading days = `(last / close[-(N+1)] - 1) * 100`
  (skip window if insufficient history or NaN).
- `score = sum(weight * tanh(return_pct / scale_pct))` over available windows,
  renormalized by the sum of used weights → roughly [-1, 1].
- Omit sectors with no usable data. Any network/yfinance failure → `logger.warning`
  and return `{}`. Never raises.

### analyzer.py
```python
def analyze(
    items: list[NewsItem],
    momentum: dict[str, SectorMomentum],
    news_weight: float = config.DEFAULT_NEWS_WEIGHT,
    profile: MarketProfile | None = None,
) -> list[SectorScore]
```
- `profile=None` resolves via `get_profile()`.
- VADER (`vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer`) with a finance
  lexicon overlay (~30 base unigrams plus `profile.lexicon`); one analyzer cached
  per `profile.key`.
- Sector keyword patterns are compiled per profile (cached by `profile.key`): one
  case-insensitive word-boundary regex per sector (`\b(?:kw1|kw2|...)\b`, each
  keyword `re.escape`d).
- Per item: sectors via the profile's compiled patterns matched against
  `title + " " + summary` (multi-label, may be empty); sentiment = compound score of
  `title + ". " + summary[:300]`, then for each `profile.phrases` entry whose
  lowercase phrase appears in the lowercase `title + " " + summary` add
  `value / 8.0`, and clamp the result to [-1, 1];
  recency weight = `0.5 ** (age_hours / config.RECENCY_HALF_LIFE_HOURS)`.
- Per sector: `raw = Σ(weight·sentiment) / Σ(weight)` over matched items (0.0 if none);
  `news_score = raw * sqrt(min(1.0, n / config.MIN_ARTICLES_FULL_CONFIDENCE))`;
  `buzz = n / total_sector_assignments` (0.0 if no assignments anywhere).
- `composite = news_weight * news_score + (1 - news_weight) * momentum[sector].score`
  when momentum is present for that sector, else `composite = news_score`.
- Returns **all `len(profile.sectors)` sectors**, sorted by composite desc. `top_items` = up to
  `config.TOP_HEADLINES_PER_SECTOR` ScoredNewsItems for the sector, sorted by
  `weight * abs(sentiment)` desc. Pure computation - no network, never raises.

### intraday.py
```python
@dataclass(frozen=True)
class IntradaySnapshot:
    ticker: str
    last_price: float
    asof: datetime                  # tz-aware, exchange tz as returned by yfinance
    day_change_pct: float           # last vs previous session close
    last_hour_change_pct: float     # last vs ~12 5-minute bars earlier
    range_position: float           # (last - day_low) / (day_high - day_low), 0..1
    avg_volume_20d: float           # 20-session mean daily volume

def get_intraday_snapshots(tickers: list[str]) -> dict[str, IntradaySnapshot]
```
- Exactly two batched downloads: one daily `period="2mo", interval="1d"` (previous
  close = second-to-last row's close when the last row is today, else the last row;
  plus 20-session mean volume) and one intraday `period="1d", interval="5m"`.
- If the intraday frame has no bars for the current calendar date in the exchange
  timezone, the market has not traded today - return `{}` (callers treat as closed).
- Defensive on MultiIndex vs flat columns (same pattern as market_data), NaN-safe,
  logs warnings, never raises; tickers with unusable data are omitted.

### decision.py
```python
@dataclass(frozen=True)
class TradeSignal:
    sector: str
    etf: str | None                 # None = no tradeable ETF (e.g. Realty/Media)
    action: str                     # "BUY" | "DON'T BUY"
    rank_score: float
    news_today: float               # news score computed on today-only items
    news_count_today: int
    momentum: float                 # multi-day momentum score (0.0 if unavailable)
    intraday: IntradaySnapshot | None
    illiquid: bool                  # avg_volume_20d < config.SIGNAL_MIN_AVG_VOLUME
    reasons: list[str]              # human-readable pass/fail per gate

def decide(items, momentum, snapshots, profile=None) -> list[TradeSignal]
def top_pick(signals) -> TradeSignal | None
```
- "Today's news trend" = items from the last `config.SIGNAL_NEWS_HOURS` hours run
  through `analyzer.analyze` for that profile; per-sector `news_score`/`news_count`
  come from that today-only run.
- **BUY gate rule** - all must pass (each recorded as a reason):
  1. sector has a trade ETF and its snapshot exists
  2. `news_today >= SIGNAL_MIN_NEWS` and `news_count_today >= SIGNAL_MIN_ARTICLES`
  3. `day_change_pct >= SIGNAL_MIN_INTRADAY_PCT` and `last_hour_change_pct >= 0`
  4. `momentum >= SIGNAL_MIN_MOMENTUM` AND
     `momentum_weight >= SIGNAL_MIN_MOMENTUM_WEIGHT` - the gate fails
     closed when the trend is unmeasurable or measured on too few
     windows, rather than treating a substituted 0.0 as neutral
  5. not illiquid (`avg_volume_20d * last_price >= SIGNAL_MIN_AVG_TURNOVER`;
     rupee turnover, not a unit count, so the bar is comparable across
     price levels)
- **Decline diagnostic** (symmetric mirror of the buy gates, evaluated only to
  enrich the explanation - it never changes the verdict): negative news over the
  article floor, day change at or below `-SIGNAL_MIN_INTRADAY_PCT` with a
  non-rising last hour, momentum at or below `-SIGNAL_MIN_MOMENTUM`, and liquid.
  Exposed as `decision.is_declining(signal)`.
- A sector that fails any buy gate is `DON'T BUY`, with `reasons` = the
  buy-gate results, then a `"--- decline check (diagnostic) ---"` separator,
  then the mirror results. A BUY carries only its own gate results.
- `decision.explain(signal)` returns a one-paragraph layman's account of the
  verdict, composed from the gate values (no LLM, no network). Both serializers
  attach it to the payload as `explanation`.
- TOP PICK = the highest-`rank_score` signal whose action is BUY; none -> no buy.
- `rank_score = 0.4*news_today + 0.4*tanh(day_change_pct / 1.0) + 0.2*momentum`,
  computed for every sector regardless of action; result sorted by rank desc.
  Pure computation, never raises.

### daily_signal.py
CLI: `python daily_signal.py [--market IN]` (default IN). Logs INFO to stdout,
fetches news (`SIGNAL_NEWS_HOURS * 2` lookback), momentum, and intraday snapshots
for `profile.trade_etfs`; empty snapshots -> prints "Market closed today - no
signal.", logs one MARKET_CLOSED row to signals.csv, exits 0. Otherwise prints the
report: a TOP SIGNAL line ("TOP SIGNAL: BUY ETF (sector) - rank +r"), then a
"WHY:" block wrapping `decision.explain(pick)` at 74 columns, then an "Also
cleared every check:" line when more than one sector qualifies. When nothing
qualifies it prints "NO BUY today - no sector passed every check." plus an
"Under real pressure today:" line naming any sector for which
`decision.is_declining` holds. Then the per-sector table, gate detail for the
top pick plus the top 2 and bottom 2 ranked sectors, and the disclaimer.
Appends one row per sector to `config.SIGNALS_CSV` (header auto-created, utf-8;
the `action` column carries "BUY" or "DON'T BUY", schema otherwise unchanged)
and writes `config.LAST_SIGNAL_JSON`, whose per-signal payloads each carry an
`explanation` string, for the app. Always exits 0. Output stays ASCII so it
survives a cp1252 Windows console.

Signal config knobs (config.py, paths resolved relative to the project root):
`SIGNAL_NEWS_HOURS=12`, `SIGNAL_MIN_NEWS=0.15`, `SIGNAL_MIN_ARTICLES=3`,
`SIGNAL_MIN_INTRADAY_PCT=0.2`, `SIGNAL_MIN_MOMENTUM=-0.2`,
`SIGNAL_MIN_AVG_TURNOVER=2_500_000`, `SIGNALS_CSV`, `LAST_SIGNAL_JSON`.

### claude_insights.py
```python
def is_available() -> bool
def generate_insights(scores: list[SectorScore], max_chars: int = 6000) -> str | None
```
- `is_available()`: True only if env `ANTHROPIC_API_KEY` is set and `anthropic`
  imports (lazy import inside the function).
- `generate_insights` builds a compact digest (ranked sector metrics + top headlines
  for the top 5 sectors, truncated to `max_chars`) and calls the Claude API:
  `anthropic.Anthropic().messages.create(model=config.CLAUDE_MODEL,
  max_tokens=config.CLAUDE_MAX_TOKENS, system=..., messages=[{"role": "user", ...}])`.
  Extract the first `text` block. **No** `temperature`/`top_p`/`top_k`/`thinking`
  params (unsupported on this model). On `ImportError` or `anthropic.APIError`:
  `logger.warning`, return `None`. Never raises.

### app.py
Streamlit dashboard. Uses only the contracts above. A `Market` selectbox is the
first sidebar control (options = `PROFILES` keys, default = `config.DEFAULT_MARKET`);
the cached loaders take a hashable `profile_key: str` and resolve `get_profile()`
inside. See the UI requirements given to the ui agent. Streamlit executes scripts
top-to-bottom - structure as small helper functions plus top-level flow (no
`main()` guard needed).

## Error-handling policy (all modules)

- The app must keep working when any single data source is down: dead feed → skip;
  yfinance down → news-only scores; no API key → no AI section.
- Catch *specific* exceptions (`requests.RequestException`, `KeyError`, etc.) close to
  the call site. Never bare `except:`; never `except Exception: pass`.
- `logging.getLogger(__name__)` per module; lazy `%s` formatting in log calls
  (no f-strings inside `logger.*(...)`).
- Close resources (`with` for requests sessions/files where applicable).

## Style rules

PEP 8. snake_case functions/vars, PascalCase classes, UPPER_SNAKE constants. f-strings
for normal formatting (never in logging calls). Type hints + a one-line docstring on
every public function. No mutable default arguments. No function over 80 lines. Imports
ordered stdlib → third-party → local with blank lines between groups.
