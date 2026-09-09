# Sector Pulse - Build Specification

A local Streamlit dashboard that gathers finance + major world news from free RSS
feeds, measures sector index/ETF price momentum from Zerodha Kite, scores a market
profile's sectors on a composite of news sentiment and momentum, and ranks where to
consider investing. Markets are config-driven **profiles** (IN today) - each
profile bundles its feeds, sector definitions, and sentiment-lexicon additions.
**Educational tool - not financial advice.** The UI must say so prominently.

Every price comes from Kite, which needs a paid subscription and an interactive
login whose token expires around 6am daily. The news and sentiment half runs
unauthenticated; momentum, the intraday scanner and the planner do not.
`ANTHROPIC_API_KEY` optionally enables an AI narrative-analysis section.

## File map and ownership

| File | Responsibility | Owner |
|---|---|---|
| `models.py` | Shared dataclasses (pre-written - do not modify) | core |
| `config.py` | Tunables + `DEFAULT_MARKET` | core |
| `profiles/__init__.py` | `MarketProfile` dataclass, `PROFILES` registry, `get_profile()` | core |
| `profiles/india.py` | IN profile: Nifty sectoral indices + Indian press feeds | news agent |
| `news_fetcher.py` | Concurrent RSS fetching, parsing, dedupe | news agent |
| `analyzer.py` | Classification, sentiment, composite scoring | analysis agent |
| `market_source.py` | the single price source: Kite symbols, bars, quotes | market agent |
| `market_data.py` | sector ETF momentum from Kite daily bars | market agent |
| `claude_insights.py` | Optional Claude narrative analysis | insights agent |
| `intraday.py` | intraday ETF snapshots from Kite | market agent |
| `indicators.py` | pure intraday maths | intraday agent |
| `levels.py` | entry/stop/target/size geometry | intraday agent |
| `setups.py` | intraday gates and ranking | intraday agent |
| `trade_costs.py` | Zerodha charge model | intraday agent |
| `scan_data.py` | batched bars, point-in-time truncation | intraday agent |
| `options_chain.py` | NSE chain and contract quality | intraday agent |
| `scan_intraday.py` | intraday scanner CLI | intraday agent |
| `instruments.py` | instrument discovery | discovery agent |
| `discover.py` | discovery CLI | discovery agent |
| `option_history.py` | option chain snapshots | discovery agent |
| `edge_lab.py` | rule measurement harness | research agent |
| `kite_instruments.py` | Zerodha instrument master (no auth) | kite agent |
| `kite_client.py` | Kite session, quotes, historical | kite agent |
| `kite_login.py` | Kite login helper | kite agent |
| `kite_bars.py` | bulk Kite bars, parquet cache | kite agent |
| `kite_ticker.py` | Kite WebSocket binary tick parser | kite agent |
| `tick_recorder.py` | background tick recorder | kite agent |
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
    key: str                       # "IN" (the US profile was removed with yfinance)
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

- **IN** (`profiles/india.py`, 12 Nifty sectoral indices), as Kite names them:
  Banks/NIFTY BANK, Financial Services/NIFTY FIN SERVICE, IT/NIFTY IT,
  Pharma/NIFTY PHARMA, Auto/NIFTY AUTO, FMCG/NIFTY FMCG, Metal/NIFTY METAL,
  Energy/NIFTY ENERGY, Realty/NIFTY REALTY, Infrastructure/NIFTY INFRA,
  PSU Banks/NIFTY PSU BANK, Media/NIFTY MEDIA. Feeds are the major Indian
  financial press plus three global feeds (category `"world"`) because Indian
  markets move on global cues.
- **US** was eleven SPDR sector ETFs and was removed with yfinance: Kite serves
  no US instrument, so the profile could not be migrated. `market_source`
  still translates the old Yahoo spellings (`^NSEI`, a trailing `.NS`) so a
  string written before the migration resolves, but new code uses Kite names.

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
  tradeable ETF is what you buy, so its own history, carrying its own premium
  and tracking error, is the relevant one.
- One `market_source.daily_bars(tickers, months=6)` call, which returns a flat
  OHLCV frame per symbol; read `frame["Close"]` per ticker.
- For each window in `config.MOMENTUM_WINDOWS` (`"5d": (weight, scale_pct)`, ...):
  pct return over the trailing N trading days = `(last / close[-(N+1)] - 1) * 100`
  (skip window if insufficient history or NaN).
- `score = sum(weight * tanh(return_pct / scale_pct))` over available windows,
  renormalized by the sum of used weights → roughly [-1, 1].
- Omit sectors with no usable data. Any network failure, and a missing Kite
  session, → `logger.warning` and return `{}`. Never raises. (`market_source`
  raises `NoSession`; this module catches it, because news-only scores are a
  survivable degradation. `market_source.bars` handles `KiteError`,
  `ValueError` and `OSError` itself, and `kite_instruments.fetch_master`
  returns `[]` rather than raising, so `NoSession` is the only escape left.)
- The planner catches `NoSession` too, but must not degrade *silently*: it
  falls back to the CSV price per symbol and names the source in its report
  (`kite (live)` / `kite 9/11, 2 from csv` / `csv (no live prices…)`), and
  `build_plan` raises `HoldingsError` when nothing at all could be priced.
  An unpriced book must never be mistaken for an empty one.

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
    asof: datetime                  # tz-aware, IST as returned by Kite
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
    etf: str | None                 # None = no tradeable ETF (e.g. Media)
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

## Intraday scanner (second, independent system)

Holds for hours where the 3:15 signal holds for weeks. Shares no threshold,
no data source and no code path with it. Everything below is a contract, not
a suggestion.

### Non-negotiables

These exist because each one was violated once and the violation was found
by audit, not by reasoning.

1. **A rule sees only what had printed.** Arrays are sliced to the current
   bar; ATR comes from prior sessions. There is no field on a context object
   that contains a future price.
2. **The replay cutoff is strictly before the instant.** A bar is stamped at
   its START, so the bar labelled 10:00 covers 10:00-10:05. `ts <= cutoff`
   made every "10:00" entry a 10:05 entry: 16 actionable setups where a true
   cutoff gives 12, 8 of them manufactured by that bar and 4 genuine ones
   deleted.
3. **A data source may inform a scan only if it predates the instant
   scanned.** Tested by timestamp, never by date. `now.date() != today` let a
   replay of 10:00 today bypass every guard and consume open interest
   captured at 20:36 plus a live option chain quoting closing prices.
4. **Absent data fails closed and says so.** Missing open interest scores
   0.0 on its rank term, not a flattering neutral. Missing relative volume,
   relative strength or turnover rejects the setup.
5. **A structural impossibility is never reported as a market reading.**
   Before 09:30 no breakout is representable, because the opening-range
   bounds are computed from the same bar as the price compared against them.
   The report says that, rather than printing an empty result.
6. **A data failure is never reported as a quiet market.** Zero bars
   returned is its own message.
7. **Replayed output is marked.** `scan_log.csv` carries a `replayed`
   column, and `append_log` rotates a file whose header predates a column
   change rather than misaligning every row.

### Module contracts

#### indicators.py
Pure maths over bar frames. Deterministic, network-free, returns `None`
rather than inventing a value. Callers must gate on `None` explicitly; a
caller treating it as zero fails open.

- `vwap` returns `None` when volume is absent or sums to zero. A zero-volume
  VWAP silently collapses to a simple mean, which is a different indicator
  wearing VWAP's name.
- `atr` smooths per-session true ranges via `true_range_by_session`, never
  across a session boundary. An overnight gap is not a tradeable move: with
  a constant 1.0 intraday range and a 20-rupee gap, whole-frame ATR read
  2.21 three bars in against a true 1.0, doubling every stop in the first
  hour. Each session's opening bar keeps its own high-low span.
- `relative_volume` compares today's cumulative volume against the **median**
  of prior sessions at the same time of day, and skips any prior session
  whose curve does not reach that clock. A mean baseline lets one frenzied
  day hide a genuine doubling; an early-ending session inflated a reading
  from 2.0 to 3.16 on nothing but missing data.
- `opening_range_closed` is a precondition, not a nicety. See non-negotiable 5.

#### levels.py
One horizon governs everything: `sigma = atr_per_bar * sqrt(bars_left)`.
Stop is a fraction of that sigma, target a multiple of the stop distance.
Sizing stop and target on different horizons is incoherent and rejected
every setup.

- The structural window is `[(1-band)*base, base]` and never exceeds `base`.
  Accepting a wider structural stop scales the target past sigma, which the
  reachability check then rejects: every structural stop in `(base,
  1.4*base]` was a guaranteed rejection.
- Quantity is capped by `capital * leverage / entry`. Risk-based sizing
  alone printed a 6.94 lakh position on 1 lakh of capital.
- `required_win_rate` solves `p*reward - (1-p)*risk = costs`. This is the
  number that makes a reward-to-risk ratio mean anything and must appear in
  every report.
- `cost_multiple` returns `0.0`, never `inf`, when cost is unknown, so the
  cost gate fails closed.

#### setups.py
Nine gates, each recording its own PASS/FAIL line. A setup is actionable
only when all pass. The rank score is a bounded weighted sum of per-symbol
readings, never scaled against the rest of the scan, so one symbol's score
does not move when another is added.

Direction requires price-vs-VWAP and the opening-range break to **agree**.
Disagreement is chop and yields nothing.

Documented weakness: of 99 rejected directional candidates in one run,
relative volume rejected 84 and reachability 8. The cost and win-rate gates
rejected 0 of 115. The trail implies nine filters; one does the work.

#### scan_data.py
The seam where a broker feed replaced yfinance, and that has now happened:
`market_source` fetches from Kite and hands back one flat OHLCV frame per
symbol. The claim that nothing downstream would change held - no indicator,
gate or level was touched by the swap. `truncate` implements
non-negotiable 2.

#### instruments.py, discover.py
No hardcoded symbol list anywhere. The index-versus-stock split derives from
the section row inside NSE's F&O master. One current file, written
atomically via `os.replace`, because with no history a truncated write
leaves nothing to fall back on.

#### edge_lab.py
Rules are measured, not argued about. Contract: `rule(ctx) -> "LONG" |
"SHORT" | None`. Reports favourable and adverse excursion distributions plus
a stop-and-target grid, and for each cell the hit rate a driftless random
walk would give that same geometry. **That difference is the only evidence
of predictive skill.** A high hit rate at a reward-to-risk below 1 proves
nothing.

Measured result for the shipped rule, which must not be quietly dropped from
this document: 3,537 signals, mean MFE 0.601 sigma against mean MAE 0.629,
so adverse excursion exceeds favourable at every percentile. 16 of 20 grid
cells lose at zero cost. The shipped geometry is 10.2 percentage points
worse than a coin flip. No rule in this repo has a demonstrated edge.

### Zerodha modules

#### kite_instruments.py
Public, unauthenticated. Contract universe only, never prices. Filter cash
equities on **segment**, not exchange: indices carry `instrument_type` EQ
and `exchange` NSE and differ by segment alone, so filtering on exchange
returns 136 indices as tradeable equities.

#### kite_client.py, kite_login.py
Credentials are read from the environment. Never logged, never printed,
never written into the repo, never placed in an exception message. The
access token travels in the WebSocket query string because Kite accepts it
nowhere else, so that URL must never reach a log. The session file is
gitignored and chmod 600.

Historical requests are chunked by an interval-aware span and paced under a
lock at Kite's documented 3 per second; the pacer serialises request starts,
not the waiting, so concurrent callers hide latency without exceeding the
limit.

#### kite_ticker.py
The wire format is not self-describing: a packet's meaning comes from its
byte length, and prices are integers whose divisor comes from the segment in
the token's low byte. A misread yields plausible wrong prices with no error.
Parsers are therefore pure functions over bytes, tested against hand-built
frames including truncated tails, unknown lengths, heartbeats and a frame
that overstates its own packet count.

## Error-handling policy (all modules)

- The app must keep working when any single data source is down: dead feed → skip;
  no Kite session → news-only scores; no API key → no AI section. The one
  deliberate exception is the planner, which must NOT degrade silently: with no
  prices it falls back to CSV values and labels them, because an unpriced book
  looks like an empty one.
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
