# Sector Pulse

A local Streamlit dashboard that gathers finance and major world news from free RSS feeds, measures sector index/ETF price momentum from Zerodha Kite, and scores a market profile's sectors on a composite of news sentiment and price momentum. It then ranks the sectors to highlight where market attention and strength are concentrated. One market profile ships today: **IN** (12 Nifty sectoral indices). An optional Claude-powered narrative section activates when `ANTHROPIC_API_KEY` is set.

> **Every price now comes from Zerodha Kite, so the app needs a Kite subscription and a login.** There is no free fallback any more. Kite issues an access token only after an interactive login with your password and 2FA, and it expires around 6am each morning, so `.venv\Scripts\python -m kite_login` is a daily step before anything that prices instruments. See [Zerodha Kite](#zerodha-kite-live-and-historical-data). What that buys: prices are live rather than roughly fifteen minutes late, minute candles reach back years instead of eight days, and historical open interest becomes available. What it costs: the news and sentiment half of the dashboard still runs unauthenticated, but momentum and the intraday scanner do not. The contribution planner degrades instead of failing - with no session it prices your book from the CSV and labels the source, and `--offline` skips the network entirely.

> **Not financial advice - educational use only.** Sector Pulse is a learning tool for exploring news sentiment and market momentum. Nothing it produces is a recommendation to buy or sell any security.

## Features

- Config-driven market profiles (India today) selectable in the app sidebar
- Aggregates free, live RSS/Atom feeds covering the selected market plus world news
- Concurrent feed fetching with timeouts, HTML stripping, and cross-feed deduplication
- Classifies headlines into the profile's sectors using rich keyword regexes
- VADER sentiment analysis with a finance-specific lexicon overlay, plus per-profile unigram and multi-word phrase boosts
- Recency-weighted news scoring (article weight halves every 24 hours by default)
- Sector momentum from Kite daily bars - the tradeable Nifty sector ETFs - across 5d / 21d / 63d windows
- Composite score blending news sentiment and momentum, with a tunable news weight
- Top headlines per sector, ranked by recency-weighted sentiment strength
- Optional AI insights section powered by the Claude API (`ANTHROPIC_API_KEY`)
- Resilient by design: a dead feed is skipped, an unavailable price source degrades to news-only scores, and a missing API key simply hides the AI section

## Architecture

| File | Responsibility |
|---|---|
| `models.py` | Shared dataclasses |
| `config.py` | Tunables + default market |
| `profiles/__init__.py` | `MarketProfile` dataclass, `PROFILES` registry, `get_profile()` |
| `profiles/india.py` | India profile: Nifty sectoral indices + Indian press feeds |
| `news_fetcher.py` | Concurrent RSS fetching, parsing, dedupe |
| `analyzer.py` | Classification, sentiment, composite scoring |
| `market_source.py` | the single price source: Kite symbols, bars, quotes |
| `market_data.py` | sector ETF momentum from daily bars |
| `intraday.py` | intraday ETF snapshots (5-minute bars) |
| `scan_intraday.py` | intraday scanner CLI: gates, levels, sizing, replay |
| `setups.py` | per-symbol gates and the ranked verdict |
| `indicators.py` | pure intraday maths: VWAP, opening range, CPR, ATR, RVOL |
| `levels.py` | entry, stop, target, size and the required win rate |
| `trade_costs.py` | Zerodha intraday equity and options charges |
| `scan_data.py` | batched bar downloads and point-in-time truncation |
| `options_chain.py` | live NSE chain, contract quality, PCR, max pain |
| `instruments.py` | instrument discovery and the current universe file |
| `discover.py` | discovery CLI |
| `option_history.py` | option chain snapshots, so replays become possible |
| `edge_lab.py` | harness measuring what price does after a signal |
| `kite_instruments.py` | Zerodha public instrument master (no auth) |
| `kite_client.py` | Kite session, quotes, paged historical candles |
| `kite_login.py` | interactive Kite login helper |
| `kite_bars.py` | bulk historical bars, cached to parquet |
| `kite_ticker.py` | Kite WebSocket tick stream and binary parser |
| `tick_recorder.py` | background tick recorder process |
| `decision.py` | Daily BUY / DON'T BUY decision engine |
| `daily_signal.py` | End-of-day signal CLI (report + `signals.csv` + `last_signal.json`) |
| `news_archive.py` | Dated JSONL archive of fetched headlines, so sentiment can be backtested |
| `claude_insights.py` | Optional Claude narrative analysis |
| `app.py` | Streamlit dashboard |
| `portfolio_models.py` | Rebalancer dataclasses (Holding, DriftRow, Order, Plan) |
| `holdings.py` | Holdings CSV and target-weight file loaders |
| `allocator.py` | Drift and whole-unit allocation maths (pure) |
| `schedule_rules.py` | Contribution cadence gates and the contributions log |
| `quotes.py` | Last traded prices for bare NSE symbols |
| `import_tradebook.py` | Builds holdings.csv from Zerodha tradebook exports |
| `plan_investment.py` | Monthly contribution planner CLI |
| `test_rebalance.py` | Rebalancer test suite |
| `test_signal.py` | Offline signal-side tests (archive, liquidity gate) |
| `smoke_test.py` | End-to-end smoke test |

## Quickstart

From the project root (Windows / PowerShell):

```powershell
# 1. Create the virtual environment (skip if .venv already exists)
python -m venv .venv

# 2. Install dependencies
.venv\Scripts\python -m pip install -r requirements.txt

# 3. Run the dashboard
.venv\Scripts\python -m streamlit run app.py
```

Streamlit will open the dashboard in your browser (typically at `http://localhost:8501`).

## Market profiles

Sector Pulse is profile-driven: each market bundles its own RSS feeds, sector
definitions (with index/ETF tickers and classification keywords), and sentiment
lexicon additions.

- **In-app selector** - the `Market` dropdown at the top of the sidebar. Only
  `IN - India (NSE)` is registered today, so it has one entry.
- **Default market** - `IN`, which both the dashboard and every `profile=None`
  call resolve to. `SECTOR_PULSE_MARKET` still overrides it, but there is
  nothing else to select.
- **The US profile was removed** when yfinance went. It was eleven SPDR sector
  ETFs, and it went with that dependency: Kite is an Indian broker and serves
  no US instrument, so keeping the profile would have meant keeping a second
  price source alive for a market this book has no position in. The
  `MarketProfile` machinery is untouched, so a second profile can return the
  day a source for it does.

- **The India profile** tracks 12 Nifty sectoral indices (Bank Nifty, Financial
  Services, IT, Pharma, Auto, FMCG, Metal, Energy, Realty, Infrastructure, PSU
  Banks, Media) and pulls headlines from the Indian financial press (Economic
  Times, Moneycontrol, LiveMint, BusinessLine, NDTV Profit, Google News India)
  plus three global feeds, since Indian markets move on global cues. It also adds
  India-specific sentiment terms (e.g. "upper circuit", "FII selling", "repo rate
  cut", monsoon phrases).
- **Adding a new market** is one file: create `profiles/<market>.py` defining a
  `MarketProfile` (key, label, currency, feeds, sectors, lexicon, phrases) and
  register it in `PROFILES` in `profiles/__init__.py`. Everything else - fetching,
  scoring, momentum, the UI, and the smoke test - picks it up automatically.

## Configuration

All tunables live in `config.py`:

| Setting | Default | What it does |
|---|---|---|
| `NEWS_MAX_AGE_HOURS` | `24` | Ignore articles older than this. Matches the signal's own fetch window so dashboard-archived days are comparable with CLI-archived ones |
| `REQUEST_TIMEOUT_SECONDS` | `10` | Per-feed HTTP timeout |
| `MAX_ITEMS_PER_FEED` | `40` | Cap on items kept per feed |
| `RECENCY_HALF_LIFE_HOURS` | `24` | Article weight halves every N hours |
| `MIN_ARTICLES_FULL_CONFIDENCE` | `5` | News score is damped below this many articles per sector |
| `DEFAULT_NEWS_WEIGHT` | `0.5` | Composite = `w * news + (1 - w) * momentum` |
| `MOMENTUM_WINDOWS` | `5d/21d/63d` | Trading-day windows with per-window weight and tanh scale |
| `CACHE_TTL_SECONDS` | `900` | Dashboard data cache lifetime (15 min) |
| `TOP_HEADLINES_PER_SECTOR` | `6` | Headlines shown per sector |
| `CLAUDE_MODEL` / `CLAUDE_MAX_TOKENS` | `claude-opus-4-8` / `4096` | Model used for the optional AI insights section |

### Optional: AI insights

Set the `ANTHROPIC_API_KEY` environment variable to enable the Claude-powered narrative analysis section:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Without the key, the app runs normally and simply omits the AI section.

## Daily 3:15 PM signal

`daily_signal.py` produces one end-of-day recommendation per run: which single
tradeable sector ETF to consider buying today, if any. It correlates
**today's news trend** (a today-only sentiment run over the last
`SIGNAL_NEWS_HOURS` hours) with each ETF's **intraday traded trend** (5-minute
bars vs the previous close), sanity-checked by multi-day momentum.

```powershell
.venv\Scripts\python daily_signal.py --market IN
```

Each run prints a report, appends one row per sector to `signals.csv`, and
writes `last_signal.json`, which the dashboard's "Today's 3:15 signal" section
displays (with a "Compute signal now" button for a live re-check).

### How the gates work

A sector is a BUY only when **all** BUY gates pass:

1. The sector has a tradeable ETF (`profile.trade_etfs`) with intraday data
   today. In the India profile, only Media has no listed sector ETF.
2. Today-only news score >= `SIGNAL_MIN_NEWS` (0.15) over at least
   `SIGNAL_MIN_ARTICLES` (3) articles.
3. Intraday confirmation: day change >= `SIGNAL_MIN_INTRADAY_PCT` (0.2%) and
   the last hour is not falling (price confirming into the close).
4. Multi-day momentum >= `SIGNAL_MIN_MOMENTUM` (-0.2), **and** computed on at
   least `SIGNAL_MIN_MOMENTUM_WEIGHT` (80%) of the configured window weight.
   Momentum is measured on the tradeable ETF, not the sector index. The
   original reason was a yfinance data gap - 1 daily bar since 2026-07-20 for
   9 of the 12 Nifty indices against 34-35 for the ETFs, silently reducing
   those sectors to the 63d window alone - and that gap is gone on Kite. The
   choice stands on its own anyway: you buy the ETF, so its own history, with
   its own premium and tracking error, is the relevant one. A missing score
   used to arrive as 0.0, and 0.0 >= -0.2, so a data outage made this gate
   PASS. It now fails closed.
5. Liquidity: 20-session average daily **turnover** >=
   `SIGNAL_MIN_AVG_TURNOVER` (Rs 25,00,000). Turnover, not a unit count: a
   50,000-unit floor was 84x stricter for INFRABEES at Rs 958/unit than for
   OILIETF at Rs 11.40, and wrongly excluded INFRABEES despite Rs 89 lakh of
   real daily turnover. This threshold is still an uncalibrated guess.

If a sector fails any buy gate the verdict is **DON'T BUY**. There is no SELL
verdict: the engine never tells you to sell something you already hold. A mirror
set of gates still runs, purely as a diagnostic, so the report can distinguish a
sector under real pressure (negative news, falling price, downward trend all
lining up) from one merely having a quiet day.

Each run also prints a **WHY** paragraph explaining the top pick in plain
English: how positive the news flow was and over how many stories, whether the
price confirmed it into the close, what the multi-week trend looks like, and
whether it trades enough volume to act on. That text is generated from the gate
values themselves, so it needs no API key and cannot cite a reason the rule did
not actually check.

All sectors are ranked by `0.4 * news_today +
0.4 * tanh(day_change_pct) + 0.2 * momentum`; the top signal is the actionable
sector with the highest rank score. If nothing clears every gate, the day's
recommendation is NO BUY. When the market has
not traded today (weekend/holiday), the run reports "Market closed".

### The news archive, and why it runs first

RSS feeds serve roughly a 48-hour window. A headline not captured on the day it
appears is gone permanently, and with it any chance of ever backtesting the news
gate. So every `daily_signal.py` run writes what it fetched to
`news_archive/YYYY-MM-DD.jsonl` before doing anything else, deduplicated by link,
and logs how many new headlines it stored.

`news_archive.load_archived_news(day)` reads a day straight back into `NewsItem`
objects, which feed `analyzer.analyze` unchanged. That is the backtest path: from
the first archived day onward, a past day's sentiment can be replayed and scored.
Before the first archived day it cannot, at any price.

The archive is gitignored (it is a data store, and it grows by roughly 270 KB a
day at current feed volumes).

### Known gaps in the signal engine

Read this before trading on it. The rebalancer half carries none of these.

- **No backtest.** Every threshold is a guess: `SIGNAL_MIN_NEWS`,
  `SIGNAL_MIN_ARTICLES`, `SIGNAL_MIN_INTRADAY_PCT`, `SIGNAL_MIN_MOMENTUM`,
  `SIGNAL_MIN_AVG_TURNOVER`, and the 0.4/0.4/0.2 rank weights. Ten-odd free
  parameters fitted to nothing.
- **No horizon and no exit rule.** The engine only ever says BUY or DON'T BUY.
  Without a holding period there is no answer to "was that BUY right?", so
  `signals.csv` can never become evidence no matter how many rows it collects.
  This is the blocker: nothing else is measurable until it is fixed.
- **No cost model.** Brokerage, STT, stamp duty, GST and the ETF NAV
  premium/discount are all absent. On a small order these dominate.
- **`buzz` is computed and never used.** `analyzer.py` produces it and `app.py`
  displays it; no gate reads it.
- **Its universe only partly overlaps your portfolio.** The tradeable tickers
  are sector ETFs chosen to be the fund you would actually buy, but most are
  not in `targets.yaml`. Acting on a signal for one of those means buying
  something the planner will treat as an untracked holding.

### Scheduling on Windows (weekdays 15:15 IST)

```powershell
schtasks /Create /TN "SectorPulseDailySignal" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:15 `
  /TR "\"C:\path\to\sector-pulse\.venv\Scripts\python.exe\" \"C:\path\to\sector-pulse\daily_signal.py\" --market IN"
```

To remove the scheduled task:

```powershell
schtasks /Delete /TN "SectorPulseDailySignal" /F
```

> **Strong caveats - read before relying on this.**
>
> - The 15:15 snapshot needs a **live Kite session**. Kite's token expires
>   around 6am, so a scheduled run started before you logged in that morning
>   will print no prices at all rather than stale ones.
> - The gate rule is **unvalidated**: it has no backtest behind it and earns
>   credibility only as `signals.csv` accumulates a real track record you can
>   evaluate yourself. Note that the *intraday* rule, which has now been
>   measured properly, turned out to have no edge - that is not evidence
>   about this rule, but it is a fair warning about how these usually end.
> - This is **decision support, not financial advice**. It is an educational
>   tool; treat every BUY line as a prompt to do your own research.
> - It **never places orders** - it only prints, logs to CSV, and writes a
>   JSON file for the dashboard.

## How the scoring works

Each headline is classified into one or more sectors via keyword matching, scored for sentiment with VADER (plus a finance lexicon overlay), and weighted by recency - an article's influence halves every `RECENCY_HALF_LIFE_HOURS`. Per sector, the recency-weighted average sentiment is damped when there are few articles, producing a news score in roughly [-1, 1]. In parallel, each sector's ETF gets a momentum score from tanh-squashed 5-day, 21-day, and 63-day returns, blended by window weights. The composite is `news_weight * news_score + (1 - news_weight) * momentum_score` (news-only when momentum data is unavailable), and sectors are ranked by composite, highest first.

## Intraday scanner

A second, separate system. The 3:15 signal above holds for weeks; this one
holds for hours. They share no thresholds, no data source and no code path,
because a rule that survives a week of drift is not the same rule that
survives an afternoon of noise.

```bash
.venv\Scripts\python discover.py                       # cache the instrument universe
.venv\Scripts\python scan_intraday.py --top 10         # scan the F&O single stocks
.venv\Scripts\python scan_intraday.py --options        # also pick a tradeable contract
.venv\Scripts\python scan_intraday.py --all-equities   # all ~2,570 listed names, slow
```

It ranks names on a transparent rule and prints, for each setup that clears
every gate, an entry, a stop, a target and a size. Direction is never
predicted: it is read off two agreeing structural facts, which side of the
session VWAP price sits and which side of the opening range it has broken.
When those disagree the symbol is chop and gets nothing, which is most
symbols on most days.

### Read this before using any of it

**No rule here has a measured edge, and one has been measured to have
none.** The harness in `edge_lab.py` ran the shipped rule over 3,537 signals
across 49 sessions and 60 names:

| | p10 | p25 | p50 | p75 | p90 | mean |
|---|---|---|---|---|---|---|
| Favourable excursion (MFE) | 0.063 | 0.178 | 0.419 | 0.785 | 1.290 | 0.601 |
| Adverse excursion (MAE) | 0.086 | 0.226 | 0.482 | 0.845 | 1.312 | 0.629 |

Both in units of sigma, where sigma is the bar ATR grown by the square root
of bars remaining. Two things follow, and both matter more than any feature
in this section.

First, **adverse excursion exceeds favourable at every percentile**. After
this breakout fires, price moves against the signal slightly more than for
it. A rule with that property cannot be rescued by choosing a better stop or
target, because no geometry manufactures edge that is not in the signal.

Second, the rule's own target sat at 1.000 sigma, which only **16.6%** of
signals ever reach. That is why live replays returned so few winners: the
target was placed near the 85th percentile of favourable movement. The
sqrt-of-time model overstated reachable range by roughly 2.4x at the median,
because intraday prices mean-revert at short horizons rather than
random-walking.

Across the whole stop-and-target grid, **16 of 20 combinations lose money
even at zero cost**, and the best gross expectancy is +0.047 R against the
roughly 0.16 R that round-trip charges consume. The shipped geometry scores
10.2 percentage points *worse* than a coin flip.

Every gate is a conventional technical-analysis choice, and the rank weights
are a judgement call. Neither is validated. Treat the output as a
description of what was measured, not a recommendation.

### The gates, and which ones actually bind

Nine gates run per symbol, each printing its own PASS/FAIL line so the
verdict can be rebuilt by hand. Honesty about their relative weight, from a
run over 210 names: of 99 directional candidates that were rejected, **the
relative-volume gate rejected 84 and reachability 8**. The cost-multiple and
required-win-rate gates rejected **zero of 115**. The trail reads like nine
independent filters; in practice one does nearly all the work.

`SCAN_REQUIRE_OI_CONFIRMATION` is deliberately `False`. Open interest rising
with price is a market convention with no backtest behind it here, so it
informs the rank and the reasons rather than vetoing a setup. Promoting an
unvalidated belief to a veto would look like rigour while being a guess.

### What is honest about it

- **The required win rate.** Every setup prints the hit rate it needs to
  break even after real charges. That number is what makes a
  reward-to-risk ratio mean anything: 2:1 sounds like an edge but only says
  a 33% hit rate breaks even *before* costs.
- **Aggregate exposure.** Each row is sized independently against its own
  stop, so the per-trade cap does not bound the total. The report states the
  combined risk and notional, which on one session came to 11.5x the assumed
  capital.
- **Nothing is computable before 09:30.** The opening range covers the first
  15 minutes, so until it closes the latest bar is one of the bars defining
  it, the range brackets the current price by construction, and no breakout
  can be represented. The scanner says exactly that instead of printing an
  empty result that looks like a quiet market.
- **A data failure is not a quiet market.** Zero bars returned prints as a
  data failure, not as an absence of setups.

### Point-in-time replay

```bash
.venv\Scripts\python scan_intraday.py --as-of "10:00"              # today
.venv\Scripts\python scan_intraday.py --as-of "2026-09-04 10:00"   # an earlier session
```

Bars after the instant are discarded, so the scan sees only what was
knowable then. Two subtleties were found by audit rather than by design, and
both are now enforced:

- The cutoff is **strictly** before the instant. A bar is stamped at its
  start, so the bar labelled 10:00 covers 10:00 to 10:05 and its close is
  the 10:05 price. Keeping it made 16 setups where a true 10:00 cutoff gives
  12, with 8 manufactured by that single bar and 4 genuine ones deleted.
- Open interest is used only when the instrument snapshot was captured at or
  before the replayed instant. NSE publishes no OI history, so a snapshot
  taken in the evening cannot inform a morning scan. The column reads `n/a`
  and says why.

**Option contracts cannot be replayed at all** unless a chain snapshot was
stored at the time. NSE serves only a live chain and keeps no archive, so a
chain fetched now describes now: an audit of a 10:00 replay found every
printed spot equal to that day's close, ten for ten. Capture chains with the
dashboard's Instrument sync tab and replays after the first capture can read
a real one.

Replayed rows are flagged in `scan_log.csv` so they can never be mistaken
for a live scan made at that time.

## Instrument discovery

Nothing in the scanner carries a hardcoded symbol list. `discover.py`
fetches the universe from NSE and Zerodha at run time and caches it to
`instruments/universe.json`, one current file rather than a history, so a
scan can say exactly which instruments were listed when it ran.

| | Count | Source |
|---|---|---|
| Listed equities | 2,571 | NSE `EQUITY_L` master |
| F&O indices / single stocks | 6 / 210 | derived from the master file's own section row |
| Underlyings with live OI and turnover | 216 | one NSE request |
| Futures contracts | 647 | Zerodha's public instrument master |

The index-versus-stock split comes from the section row inside NSE's F&O
file, not a maintained list. That is not pedantry: the hand-written list it
replaced was missing `NIFTYFPI`, so every scan requested it as an equity and
logged a 404, and no test could have caught it because the list *was* the
definition of truth. Deriving found 6 indices where 5 were hardcoded.

## Zerodha Kite

Two tiers, and the boundary matters.

**No credentials needed.** `kite_instruments.py` reads Zerodha's public
instrument master: 647 futures contracts, 32,437 option contracts, 10,111
cash equities and 236 indices in one unauthenticated request. This closes a
gap that looked permanent - NSE serves no bulk per-contract futures data,
because the derivatives bhavcopy 404s on every published URL and
`/api/quote-derivative` has been withdrawn. Lot sizes cross-check exactly
against NSE's own market-lot file on all 216 underlyings.

Note that indices carry `instrument_type` EQ and `exchange` NSE, differing
from equities by **segment alone**. Filtering on exchange returns 136 of
them as tradeable equities, NIFTY 50 and NIFTY BANK among them.

**Your own subscription and login.** Quotes, historical candles and the tick
stream.

```bash
setx KITE_API_KEY     your_key
setx KITE_API_SECRET  your_secret
# open a NEW terminal - setx only affects shells started afterwards
.venv\Scripts\python -m kite_login
```

Credentials are read from the environment and never written into this repo,
never logged and never echoed. The login is yours to perform: Kite issues a
`request_token` only after an interactive login with your password and 2FA,
so no tool can authenticate as you unattended. The resulting access token
lands in a gitignored, owner-readable file and expires around 6am the next
day, with no non-interactive refresh for retail apps.

### Why the historical API matters more than the tick feed

`kite_client.historical()` serves minute candles reaching back **at least
six years**, where yfinance served eight days of 1-minute and about sixty of
5-minute. That gap is the whole reason this migration was worth doing.
Frames come back in the same column shape the rest of the project consumes,
so they drop into `edge_lab` without translation, and `oi=True` adds
historical open interest - the one field NSE never publishes. Note that no
caller passes `oi=True` yet: the scanner and the dashboard still drop open
interest for a replayed session and say so. The capability is there, unused.

That depth is what turns the edge search from suggestive into conclusive.
The current measurement rests on 49 correlated sessions; a year of Kite data
is 249. Speed does not fix a thin sample.

### Live ticks

```bash
.venv\Scripts\python -m tick_recorder --fo --futures --mode full
```

`kite_ticker.py` implements Kite's binary protocol directly. The format is
not self-describing: a packet's *meaning* comes from its byte *length* (8 is
last price, 28 and 32 an index, 44 a quote, 184 a quote plus open interest
and depth), and prices are integers whose divisor comes from the segment
encoded in the instrument token's low byte. Get that wrong and you get
plausible prices off by a factor of a hundred, silently, so the parsers are
pure functions tested against hand-built frames rather than against a live
market.

The recorder is a **separate process**, not a thread inside Streamlit.
Streamlit re-executes its script on every interaction, so a thread started
there duplicates across reruns and dies with the session. This one owns one
socket and one file, and keeps recording while nobody is watching.

The tick feed is push rather than poll, which is what separates it from
`kite_client.quote()`. The quote endpoint is live but sampled: you learn the
last price at the moment you asked. The tick stream delivers every change,
which is the only way to reconstruct what happened between two polls.

## Measuring a rule before trusting it

`edge_lab.py` exists because the first rule failed in a way that could only
be diagnosed by measurement. You write a rule, it measures what price did
afterwards:

```python
def my_rule(ctx):
    """Return "LONG", "SHORT" or None for the bar at ctx.i."""
    if not ctx.orb_closed:
        return None
    return "LONG" if ctx.price > ctx.vwap[-1] and ctx.price > ctx.orb_high else None
```

`ctx` exposes only what had printed by bar `i` - arrays are pre-sliced and
the ATR comes from prior sessions - so a rule cannot look ahead even by
accident. Three disciplines are enforced by the harness rather than left to
each rule: one signal per direction per session taken on the first bar it
fires, a bar spanning both stop and target counted as a stop, and unresolved
positions marked out at the close rather than discarded.

The output is a grid of stop and target sizes with, for each, the hit rate,
the hit rate a driftless random walk would give that same geometry, and the
difference. That difference is the only evidence of predictive skill. A high
hit rate at a reward-to-risk below 1 proves nothing: 87% wins at 0.33 R:R is
what a random walk already pays.

## Monthly contribution planner (target weights)

`plan_investment.py` answers one question: **where should this month's money go?**
It compares what you hold against target weights you set yourself, then splits the
contribution into whole-unit buy orders against the largest rupee shortfalls.

This is deliberately the opposite of the daily signal above. The signal engine ranks
sectors by news and momentum; the planner ignores both. Position sizing is decided by
your target weights alone, so a contribution can never be talked into a falling sector
beyond the ceiling you set for it.

```powershell
.venv\Scripts\python plan_investment.py
```

### Building holdings.csv from your tradebooks

Rather than maintaining `holdings.csv` by hand, derive it from Zerodha tradebook
exports (Console -> Reports -> Tradebook, one CSV per financial year). The tradebook is
the authoritative record of what you bought and sold, so net quantity and average cost
come from replaying it:

```powershell
.venv\Scripts\python import_tradebook.py "tradebook-FY24-25.csv" "tradebook-FY25-26.csv" "tradebook-FY26-27.csv"
```

Pass every year you have. Trades are de-duplicated on their full identity rather than
on `trade_id` alone, because that id is only unique per exchange per day - the NSE and
BSE ranges genuinely overlap. Overlapping exports are therefore safe, including rows
carrying no `trade_id`, and two different trades sharing one identity are reported as an
error rather than silently merged. Ordering is by parsed timestamp, with buys settled
before sells on a tie, so the order of the files on the command line does not change the
result. The importer replays trades using the running weighted-average convention: a
sell removes quantity at the prevailing average and leaves that average unchanged.

It prints a reconciliation (quantity, average cost, invested, LTP, value, P/L per
symbol), reports realised profit on every sell whether the position survived or not,
writes the open positions to `holdings.csv`, and fetches live prices to fill the LTP
column. Add `--dry-run` to see the summary without writing, or `--no-prices` to skip the
network.

A position you sold out of is simply absent from `holdings.csv`, because you hold none of
it. That is not a retirement: add the symbol to `targets.yaml` with a weight and
contributions will buy it again from zero.

Three caveats:

- **Charges are not in the tradebook.** Average cost is built from trade prices only, so
  it excludes brokerage, STT, stamp duty and GST. It will read very slightly below the
  figure Kite shows.
- **Missing history is flagged, not guessed.** If a sell exceeds everything the supplied
  files account for, the opening position predates them. The importer names the symbols
  and *refuses to write*, so an incomplete set cannot quietly replace a good
  `holdings.csv`; pass `--force` if you want it written anyway. Handing it only a
  sells-only tradebook is refused for the same reason.
- **Corporate actions are invisible.** Splits, bonuses and consolidations never appear
  in a tradebook, so a replay cannot see them. If one has affected a holding, that
  symbol's quantity and average cost will be wrong and nothing here can detect it.
  Cross-check against Kite after any such event.

### Inputs

| File | What it is |
|---|---|
| `holdings.csv` | Your positions. Export from Zerodha Console or Kite and drop it here. |
| `targets.yaml` | Your target weights in percent. Must sum to 100. Copy `targets.example.yaml` to start. |
| `contributions.csv` | Written by `--record`. The cadence gates read the last date from it. |

The holdings loader matches columns generously, so both the Zerodha Console export
(`Symbol`, `Quantity Available`, `Average Price`, `Previous Closing Price`) and the Kite
export (`Instrument`, `Qty.`, `Avg. cost`, `LTP`) load without editing. Zerodha's other
quantity columns (discrepant, pledged, long term) are ignored. `holdings.example.csv`
shows the shape. `holdings.csv`, `targets.yaml`, `contributions.csv`, and
`last_plan.json` are gitignored because they are personal financial data, so a
fresh clone starts from the two `.example.` files:

```powershell
Copy-Item holdings.example.csv holdings.csv
Copy-Item targets.example.yaml targets.yaml
```

Live prices for the planner come from Kite's quote endpoint - the actual
last traded price, not the previous close, which is what you want when
sizing an order. Symbols are bare NSE tradingsymbols; the `.NS` suffix went
with yfinance.

This means the planner needs a Kite session. If there is none, or a symbol
returns no price, the planner falls back to the price column in your CSV and
says which prices came from where in the report - look for the `Prices:`
line, which reads `kite (live)`, `kite 9/11, 2 from csv`, or `csv (no live
prices - is there a Kite session?)`. `--offline` skips the network entirely
and is the way to run this without logging in at all.

### The cadence: why it is safe to run daily

Running this every day will not make you buy every day. The contribution is gated:

- **First run** with no `contributions.csv` history: INVEST.
- **Scheduled**: INVEST once the calendar month has rolled over from the last
  recorded buy (`CONTRIBUTION_INTERVAL_MONTHS`, default 1). Buying on the 4th means
  the next scheduled buy is the 4th of next month; a fixed 30-day interval would
  instead walk backwards through the calendar (Jan 1, Jan 31, Mar 2, ...).
- **Off-cycle**: INVEST early only if the largest *closeable* drift breaches
  `REBALANCE_BAND_PP` (5pp) *and* at least `MIN_DAYS_BETWEEN_BUYS` (7) days have
  passed. This is the only path to an unscheduled buy.

  **This accelerates deployment while you are far from target, by design.** A book
  19.5pp under on one holding keeps re-breaching the band, so it deploys roughly
  60,000 a month for two months and then 20,000 once drift falls inside the band -
  about 319,000 over a year against 240,000 at a pure monthly cadence. Raise
  `MIN_DAYS_BETWEEN_BUYS` toward 30 if you want a strictly monthly rate instead.
  Note the planner has no view of your available cash: it prints an order every 7
  days whether or not the money is there.

  "Closeable" matters: the band looks only at holdings that are below target *and*
  have a usable price, because those are the only ones a purchase can move. Drift on
  an over-target, untracked, or unpriced holding would otherwise breach the band
  forever while every run placed no orders, firing an off-cycle buy every 7 days.
- **Otherwise**: HOLD, with the next due date and the drift table printed for
  information.

Every gate is printed as a PASS or FAIL line, so a HOLD always explains itself.

### Holdings outside your targets

Target weights describe the portfolio you are actually managing, so weights are
measured against the value of the *tracked* symbols only. A holding that is absent from
`targets.yaml` sits outside the plan: it is listed with a `*`, its value is reported
separately, and it is excluded from the drift band.

That exclusion matters. If off-plan holdings counted toward the band, their weight would
register as permanent drift that buying can never close, and the planner would fire an
off-cycle buy every 7 days forever. Either add such a symbol to `targets.yaml` or sell it
down; leaving it out simply means the tool ignores it when sizing contributions.

### Allocation modes

- `--mode fill` (default) is your rule: waterfall into whichever holding is furthest
  below target in rupees, then the next, until the cash runs out. A unit is only bought
  when at least half its price still fits inside that symbol's shortfall, so a buy
  cannot blow past a target.
- `--mode spread` divides the contribution across every shortfall in proportion, then
  tops up with whatever integer units the remainder can still buy.

ETFs trade in whole units, so the planner floors every order and reports the undeployed
remainder rather than pretending it was invested, along with the reason it is idle.
`--min-order` (default 500) suppresses dribble orders: that symbol's share goes to the
next real gap, and if every gap is too small the cash stays put and the report says so.
Run with `--min-order 0` to deploy into small gaps as well.

### Recording a contribution

Once you have actually placed the orders in Kite:

```powershell
.venv\Scripts\python plan_investment.py --record
```

That appends one row per order to `contributions.csv`, which restarts the 30-day clock.
Nothing is recorded unless the run actually planned orders. The planner never places
orders itself.

### Useful flags

| Flag | Effect |
|---|---|
| `--amount 20000` | Rupees to deploy. Defaults to `config.MONTHLY_CONTRIBUTION`. |
| `--mode fill\|spread` | Allocation strategy. |
| `--force` | Plan orders as a dry run even when the cadence says HOLD. |
| `--offline` | Use CSV prices, no network. |
| `--json` | Machine-readable plan. Also always written to `last_plan.json`. |
| `--min-order` | Suppress orders below this rupee value. |

### Setting your target weights

The weights in `targets.example.yaml` are a neutral illustration, not a recommendation.
Set your own in `targets.yaml` before the first real run: if the targets simply mirror
your current allocation, the planner will faithfully hold that allocation in place,
including whatever is already underwater. Symbols you hold but leave out of the file are
treated as target 0% and flagged with `*` in the drift table.

### Tests

```powershell
.venv\Scripts\python test_rebalance.py
```

Runs standalone, or under pytest if you prefer. Covers the drift maths, whole-unit
allocation, both modes, the CSV and targets loaders, and every cadence boundary.

## Disclaimer

**Not financial advice - educational use only.** This project demonstrates news aggregation, sentiment analysis, and market-data processing techniques. Its scores and rankings are illustrative, may be inaccurate or stale, and must not be used as the basis for investment decisions.
