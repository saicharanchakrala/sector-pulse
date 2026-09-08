# Sector Pulse

A local Streamlit dashboard that gathers finance and major world news from free RSS feeds, measures sector index/ETF price momentum via yfinance, and scores a market profile's sectors on a composite of news sentiment and price momentum. It then ranks the sectors to highlight where market attention and strength are concentrated. Two market profiles ship out of the box - **US** (11 GICS sectors via SPDR ETFs) and **IN** (12 Nifty sectoral indices). No API keys are required for the core app; an optional Claude-powered narrative section activates when `ANTHROPIC_API_KEY` is set.

> **Not financial advice - educational use only.** Sector Pulse is a learning tool for exploring news sentiment and market momentum. Nothing it produces is a recommendation to buy or sell any security.

## Features

- Config-driven market profiles (US and India today) selectable in the app sidebar
- Aggregates free, live RSS/Atom feeds covering the selected market plus world news
- Concurrent feed fetching with timeouts, HTML stripping, and cross-feed deduplication
- Classifies headlines into the profile's sectors using rich keyword regexes
- VADER sentiment analysis with a finance-specific lexicon overlay, plus per-profile unigram and multi-word phrase boosts
- Recency-weighted news scoring (article weight halves every 24 hours by default)
- Sector momentum from yfinance - SPDR ETFs for the US, Nifty sectoral indices for India - across 5d / 21d / 63d windows
- Composite score blending news sentiment and momentum, with a tunable news weight
- Top headlines per sector, ranked by recency-weighted sentiment strength
- Optional AI insights section powered by the Claude API (`ANTHROPIC_API_KEY`)
- Resilient by design: a dead feed is skipped, a yfinance outage degrades to news-only scores, and a missing API key simply hides the AI section

## Architecture

| File | Responsibility |
|---|---|
| `models.py` | Shared dataclasses |
| `config.py` | Tunables + default market |
| `profiles/__init__.py` | `MarketProfile` dataclass, `PROFILES` registry, `get_profile()` |
| `profiles/us.py` | US profile: SPDR sector ETFs + US/world feeds |
| `profiles/india.py` | India profile: Nifty sectoral indices + Indian press feeds |
| `news_fetcher.py` | Concurrent RSS fetching, parsing, dedupe |
| `analyzer.py` | Classification, sentiment, composite scoring |
| `market_data.py` | yfinance sector index/ETF momentum |
| `intraday.py` | yfinance intraday ETF snapshots (5-minute bars) |
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

- **In-app selector** - the `Market` dropdown at the top of the sidebar switches
  between profiles (`US - United States`, `IN - India (NSE)`).
- **Default market** - set the `SECTOR_PULSE_MARKET` environment variable (`US` or
  `IN`) to choose which profile loads by default:

  ```powershell
  $env:SECTOR_PULSE_MARKET = "IN"
  ```

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
| `NEWS_MAX_AGE_HOURS` | `48` | Ignore articles older than this |
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
   today. In the India profile, Realty and Media have no listed sector ETF.
2. Today-only news score >= `SIGNAL_MIN_NEWS` (0.15) over at least
   `SIGNAL_MIN_ARTICLES` (3) articles.
3. Intraday confirmation: day change >= `SIGNAL_MIN_INTRADAY_PCT` (0.2%) and
   the last hour is not falling (price confirming into the close).
4. Multi-day momentum >= `SIGNAL_MIN_MOMENTUM` (-0.2), **and** computed on at
   least `SIGNAL_MIN_MOMENTUM_WEIGHT` (80%) of the configured window weight.
   Momentum is measured on the tradeable ETF, not the sector index: yfinance
   served only 1 daily bar since 2026-07-20 for 9 of the 12 Nifty indices while
   the ETFs had 34-35, which had silently reduced those sectors to the 63d
   window alone. A missing score used to arrive as 0.0, and 0.0 >= -0.2, so a
   data outage made this gate PASS. It now fails closed.
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
- **Its universe is not your portfolio.** The ten tradeable tickers are sector
  ETFs; `targets.yaml` holds different instruments. Acting on a signal means
  buying something the planner will treat as an untracked holding.

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
> - Yahoo Finance intraday data **may be delayed** by several minutes; the
>   15:15 snapshot may not reflect the true last traded price.
> - The gate rule is **unvalidated**: it has no backtest behind it and earns
>   credibility only as `signals.csv` accumulates a real track record you can
>   evaluate yourself.
> - This is **decision support, not financial advice**. It is an educational
>   tool; treat every BUY line as a prompt to do your own research.
> - It **never places orders** - it only prints, logs to CSV, and writes a
>   JSON file for the dashboard.

## How the scoring works

Each headline is classified into one or more sectors via keyword matching, scored for sentiment with VADER (plus a finance lexicon overlay), and weighted by recency - an article's influence halves every `RECENCY_HALF_LIFE_HOURS`. Per sector, the recency-weighted average sentiment is damped when there are few articles, producing a news score in roughly [-1, 1]. In parallel, each sector's ETF gets a momentum score from tanh-squashed 5-day, 21-day, and 63-day returns, blended by window weights. The composite is `news_weight * news_score + (1 - news_weight) * momentum_score` (news-only when momentum data is unavailable), and sectors are ranked by composite, highest first.

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

Live prices come from yfinance (bare NSE symbols get a `.NS` suffix). If that is
unreachable the planner falls back to the price column in your CSV and says so in the
report. `--offline` skips the network entirely.

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
