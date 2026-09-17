"""Bar loading for the intraday scanner, on top of the Kite source.

This module was the seam where a real broker feed would replace yfinance,
and that has now happened: market_source fetches from Kite and hands back
one flat OHLCV frame per symbol. The claim that nothing downstream would
change held - no indicator, gate or level was touched by the swap.

Nothing here raises. A symbol that fails is logged and omitted, and the
report says how many actually returned data, so a half-empty scan cannot be
mistaken for a quiet market. The one thing that DOES now raise upstream is
the absence of a Kite session, and market_source raises rather than
returning empty precisely so it cannot be mistaken for a quiet market
either.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

import config
import indicators
import market_source
logger = logging.getLogger(__name__)


@dataclass
class BarSet:
    """Intraday and daily frames for every symbol that returned data."""

    intraday: dict[str, pd.DataFrame]
    daily: dict[str, pd.DataFrame]
    requested: int
    failed: list[str]
    # Where the intraday bars came from. Defaults to the download so any
    # caller that does not set it cannot accidentally claim to be live.
    source: str = "download"
    # How many symbols actually had bars built from ticks today, and how old
    # the newest of those bars is. Separate fields because `source` was
    # being asked to carry both and got them wrong: a live path covering 3
    # of 216 symbols, or running on a file hours stale, is not the same
    # thing as a healthy stream, and the UI has to be able to tell.
    live_symbols: int = 0
    live_age_seconds: float = float("nan")
    # Symbols that got intraday bars but no daily context. They are NOT in
    # `failed` - they scan, and every gate needing a previous close, a
    # pivot range or twenty-session turnover fails closed for them. Carried
    # separately so a silent failed-closed gate can be told apart from a
    # real refusal.
    daily_failed: list = field(default_factory=list)

    @property
    def covered(self) -> int:
        """Symbols with usable intraday bars."""
        return len(self.intraday)


def _from_store(symbols: list[str], start: date, end: date,
                interval: str) -> dict[str, pd.DataFrame]:
    """Bars the consolidated store can PROVE it holds, or {}.

    The nightly job folds every per-symbol cache file into one parquet per
    interval, and reading it is a file-layout win rather than a different
    answer - the store is folded FROM the same bars a download would
    return. Measured 2026-09-16 on 2,302 symbols: 3.6 seconds from the
    consolidated file against 33 from the same data as separate files, and
    against minutes from Kite.

    DELEGATES TO live_bars.from_store rather than reimplementing it, and
    that matters: the proof is the hard part. It refuses the store outright
    when its freshest session is more than STORE_MAX_STALE_DAYS behind the
    window's end, and drops any symbol whose own last bar does not reach
    that session.

    Without those two guards a frame ending yesterday is served as today's.
    setups.measure takes the LAST session present in the frame - it never
    compares it to the clock - so every reading, the opening range, rvol,
    ATR and the day's high and low, would be yesterday's, combined with a
    live minutes_left, and ranked into the table as a current setup with
    nothing anywhere reporting the shortfall. An earlier version of this
    function promised that proof in its docstring and did not implement it.
    """
    # KITE'S SPELLING, NOT THE CONFIG'S. config.SCAN_BAR_INTERVAL is "3m"
    # while the store is folded and named "3minute" by the nightly job, so
    # passing the config value straight through finds no file, returns {},
    # and every symbol silently falls through to the download path this
    # function exists to avoid.
    resolved = market_source.kite_interval(interval)
    try:
        import live_bars
    except Exception as exc:                      # optional dependency path
        logger.info("No consolidated store available (%s)", exc)
        return {}
    try:
        usable = live_bars.from_store(list(symbols), start, end, resolved)
    except Exception as exc:
        logger.warning("Consolidated store unreadable at %s: %s",
                       resolved, exc)
        return {}
    if usable:
        logger.info("Served %d/%d %s frames from the consolidated store",
                    len(usable), len(symbols), resolved)
    return usable


def _fetch(symbols: list[str], start: date, end: date, interval: str,
           may_download: bool = True) -> dict[str, pd.DataFrame]:
    """Bars per symbol from Kite, or {} rather than raising.

    `may_download` is the CALLER'S policy, passed in rather than read from
    config here. It exists for the Streamlit UI, where on 2026-09-17 a
    scope wider than the store answered its misses by fetching about 2,570
    symbols one at a time: 31,537 cache files, an unresponsive tab, and a
    machine out of CPU and memory, with the requests competing against the
    live feed for Kite's three-a-second budget.

    It defaults to True because most callers here are not the UI.
    scan_intraday and instrument_report are command-line tools whose whole
    job is to answer for a symbol, and silently refusing to fetch turns
    that into "no intraday bars for this symbol" - which
    instrument_report renders as a VERDICT rather than as an error.

    market_source returns one flat frame per symbol, so the MultiIndex
    unpacking the yfinance path needed is gone from this route entirely.
    """
    if not may_download:
        if symbols:
            logger.warning(
                "%d symbol(s) are not covered by the consolidated store at "
                "%s and this caller may not download, so they are reported "
                "uncovered. Run the nightly job to fold them in.",
                len(symbols), interval)
        return {}
    try:
        return market_source.bars(symbols, start, end, interval=interval)
    except market_source.NoSession as exc:
        logger.warning("%s", exc)
        return {}
    except Exception as exc:  # network and parse failures are heterogeneous
        logger.warning("Fetch failed for %d symbol(s) at %s: %s",
                       len(symbols), interval, exc)
        return {}


def replay_window(target: date,
                  lookback_days: int = config.SCAN_REPLAY_LOOKBACK_DAYS
                  ) -> tuple[date, date]:
    """The (start, end) span covering one past session plus its context.

    Both ends are inclusive, which differs from the yfinance route this
    replaced: Kite treats `to` as inclusive, so the end is the target date
    itself rather than the day after it.
    """
    start = target - timedelta(days=max(1, lookback_days))
    return start, target


def _lookback_days(spec: str, default: int) -> int:
    """Calendar days from a '10d' or '3mo' style span.

    Delegates so this and edge_lab cannot drift apart; see
    market_source.calendar_days for why "10d" is not ten days.
    """
    return market_source.calendar_days(spec, default)


def fetch_bars(tickers: list[str], batch_size: int = config.SCAN_BATCH_SIZE,
               target: "date | None" = None,
               may_download: "bool | None" = None) -> BarSet:
    """Fetch intraday and daily bars for every symbol from Kite.

    With `target` set, the window ends at that date instead of today, which
    is what makes a past-date replay possible at all.

    `batch_size` is retained for callers that still pass it but no longer
    batches anything: Kite is queried per instrument and paced centrally at
    its documented rate, so grouping symbols buys nothing. It is kept rather
    than removed to avoid breaking a caller for no gain.
    """
    unique = list(dict.fromkeys(t for t in tickers if t))
    if not unique:
        return BarSet({}, {}, 0, [])
    if may_download is None:
        may_download = bool(getattr(config, "SCAN_UI_MAY_DOWNLOAD", False))
    if target is not None:
        # A REPLAY MAY ALWAYS DOWNLOAD. There is no other source for a past
        # instant: the live feed holds today, and the consolidated store
        # ends at the last nightly fold. Safe now in a way it was not
        # before, because the scope gate bounds a replay at 300 symbols
        # rather than the ~2,570 that took the machine down, and because
        # _from_store refuses partial coverage rather than handing back a
        # frame from the wrong session.
        may_download = True
    end = target or date.today()
    fine_days = _lookback_days(config.SCAN_BAR_LOOKBACK, 10)
    coarse_days = _lookback_days(config.SCAN_DAILY_LOOKBACK, 92)
    if target is not None:
        # One definition of the replay window, not two. This used to compute
        # its own span inline while replay_window sat unused beside it under
        # a different rule.
        fine_start, end = replay_window(target, max(
            fine_days, config.SCAN_REPLAY_LOOKBACK_DAYS))
        fine_days = (end - fine_start).days

    intraday = {}
    source = "download"
    streamed, age = 0, float("nan")
    if target is None:
        # Live scan: prefer bars the feed has already built from ticks.
        # Costs no network and the newest bar is seconds old rather than
        # minutes. A replay skips this entirely - there is no live feed for
        # a past instant.
        intraday, streamed, age = _live_intraday(unique)
        if intraday:
            source = "live feed"
    if not intraday:
        # THE STORE BEFORE THE NETWORK. Whatever it cannot prove goes to
        # _fetch, which downloads only when the UI is allowed to.
        fine_start = end - timedelta(days=fine_days)
        intraday = _from_store(unique, fine_start, end,
                               config.SCAN_BAR_INTERVAL)
        from_store_count = len(intraday)
        absent = [s for s in unique if s not in intraday]
        downloaded = {}
        if absent:
            downloaded = _fetch(absent, fine_start, end,
                                config.SCAN_BAR_INTERVAL,
                                may_download=may_download)
            intraday.update(downloaded)
        # LABELLED ON WHAT ARRIVED, not on what was asked for. `absent` is
        # computed before the fetch, so keying the label on it called a
        # scan "download" when the fetch was refused and every bar came
        # from the store - and the UI then warned that bars "can be
        # several minutes old" when nothing had been downloaded at all.
        if downloaded:
            source = "download"
        elif from_store_count:
            source = "store"
    # ENDS THE DAY BEFORE `end`, so the span is stable for the whole
    # session and caches once instead of re-fetching every symbol every
    # time the today-ending cache goes stale.
    #
    # Nothing is lost. The bar for `end` is DISCARDED twice over - once
    # below in the cutoff filter and again in setups.measure - because a
    # partial daily bar leaks the future into prev_close and the CPR. So
    # the old span asked Kite for a bar, paid the request, and threw it
    # away; the only thing it bought was a cache key that expired every
    # CACHE_TODAY_MAX_AGE_SECONDS. Measured on 2026-09-15: 210 F&O
    # symbols re-fetching their daily bars mid-session, which is what made
    # a scan of an entirely streamed universe take minutes.
    #
    # live_bars.history_window does the same thing for intraday, and says
    # so in the same words: a window that ends yesterday caches all day.
    daily_end = end - timedelta(days=1)
    daily_start = daily_end - timedelta(days=coarse_days)
    # The same store-first rule. This span is the one that cost the most:
    # it ends yesterday and so is entirely settled history, yet a rolling
    # window meant the cache key changed every morning and all of it was
    # refetched on the first page load of each trading day - 222 spans on
    # 2026-09-17 before the scan proper had begun.
    daily = _from_store(unique, daily_start, daily_end, "day")
    missing_daily = [s for s in unique if s not in daily]
    if missing_daily:
        daily.update(_fetch(missing_daily, daily_start, daily_end, "day",
                            may_download=may_download))
    logger.info("Intraday bars for %d/%d symbols (%s)",
                len(intraday), len(unique), source)
    failed = [t for t in unique if t not in intraday]
    if failed:
        logger.warning("No intraday bars for %d symbol(s): %s",
                       len(failed), ", ".join(failed[:10]))
    # DAILY GAPS ARE REPORTED SEPARATELY. A symbol with intraday bars but
    # no daily context still reaches setups.measure, where prev_close,
    # turnover_20d and the pivot range all come back None and every gate
    # that needs them fails closed. That used to be rare because daily was
    # always downloaded; with the store-first rule it is routine, and a
    # silent failed-closed gate is indistinguishable from a real refusal.
    daily_failed = [t for t in unique if t not in daily]
    if daily_failed:
        logger.warning("No daily context for %d symbol(s): %s",
                       len(daily_failed), ", ".join(daily_failed[:10]))
    return BarSet(intraday=intraday, daily=daily,
                  requested=len(unique), failed=failed, source=source,
                  live_symbols=streamed, live_age_seconds=age,
                  daily_failed=daily_failed)


def auto_refresh_interval(*, enabled: bool, requested: int,
                          replaying: bool, want_options: bool,
                          in_trading_hours: bool, feed_running: bool,
                          feed_age: float,
                          last_scan_seconds: float = 0.0) -> tuple:
    """(seconds to re-scan at, note) for the intraday scan's auto-refresh.

    `None` for the interval means do not refresh, and `note` says why. Both
    are returned together because a toggle the user switched on that then
    quietly does nothing is worse than having no toggle: the reason has to
    reach the screen.

    Pure - every input is passed in - so the rules can be tested without a
    browser, a feed or a clock.
    """
    if not enabled:
        return None, ""
    if replaying:
        return None, ("a replay is a fixed instant in the past, so there is "
                      "nothing for a timer to refresh")
    if want_options:
        return None, ("picking option contracts costs one NSE chain request "
                      "per setup, and a timer would repeat that indefinitely")
    if not in_trading_hours:
        return None, ("the market is closed, so no refresh can change a bar")
    if not feed_running:
        return None, ("the live feed is not running, so each refresh would "
                      "re-download every symbol at 3 requests a second")
    if not (feed_age == feed_age) or feed_age > config.SCAN_LIVE_MAX_AGE_SECONDS:
        return None, ("the newest live bar is too old - the feed looks "
                      "stopped, so a refresh would fall back to downloading")
    # Never re-scan faster than a scan takes to finish, or the reruns queue
    # up behind each other and the page is permanently mid-scan. Doubling
    # the measured time leaves the CPU idle at least half the interval.
    floor = int(last_scan_seconds * 2) if last_scan_seconds else 0
    if floor > requested:
        return floor, (f"raised to {floor}s: the last scan took "
                       f"{last_scan_seconds:.0f}s, and refreshing faster "
                       f"than that would leave it permanently re-scanning")
    return requested, ""


def _live_intraday(symbols: list[str]) -> tuple:
    """(frames, symbols that streamed, age of newest bar) or ({}, 0, nan).

    Returns {} rather than a partial answer when the feed has written
    nothing for today, so the caller downloads instead of silently scanning
    yesterday's close as though it were now.

    The two extra return values exist because `combined()` merges cached
    prior sessions and the startup seed with the stream, and yields a frame
    for every symbol having ANY of the three. Its emptiness therefore says
    nothing about whether the feed is alive, which is precisely what the
    caller needs to know.
    """
    empty = ({}, 0, float("nan"))
    try:
        import live_bars
        import market_source
    except Exception as exc:                      # optional dependency path
        logger.info("Live bars unavailable (%s); downloading", exc)
        return empty
    state = live_bars.status()
    # AN UNREACHABLE STORE IS NOT AN ABSENT ONE. status() reports a storage
    # fault rather than raising, because it renders on every page load - so
    # this gate, which is what the scan actually depends on, has to read
    # that field. Without it an expired token or a denied read arrives here
    # as "no live bars", and the scan answers by downloading the whole
    # universe at three requests a second while reporting nothing wrong.
    if state.get("error"):
        logger.error("Live bar store unreachable (%s). NOT falling back to "
                     "a download: fix the store rather than re-fetching "
                     "what the feed has already published.", state["error"])
        return empty
    if not state.get("present") or not state.get("bars"):
        logger.info("No live bars written today; downloading instead")
        return empty
    age = state.get("age_seconds", float("nan"))
    # A stale file is worse than no file: it looks live, so nothing warns,
    # and the levels are computed against a price that stopped moving when
    # the feed died. `age == age` rules out NaN, where the age is unknowable
    # and the download is the safe answer too.
    if not (age == age) or age > config.SCAN_LIVE_MAX_AGE_SECONDS:
        logger.warning(
            "Live bars are %.0fs old (limit %ds) - the feed looks stopped, "
            "so downloading instead", age, config.SCAN_LIVE_MAX_AGE_SECONDS)
        return empty
    tokens = {}
    for symbol in symbols:
        token = market_source.token_for(symbol)
        if token:
            tokens[symbol] = token
    try:
        combined = live_bars.combined(symbols, tokens)
    except Exception as exc:
        logger.warning("Live bar assembly failed (%s); downloading", exc)
        return empty
    live_today = live_bars.load_today(tokens)
    if not live_today:
        return empty
    # Only symbols the caller ASKED for and that actually streamed. The
    # store holds whatever the feed was subscribed to, which is not the
    # same universe as this scan.
    streamed = len([s for s in symbols if s in live_today])
    if not streamed:
        logger.info("Live store holds no requested symbol; downloading")
        return empty
    logger.info("Live bars: %d symbols assembled, %d of %d requested "
                "streaming, newest %.0fs old",
                len(combined), streamed, len(symbols), age)
    return combined, streamed, age


def _carry_source(original: BarSet, intraday: dict, daily: dict,
                  failed: "list[str] | None" = None) -> BarSet:
    """Rebuild a BarSet without losing where its bars came from.

     must be passed by any caller that changes which symbols have
    usable bars. truncate does: a symbol whose every bar sits after the
    cutoff has no bars left and has to be reported as failed, or a replay
    silently shrinks its own universe.
    """
    return BarSet(intraday=intraday, daily=daily,
                  requested=original.requested,
                  failed=list(original.failed if failed is None else failed),
                  source=original.source,
                  live_symbols=original.live_symbols,
                  live_age_seconds=original.live_age_seconds)


def truncate(bars: BarSet, cutoff) -> BarSet:
    """A point-in-time view of a BarSet: nothing after `cutoff` survives.

    This is what makes a replay honest. Passing an earlier clock to the gates
    while leaving the frames whole would compute the "10am signal" from the
    last bar of the day, a full-session VWAP and an ATR that has seen the
    afternoon - every reading contaminated by the future it is supposed to
    predict. Daily bars are cut on date so a replay of an earlier session
    cannot see later sessions' closes either.

    The intraday comparison is STRICTLY less than the cutoff, and that
    matters more than it looks. A bar is stamped at its START, so the bar
    labelled 10:00 covers 10:00 to 10:05 and its close is the 10:05 price.
    Keeping it made every "10:00" entry a 10:05 entry: an independent audit
    measured 16 actionable setups against 12 at a true 10:00 cutoff, with 8
    of the 16 existing only because of that single bar and 4 genuine ones
    deleted. Excluding it leaves the 09:55 bar last, whose close IS the
    price at 10:00.
    """
    fine: dict[str, pd.DataFrame] = {}
    for ticker, frame in bars.intraday.items():
        keep = frame.loc[[ts for ts in frame.index if ts < cutoff]]
        if not keep.empty:
            fine[ticker] = keep
    coarse: dict[str, pd.DataFrame] = {}
    for ticker, frame in bars.daily.items():
        # Strictly before the cutoff DATE. The replay date's own daily bar
        # is complete by the time a replay runs, so keeping it and relying
        # on each consumer to drop it again is how a leak got in.
        keep = frame.loc[[ts for ts in frame.index if ts.date() < cutoff.date()]]
        if not keep.empty:
            coarse[ticker] = keep
    return _carry_source(
        bars, fine, coarse,
        failed=[t for t in bars.failed]
               + [t for t in bars.intraday if t not in fine])


def benchmark_change_pct(bars: BarSet, ticker: str = config.SCAN_BENCHMARK
                         ) -> "float | None":
    """Today's percent change for the relative-strength benchmark.

    None when the benchmark did not download, which makes every
    relative-strength gate fail closed rather than silently comparing every
    symbol against zero and calling a falling market strength.
    """
    frame = bars.intraday.get(ticker)
    daily = bars.daily.get(ticker)
    if frame is None or daily is None:
        return None
    today = indicators.session_bars(frame)
    if today is None:
        return None
    closes = today["Close"].dropna()
    prior = daily.dropna(subset=["Close"])
    try:
        # Same clock anchor as setups.measure, for the same reason.
        cutoff_day = closes.index[-1].date()
        prior = prior.loc[[ts for ts in prior.index
                           if ts.date() < cutoff_day]]
    except AttributeError:
        logger.warning("Benchmark daily index is not timestamped; refusing to "
                       "compare against a partial bar")
        return None
    if closes.empty or prior.empty:
        return None
    return indicators.percent_change(float(closes.iloc[-1]),
                                     float(prior["Close"].iloc[-1]))
