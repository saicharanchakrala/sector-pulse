@echo off
setlocal
REM Nightly maintenance for sector-pulse. Runs AFTER the close on purpose:
REM every Kite request here competes with the live feed for the same
REM 3-a-second budget, and during a session that budget is what keeps the
REM scan quick.
REM
REM   fetch_tail  - daily bars for listed equities the store has never seen,
REM                 then folds the per-symbol cache into the consolidated
REM                 store so it stops drifting stale.
REM   fetch_announcements - two years of NSE filings, resumable.
REM   prune_cache - deletes intraday cache files another file already
REM                 covers, and any past every reader's window. Runs
REM                 BEFORE the fold on purpose: bar_store.rebuild folds
REM                 the consolidated store from the per-symbol files
REM                 ALONE and never reads the store it replaces, so
REM                 pruning first is what makes the two agree. It never
REM                 touches "day" - bars_day.parquet reaches back to about
REM                 2001 and is rebuilt the same way, so pruning those
REM                 would delete twenty-five years of daily bars.
REM                 Measured 2026-09-17: 6,133 of 19,515 intraday files
REM                 were spans another file already contained, 102 MB.
REM   refresh_intraday - fetches fresh intraday bars so the 3-minute store
REM                 can advance. Nothing on this machine was doing it:
REM                 the cache was filled as a SIDE EFFECT of the UI's own
REM                 scans until SCAN_UI_MAY_DOWNLOAD went False on
REM                 2026-09-18, and the only other caller of prewarm runs
REM                 in a container whose disk is discarded. Measured
REM                 2026-09-21, the store's newest session was 2026-09-17
REM                 - it did not even hold Friday - so prewarm refused it
REM                 and the feed refetched from Kite on every cold start.
REM                 Same failure the daily store had, in the other store.
REM                 Runs BEFORE the fold, which consolidates what it gets.
REM   bar_store --intervals 3minute - folds the per-symbol intraday cache
REM                 into one consolidated file. NOT optional for speed:
REM                 a full-universe scan reads prior sessions for ~2,300
REM                 symbols, and measured 2026-09-16 that is 3.6 seconds
REM                 from the consolidated file against 33 from the same
REM                 data as separate per-symbol files. live_bars.prewarm
REM                 REFUSES a store more than one session behind, so if
REM                 this step does not run the scan is correct and slow
REM                 rather than fast and wrong.
REM                 --publish uploads it, which is what lets a CONTAINER
REM                 prewarm from it. Without that the feed refetches 17
REM                 days for ~2,500 symbols from Kite on every cold
REM                 start - measured 2026-09-18 at 836 seconds, so three
REM                 deploys that day cost about an hour of the session.
REM                 The daily store is NOT published: 191 MB, and the
REM                 container reads daily_context.parquet instead.
REM   daily_context - the twenty-five daily bars per symbol that the
REM                 intraday scan actually reads. About 1.4 MB against the
REM                 559 MB store it comes from, so a container can compute
REM                 prev_close, the pivot range and turnover without one.
REM                 Verified drop-in: 216 symbols, every reading identical.
REM   fetch_scan_log - downloads the FEED's own scan logs. Needed because
REM                 scan_intraday.append_log is called only by the
REM                 command-line scanner: the UI evaluates setups without
REM                 logging them and the feed publishes a parquet table
REM                 instead, so scan_log.csv sat at 8 rows from 9 Sep
REM                 while the feed scanned 2,485 symbols every 45 seconds
REM                 for days. Runs BEFORE outcomes, which resolves
REM                 whatever it finds.
REM   outcomes    - resolves every setup in scan_log.csv against what the
REM                 market actually did, turning the log from a record of
REM                 intentions into a track record. Idempotent, and only
REM                 touches rows whose session has already closed.
REM   publish_turnover - the ranking the CONTAINERISED feed picks its
REM                 universe from. It has no 559 MB daily store to rank on,
REM                 so without this it streams the ~216 F&O underlyings
REM                 instead of ~2,500. Runs after fetch_tail because the
REM                 ranking is only as current as the store it reads, and
REM                 is a no-op when SECTOR_PULSE_S3_BUCKET is unset.
REM
REM EVERY STEP IS CHECKED. It used to run all nine unconditionally and
REM report nothing, which failed twice in three days in exactly the way
REM that is hardest to notice:
REM   2026-09-16..19  fetch_tail only ever backfilled symbols the store had
REM                   never seen, so the store fell four days behind and
REM                   prev_close became a different date per symbol while
REM                   the log said "daily store now holds 2,521 symbols".
REM   2026-09-20      fetch_tail died on an expired Kite token and the job
REM                   carried on through eight more steps, finishing with
REM                   no indication that the one step that mattered had
REM                   aborted.
REM Steps still all RUN rather than stopping at the first failure - the
REM 3-minute fold is worth having even when the token is dead - but a
REM failure is banner-marked in the log, listed in a summary at the end,
REM and returned as the exit code.
cd /d "C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse"

REM THE OBJECT STORE VARIABLES, read from the same file deploy\env.ps1
REM reads. Without them publish_turnover.py and daily_context.py refuse
REM outright and this job silently accomplishes most of its nine steps -
REM leaving the feed to stream ~216 names instead of ~2,500 and the scan
REM with no previous close, pivot range or turnover. cmd cannot
REM dot-source a PowerShell script, so both parse one plain KEY=VALUE
REM file rather than each carrying a copy that can drift.
if not exist "deploy\env.vars" (
    echo MISSING deploy\env.vars - the publishing steps will be skipped>>run\nightly.log
) else (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("deploy\env.vars") do (
        if not "%%a"=="" set "%%a=%%b"
    )
)
set "PY=C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe"
set "LOG=run\nightly.log"
set /a FAILS=0
set "BROKEN="

REM A DATED BANNER PER RUN. Runs were concatenated with nothing between
REM them, so the tail of a finished run and the tail of a running one look
REM identical - which is how a stale run's final line was read as the
REM current one's on 2026-09-20.
>>%LOG% echo.
>>%LOG% echo ================================================================
>>%LOG% echo NIGHTLY RUN STARTED %DATE% %TIME%
>>%LOG% echo ================================================================

"%PY%" fetch_tail.py >> %LOG% 2>&1
call :check fetch_tail
"%PY%" fetch_announcements.py 2 >> %LOG% 2>&1
call :check fetch_announcements
"%PY%" publish_turnover.py >> %LOG% 2>&1
call :check publish_turnover
"%PY%" prune_cache.py --apply >> %LOG% 2>&1
call :check prune_cache
"%PY%" refresh_intraday.py >> %LOG% 2>&1
call :check refresh_intraday
"%PY%" -m bar_store --intervals 3minute --publish 3minute >> %LOG% 2>&1
call :check bar_store_3minute
"%PY%" fetch_scan_log.py >> %LOG% 2>&1
call :check fetch_scan_log
"%PY%" outcomes.py >> %LOG% 2>&1
call :check outcomes
"%PY%" -c "import premarket; print('watchlist:', premarket.publish(), 'names')" >> %LOG% 2>&1
call :check premarket
"%PY%" daily_context.py >> %LOG% 2>&1
call :check daily_context

>>%LOG% echo.
if %FAILS%==0 goto :allgood
>>%LOG% echo ================================================================
>>%LOG% echo NIGHTLY FINISHED WITH %FAILS% FAILED STEP^(S^):%BROKEN%
>>%LOG% echo Anything downstream of a failed step worked from whatever was
>>%LOG% echo already on disk, so its output is not wrong so much as UNCHANGED.
>>%LOG% echo Fix the cause and re-run the whole job - every step is idempotent.
>>%LOG% echo ================================================================
echo.
echo NIGHTLY FINISHED WITH %FAILS% FAILED STEP^(S^):%BROKEN%
echo See run\nightly.log for the banner-marked failures.
exit /b %FAILS%

:allgood
>>%LOG% echo NIGHTLY COMPLETE - all 10 steps succeeded %DATE% %TIME%
echo.
echo Nightly complete - all 10 steps succeeded.
exit /b 0

REM ---------------------------------------------------------------------
REM Records a step's exit code. ERRORLEVEL IS CAPTURED FIRST, because
REM `set /a` below resets it and the code would be lost before it was read.
:check
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" goto :eof
set /a FAILS+=1
set "BROKEN=%BROKEN% %1"
>>%LOG% echo.
>>%LOG% echo ***************************************************************
>>%LOG% echo ***** STEP FAILED: %1 - exit code %CODE%
>>%LOG% echo ***************************************************************
>>%LOG% echo.
goto :eof
