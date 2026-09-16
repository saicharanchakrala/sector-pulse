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
REM   bar_store --intervals 3minute - folds the per-symbol intraday cache
REM                 into one consolidated file. NOT optional for speed:
REM                 a full-universe scan reads prior sessions for ~2,300
REM                 symbols, and measured 2026-09-16 that is 3.6 seconds
REM                 from the consolidated file against 33 from the same
REM                 data as separate per-symbol files. live_bars.prewarm
REM                 REFUSES a store more than one session behind, so if
REM                 this step does not run the scan is correct and slow
REM                 rather than fast and wrong.
REM   daily_context - the twenty-five daily bars per symbol that the
REM                 intraday scan actually reads. About 1.4 MB against the
REM                 559 MB store it comes from, so a container can compute
REM                 prev_close, the pivot range and turnover without one.
REM                 Verified drop-in: 216 symbols, every reading identical.
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
cd /d "C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse"

REM THE OBJECT STORE VARIABLES, read from the same file deploy\env.ps1
REM reads. Without them publish_turnover.py and daily_context.py refuse
REM outright and this job silently accomplishes five of its seven steps -
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
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" fetch_tail.py           >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" fetch_announcements.py 2 >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" publish_turnover.py     >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" -m bar_store --intervals 3minute >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" outcomes.py             >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" -c "import premarket; print('watchlist:', premarket.publish(), 'names')" >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" daily_context.py     >> run\nightly.log 2>&1
