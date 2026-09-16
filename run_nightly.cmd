@echo off
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
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" fetch_tail.py           >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" fetch_announcements.py 2 >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" publish_turnover.py     >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" -m bar_store --intervals 3minute >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" outcomes.py             >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" -c "import premarket; print('watchlist:', premarket.publish(), 'names')" >> run\nightly.log 2>&1
