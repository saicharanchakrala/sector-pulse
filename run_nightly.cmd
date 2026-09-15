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
cd /d "C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse"
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" fetch_tail.py           >> run\nightly.log 2>&1
"C:\Users\Sai Charan Chakrala\PycharmProjects\sector-pulse\.venv\Scripts\python.exe" fetch_announcements.py 2 >> run\nightly.log 2>&1
