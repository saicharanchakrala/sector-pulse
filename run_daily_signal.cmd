@echo off
rem Launcher for the Sector Pulse daily 3:15 PM signal (used by Windows Task Scheduler).
cd /d "%~dp0"
".venv\Scripts\python.exe" daily_signal.py --market IN >> signal_log.txt 2>&1
