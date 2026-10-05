@echo off
rem Double-click to start VideoScout in your browser. Close this window to stop it.
cd /d "%~dp0"
if not exist ".tmp" mkdir ".tmp"
rem Keep temporary files (uploads in progress) on this drive instead of C:
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"
".venv\Scripts\python.exe" -m videoscout.web
pause
