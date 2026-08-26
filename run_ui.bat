@echo off
setlocal
cd /d "%~dp0code"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Activate your BACH environment first, then run this file again.
  pause
  exit /b 1
)

echo Starting BACH Studio...
python ui.py

if errorlevel 1 (
  echo.
  echo BACH Studio exited with an error.
  pause
)
endlocal
