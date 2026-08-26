@echo off
setlocal
cd /d "%~dp0code"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Install/activate Python 3.10+ and run this file again.
  pause
  exit /b 1
)

echo ============================================================
echo BACH Studio - dependency and model setup
echo ============================================================
echo.

echo [1/3] Ensuring Python dependencies are installed...
python -m pip install -r requirements.txt
if errorlevel 1 goto :setup_failed

echo.
echo [2/3] Ensuring the complete generation runtime is installed...
python setup_runtime.py
if errorlevel 1 goto :setup_failed

echo.
echo [3/3] Starting BACH Studio...
python ui.py
if errorlevel 1 (
  echo.
  echo BACH Studio exited with an error.
  pause
  exit /b 1
)

exit /b 0

:setup_failed
echo.
echo BACH Studio setup failed. The UI was not started because the
echo generation runtime did not pass setup/verification.
echo Re-run run_ui.bat to resume interrupted Hugging Face downloads.
pause
exit /b 1
