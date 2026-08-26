@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo BACH Studio - complete setup and launch
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [0/5] Creating an isolated BACH Studio Python environment...

  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.10 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
      py -3.10 -m venv .venv
      goto :venv_created
    )
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
      py -3.11 -m venv .venv
      goto :venv_created
    )
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
      py -3.12 -m venv .venv
      goto :venv_created
    )
    py -3.13 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
      py -3.13 -m venv .venv
      goto :venv_created
    )
  )

  where python >nul 2>nul
  if errorlevel 1 (
    echo No compatible Python installation was found.
    goto :setup_failed
  )

  python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12),(3,13)] else 1)" >nul 2>nul
  if errorlevel 1 (
    echo BACH Studio requires Python 3.10 through 3.13 for the pinned audio stack.
    goto :setup_failed
  )
  python -m venv .venv
)

:venv_created
if not exist ".venv\Scripts\python.exe" goto :setup_failed
set "BACH_PY=%CD%\.venv\Scripts\python.exe"

echo [1/5] Updating pip in the isolated environment...
"%BACH_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :setup_failed

echo.
echo [2/5] Installing and verifying CUDA-enabled PyTorch...
"%BACH_PY%" code\setup_cuda.py
if errorlevel 1 goto :setup_failed

echo.
echo [3/5] Installing remaining BACH Studio Python dependencies...
"%BACH_PY%" -m pip install -r code\requirements.txt
if errorlevel 1 goto :setup_failed

echo.
echo [4/5] Installing/verifying codec and generation models...
"%BACH_PY%" code\setup_runtime.py
if errorlevel 1 goto :setup_failed

echo.
echo [5/5] Starting BACH Studio...
cd /d "%~dp0code"
"%BACH_PY%" ui.py
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
echo generation environment did not pass setup/verification.
echo Re-run run_ui.bat to resume interrupted installs or model downloads.
pause
exit /b 1
