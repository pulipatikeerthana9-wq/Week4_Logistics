@echo off
setlocal
cd /d "%~dp0"
set "PYVER="
for %%V in (3.13 3.12 3.11) do (
  if not defined PYVER (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYVER=%%V"
  )
)
if not defined PYVER (
  echo Please install 64-bit Python 3.13 from python.org with the Python launcher.
  echo This package supports Python 3.11-3.13, not the default Python 3.14.
  goto fail
)
if not exist ".venv\Scripts\python.exe" (
  py -%PYVER% -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,13)"
if errorlevel 1 (
  echo Existing .venv uses an unsupported Python version. Send this error for help.
  goto fail
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto fail
".venv\Scripts\python.exe" run_week4.py
if errorlevel 1 goto fail
echo SUCCESS - Save your genuine terminal output for submission evidence.
pause
exit /b 0
:fail
echo FAILED - Save the error above and ask for help.
pause
exit /b 1
