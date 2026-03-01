@echo off
REM QuietHorizon Frontend Startup Script for Windows

set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..

echo Starting QuietHorizon Frontend...
echo.

REM Prefer root project virtual environment for model compatibility.
if exist "%ROOT_DIR%\venv\Scripts\activate.bat" (
    echo Activating root virtual environment...
    call "%ROOT_DIR%\venv\Scripts\activate.bat"
) else (
    echo Root virtual environment not found.
    echo Falling back to frontend virtual environment...
    if not exist "%SCRIPT_DIR%venv\Scripts\activate.bat" (
        echo Creating frontend virtual environment...
        python -m venv "%SCRIPT_DIR%venv"
    )
    call "%SCRIPT_DIR%venv\Scripts\activate.bat"
    echo.
)

cd /d "%SCRIPT_DIR%"

REM Install/update dependencies
echo Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo ====================================
echo  QuietHorizon Frontend
echo ====================================
echo.
echo Starting Streamlit application...
echo The app will open in your browser.
echo Press Ctrl+C to stop the server.
echo.

REM Run Streamlit
streamlit run app.py
