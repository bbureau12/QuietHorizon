@echo off
REM QuietHorizon Frontend Startup Script for Windows

echo Starting QuietHorizon Frontend...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo No virtual environment found.
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

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
