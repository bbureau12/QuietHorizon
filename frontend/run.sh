#!/bin/bash
# QuietHorizon Frontend Startup Script for Unix/Linux/Mac

echo "Starting QuietHorizon Frontend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "No virtual environment found."
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "===================================="
echo "  QuietHorizon Frontend"
echo "===================================="
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your browser."
echo "Press Ctrl+C to stop the server."
echo ""

# Run Streamlit
streamlit run app.py
