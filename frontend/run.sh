#!/bin/bash
# QuietHorizon Frontend Startup Script for Unix/Linux/Mac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting QuietHorizon Frontend..."
echo ""

# Prefer root project virtual environment for model compatibility.
if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    echo "Activating root virtual environment..."
    source "$ROOT_DIR/venv/bin/activate"
else
    echo "Root virtual environment not found."
    echo "Falling back to frontend virtual environment..."
    if [ ! -d "$SCRIPT_DIR/venv" ]; then
        echo "Creating frontend virtual environment..."
        python3 -m venv "$SCRIPT_DIR/venv"
        echo ""
    fi
    source "$SCRIPT_DIR/venv/bin/activate"
fi

cd "$SCRIPT_DIR"

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
