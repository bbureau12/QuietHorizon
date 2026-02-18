# 🚀 QuickStart Guide - QuietHorizon Frontend

Get the QuietHorizon web interface up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~500 MB free disk space (for dependencies)
- Internet connection (for first-time model download)

## Installation

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web framework)
- TensorFlow (deep learning)
- Librosa (audio processing)
- Matplotlib (visualizations)
- And other supporting libraries

_Installation takes 2-5 minutes depending on your connection._

## Running the App

### Using Startup Scripts (Easiest)

**Windows:**
```bash
run.bat
```

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

### Or Manually

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## First Time Setup

On first run, the app will:
1. Check for a local model in `../models/`
2. If not found, download from HuggingFace (~4 MB)
3. Cache the model for future use

**This only happens once!** Subsequent launches are instant.

## Using the Interface

### Single File Classification

1. Click the **"Single File"** tab
2. Click **"Browse files"** or drag & drop an audio file
3. Click **"Classify Audio"** button
4. View results:
   - Prediction (Nature or Anthropogenic)
   - Confidence score
   - Waveform and spectrogram
   - Audio playback

### Batch Processing

1. Click the **"Batch Processing"** tab
2. Upload multiple files (up to 100)
3. Click **"Process All Files"**
4. View summary statistics and charts
5. Download results as CSV

### Supported Formats

- WAV (.wav)
- MP3 (.mp3)
- OGG (.ogg)
- FLAC (.flac)
- M4A (.m4a)

**Max file size:** 50 MB per file

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Model download failed"
- Check internet connection
- Ensure HuggingFace is accessible
- Try loading from local path in sidebar

### "Audio processing error"
- Verify audio file is not corrupted
- Check file format is supported
- Ensure file size is under 50 MB

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Clear cache
In the app: **Hamburger menu (☰) → Settings → Clear cache**

## Tips & Tricks

### Performance
- First classification takes longer (model loading)
- Subsequent classifications are fast (~1-3 seconds)
- Use batch mode for multiple files

### Best Results
- Use clear, uncompressed audio when possible
- Recordings should be at least 1-2 seconds long
- Mixed sounds (nature + anthro) may show intermediate probabilities

### Customization
Edit `config.py` to change:
- Classification threshold (default: 0.5)
- High confidence threshold (default: 0.85)
- Max file size
- Color scheme
- Audio processing parameters

## Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🏗️ Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand the code structure
- 🔧 Modify `config.py` for custom settings
- 🌲 Visit the main [QuietHorizon README](../quiet_horizon/README.md) for project background

## Getting Help

**Common Issues:**
1. Dependencies not installing → Update pip: `pip install --upgrade pip`
2. TensorFlow errors → Ensure Python 3.8-3.11 (TF compatibility)
3. Audio won't play → Try different browser (Chrome/Firefox recommended)

**Still stuck?** Check the browser console (F12) for JavaScript errors.

## Stopping the App

Press **Ctrl + C** in the terminal where Streamlit is running.

---

**Enjoy using QuietHorizon! 🌲🎵**
