# QuietHorizon Frontend

A Streamlit web application for the QuietHorizon environmental audio classifier.

## Features

- 🎵 **Single File Classification**: Upload and classify individual audio files
- 📦 **Batch Processing**: Process multiple files at once with summary statistics
- 📊 **Rich Visualizations**: Waveforms, spectrograms, and probability gauges
- 🔊 **Audio Playback**: Listen to your audio directly in the browser
- 📥 **Export Results**: Download batch processing results as CSV

## Quick Start

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
frontend/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── model_loader.py      # Model loading and caching
│   ├── audio_processor.py   # Audio processing pipeline
│   └── visualization.py     # Plotting and visualization
└── components/              # UI components
    ├── __init__.py
    ├── upload.py           # File upload component
    ├── results.py          # Results display
    └── batch.py            # Batch processing interface
```

## Configuration

Edit `config.py` to customize:

- Model paths and Hugging Face repository
- Audio processing parameters (sample rate, spectrogram size)
- Classification thresholds
- UI settings (max file size, supported formats)
- Color scheme

## Usage

### Single File Classification

1. Navigate to the "Single File" tab
2. Upload an audio file (WAV, MP3, OGG, FLAC, or M4A)
3. Click "Classify Audio"
4. View results including:
   - Prediction (Nature or Anthropogenic)
   - Confidence level and probabilities
   - Audio waveform and spectrogram visualizations
   - Playback controls

### Batch Processing

1. Navigate to the "Batch Processing" tab
2. Upload multiple audio files (up to 100)
3. Click "Process All Files"
4. View summary statistics and visualizations
5. Download results as CSV

## Model Loading

The app automatically:
- Checks for a local model in `../models/`
- Downloads from Hugging Face if not found
- Caches the model for fast subsequent loads

You can also specify a custom model path in the sidebar.

## Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- OGG (`.ogg`)
- FLAC (`.flac`)
- M4A (`.m4a`)

Maximum file size: 50 MB (configurable in `config.py`)

## Performance

- Model loading: ~2-5 seconds (first time)
- Single file processing: ~1-3 seconds
- Batch processing: ~1-3 seconds per file

## Troubleshooting

### Model Not Loading
- Ensure you have internet connection for Hugging Face download
- Check that the model repository exists: https://huggingface.co/bbureau12/QuietHorizon
- Try specifying a local model path in the sidebar

### Audio Processing Errors
- Verify the audio file is not corrupted
- Check file size is under the limit
- Ensure the format is supported

### Display Issues
- Clear Streamlit cache: Settings → Clear Cache
- Refresh the browser page
- Check browser console for JavaScript errors

## Development

### Adding New Features

1. **New Utility Function**: Add to appropriate module in `utils/`
2. **New UI Component**: Create new file in `components/`
3. **Configuration**: Add settings to `config.py`
4. **Main App**: Update `app.py` to integrate new features

### Code Organization

- **Separation of Concerns**: UI logic separated from business logic
- **Modular Design**: Each component is self-contained
- **Caching**: Model and expensive operations are cached
- **Type Hints**: Use type hints for better code clarity
- **Documentation**: Docstrings for all functions

## License

MIT License (same as parent project)
