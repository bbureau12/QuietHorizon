# QuietHorizon Frontend Architecture

## Directory Structure

```
frontend/
├── app.py                      # Main Streamlit application entry point
├── config.py                   # Central configuration (paths, thresholds, UI settings)
├── requirements.txt            # Python dependencies
├── README.md                   # Frontend documentation
├── .gitignore
├── run.bat                     # Windows startup script
├── run.sh                      # Unix/Mac startup script
│
├── utils/                      # Business logic & processing
│   ├── __init__.py
│   ├── model_loader.py        # TensorFlow model loading with Streamlit caching
│   ├── audio_processor.py     # Audio → Spectrogram pipeline (librosa)
│   └── visualization.py       # Matplotlib plotting functions
│
└── components/                 # UI components (Streamlit widgets)
    ├── __init__.py
    ├── upload.py              # File upload interface
    ├── results.py             # Results display with visualizations
    └── batch.py               # Batch processing UI and logic
```

## Component Responsibilities

### `app.py` - Main Application
- Streamlit page configuration
- Sidebar setup (model selection, settings)
- Tab management (Single File, Batch, Info)
- Orchestrates components and utilities

### `config.py` - Configuration
- Project paths (model directory, HF repo)
- Audio settings (sample rate, spectrogram size)
- Classification thresholds
- UI settings (max file size, supported formats)
- Color scheme for visualizations

### `utils/` - Processing Layer

#### `model_loader.py`
- `load_model()`: Cached TensorFlow model loading
- `download_model_from_hf()`: HuggingFace Hub integration
- `predict_from_spectrogram()`: Inference wrapper

#### `audio_processor.py`
- `load_audio()`: Load and resample audio files
- `create_mel_spectrogram()`: Generate mel-spectrogram
- `save_spectrogram_image()`: Convert to CNN-ready RGB image
- `process_audio_file()`: Complete pipeline
- `validate_audio_file()`: Input validation

#### `visualization.py`
- `plot_waveform()`: Time-domain audio visualization
- `plot_spectrogram()`: Mel-spectrogram heatmap
- `plot_prediction_gauge()`: Probability distribution bar
- `create_results_summary()`: HTML results card
- `plot_batch_results()`: Batch statistics charts

### `components/` - UI Layer

#### `upload.py`
- File uploader widget
- File info display
- Format and size validation messages

#### `results.py`
- Results summary card
- Probability gauge
- Audio player
- Tabbed visualizations (waveform, spectrogram, model input)
- Expandable info section

#### `batch.py`
- Multi-file uploader
- Progress tracking
- Results table with pandas DataFrame
- CSV export
- Summary statistics and charts

## Data Flow

```
User Upload
    ↓
[upload.py] File Validation
    ↓
[audio_processor.py] Audio → Spectrogram → RGB Image
    ↓
[model_loader.py] CNN Prediction
    ↓
[results.py / batch.py] Display Results
    ↓
[visualization.py] Generate Charts
```

## Key Design Principles

1. **Separation of Concerns**
   - UI components don't handle business logic
   - Processing utilities are UI-agnostic
   - Configuration is centralized

2. **Modularity**
   - Each module has a single responsibility
   - Easy to test and maintain
   - New features can be added without major refactoring

3. **Caching**
   - Model loaded once and cached (`@st.cache_resource`)
   - Expensive operations are minimized
   - Fast subsequent runs

4. **Error Handling**
   - Validation at multiple levels
   - User-friendly error messages
   - Graceful degradation

5. **Reusability**
   - Utility functions are generic
   - Components can be reused across tabs
   - Consistent API design

## Dependencies

### Core
- `streamlit`: Web framework
- `tensorflow`: Model inference
- `librosa`: Audio processing
- `matplotlib`: Visualizations

### Supporting
- `pandas`: Data handling
- `numpy`: Numerical operations
- `Pillow`: Image processing
- `huggingface-hub`: Model downloading
- `scipy`: Audio I/O

## Performance Considerations

- **First Load**: 2-5s (model download + loading)
- **Subsequent Loads**: <1s (cached model)
- **Single File Processing**: 1-3s
- **Batch Processing**: ~2s per file (parallelizable in future)

## Future Enhancements

- [ ] Real-time audio recording
- [ ] Parallel batch processing
- [ ] Audio trimming/segmentation
- [ ] Advanced filtering options
- [ ] Export annotated spectrograms
- [ ] API mode (REST endpoint)
- [ ] Multi-language support
- [ ] Dark mode theme
