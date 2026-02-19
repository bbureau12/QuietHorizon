"""
Configuration settings for QuietHorizon Frontend
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "quiet_horizon_cnn.keras"

# Hugging Face model repository
HF_REPO_ID = "bbureau12/QuietHorizon"
HF_MODEL_FILENAME = "quiet_horizon_cnn.keras"

# Audio processing settings
TARGET_SAMPLE_RATE = 22050
SPECTROGRAM_SIZE = (128, 128)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Classification thresholds
ANTHRO_THRESHOLD = 0.5  # Probability threshold for anthropogenic classification
HIGH_CONFIDENCE_THRESHOLD = 0.85  # For highlighting high-confidence predictions

# UI Settings
MAX_FILE_SIZE_MB = 50
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "ogg", "flac", "m4a"]
SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS  # Alias for backward compatibility
BATCH_PROCESSING_LIMIT = 100

# Color scheme
COLOR_NATURE = "#4CAF50"  # Green
COLOR_ANTHRO = "#F44336"  # Red
COLOR_NEUTRAL = "#9E9E9E"  # Grey
