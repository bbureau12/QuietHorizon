"""
Audio processing utilities for QuietHorizon Frontend

This module provides a thin wrapper around the shared preprocessing pipeline
with additional frontend-specific utilities like validation.
"""
import sys
from pathlib import Path

# Add parent directory to path to import quiet_horizon
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import librosa.display

# Import from shared audio preprocessing module
from quiet_horizon.audio import (
    audio_to_spectrogram,
    load_audio,
    create_mel_spectrogram,
    spectrogram_to_image,
)

# Add frontend to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# Re-export shared functions for backward compatibility
__all__ = [
    "load_audio",
    "create_mel_spectrogram", 
    "save_spectrogram_image",
    "process_audio_file",
    "validate_audio_file",
]


def save_spectrogram_image(mel_spec_db, output_size=None):
    """
    Convert mel-spectrogram to RGB image suitable for CNN input.
    
    Wrapper around shared spectrogram_to_image function.
    
    Args:
        mel_spec_db: Mel-spectrogram in dB scale
        output_size: Tuple (width, height), default from config
    
    Returns:
        numpy array of shape (height, width, 3)
    """
    if output_size is None:
        output_size = config.SPECTROGRAM_SIZE
    
    return spectrogram_to_image(mel_spec_db, output_size)


def process_audio_file(audio_file):
    """
    Complete pipeline: audio file -> spectrogram image suitable for CNN.
    
    Uses the shared preprocessing pipeline to ensure consistency with CLI.
    
    Args:
        audio_file: File path or file-like object
    
    Returns:
        dict with audio_data, sample_rate, mel_spectrogram, and spectrogram_image
    """
    # Use canonical preprocessing with metadata
    return audio_to_spectrogram(
        audio_file,
        target_sr=config.TARGET_SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        output_size=config.SPECTROGRAM_SIZE,
        return_metadata=True
    )


def validate_audio_file(filename, file_size=None, uploaded_file=None):
    """
    Validate audio file by filename, size, or Streamlit UploadedFile object.
    
    Args:
        filename: File name (str) or Streamlit UploadedFile object
        file_size: File size in bytes (optional, required if filename is str)
        uploaded_file: Streamlit UploadedFile object (deprecated, use filename)
    
    Returns:
        dict: {"valid": bool, "error": str or None}
    """
    # Handle legacy uploaded_file parameter
    if uploaded_file is not None:
        filename = uploaded_file.name
        file_size = uploaded_file.size
    
    # Handle case where filename is actually an UploadedFile object
    if hasattr(filename, 'name') and hasattr(filename, 'size'):
        file_size = filename.size
        filename = filename.name
    
    # Validate filename
    if not filename or filename.strip() == '':
        return {"valid": False, "error": "Empty filename"}
    
    # Check file extension
    if '.' not in filename:
        return {"valid": False, "error": "No file extension found"}
    
    file_extension = filename.split('.')[-1].lower()
    if file_extension not in config.SUPPORTED_AUDIO_FORMATS:
        return {"valid": False, "error": f"Unsupported format: .{file_extension}. Supported: {', '.join(config.SUPPORTED_AUDIO_FORMATS)}"}
    
    # Check file size if provided
    if file_size is not None:
        if file_size == 0:
            return {"valid": False, "error": "File is empty (0 bytes)"}
        
        file_size_mb = file_size / (1024 * 1024)
        max_size_mb = config.MAX_FILE_SIZE_MB
        
        if file_size_mb > max_size_mb:
            return {"valid": False, "error": f"File too large ({file_size_mb:.1f} MB). Maximum size: {max_size_mb} MB"}
    
    return {"valid": True, "error": None}
    
    return True, None
