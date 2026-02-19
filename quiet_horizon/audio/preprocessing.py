"""
Canonical audio preprocessing for QuietHorizon.

This module provides the single source of truth for converting audio files
to spectrogram images for CNN inference. Used by both CLI and frontend.
"""
import numpy as np
import librosa
from PIL import Image
from typing import Tuple, Union, Dict, Optional, Any

# Constants matching training setup
# These MUST match the values used during model training
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_OUTPUT_SIZE = (128, 128)


def load_audio(
    audio_file: Union[str, Any],
    target_sr: int = DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to target sample rate.
    
    Args:
        audio_file: File path (str) or file-like object
        target_sr: Target sample rate in Hz
    
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        ValueError: If audio file cannot be loaded
    """
    try:
        y, sr = librosa.load(audio_file, sr=target_sr)
        return y, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def create_mel_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """
    Create mel-spectrogram from audio data.
    
    Args:
        audio_data: Audio time series (1D numpy array)
        sample_rate: Sample rate of audio in Hz
        n_mels: Number of mel frequency bands
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
    
    Returns:
        Mel-spectrogram in dB scale (2D numpy array)
    """
    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def spectrogram_to_image(
    mel_spec_db: np.ndarray,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE
) -> np.ndarray:
    """
    Convert mel-spectrogram to RGB image suitable for CNN input.
    
    Args:
        mel_spec_db: Mel-spectrogram in dB scale (2D array)
        output_size: Target size as (width, height)
    
    Returns:
        RGB image as numpy array with shape (height, width, 3) and dtype uint8
    """
    # Normalize to 0-255 range
    spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                 (mel_spec_db.max() - mel_spec_db.min()) * 255)
    spec_norm = spec_norm.astype(np.uint8)
    
    # Convert to PIL Image for resizing
    img = Image.fromarray(spec_norm)
    
    # Resize to target size using high-quality LANCZOS resampling
    img = img.resize(output_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB (3 channels)
    img_rgb = img.convert('RGB')
    
    # Convert back to numpy array
    img_array = np.array(img_rgb)
    
    return img_array


def audio_to_spectrogram(
    audio_file: Union[str, Any],
    target_sr: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    return_metadata: bool = False
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Complete pipeline: Convert audio file to CNN-ready spectrogram image.
    
    This is the canonical implementation used by both CLI and frontend.
    All preprocessing steps are performed in a consistent manner.
    
    Args:
        audio_file: Path to audio file or file-like object
        target_sr: Target sample rate in Hz
        n_mels: Number of mel frequency bands
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        output_size: Output image size as (width, height)
        return_metadata: If True, return dict with audio data and metadata
    
    Returns:
        If return_metadata=False: 
            numpy array with shape (height, width, 3) and dtype uint8
        If return_metadata=True: 
            dict containing:
                - spectrogram_image: The RGB spectrogram image
                - audio_data: The loaded audio waveform
                - sample_rate: Sample rate used
                - mel_spectrogram: The mel-spectrogram in dB scale
                - duration: Duration of audio in seconds
    
    Raises:
        ValueError: If audio file cannot be loaded or processed
    
    Example:
        >>> # Simple usage
        >>> img = audio_to_spectrogram("path/to/audio.wav")
        >>> img.shape
        (128, 128, 3)
        
        >>> # With metadata
        >>> result = audio_to_spectrogram("path/to/audio.wav", return_metadata=True)
        >>> result['duration']
        2.5  # seconds
    """
    # 1. Load audio
    audio_data, sr = load_audio(audio_file, target_sr)
    
    # 2. Create mel-spectrogram
    mel_spec_db = create_mel_spectrogram(
        audio_data, sr, n_mels, n_fft, hop_length
    )
    
    # 3. Convert to RGB image
    spec_image = spectrogram_to_image(mel_spec_db, output_size)
    
    if return_metadata:
        return {
            "spectrogram_image": spec_image,
            "audio_data": audio_data,
            "sample_rate": sr,
            "mel_spectrogram": mel_spec_db,
            "duration": len(audio_data) / sr
        }
    
    return spec_image
