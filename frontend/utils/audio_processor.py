"""
Audio processing utilities for QuietHorizon
"""
import numpy as np
import librosa
import librosa.display
from PIL import Image
import io
import config


def load_audio(audio_file, target_sr=None):
    """
    Load audio file and convert to target sample rate.
    
    Args:
        audio_file: File path or file-like object
        target_sr: Target sample rate (default from config)
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    if target_sr is None:
        target_sr = config.TARGET_SAMPLE_RATE
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=target_sr)
        return y, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def create_mel_spectrogram(audio_data, sample_rate=None):
    """
    Create mel-spectrogram from audio data.
    
    Args:
        audio_data: Audio time series
        sample_rate: Sample rate of audio
    
    Returns:
        Mel-spectrogram as numpy array
    """
    if sample_rate is None:
        sample_rate = config.TARGET_SAMPLE_RATE
    
    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def save_spectrogram_image(mel_spec_db, output_size=None):
    """
    Convert mel-spectrogram to RGB image suitable for CNN input.
    
    Args:
        mel_spec_db: Mel-spectrogram in dB scale
        output_size: Tuple (width, height), default from config
    
    Returns:
        numpy array of shape (height, width, 3)
    """
    if output_size is None:
        output_size = config.SPECTROGRAM_SIZE
    
    # Normalize to 0-255 range
    spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                 (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(spec_norm)
    
    # Resize to target size
    img = img.resize(output_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB (3 channels)
    img_rgb = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img_rgb)
    
    return img_array


def process_audio_file(audio_file):
    """
    Complete pipeline: audio file -> spectrogram image suitable for CNN.
    
    Args:
        audio_file: File path or file-like object
    
    Returns:
        dict with audio_data, sample_rate, mel_spectrogram, and spectrogram_image
    """
    # Load audio
    audio_data, sr = load_audio(audio_file)
    
    # Create mel-spectrogram
    mel_spec = create_mel_spectrogram(audio_data, sr)
    
    # Convert to image
    spec_image = save_spectrogram_image(mel_spec)
    
    return {
        "audio_data": audio_data,
        "sample_rate": sr,
        "mel_spectrogram": mel_spec,
        "spectrogram_image": spec_image,
        "duration": len(audio_data) / sr
    }


def validate_audio_file(uploaded_file):
    """
    Validate uploaded audio file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f} MB). Maximum size: {config.MAX_FILE_SIZE_MB} MB"
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in config.SUPPORTED_AUDIO_FORMATS:
        return False, f"Unsupported format: .{file_extension}. Supported: {', '.join(config.SUPPORTED_AUDIO_FORMATS)}"
    
    return True, None
