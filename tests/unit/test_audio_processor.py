"""
Unit tests for frontend audio processing utilities.

These tests verify audio loading, spectrogram generation, and validation.
"""
import pytest
import numpy as np
from frontend.utils.audio_processor import (
    create_mel_spectrogram,
    save_spectrogram_image,
)


class TestMelSpectrogram:
    """Tests for mel-spectrogram generation."""
    
    def test_create_mel_spectrogram_shape(self, sample_audio_array):
        """Test that mel-spectrogram has correct shape."""
        audio, sr = sample_audio_array
        mel_spec = create_mel_spectrogram(audio, sr)
        
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == 128  # n_mels
        
    def test_create_mel_spectrogram_values(self, sample_audio_array):
        """Test that mel-spectrogram values are in dB scale."""
        audio, sr = sample_audio_array
        mel_spec = create_mel_spectrogram(audio, sr)
        
        # dB values should be negative or zero
        assert mel_spec.max() <= 0
        assert not np.isnan(mel_spec).any()
        assert not np.isinf(mel_spec).any()


class TestSpectrogramImage:
    """Tests for spectrogram to image conversion."""
    
    def test_save_spectrogram_image_shape(self):
        """Test that output image has correct shape."""
        # Create dummy mel-spectrogram
        mel_spec = np.random.randn(128, 256) * 10
        
        img = save_spectrogram_image(mel_spec, output_size=(128, 128))
        
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8
        
    def test_save_spectrogram_image_range(self):
        """Test that image values are in valid range."""
        mel_spec = np.random.randn(128, 256) * 10
        
        img = save_spectrogram_image(mel_spec)
        
        assert img.min() >= 0
        assert img.max() <= 255


# TODO: Add tests for:
# - load_audio with different file formats
# - process_audio_file end-to-end
# - validate_audio_file with various inputs
# - Error handling for corrupted files
# - Edge cases (empty audio, very short audio, etc.)
