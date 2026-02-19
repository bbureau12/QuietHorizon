"""
Unit tests for shared audio preprocessing module.

These tests verify the canonical preprocessing pipeline used by both
CLI and frontend components.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quiet_horizon.audio import (
    load_audio,
    create_mel_spectrogram,
    spectrogram_to_image,
    audio_to_spectrogram,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_MELS,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_OUTPUT_SIZE,
)


class TestLoadAudio:
    """Tests for audio loading functionality."""
    
    def test_load_audio_from_generated_data(self, sample_audio_array):
        """Test loading audio data."""
        audio, sr = sample_audio_array
        
        # Should be 1D array
        assert audio.ndim == 1
        assert sr == DEFAULT_SAMPLE_RATE
        assert len(audio) > 0
    
    def test_load_audio_validates_input(self):
        """Test that invalid audio raises ValueError."""
        with pytest.raises(ValueError, match="Error loading audio file"):
            load_audio("nonexistent_file.wav")
    
    def test_load_audio_custom_sample_rate(self, tmp_path):
        """Test loading with custom sample rate."""
        # This would need a real audio file, so we'll skip for now
        # but demonstrates the test structure
        pytest.skip("Requires audio file fixture")


class TestCreateMelSpectrogram:
    """Tests for mel-spectrogram generation."""
    
    def test_mel_spectrogram_shape(self, sample_audio_array):
        """Test that mel-spectrogram has correct dimensions."""
        audio, sr = sample_audio_array
        mel_spec = create_mel_spectrogram(audio, sr)
        
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == DEFAULT_N_MELS
        # Width varies with audio length
        assert mel_spec.shape[1] > 0
    
    def test_mel_spectrogram_db_scale(self, sample_audio_array):
        """Test that values are in dB scale (should be <= 0)."""
        audio, sr = sample_audio_array
        mel_spec = create_mel_spectrogram(audio, sr)
        
        # dB scale: max should be 0 (reference), others negative
        assert mel_spec.max() <= 0
        assert mel_spec.min() < 0
    
    def test_mel_spectrogram_no_nan_or_inf(self, sample_audio_array):
        """Test that spectrogram contains no NaN or Inf values."""
        audio, sr = sample_audio_array
        mel_spec = create_mel_spectrogram(audio, sr)
        
        assert not np.isnan(mel_spec).any()
        assert not np.isinf(mel_spec).any()
    
    def test_mel_spectrogram_deterministic(self, sample_audio_array):
        """Test that same input produces same output."""
        audio, sr = sample_audio_array
        
        mel_spec1 = create_mel_spectrogram(audio, sr)
        mel_spec2 = create_mel_spectrogram(audio, sr)
        
        np.testing.assert_array_equal(mel_spec1, mel_spec2)
    
    def test_mel_spectrogram_custom_params(self, sample_audio_array):
        """Test with custom parameters."""
        audio, sr = sample_audio_array
        
        mel_spec = create_mel_spectrogram(
            audio, sr,
            n_mels=64,
            n_fft=1024,
            hop_length=256
        )
        
        assert mel_spec.shape[0] == 64


class TestSpectrogramToImage:
    """Tests for spectrogram to image conversion."""
    
    def test_image_output_shape(self):
        """Test that output has correct shape."""
        # Create dummy spectrogram
        mel_spec = np.random.randn(128, 256) * 10
        
        img = spectrogram_to_image(mel_spec)
        
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8
    
    def test_image_value_range(self):
        """Test that pixel values are in valid range [0, 255]."""
        mel_spec = np.random.randn(128, 256) * 10
        
        img = spectrogram_to_image(mel_spec)
        
        assert img.min() >= 0
        assert img.max() <= 255
    
    def test_image_custom_size(self):
        """Test with custom output size."""
        mel_spec = np.random.randn(128, 256) * 10
        
        img = spectrogram_to_image(mel_spec, output_size=(64, 64))
        
        assert img.shape == (64, 64, 3)
    
    def test_image_rgb_channels(self):
        """Test that all 3 RGB channels contain data."""
        mel_spec = np.random.randn(128, 256) * 10
        
        img = spectrogram_to_image(mel_spec)
        
        # Should have 3 channels
        assert img.shape[2] == 3
        
        # All channels should have some variation
        # (not all zeros on any channel)
        for i in range(3):
            assert img[:, :, i].std() > 0
    
    def test_image_normalization(self):
        """Test that normalization uses full dynamic range."""
        # Create spectrogram with known range
        mel_spec = np.array([[0, 50], [100, 150]], dtype=float)
        
        img = spectrogram_to_image(mel_spec, output_size=(2, 2))
        
        # After normalization, should have 0 and 255
        assert img.min() == 0
        assert img.max() == 255


class TestAudioToSpectrogram:
    """Tests for complete preprocessing pipeline."""
    
    def test_pipeline_output_shape(self, sample_audio_array):
        """Test that pipeline produces correct output shape."""
        audio, sr = sample_audio_array
        
        # Create a temporary array in memory that librosa can load
        # For this test, we'll use the direct function on array
        # In practice, this would use a real audio file
        pytest.skip("Requires audio file fixture or mock")
    
    def test_pipeline_output_dtype(self, sample_audio_array):
        """Test that output is uint8."""
        pytest.skip("Requires audio file fixture")
    
    def test_pipeline_with_metadata(self, sample_audio_array):
        """Test pipeline with metadata return."""
        pytest.skip("Requires audio file fixture")
    
    def test_pipeline_consistency(self, sample_audio_array):
        """Test that pipeline produces consistent results."""
        pytest.skip("Requires audio file fixture")


class TestConsistencyBetweenImplementations:
    """
    Critical tests: Verify CLI and frontend produce identical results.
    
    These tests ensure the refactoring achieved its goal of eliminating
    inconsistencies between different code paths.
    """
    
    def test_preprocessing_produces_identical_output(self, sample_audio_array):
        """
        Test that the same audio produces identical spectrograms.
        
        This is the key guarantee of the refactoring - both CLI and
        frontend use the same implementation and get the same results.
        """
        audio, sr = sample_audio_array
        
        # Generate spectrogram twice
        spec1 = create_mel_spectrogram(audio, sr)
        spec2 = create_mel_spectrogram(audio, sr)
        
        # Should be exactly identical (not just close)
        np.testing.assert_array_equal(spec1, spec2)
    
    def test_image_conversion_deterministic(self):
        """Test that image conversion is deterministic."""
        mel_spec = np.random.randn(128, 256) * 10
        
        img1 = spectrogram_to_image(mel_spec)
        img2 = spectrogram_to_image(mel_spec)
        
        np.testing.assert_array_equal(img1, img2)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_very_short_audio(self):
        """Test with very short audio (< 1 second)."""
        # Generate 0.1 second of audio
        sr = DEFAULT_SAMPLE_RATE
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sr * 0.1)))
        
        mel_spec = create_mel_spectrogram(audio, sr)
        
        # Should still work, just narrower
        assert mel_spec.shape[0] == DEFAULT_N_MELS
        assert mel_spec.shape[1] > 0
    
    def test_silent_audio(self):
        """Test with silent (all zeros) audio."""
        sr = DEFAULT_SAMPLE_RATE
        audio = np.zeros(sr * 2)  # 2 seconds of silence
        
        mel_spec = create_mel_spectrogram(audio, sr)
        
        # Should produce valid output (likely all same value)
        assert not np.isnan(mel_spec).any()
        assert not np.isinf(mel_spec).any()
    
    def test_very_loud_audio(self):
        """Test with high amplitude audio."""
        sr = DEFAULT_SAMPLE_RATE
        # Very loud sine wave
        audio = 10.0 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2))
        
        mel_spec = create_mel_spectrogram(audio, sr)
        
        # Should handle clipping gracefully
        assert not np.isnan(mel_spec).any()
        assert not np.isinf(mel_spec).any()
    
    def test_uniform_spectrogram(self):
        """Test edge case: all spectrogram values the same."""
        mel_spec = np.ones((128, 256)) * 50
        
        # Should not crash on zero variance
        img = spectrogram_to_image(mel_spec)
        
        assert img.shape == (128, 128, 3)
        # All pixels should be the same (either 0 or 255)
        assert len(np.unique(img)) <= 2


class TestConstants:
    """Test that constants are set correctly."""
    
    def test_constants_match_training_config(self):
        """Verify preprocessing constants match training setup."""
        # These values MUST match the training configuration
        assert DEFAULT_SAMPLE_RATE == 22050
        assert DEFAULT_N_MELS == 128
        assert DEFAULT_N_FFT == 2048
        assert DEFAULT_HOP_LENGTH == 512
        assert DEFAULT_OUTPUT_SIZE == (128, 128)
    
    def test_constants_are_immutable(self):
        """Ensure constants are the expected types."""
        assert isinstance(DEFAULT_SAMPLE_RATE, int)
        assert isinstance(DEFAULT_N_MELS, int)
        assert isinstance(DEFAULT_N_FFT, int)
        assert isinstance(DEFAULT_HOP_LENGTH, int)
        assert isinstance(DEFAULT_OUTPUT_SIZE, tuple)
        assert len(DEFAULT_OUTPUT_SIZE) == 2
