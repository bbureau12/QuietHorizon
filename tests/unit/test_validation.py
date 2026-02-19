"""
Unit tests for audio file validation.

Tests validation logic for file formats, sizes, durations, and quality.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.utils.audio_processor import validate_audio_file
import config


class TestFileFormatValidation:
    """Tests for audio file format validation."""
    
    def test_valid_wav_extension(self):
        """Test that WAV files are accepted."""
        result = validate_audio_file("test_audio.wav", file_size=1024*1024)
        assert result["valid"] is True
    
    def test_valid_mp3_extension(self):
        """Test that MP3 files are accepted."""
        result = validate_audio_file("test_audio.mp3", file_size=1024*1024)
        assert result["valid"] is True
    
    def test_valid_flac_extension(self):
        """Test that FLAC files are accepted."""
        result = validate_audio_file("test_audio.flac", file_size=1024*1024)
        assert result["valid"] is True
    
    def test_valid_ogg_extension(self):
        """Test that OGG files are accepted."""
        result = validate_audio_file("test_audio.ogg", file_size=1024*1024)
        assert result["valid"] is True
    
    def test_invalid_extension(self):
        """Test that invalid file extensions are rejected."""
        result = validate_audio_file("test_audio.txt", file_size=1024)
        assert result["valid"] is False
        assert "format" in result["error"].lower()
    
    def test_case_insensitive_extension(self):
        """Test that extensions are case-insensitive."""
        result = validate_audio_file("test_audio.WAV", file_size=1024*1024)
        assert result["valid"] is True
        
        result = validate_audio_file("test_audio.Mp3", file_size=1024*1024)
        assert result["valid"] is True


class TestFileSizeValidation:
    """Tests for file size validation."""
    
    def test_file_size_within_limit(self):
        """Test that files within size limit are accepted."""
        # 10 MB file
        result = validate_audio_file("test.wav", file_size=10*1024*1024)
        assert result["valid"] is True
    
    def test_file_size_at_limit(self):
        """Test file at exact size limit."""
        # Exactly at max size
        max_size = config.MAX_FILE_SIZE_MB * 1024 * 1024
        result = validate_audio_file("test.wav", file_size=max_size)
        assert result["valid"] is True
    
    def test_file_size_exceeds_limit(self):
        """Test that files exceeding size limit are rejected."""
        # Over max size
        max_size = config.MAX_FILE_SIZE_MB * 1024 * 1024
        result = validate_audio_file("test.wav", file_size=max_size + 1024)
        assert result["valid"] is False
        assert "size" in result["error"].lower()
    
    def test_very_small_file(self):
        """Test that very small files are handled."""
        # 1 KB file
        result = validate_audio_file("test.wav", file_size=1024)
        # Might be valid or invalid depending on implementation
        # Just ensure it doesn't crash
        assert "valid" in result
    
    def test_zero_size_file(self):
        """Test that zero-size files are rejected."""
        result = validate_audio_file("test.wav", file_size=0)
        assert result["valid"] is False


class TestDurationValidation:
    """Tests for audio duration validation."""
    
    def test_duration_within_limit(self):
        """Test that audio within duration limit is accepted."""
        # This would require actual audio data
        pytest.skip("Requires audio file fixture")
    
    def test_duration_exceeds_limit(self):
        """Test that audio exceeding duration limit is rejected."""
        pytest.skip("Requires audio file fixture")
    
    def test_very_short_duration(self):
        """Test handling of very short audio (< 1 second)."""
        pytest.skip("Requires audio file fixture")


class TestContentValidation:
    """Tests for audio content validation."""
    
    def test_silent_audio_detection(self):
        """Test detection of silent/empty audio."""
        pytest.skip("Requires audio generation utilities")
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted audio files."""
        pytest.skip("Requires corrupted audio fixtures")
    
    def test_mono_vs_stereo(self):
        """Test handling of mono and stereo audio."""
        # Both should be accepted (librosa converts to mono)
        pytest.skip("Requires audio fixtures")


class TestValidationErrorMessages:
    """Tests for validation error messages."""
    
    def test_error_message_for_invalid_format(self):
        """Test that error messages are helpful."""
        result = validate_audio_file("test.xyz", file_size=1024)
        assert result["valid"] is False
        assert result["error"] is not None
        assert len(result["error"]) > 0
    
    def test_error_message_for_oversized_file(self):
        """Test error message for file too large."""
        result = validate_audio_file("test.wav", file_size=500*1024*1024)
        assert result["valid"] is False
        assert "size" in result["error"].lower()
    
    def test_multiple_validation_issues(self):
        """Test that multiple issues are reported clearly."""
        # Invalid format AND too large
        result = validate_audio_file("test.xyz", file_size=500*1024*1024)
        assert result["valid"] is False
        # Should report at least one issue
        assert result["error"] is not None


class TestEdgeCases:
    """Tests for edge cases in validation."""
    
    def test_filename_with_multiple_dots(self):
        """Test filename with multiple dots."""
        result = validate_audio_file("my.test.audio.wav", file_size=1024*1024)
        # Should use last extension
        assert result["valid"] is True
    
    def test_filename_no_extension(self):
        """Test filename without extension."""
        result = validate_audio_file("audiofile", file_size=1024*1024)
        assert result["valid"] is False
    
    def test_empty_filename(self):
        """Test empty filename."""
        result = validate_audio_file("", file_size=1024)
        assert result["valid"] is False
    
    def test_filename_only_extension(self):
        """Test filename that is only an extension."""
        result = validate_audio_file(".wav", file_size=1024*1024)
        # Might be valid or invalid depending on OS
        assert "valid" in result


class TestBatchValidation:
    """Tests for validating multiple files."""
    
    def test_validate_multiple_files(self):
        """Test validating a batch of files."""
        files = [
            ("valid1.wav", 1024*1024),
            ("valid2.mp3", 2*1024*1024),
            ("invalid.txt", 1024),
            ("toolarge.wav", 500*1024*1024),
        ]
        
        results = [validate_audio_file(name, size) for name, size in files]
        
        # First two should be valid
        assert results[0]["valid"] is True
        assert results[1]["valid"] is True
        
        # Last two should be invalid
        assert results[2]["valid"] is False
        assert results[3]["valid"] is False
    
    def test_all_valid_batch(self):
        """Test batch where all files are valid."""
        files = [
            ("audio1.wav", 1024*1024),
            ("audio2.mp3", 2*1024*1024),
            ("audio3.flac", 3*1024*1024),
        ]
        
        results = [validate_audio_file(name, size) for name, size in files]
        
        assert all(r["valid"] for r in results)
    
    def test_all_invalid_batch(self):
        """Test batch where all files are invalid."""
        files = [
            ("audio1.txt", 1024),
            ("audio2.xyz", 2*1024),
            ("audio3.doc", 3*1024),
        ]
        
        results = [validate_audio_file(name, size) for name, size in files]
        
        assert not any(r["valid"] for r in results)


class TestConfigurationRespect:
    """Tests that validation respects configuration."""
    
    def test_uses_config_max_file_size(self):
        """Test that MAX_FILE_SIZE_MB from config is used."""
        # File at config limit should be valid
        max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
        result = validate_audio_file("test.wav", file_size=max_bytes)
        assert result["valid"] is True
        
        # File over config limit should be invalid
        result = validate_audio_file("test.wav", file_size=max_bytes + 1)
        assert result["valid"] is False
    
    def test_uses_config_supported_formats(self):
        """Test that SUPPORTED_FORMATS from config is used."""
        # Check each supported format
        for fmt in config.SUPPORTED_FORMATS:
            result = validate_audio_file(f"test.{fmt}", file_size=1024*1024)
            assert result["valid"] is True
