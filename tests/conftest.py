"""
Pytest configuration and shared fixtures for QuietHorizon tests.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_audio_array():
    """Generate a simple sine wave for testing."""
    sr = 22050
    duration = 2.0  # seconds
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    return audio, sr


@pytest.fixture
def sample_short_audio():
    """Generate a short audio clip (0.5 seconds)."""
    sr = 22050
    duration = 0.5
    frequency = 880  # Hz
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    return audio, sr


@pytest.fixture
def sample_silent_audio():
    """Generate silent audio (all zeros)."""
    sr = 22050
    duration = 2.0
    audio = np.zeros(int(sr * duration))
    return audio, sr


@pytest.fixture
def sample_spectrogram():
    """Generate a sample spectrogram for testing."""
    return np.random.rand(128, 128, 3).astype(np.uint8)


@pytest.fixture
def sample_mel_spectrogram():
    """Generate a sample mel-spectrogram in dB scale."""
    # Realistic mel-spectrogram: negative dB values
    mel_spec = np.random.randn(128, 256) * 20 - 40
    return mel_spec


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_model():
    """Mock TensorFlow model for testing without loading real model."""
    class MockModel:
        def predict(self, x, verbose=0):
            # Return random prediction (nature probability)
            return np.array([[0.5]])
    
    return MockModel()


@pytest.fixture
def mock_nature_model():
    """Mock model that always predicts nature."""
    class MockNatureModel:
        def predict(self, x, verbose=0):
            return np.array([[0.95]])  # High nature probability
    
    return MockNatureModel()


@pytest.fixture
def mock_anthro_model():
    """Mock model that always predicts anthropogenic."""
    class MockAnthroModel:
        def predict(self, x, verbose=0):
            return np.array([[0.05]])  # Low nature probability (high anthro)
    
    return MockAnthroModel()


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Create temporary directory for audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def real_nature_audio():
    """Path to real nature audio file (Northern Cardinal)."""
    audio_path = Path(__file__).parent.parent / "quiet_horizon" / "test_data" / "240404__itinerantmonk108__northern-cardinal-closeup.wav"
    if audio_path.exists():
        return audio_path
    pytest.skip(f"Nature audio file not found: {audio_path}")


@pytest.fixture
def real_anthro_audio():
    """Path to real anthropogenic audio file (Heavy Traffic)."""
    audio_path = Path(__file__).parent.parent / "quiet_horizon" / "test_data" / "691513__ania635__heavy_traffic_03.wav"
    if audio_path.exists():
        return audio_path
    pytest.skip(f"Anthro audio file not found: {audio_path}")


@pytest.fixture
def test_audio_files(real_nature_audio, real_anthro_audio):
    """Dictionary of test audio files."""
    return {
        "nature": real_nature_audio,
        "anthro": real_anthro_audio
    }


@pytest.fixture
def sample_valid_filenames():
    """List of valid audio filenames."""
    return [
        "test.wav",
        "audio.mp3",
        "sound.flac",
        "recording.ogg",
        "UPPERCASE.WAV",
        "mixed.Mp3",
    ]


@pytest.fixture
def sample_invalid_filenames():
    """List of invalid filenames."""
    return [
        "test.txt",
        "audio.doc",
        "sound.xyz",
        "noextension",
        ".wav",
        "",
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require trained model"
    )
    config.addinivalue_line(
        "markers", "requires_audio: marks tests that require audio files"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )

