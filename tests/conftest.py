"""
Pytest configuration and shared fixtures for QuietHorizon tests.
"""
import pytest
import numpy as np
from pathlib import Path


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
def sample_spectrogram():
    """Generate a sample spectrogram for testing."""
    return np.random.rand(128, 128, 3).astype(np.uint8)


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_model():
    """Mock TensorFlow model for testing without loading real model."""
    class MockModel:
        def predict(self, x, verbose=0):
            # Return random prediction
            return np.array([[0.5]])
    
    return MockModel()
