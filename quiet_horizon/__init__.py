"""
QuietHorizon - Environmental Audio Classifier

Detecting Human Noise Intrusion in Natural Soundscapes
"""

__version__ = "0.1.0"

# Import key functions for public API
from quiet_horizon.audio import audio_to_spectrogram

__all__ = [
    "audio_to_spectrogram",
    "__version__",
]
