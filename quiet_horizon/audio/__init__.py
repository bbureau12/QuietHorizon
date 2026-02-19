"""
Shared audio processing utilities for QuietHorizon.

This module provides the canonical implementation for audio preprocessing
used by both the CLI inference tool and the web frontend.
"""
from .preprocessing import (
    audio_to_spectrogram,
    load_audio,
    create_mel_spectrogram,
    spectrogram_to_image,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_MELS,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_OUTPUT_SIZE,
)

__all__ = [
    "audio_to_spectrogram",
    "load_audio",
    "create_mel_spectrogram",
    "spectrogram_to_image",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_N_MELS",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_OUTPUT_SIZE",
]
