# dsp/engine.py
from dataclasses import dataclass
import numpy as np
import librosa

@dataclass
class DSPConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512

class DSPEngine:
    def __init__(self, config: DSPConfig | None = None):
        self.config = config or DSPConfig()

    def extract_all(self, path: str) -> dict:
        y, sr = librosa.load(path, sr=self.config.sr, mono=True)

        stft = librosa.stft(
            y,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        S = np.abs(stft) ** 2
        freqs = librosa.fft_frequencies(
            sr=sr,
            n_fft=self.config.n_fft,
        )

        return {
            "path": path,
            "y": y,
            "sr": sr,
            "stft": stft,
            "S": S,
            "freqs": freqs,
        }
