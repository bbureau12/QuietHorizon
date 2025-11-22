from dataclasses import dataclass
import librosa
import numpy as np

@dataclass
class FrequencyThresholds:
    low_frac: float = 0.30
    high_frac: float = 0.25

class FrequencyFilter:
    def __init__(self, thresholds: FrequencyThresholds | None = None):
        self.thresholds = thresholds or FrequencyThresholds()

    def load_audio(self, path: str, sr: int = 22050):
        y, sr = librosa.load(path, sr=sr, mono=True)
        return y, sr

    def compute_band_energies(self, y: np.ndarray, sr: int):
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        S = np.abs(stft) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        low_mask = freqs < 200
        high_mask = freqs > 10000

        total_energy = S.sum() or 1e-12
        low_frac  = S[low_mask].sum()  / total_energy
        high_frac = S[high_mask].sum() / total_energy

        return {
            "low_frac": low_frac,
            "high_frac": high_frac,
            "total_energy": float(total_energy),
        }

    def analyze(self, path: str):
        y, sr = self.load_audio(path)
        bands = self.compute_band_energies(y, sr)
        
        unnatural = (
            bands["low_frac"]  > self.thresholds.low_frac or
            bands["high_frac"] > self.thresholds.high_frac
        )

        return {
            "path": path,
            "bands": bands,
            "unnatural": bool(unnatural),
        }
