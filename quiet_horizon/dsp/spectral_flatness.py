# dsp/flatness.py
from dataclasses import dataclass
import numpy as np
import librosa

@dataclass
class FlatnessThresholds:
    mean_flatness: float = 0.45
    high_frame_frac: float = 0.3

class FlatnessFilter:
    def __init__(self, thresholds: FlatnessThresholds | None = None):
        self.thresholds = thresholds or FlatnessThresholds()

    def compute_flatness(self, S: np.ndarray) -> dict:
        """
        S: magnitude or power spectrogram (freq_bins x frames)
        """
        # librosa expects shape (freqs, frames)
        flat = librosa.feature.spectral_flatness(S=S)[0]  # (frames,)

        mean_flatness = float(np.mean(flat))
        high_flat_frames = float(np.mean(flat > 0.6))

        return {
            "flatness_per_frame": flat,
            "mean_flatness": mean_flatness,
            "high_flat_frame_frac": high_flat_frames,
        }

    def analyze(self, features: dict) -> dict:
        S = features["S"]
        stats = self.compute_flatness(S)

        is_noise_like = (
            stats["mean_flatness"] > self.thresholds.mean_flatness
            or stats["high_flat_frame_frac"] > self.thresholds.high_flat_frame_frac
        )

        return {
            "mean_flatness": stats["mean_flatness"],
            "high_flat_frame_frac": stats["high_flat_frame_frac"],
            "noise_like": bool(is_noise_like),
        }
