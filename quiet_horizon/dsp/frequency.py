# dsp/frequency.py
from dataclasses import dataclass
import numpy as np

@dataclass
class FrequencyThresholds:
    low_frac: float = 0.30
    high_frac: float = 0.25

class FrequencyFilter:
    def __init__(self, thresholds: FrequencyThresholds | None = None):
        self.thresholds = thresholds or FrequencyThresholds()

    def compute_band_energies(self, S: np.ndarray, freqs: np.ndarray) -> dict:
        """
        S: power spectrogram (freq_bins x frames)
        freqs: frequency of each bin (freq_bins,)
        """
        low_mask = freqs < 200
        high_mask = freqs > 10000

        total_energy = float(S.sum() or 1e-12)
        low_frac  = float(S[low_mask].sum()  / total_energy)
        high_frac = float(S[high_mask].sum() / total_energy)

        return {
            "low_frac": low_frac,
            "high_frac": high_frac,
            "total_energy": total_energy,
        }

    def analyze(self, features: dict) -> dict:
        S = features["S"]
        freqs = features["freqs"]

        bands = self.compute_band_energies(S, freqs)

        unnatural = (
            bands["low_frac"]  > self.thresholds.low_frac
            or bands["high_frac"] > self.thresholds.high_frac
        )

        return {
            "bands": bands,
            "unnatural": bool(unnatural),
        }
