# dsp/rhythm.py
import numpy as np
import librosa

class RhythmDetector:
    def __init__(self, threshold_peak_height=0.2, threshold_peak_ratio=0.1):
        self.threshold_peak_height = threshold_peak_height
        self.threshold_peak_ratio = threshold_peak_ratio

    def autocorrelation(self, y: np.ndarray) -> np.ndarray:
        y = y - np.mean(y)
        corr = np.correlate(y, y, mode='full')
        corr = corr[len(corr)//2:]  # keep positive lags only
        if corr[0] != 0:
            corr = corr / corr[0]
        return corr

    def detect_rhythm(self, y: np.ndarray) -> dict:
        corr = self.autocorrelation(y)

        peaks = librosa.util.peak_pick(
            corr,
            pre_max=20, post_max=20,
            pre_avg=20, post_avg=20,
            delta=self.threshold_peak_ratio,
            wait=20,
        )

        strong_peaks = [p for p in peaks if corr[p] > self.threshold_peak_height]

        return {
            "peaks": strong_peaks,
            "peak_strengths": corr[strong_peaks].tolist() if len(strong_peaks) else [],
            "rhythmic": len(strong_peaks) >= 2,
        }

    def analyze(self, features: dict) -> dict:
        y = features["y"]
        result = self.detect_rhythm(y)
        return result
