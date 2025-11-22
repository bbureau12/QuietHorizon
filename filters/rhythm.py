import numpy as np
import librosa

class RhythmDetector:
    def __init__(self, threshold_peak_height=0.2, threshold_peak_ratio=0.1):
        """
        threshold_peak_height: how strong a peak must be to count
        threshold_peak_ratio: ratio of main peak to surrounding noise floor
        """
        self.threshold_peak_height = threshold_peak_height
        self.threshold_peak_ratio = threshold_peak_ratio

    def load_audio(self, path, sr=22050):
        y, sr = librosa.load(path, sr=sr, mono=True)
        return y, sr

    def autocorrelation(self, y):
        y = y - np.mean(y)
        corr = np.correlate(y, y, mode='full')
        corr = corr[len(corr)//2:]  # keep positive lags only
        corr /= corr[0]  # normalize by max
        return corr

    def detect_rhythm(self, y):
        corr = self.autocorrelation(y)

        # Ignore the 0-lag peak (always 1.0)
        peaks = librosa.util.peak_pick(
            corr, 
            pre_max=20, post_max=20,
            pre_avg=20, post_avg=20,
            delta=self.threshold_peak_ratio,
            wait=20
        )

        # Filter out tiny peaks
        strong_peaks = [p for p in peaks if corr[p] > self.threshold_peak_height]

        return {
            "peaks": strong_peaks,
            "peak_strengths": corr[strong_peaks].tolist(),
            "rhythmic": len(strong_peaks) >= 2  # two periodic peaks = repeating pattern
        }

    def analyze(self, path):
        y, sr = self.load_audio(path)
        result = self.detect_rhythm(y)
        result["path"] = path
        return result
