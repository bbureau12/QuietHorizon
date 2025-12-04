from core.dsp_engine import DSPEngine
from dsp.frequency import FrequencyFilter
from dsp.rhythm import RhythmDetector
from dsp.spectral_flatness import FlatnessFilter

class QuietHorizonDSP:
    def __init__(
        self,
        train_peak_threshold: int = 12,
        train_flatness_threshold: float = 0.05,
    ):
        self.engine = DSPEngine()
        self.freq = FrequencyFilter()
        self.rhythm = RhythmDetector()
        self.flat = FlatnessFilter()

        # heuristic knobs
        self.train_peak_threshold = train_peak_threshold
        self.train_flatness_threshold = train_flatness_threshold

    def analyze(self, path: str) -> dict:
        features = self.engine.extract_all(path)
        if features.get("too_short"):
            return {
                "path": path,
                "frequency": {},
                "rhythm": {},
                "flatness": {},
                "anthropogenic": False,
                "reasons": {"too_short": True},
            }
        freq_result = self.freq.analyze(features)
        rhythm_result = self.rhythm.analyze(features)
        flat_result = self.flat.analyze(features)

        # ---- core filter decisions ----
        freq_unnatural = bool(freq_result.get("unnatural", False))
        rhythmic = bool(rhythm_result.get("rhythmic", False))
        noise_like = bool(flat_result.get("noise_like", False))

        # we already have peaks + flatness stats
        peaks = rhythm_result.get("peaks", []) or []
        peak_count = len(peaks)

        anthropogenic = freq_unnatural or rhythmic or noise_like

        return {
            "path": path,
            "frequency": freq_result,
            "rhythm": {
                **rhythm_result,
                "peak_count": peak_count,
            },
            "flatness": flat_result,
            "anthropogenic": anthropogenic,
            "reasons": {
                "freq_unnatural": freq_unnatural,
                "rhythmic": rhythmic,
                "noise_like": noise_like
            },
        }
