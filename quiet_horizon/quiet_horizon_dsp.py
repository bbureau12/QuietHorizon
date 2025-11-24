from core.dsp_engine import DSPEngine
from dsp.frequency import FrequencyFilter
from dsp.rhythm import RhythmDetector
from dsp.spectral_flatness import FlatnessFilter

class QuietHorizonDSP:
    def __init__(self):
        self.engine = DSPEngine()       # shared audio + spectro workflow
        self.freq = FrequencyFilter()   # subsystem 1
        self.rhythm = RhythmDetector()  # subsystem 2
        self.flat = FlatnessFilter()    # subsystem 3

    def analyze(self, path):
        features = self.engine.extract_all(path)  # load and compute once

        # Each subsystem receives the shared data
        freq_result = self.freq.analyze(features["y"], features["sr"], features["S"])
        rhythm_result = self.rhythm.analyze(features["y"], features["sr"])
        flat_result = self.flat.analyze(features["y"], features["sr"], features["S"])

        return {
            "path": path,
            "frequency": freq_result,
            "rhythm": rhythm_result,
            "flatness": flat_result,
            "anthropogenic": (
                freq_result["unnatural"]
                or rhythm_result["rhythmic"]
                or flat_result["noise_like"]
            )
        }
