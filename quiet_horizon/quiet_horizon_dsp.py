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
        # Get the shared audio + spectral features
        features = self.engine.extract_all(path)

        # Each subsystem now takes the whole feature dict
        freq_result = self.freq.analyze(features)
        rhythm_result = self.rhythm.analyze(features)
        flat_result = self.flat.analyze(features)

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
