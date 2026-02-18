# Test Fixtures

This directory contains sample data for testing:

## Audio Files

- `sample_nature.wav` - Clean nature recording (bird call)
- `sample_anthro.wav` - Anthropogenic noise (vehicle)
- `sample_mixed.wav` - Mixed nature + anthro
- `corrupted.wav` - Corrupted file for error testing

## Spectrograms

- `sample_spec.npy` - Numpy array of sample spectrogram

## Models (if needed for testing)

- Small test models should go here (not the full 4MB model)

---

**Note**: Large audio files should be downloaded separately or generated programmatically in tests. Do not commit large binary files to the repository.

## Generating Test Fixtures

```python
# Generate test audio
import numpy as np
import scipy.io.wavfile as wav

sr = 22050
duration = 2.0
t = np.linspace(0, duration, int(sr * duration))
audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

wav.write('tests/fixtures/sample_sine.wav', sr, audio.astype(np.float32))
```
