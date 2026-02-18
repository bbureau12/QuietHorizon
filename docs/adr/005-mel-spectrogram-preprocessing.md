# 5. Mel-Spectrogram Audio Preprocessing

**Date**: 2026-02-18

**Status**: Accepted

## Context

Audio must be converted to a suitable representation for the CNN. Options include:
- Raw waveforms
- Short-Time Fourier Transform (STFT)
- Mel-spectrograms
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Constant-Q Transform (CQT)

Requirements:
- Capture relevant audio features (frequency, time, energy)
- Compatible with CNN input
- Computationally efficient
- Standard in audio ML community

## Decision

Use mel-spectrograms converted to 128×128 RGB images as CNN input.

**Preprocessing Pipeline**:
1. Load audio at 22,050 Hz sample rate
2. Compute mel-spectrogram (128 mel bins, n_fft=2048, hop=512)
3. Convert to dB scale
4. Normalize to 0-255 range
5. Resize to 128×128 pixels
6. Convert to 3-channel RGB image

## Consequences

### Positive Consequences

- **Perceptually Relevant**: Mel scale matches human hearing
- **Time-Frequency Representation**: Captures both temporal and spectral features
- **Visual Patterns**: Distinct patterns for nature vs anthropogenic sounds
- **Standard Approach**: Well-established in audio classification
- **Library Support**: Excellent support in librosa
- **CNN-Ready**: Can leverage computer vision architectures
- **Efficient**: Relatively fast computation
- **Compact**: 128×128 images are small yet informative

### Negative Consequences

- **Information Loss**: Some audio details lost in conversion
- **Fixed Resolution**: 128×128 may miss fine details or compress long audio
- **Parameter Sensitivity**: Results depend on n_fft, hop_length, n_mels choices
- **Not Invertible**: Can't reconstruct original audio from spectrogram
- **RGB Redundancy**: 3 channels are redundant (grayscale would suffice technically)

## Alternatives Considered

### Alternative 1: Raw Waveform

Feed raw audio samples directly to 1D CNN.

**Rejected because**:
- Requires much larger models
- Less interpretable
- Harder to visualize
- Doesn't leverage human understanding of frequency content
- More difficult to train

### Alternative 2: MFCC Features

Use MFCCs (standard in speech recognition).

**Rejected because**:
- Designed for speech, not environmental audio
- Uses DCT which discards information
- Less visually interpretable
- MFCCs compress frequency information
- Not ideal for distinguishing environmental sounds

### Alternative 3: Linear STFT Spectrogram

Use linear-frequency STFT instead of mel scale.

**Rejected because**:
- Most energy in lower frequencies gets lost in linear scale
- Not perceptually aligned
- Mel scale better for biological and environmental sounds
- Larger input size needed for same frequency resolution

### Alternative 4: Constant-Q Transform (CQT)

Use CQT for better low-frequency resolution.

**Rejected because**:
- More computationally expensive
- Less standard in ML community
- Similar performance to mel-spectrograms in testing
- Fewer library implementations

### Alternative 5: Wavelet Transform

Use continuous or discrete wavelet transforms.

**Rejected because**:
- More complex to implement
- Less interpretable
- Not standard in audio ML
- No clear advantage for this task
- Harder to visualize

## Parameters Chosen

```python
TARGET_SAMPLE_RATE = 22050    # Good balance of quality and size
N_MELS = 128                  # Standard choice, good resolution
HOP_LENGTH = 512              # ~23ms windows
N_FFT = 2048                  # ~93ms FFT window
SPECTROGRAM_SIZE = (128, 128) # Square, CNN-friendly
```

## Why RGB (3 channels)?

While grayscale would be technically sufficient, RGB was chosen because:
- Many pretrained CNNs expect 3-channel input
- Minimal size overhead
- Potential for future color mapping of features
- Consistent with standard image processing pipelines

## References

- [Mel-Frequency Scale](https://en.wikipedia.org/wiki/Mel_scale)
- [Librosa Mel-Spectrogram Documentation](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)
- [Audio Processing for ML](https://wiki.aalto.fi/display/ITSP/Audio+Features)
- `audio_processor.py` implementation
