# 1. CNN-Based Audio Classification Using Spectrograms

**Date**: 2026-02-18

**Status**: Accepted

## Context

QuietHorizon needs to classify audio recordings as either nature sounds or anthropogenic (human-made) noise. The classification must be:
- Accurate (>90% accuracy target)
- Lightweight (<10 MB model size for deployment)
- Fast enough for real-time or near-real-time processing
- Robust to various audio conditions

Traditional signal processing approaches (DSP) can detect some features but struggle with the complexity and variability of real-world audio.

## Decision

We will use a Convolutional Neural Network (CNN) operating on mel-spectrogram images rather than raw audio waveforms or hand-crafted DSP features.

The architecture:
- Input: 128×128×3 RGB mel-spectrogram images
- CNN layers: Conv(32) → Conv(64) → Conv(128) → Conv(256)
- Global Average Pooling
- Dense layers with dropout
- Binary output (sigmoid activation)

## Consequences

### Positive Consequences

- **High Accuracy**: Achieved ~95% accuracy, exceeding the target
- **Pattern Recognition**: CNNs excel at learning visual patterns in spectrograms (harmonic structures, temporal patterns)
- **Small Model Size**: ~4 MB on disk, suitable for deployment
- **Transfer Learning**: Can leverage computer vision techniques and pretrained models
- **End-to-End Learning**: No need for manual feature engineering
- **Robust**: Learns invariant features across different recording conditions

### Negative Consequences

- **Black Box**: Less interpretable than explicit DSP features
- **Training Data**: Requires large labeled dataset (~20,000 samples)
- **Computational Cost**: Requires GPU for efficient training
- **Spectrogram Dependency**: Must convert audio to spectrogram for inference
- **Fixed Input Size**: All audio segments must be converted to 128×128 images

## Alternatives Considered

### Alternative 1: Traditional DSP Features with Classical ML

Extract hand-crafted features (spectral flatness, rhythm, frequency characteristics) and use SVM, Random Forest, or similar.

**Rejected because**: 
- Lower accuracy in testing (~80-85%)
- Extensive feature engineering required
- Difficulty capturing complex temporal patterns
- Less robust to variations in recording conditions

### Alternative 2: Recurrent Neural Networks (RNN/LSTM) on Raw Audio

Use RNN/LSTM to process audio waveforms directly or spectrogram slices sequentially.

**Rejected because**:
- Larger model size (>20 MB)
- Slower inference time
- More complex training
- Spectrograms already capture temporal information effectively

### Alternative 3: Transformer-Based Models

Use attention mechanisms and transformer architectures for audio classification.

**Rejected because**:
- Much larger model size (>50 MB)
- Higher computational requirements
- Overkill for binary classification task
- Harder to deploy on edge devices

## References

- [Deep Learning for Audio Classification](https://towardsdatascience.com/audio-deep-learning-made-simple-633971f5df3e)
- [Mel-Spectrograms for Deep Learning](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-f)
- TensorFlow Keras documentation
- Model performance metrics in main README.md
