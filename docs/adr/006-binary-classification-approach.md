# 6. Binary Classification (Nature vs Anthropogenic)

**Date**: 2026-02-18

**Status**: Accepted

## Context

QuietHorizon needs to categorize environmental audio. Categorization could be:
- Binary: Nature vs Anthropogenic
- Multi-class: Nature, Road Vehicle, Aircraft, Construction, etc.
- Hierarchical: Nature → (Birds, Frogs, Mammals), Anthro → (Vehicles → Road/Air, Tools, etc.)
- Multi-label: Audio could belong to multiple categories

## Decision

Implement binary classification: Nature (negative class) vs Anthropogenic (positive class).

**Output**: Sigmoid activation producing probability score 0-1, threshold at 0.5.

## Consequences

### Positive Consequences

- **Simplicity**: Easier to train, deploy, and interpret
- **Clear Use Case**: Matches primary need (filter noisy recordings)
- **High Accuracy**: Achieved ~95% accuracy on binary task
- **Small Model**: Binary classification requires fewer parameters
- **Fast Inference**: Single forward pass, immediate result
- **Easy Threshold Tuning**: Can adjust sensitivity via threshold
- **Clear Metrics**: Precision, recall, AUC are straightforward
- **Balanced Dataset**: Easier to balance two classes

### Negative Consequences

- **Limited Granularity**: Can't distinguish between types of anthropogenic noise
- **Mixed Sounds**: Ambiguous for recordings with both nature and anthro
- **No Species Info**: Doesn't identify which animal species
- **Future Expansion**: Would require retraining for multi-class
- **Threshold Sensitivity**: Edge cases near 0.5 may be unreliable

## Alternatives Considered

### Alternative 1: Multi-Class Classification

Classify into specific categories: Bird, Frog, Vehicle, Aircraft, Construction, Rain, etc.

**Rejected because**:
- Much more complex model and training
- Requires detailed labeling of all audio
- Harder to balance classes
- Lower per-class accuracy
- Not aligned with primary use case (noise filtering)
- Can add later as separate model

### Alternative 2: Hierarchical Classification

Two-stage: First nature/anthro, then sub-classify.

**Rejected because**:
- More complex inference pipeline
- Two models to maintain
- Slower inference
- Overkill for current requirements
- Can add later if needed

### Alternative 3: Multi-Label Classification

Allow multiple labels per audio sample.

**Rejected because**:
- More complex training (multi-hot encoding)
- Harder to interpret results
- Not needed for primary use case
- Dataset not labeled for multiple simultaneous sources
- Evaluation metrics more complex

### Alternative 4: Anomaly Detection

Train only on nature sounds, flag everything else as anomaly.

**Rejected because**:
- Lower accuracy than supervised learning
- Harder to tune
- Requires only clean nature data (limiting)
- No explicit modeling of anthropogenic patterns
- Less predictable behavior

### Alternative 5: Regression on "Naturalness Score"

Continuous score 0-1 for how "natural" audio is.

**Rejected because**:
- Subjective labeling required
- Harder to evaluate
- Users want clear classification, not scores
- Doesn't match available training data
- Similar to binary with probability anyway

## Design for Future Multi-Class

The architecture is designed to easily extend to multi-class:
- Change final layer from Dense(1, sigmoid) to Dense(N, softmax)
- Update data pipeline for one-hot encoding
- Retrain with multi-class labels
- Frontend already supports displaying probabilities

## Use Case Alignment

**Primary Use Case**: Filter out anthropogenic noise from wildlife recordings.

Binary classification directly solves this:
- Researchers can automatically filter recordings
- Clear actionable output: keep or discard
- Statistics: "X% of recordings contain human noise"

Future multi-class would enable:
- "Noise was caused by aircraft vs road traffic"
- More detailed environmental impact studies
- But these are secondary to primary filtering need

## References

- Main README.md performance metrics
- Training notebook: `cnn_generation/cnn_trainer.ipynb`
- Model architecture in ADR-001
