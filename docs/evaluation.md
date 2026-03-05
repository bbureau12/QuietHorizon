# Model Evaluation CLI

QuietHorizon now includes a dedicated evaluation script for labeled audio datasets.

- Module: `quiet_horizon.evaluation.evaluate_cnn`
- Console script (after `pip install -e .`): `quiet-horizon-eval`

## What It Reports

- Accuracy
- Precision/Recall/F1 (anthro as positive class)
- Precision/Recall/F1 for each class (`anthro`, `nature`)
- Support per class (sample counts)
- ROC-AUC (anthro)
- Confusion matrix (`TP`, `FP`, `FN`, `TN`)
- Confusion matrix PNG image (auto-generated)
- Per-file predictions
- Failed files (decode/preprocessing errors)

## Input Modes

You can evaluate from either:

1. `--dataset-root`
- Expects files under paths containing `nature` or `anthro`.
- Works well with datasets like `quiet_horizon/dataset_cnn`.

2. `--manifest`
- CSV with columns: `path,label`
- `label` values: `nature` or `anthro` (aliases: `natural`, `anthropogenic`, `human`)

## Examples

Evaluate a dataset directory recursively:

```bash
python -m quiet_horizon.evaluation.evaluate_cnn \
  --dataset-root quiet_horizon/dataset_cnn \
  --recursive \
  --model-path models/quiet_horizon_cnn.weights.h5
```

Generate a precomputed benchmark snapshot for the website:

```bash
python -m quiet_horizon.evaluation.evaluate_cnn \
  --dataset-root quiet_horizon/dataset_cnn \
  --recursive \
  --model-path models/quiet_horizon_cnn.weights.h5 \
  --output-json reports/dataset_cnn_benchmark.json \
  --output-confusion-matrix reports/dataset_cnn_benchmark_cm.png
```

Evaluate using a manifest and save JSON output:

```bash
python -m quiet_horizon.evaluation.evaluate_cnn \
  --manifest data/eval_manifest.csv \
  --model-path models/quiet_horizon_cnn.weights.h5 \
  --output-json reports/eval_report.json
```

Write confusion matrix image to a custom location:

```bash
python -m quiet_horizon.evaluation.evaluate_cnn \
  --manifest data/eval_manifest.csv \
  --output-confusion-matrix reports/confusion_matrix.png
```

Quick smoke evaluation:

```bash
python -m quiet_horizon.evaluation.evaluate_cnn \
  --dataset-root quiet_horizon/dataset_cnn \
  --recursive \
  --max-files 100
```

Use installed console script:

```bash
quiet-horizon-eval --dataset-root quiet_horizon/dataset_cnn --recursive
```

## Notes

- Threshold is on `P(nature)` (`--threshold`, default `0.5`).
- Prediction rule: `nature` if `P(nature) >= threshold`, else `anthro`.
- For metrics, anthro is treated as the positive class.
- By default, confusion matrix PNG is written to `confusion_matrix.png`.
- ROC-AUC in reports is one-vs-rest with anthro as positive class.
- For public claims, prefer reporting ROC-AUC on a held-out test set with split strategy documented.
- Confusion matrix quality depends on your dataset labels and class distribution.
