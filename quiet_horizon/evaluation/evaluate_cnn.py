"""CLI evaluation script for QuietHorizon CNN audio classifier.

Usage:
    python -m quiet_horizon.evaluation.evaluate_cnn --dataset-root quiet_horizon/dataset_cnn
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from quiet_horizon.audio import audio_to_spectrogram
from quiet_horizon.inference_cnn import load_model as load_weights_model

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


@dataclass
class Sample:
    path: Path
    label: str  # "nature" or "anthro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QuietHorizon model on labeled audio datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root folder containing labeled audio (expected path parts include "
            "'nature' or 'anthro')."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV manifest with columns: path,label. "
            "Label values: nature or anthro."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/quiet_horizon_cnn.weights.h5"),
        help="Model path (.weights.h5 or .keras).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Nature threshold for classification. Default: 0.5",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for full JSON report.",
    )
    parser.add_argument(
        "--output-confusion-matrix",
        type=Path,
        default=Path("confusion_matrix.png"),
        help="PNG output path for confusion matrix image.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional max files to evaluate (for quick smoke checks).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan dataset root for audio files.",
    )
    return parser.parse_args()


def write_confusion_matrix_image(
    cm: dict[str, int],
    output_path: Path,
) -> None:
    """Render a simple confusion matrix image as PNG."""
    width, height = 780, 520
    margin_left = 170
    margin_top = 120
    cell_w = 260
    cell_h = 160

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    tp = cm["tp"]
    fp = cm["fp"]
    fn = cm["fn"]
    tn = cm["tn"]
    total = max(tp + fp + fn + tn, 1)

    # Cells ordered as:
    # [ TN | FP ]
    # [ FN | TP ]
    cells = [
        ("TN", tn, (212, 245, 214)),  # green-ish
        ("FP", fp, (255, 224, 224)),  # red-ish
        ("FN", fn, (255, 224, 224)),  # red-ish
        ("TP", tp, (212, 245, 214)),  # green-ish
    ]

    for idx, (name, value, color) in enumerate(cells):
        row = idx // 2
        col = idx % 2
        x0 = margin_left + (col * cell_w)
        y0 = margin_top + (row * cell_h)
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black", width=2)
        pct = (value / total) * 100.0
        draw.text((x0 + 16, y0 + 24), f"{name}", fill="black", font=font)
        draw.text((x0 + 16, y0 + 54), f"Count: {value}", fill="black", font=font)
        draw.text((x0 + 16, y0 + 84), f"Share: {pct:.1f}%", fill="black", font=font)

    draw.text((margin_left + 120, 36), "QuietHorizon Confusion Matrix", fill="black", font=font)
    draw.text((margin_left + 28, 90), "Predicted label", fill="black", font=font)
    draw.text((margin_left + 90, 445), "True label", fill="black", font=font)

    draw.text((margin_left + 95, margin_top - 24), "Nature", fill="black", font=font)
    draw.text((margin_left + cell_w + 95, margin_top - 24), "Anthro", fill="black", font=font)
    draw.text((margin_left - 70, margin_top + 70), "Nature", fill="black", font=font)
    draw.text((margin_left - 70, margin_top + cell_h + 70), "Anthro", fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def normalize_label(raw: str) -> str | None:
    value = raw.strip().lower()
    if value in {"nature", "natural"}:
        return "nature"
    if value in {"anthro", "anthropogenic", "human"}:
        return "anthro"
    return None


def infer_label_from_path(path: Path) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    if any(part in {"nature", "natural"} for part in lowered_parts):
        return "nature"
    if any(part in {"anthro", "anthropogenic"} for part in lowered_parts):
        return "anthro"
    return None


def collect_samples_from_manifest(manifest_path: Path) -> list[Sample]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: list[Sample] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Manifest must contain 'path' and 'label' columns.")

        for row in reader:
            sample_path = Path(row["path"]).expanduser()
            if not sample_path.is_absolute():
                sample_path = (manifest_path.parent / sample_path).resolve()
            label = normalize_label(row["label"])
            if label is None:
                continue
            samples.append(Sample(path=sample_path, label=label))

    return samples


def collect_samples_from_dataset_root(dataset_root: Path, recursive: bool) -> list[Sample]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    pattern = "**/*" if recursive else "*"
    samples: list[Sample] = []

    for candidate in dataset_root.glob(pattern):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue

        label = infer_label_from_path(candidate)
        if label is None:
            continue
        samples.append(Sample(path=candidate.resolve(), label=label))

    return samples


def load_eval_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_path_str = str(model_path)
    if model_path_str.lower().endswith(".weights.h5"):
        return load_weights_model(model_path_str)
    return tf.keras.models.load_model(model_path_str)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = [score for score, label in zip(y_score, y_true) if label == 1]
    negatives = [score for score, label in zip(y_score, y_true) if label == 0]
    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0 or n_neg == 0:
        return None

    better = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                better += 1.0
            elif pos == neg:
                better += 0.5

    return better / (n_pos * n_neg)


def evaluate_dataset(
    model: Any,
    samples: list[Sample],
    threshold: float,
) -> dict[str, Any]:
    y_true_anthro: list[int] = []
    y_pred_anthro: list[int] = []
    y_score_anthro: list[float] = []
    per_file: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for sample in samples:
        try:
            spectrogram_image = audio_to_spectrogram(str(sample.path))
            batched = np.expand_dims(spectrogram_image, axis=0)
            pred = model.predict(batched, verbose=0)
            prob_nature = float(pred[0][0])
            prob_anthro = 1.0 - prob_nature
            pred_label = "nature" if prob_nature >= threshold else "anthro"

            truth_anthro = 1 if sample.label == "anthro" else 0
            pred_anthro = 1 if pred_label == "anthro" else 0

            y_true_anthro.append(truth_anthro)
            y_pred_anthro.append(pred_anthro)
            y_score_anthro.append(prob_anthro)

            per_file.append(
                {
                    "path": str(sample.path),
                    "true_label": sample.label,
                    "predicted_label": pred_label,
                    "prob_nature": prob_nature,
                    "prob_anthro": prob_anthro,
                    "correct": sample.label == pred_label,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed_files.append({"path": str(sample.path), "error": str(exc)})

    tp = sum(1 for t, p in zip(y_true_anthro, y_pred_anthro) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true_anthro, y_pred_anthro) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true_anthro, y_pred_anthro) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_anthro, y_pred_anthro) if t == 1 and p == 0)

    total = len(y_true_anthro)
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    roc_auc = compute_roc_auc(y_true_anthro, y_score_anthro)

    # Nature as positive class for per-class metrics.
    tp_nature = tn
    fp_nature = fn
    fn_nature = fp
    support_anthro = tp + fn
    support_nature = tp_nature + fn_nature
    precision_nature = safe_div(tp_nature, tp_nature + fp_nature)
    recall_nature = safe_div(tp_nature, tp_nature + fn_nature)
    f1_nature = safe_div(
        2 * precision_nature * recall_nature, precision_nature + recall_nature
    )

    return {
        "summary": {
            "total_samples": len(samples),
            "evaluated_samples": total,
            "failed_samples": len(failed_files),
            "threshold_nature": threshold,
            "dataset_note": (
                "Metrics are computed on the provided dataset/manifest and depend on "
                "label quality and class distribution."
            ),
            "auc_note": "ROC-AUC is computed one-vs-rest with anthro as positive class.",
        },
        "metrics": {
            "accuracy": accuracy,
            "precision_anthro": precision,
            "recall_anthro": recall,
            "f1_anthro": f1,
            "roc_auc_anthro": roc_auc,
        },
        "per_class": {
            "anthro": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support_anthro,
            },
            "nature": {
                "precision": precision_nature,
                "recall": recall_nature,
                "f1": f1_nature,
                "support": support_nature,
            },
        },
        "confusion_matrix_anthro": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "failed_files": failed_files,
        "predictions": per_file,
    }


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    metrics = report["metrics"]
    per_class = report["per_class"]
    cm = report["confusion_matrix_anthro"]

    print("\nQuietHorizon Evaluation Report")
    print("=" * 32)
    print(f"Total samples discovered: {summary['total_samples']}")
    print(f"Samples evaluated:       {summary['evaluated_samples']}")
    print(f"Samples failed:          {summary['failed_samples']}")
    print(f"Nature threshold:        {summary['threshold_nature']:.2f}")
    print(f"Dataset note:            {summary['dataset_note']}")

    print("\nMetrics (anthro as positive class)")
    print(f"Accuracy:                {metrics['accuracy']:.4f}")
    print(f"Precision (anthro):      {metrics['precision_anthro']:.4f}")
    print(f"Recall (anthro):         {metrics['recall_anthro']:.4f}")
    print(f"F1 (anthro):             {metrics['f1_anthro']:.4f}")
    roc_auc = metrics["roc_auc_anthro"]
    print(f"ROC-AUC (anthro):        {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC (anthro):        n/a")
    print(f"AUC note:                {summary['auc_note']}")

    print("\nPer-class metrics")
    print(
        "Anthro -> "
        f"P: {per_class['anthro']['precision']:.4f}  "
        f"R: {per_class['anthro']['recall']:.4f}  "
        f"F1: {per_class['anthro']['f1']:.4f}  "
        f"Support: {per_class['anthro']['support']}"
    )
    print(
        "Nature -> "
        f"P: {per_class['nature']['precision']:.4f}  "
        f"R: {per_class['nature']['recall']:.4f}  "
        f"F1: {per_class['nature']['f1']:.4f}  "
        f"Support: {per_class['nature']['support']}"
    )

    print("\nConfusion Matrix (anthro positive)")
    print(f"TP: {cm['tp']}  FP: {cm['fp']}")
    print(f"FN: {cm['fn']}  TN: {cm['tn']}")


def main() -> None:
    args = parse_args()

    if args.dataset_root is None and args.manifest is None:
        raise ValueError("Provide either --dataset-root or --manifest.")

    samples: list[Sample] = []
    if args.dataset_root is not None:
        samples.extend(
            collect_samples_from_dataset_root(args.dataset_root, args.recursive)
        )
    if args.manifest is not None:
        samples.extend(collect_samples_from_manifest(args.manifest))

    if args.max_files is not None:
        samples = samples[: args.max_files]

    if not samples:
        raise ValueError("No labeled audio samples found for evaluation.")

    model = load_eval_model(args.model_path)
    report = evaluate_dataset(model=model, samples=samples, threshold=args.threshold)
    print_report(report)

    cm_path = args.output_confusion_matrix
    write_confusion_matrix_image(report["confusion_matrix_anthro"], cm_path)
    print(f"\nWrote confusion matrix image to: {cm_path}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
