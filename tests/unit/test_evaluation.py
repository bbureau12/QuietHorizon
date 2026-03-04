"""Unit tests for evaluation script utilities."""

from quiet_horizon.evaluation.evaluate_cnn import (
    compute_roc_auc,
    normalize_label,
    write_confusion_matrix_image,
)


def test_normalize_label_supported_values():
    assert normalize_label("nature") == "nature"
    assert normalize_label("Natural") == "nature"
    assert normalize_label("anthro") == "anthro"
    assert normalize_label("Anthropogenic") == "anthro"


def test_normalize_label_unsupported_value():
    assert normalize_label("bird") is None


def test_compute_roc_auc_perfect_separation():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    assert compute_roc_auc(y_true, y_score) == 1.0


def test_compute_roc_auc_tied_scores():
    y_true = [1, 0]
    y_score = [0.5, 0.5]
    assert compute_roc_auc(y_true, y_score) == 0.5


def test_compute_roc_auc_single_class_returns_none():
    y_true = [1, 1, 1]
    y_score = [0.1, 0.4, 0.9]
    assert compute_roc_auc(y_true, y_score) is None


def test_write_confusion_matrix_image(tmp_path):
    output = tmp_path / "cm.png"
    cm = {"tp": 7, "tn": 11, "fp": 2, "fn": 3}
    write_confusion_matrix_image(cm, output)
    assert output.exists()
    assert output.stat().st_size > 0
