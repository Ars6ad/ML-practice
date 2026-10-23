import pytest
import numpy as np
from src.metrics import (
    compute_classification_metrics,
    bootstrap_confidence_intervals,
    mcnemar_test,
)


def test_compute_metrics_perfect():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    class_names = ["Good", "Bad", "Mixed"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["mixed_class"]["recall"] == 1.0
    assert metrics["mixed_class"]["precision"] == 1.0


def test_compute_metrics_zero_division():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    class_names = ["Good", "Bad", "Mixed"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["mixed_class"]["recall"] == 0.0
    assert metrics["mixed_class"]["precision"] == 0.0


def test_bootstrap_intervals():
    y_true = np.array([0, 1, 2, 0, 1, 2] * 20)
    y_pred = np.array([0, 1, 2, 0, 1, 0] * 20)
    class_names = ["Good", "Bad", "Mixed"]

    ci = bootstrap_confidence_intervals(y_true, y_pred, class_names=class_names, n_bootstraps=50)
    assert "accuracy" in ci
    assert "mixed_recall" in ci
    assert ci["accuracy"]["ci_lower"] <= ci["accuracy"]["ci_upper"]


def test_mcnemar_identical_models():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred_a = np.array([0, 1, 2, 0, 1, 2])
    y_pred_b = np.array([0, 1, 2, 0, 1, 2])

    res = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert res["b_only_a_correct"] == 0
    assert res["c_only_b_correct"] == 0
    assert res["p_value"] == 1.0
