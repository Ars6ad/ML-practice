from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from scipy.stats import chi2


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics with a primary focus on minority class performance.
    
    Args:
        y_true (np.ndarray): 1D array of ground truth class indices.
        y_pred (np.ndarray): 1D array of predicted class indices.
        y_prob (np.ndarray, optional): 2D array of class probabilities [N, num_classes].
        class_names (list): Ordered list of class names (e.g. ['Good', 'Bad', 'Mixed']).
        
    Returns:
        dict: Hierarchical dictionary containing overall and per-class metrics.
    """
    if class_names is None:
        class_names = ["Good", "Bad", "Mixed"]
        
    num_classes = len(class_names)
    labels = list(range(num_classes))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Macro & Weighted metrics
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_metrics = {}
    for i, c_name in enumerate(class_names):
        per_class_metrics[c_name] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
        }

    # Dedicated minority class (Mixed) analysis
    mixed_idx = class_names.index("Mixed") if "Mixed" in class_names else 2
    mixed_metrics = {
        "precision": float(prec_per_class[mixed_idx]),
        "recall": float(rec_per_class[mixed_idx]),
        "f1": float(f1_per_class[mixed_idx]),
        "support": int(support[mixed_idx]),
    }

    report_str = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
        "mixed_class": mixed_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Estimates 95% bootstrap confidence intervals for key performance metrics.
    """
    if class_names is None:
        class_names = ["Good", "Bad", "Mixed"]
    mixed_idx = class_names.index("Mixed") if "Mixed" in class_names else 2

    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    boot_acc = []
    boot_bal_acc = []
    boot_macro_f1 = []
    boot_mixed_rec = []
    boot_mixed_f1 = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        yt_sample = y_true[indices]
        yp_sample = y_pred[indices]

        boot_acc.append(accuracy_score(yt_sample, yp_sample))
        boot_bal_acc.append(balanced_accuracy_score(yt_sample, yp_sample))

        _, _, mf1, _ = precision_recall_fscore_support(
            yt_sample, yp_sample, average="macro", zero_division=0
        )
        boot_macro_f1.append(mf1)

        _, rec, f1, _ = precision_recall_fscore_support(
            yt_sample, yp_sample, labels=[mixed_idx], zero_division=0
        )
        boot_mixed_rec.append(rec[0])
        boot_mixed_f1.append(f1[0])

    def calc_ci(vals):
        lower = float(np.percentile(vals, 100 * (alpha / 2)))
        upper = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        mean = float(np.mean(vals))
        return {"mean": mean, "ci_lower": lower, "ci_upper": upper}

    return {
        "accuracy": calc_ci(boot_acc),
        "balanced_accuracy": calc_ci(boot_bal_acc),
        "macro_f1": calc_ci(boot_macro_f1),
        "mixed_recall": calc_ci(boot_mixed_rec),
        "mixed_f1": calc_ci(boot_mixed_f1),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Performs McNemar's paired test with continuity correction to evaluate
    whether difference in error distribution between two models is statistically significant.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B incorrect
    b = int(np.sum(correct_a & ~correct_b))
    # c: A incorrect, B correct
    c = int(np.sum(~correct_a & correct_b))

    total = b + c
    if total == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        stat = ((abs(b - c) - 1.0) ** 2) / total
        p_val = float(chi2.sf(stat, df=1))

    return {
        "b_only_a_correct": b,
        "c_only_b_correct": c,
        "statistic": float(stat),
        "p_value": p_val,
        "significant_at_05": p_val < 0.05,
    }
