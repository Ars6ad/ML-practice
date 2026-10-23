import os
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from src.models import get_model
from src.dataset import create_dataloaders
from src.metrics import compute_classification_metrics, bootstrap_confidence_intervals
from src.utils import get_device, save_json


def evaluate_checkpoint(
    checkpoint_path: str,
    config: dict,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Evaluates a saved model checkpoint on the specified split (default: untouched test set).
    
    Args:
        checkpoint_path (str): Path to .pth checkpoint.
        config (dict): Master configuration dictionary.
        split (str): 'test' or 'validation'.
        
    Returns:
        dict: Complete evaluation metrics, predictions DataFrame, and error analysis.
    """
    device = get_device(verbose=False)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    experiment_type = ckpt.get("experiment_type", "unknown")
    class_order = ckpt.get("class_order", config.get("dataset", {}).get("class_order", ["Good", "Bad", "Mixed"]))
    num_classes = len(class_order)

    # Initialize model and load weights
    model = get_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create dataloaders
    dataloaders, datasets, class_to_idx = create_dataloaders(config, experiment_type=experiment_type)
    loader = dataloaders[split]
    dataset = datasets[split]

    all_preds = []
    all_targets = []
    all_probs = []
    all_paths = []

    start_time = time.time()
    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(paths)

    inference_duration = time.time() - start_time
    throughput = len(all_paths) / inference_duration if inference_duration > 0 else 0.0

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Calculate metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob, class_names=class_order)
    metrics["inference_time_seconds"] = float(inference_duration)
    metrics["throughput_images_per_sec"] = float(throughput)
    metrics["experiment_type"] = experiment_type
    metrics["checkpoint_path"] = checkpoint_path
    metrics["split"] = split
    metrics["total_samples"] = len(y_true)

    # Bootstrap 95% confidence intervals
    ci_results = bootstrap_confidence_intervals(y_true, y_pred, class_names=class_order, n_bootstraps=1000)
    metrics["confidence_intervals_95"] = ci_results

    # Build per-image prediction DataFrame
    pred_df = pd.DataFrame({
        "image_path": all_paths,
        "ground_truth": [class_order[t] for t in y_true],
        "predicted": [class_order[p] for p in y_pred],
        "is_correct": (y_true == y_pred),
    })
    for i, c in enumerate(class_order):
        pred_df[f"prob_{c}"] = y_prob[:, i]

    # Detailed Mixed class error analysis
    mixed_idx = class_order.index("Mixed")
    is_mixed_true = (y_true == mixed_idx)
    is_mixed_pred = (y_pred == mixed_idx)

    # False Negatives: True is Mixed, but predicted as Good or Bad
    fn_mask = is_mixed_true & (~is_mixed_pred)
    fn_df = pred_df[fn_mask].copy()
    fn_df["error_type"] = "False Negative (True Mixed missed)"

    # False Positives: True is Good or Bad, but predicted as Mixed
    fp_mask = (~is_mixed_true) & is_mixed_pred
    fp_df = pred_df[fp_mask].copy()
    fp_df["error_type"] = "False Positive (False Mixed alarm)"

    error_analysis = {
        "mixed_total_true": int(np.sum(is_mixed_true)),
        "mixed_true_positives": int(np.sum(is_mixed_true & is_mixed_pred)),
        "mixed_false_negatives": int(len(fn_df)),
        "mixed_false_positives": int(len(fp_df)),
        "sample_false_negatives": fn_df.head(10).to_dict(orient="records"),
        "sample_false_positives": fp_df.head(10).to_dict(orient="records"),
    }
    metrics["error_analysis"] = error_analysis

    # Save predictions CSV and metrics JSON
    preds_dir = config.get("paths", {}).get("predictions_dir", "outputs/predictions")
    reports_dir = config.get("paths", {}).get("reports_dir", "outputs/reports")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    pred_csv_path = os.path.join(preds_dir, f"{experiment_type}_{split}_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    metrics_json_path = os.path.join(reports_dir, f"{experiment_type}_metrics.json")
    save_json(metrics, metrics_json_path)

    print(f"\nEvaluation on [{split}] for {experiment_type.upper()}:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}% | Balanced Acc: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.4f} | Macro Recall: {metrics['macro_recall']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Mixed Precision: {metrics['mixed_class']['precision']:.4f} | Mixed Recall: {metrics['mixed_class']['recall']*100:.2f}% | Mixed F1: {metrics['mixed_class']['f1']:.4f}")
    print(f"Throughput: {throughput:.1f} imgs/sec | Saved to {pred_csv_path}")

    return {
        "metrics": metrics,
        "predictions_df": pred_df,
        "metrics_json_path": metrics_json_path,
        "predictions_csv_path": pred_csv_path,
    }
