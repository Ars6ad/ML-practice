import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.models import get_model
from src.dataset import create_dataloaders
from src.metrics import compute_classification_metrics
from src.utils import get_device, save_json


CORRUPTIONS = [
    {"name": "clean", "type": "none"},
    {"name": "brightness_plus_20", "type": "brightness", "factor": 1.2},
    {"name": "brightness_minus_20", "type": "brightness", "factor": 0.8},
    {"name": "brightness_plus_40", "type": "brightness", "factor": 1.4},
    {"name": "brightness_minus_40", "type": "brightness", "factor": 0.6},
    {"name": "contrast_low", "type": "contrast", "factor": 0.7},
    {"name": "contrast_high", "type": "contrast", "factor": 1.3},
    {"name": "gaussian_noise", "type": "noise", "std": 0.08},
    {"name": "gaussian_blur", "type": "blur", "kernel_size": 5, "sigma": 1.2},
]


def apply_perturbation(images: torch.Tensor, corruption_info: dict) -> torch.Tensor:
    """Applies a synthetic environmental perturbation to a normalized image tensor batch."""
    c_type = corruption_info["type"]
    if c_type == "none":
        return images

    out = images.clone()
    
    # We denormalize roughly to [0, 1] range to apply visual perturbations cleanly, then re-normalize
    # ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)

    denorm = torch.clamp(out * std + mean, 0.0, 1.0)

    if c_type == "brightness":
        factor = corruption_info["factor"]
        denorm = torch.clamp(denorm * factor, 0.0, 1.0)
    elif c_type == "contrast":
        factor = corruption_info["factor"]
        mean_lum = torch.mean(denorm, dim=(2, 3), keepdim=True)
        denorm = torch.clamp((denorm - mean_lum) * factor + mean_lum, 0.0, 1.0)
    elif c_type == "noise":
        noise = torch.randn_like(denorm) * corruption_info["std"]
        denorm = torch.clamp(denorm + noise, 0.0, 1.0)
    elif c_type == "blur":
        k = corruption_info["kernel_size"]
        sig = corruption_info["sigma"]
        denorm = TF.gaussian_blur(denorm, kernel_size=[k, k], sigma=[sig, sig])

    # Re-normalize
    renorm = (denorm - mean) / std
    return renorm


def evaluate_robustness(
    checkpoints: Dict[str, str],
    config: dict,
) -> pd.DataFrame:
    """
    Evaluates multiple trained models on the test set across controlled environmental corruptions.
    
    Args:
        checkpoints (dict): {'baseline': path, 'augmentation': path, 'class_weighted': path}
        config (dict): Master config.
        
    Returns:
        pd.DataFrame: Table comparing models across corruptions.
    """
    device = get_device(verbose=False)
    dataloaders, datasets, class_to_idx = create_dataloaders(config, experiment_type="baseline")
    test_loader = dataloaders["test"]
    class_order = datasets["test"].class_order

    models_dict = {}
    for exp_name, ckpt_path in checkpoints.items():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = get_model(num_classes=len(class_order), pretrained=False).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models_dict[exp_name] = m

    records = []

    print("\n--- Running Environmental Robustness Analysis ---")
    for corr in CORRUPTIONS:
        corr_name = corr["name"]
        print(f"Evaluating Perturbation: {corr_name}")

        corr_preds = {exp: [] for exp in models_dict.keys()}
        corr_targets = []

        with torch.inference_mode():
            for images, labels, _ in test_loader:
                images = images.to(device, non_blocking=True)
                pert_images = apply_perturbation(images, corr)
                corr_targets.extend(labels.numpy().tolist())

                for exp_name, model in models_dict.items():
                    outputs = model(pert_images)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                    corr_preds[exp_name].extend(preds)

        y_true = np.array(corr_targets)

        for exp_name in models_dict.keys():
            y_pred = np.array(corr_preds[exp_name])
            m = compute_classification_metrics(y_true, y_pred, class_names=class_order)
            records.append({
                "experiment": exp_name,
                "corruption": corr_name,
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "mixed_recall": m["mixed_class"]["recall"],
                "mixed_precision": m["mixed_class"]["precision"],
                "mixed_f1": m["mixed_class"]["f1"],
            })

    rob_df = pd.DataFrame(records)
    reports_dir = config.get("paths", {}).get("reports_dir", "outputs/reports")
    os.makedirs(reports_dir, exist_ok=True)
    rob_csv_path = os.path.join(reports_dir, "robustness_results.csv")
    rob_df.to_csv(rob_csv_path, index=False)
    print(f"Saved environmental robustness results to {rob_csv_path}")

    return rob_df
