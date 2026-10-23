import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.models import get_model, get_parameter_groups
from src.dataset import create_dataloaders
from src.metrics import compute_classification_metrics
from src.utils import set_seed, get_device, save_json


def compute_class_weights(dataset) -> torch.Tensor:
    """
    Computes balanced class weights exclusively from training split:
    w_c = N / (K * n_c)
    """
    counts = dataset.get_class_counts()
    class_order = dataset.class_order
    n_total = len(dataset)
    k_classes = len(class_order)

    weights = []
    print("\n--- Training Class Frequencies and Balanced Weights ---")
    for c in class_order:
        n_c = counts[c]
        w = n_total / (k_classes * n_c) if n_c > 0 else 1.0
        weights.append(w)
        print(f"Class: {c:<8} | Count: {n_c:<6} | Balanced Weight: {w:.4f}")
        
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list,
) -> Tuple[float, Dict[str, Any]]:
    """Runs evaluation on validation set."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    metrics = compute_classification_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        class_names=class_names,
    )
    return val_loss, metrics


def train_experiment(
    config: dict,
    experiment_type: str = "baseline",
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Executes a complete training experiment.
    
    Args:
        config (dict): Master configuration.
        experiment_type (str): 'baseline', 'augmentation', or 'class_weighted'.
        resume (bool): Whether to skip training if a valid checkpoint already exists.
        
    Returns:
        dict: Training summary and path to best checkpoint.
    """
    print(f"\n=======================================================")
    print(f"  STARTING EXPERIMENT: {experiment_type.upper()}")
    print(f"=======================================================")

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(verbose=True)

    ckpt_dir = config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
    logs_dir = config.get("paths", {}).get("logs_dir", "outputs/logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, f"{experiment_type}_best.pth")
    history_path = os.path.join(logs_dir, f"{experiment_type}_history.json")

    # If checkpoint exists and valid, allow resuming or skipping
    if resume and os.path.exists(best_ckpt_path):
        print(f"Found existing valid checkpoint for {experiment_type} at {best_ckpt_path}.")
        try:
            ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            if ckpt.get("experiment_type") == experiment_type:
                print(f"Checkpoint matches experiment '{experiment_type}' (best val macro F1: {ckpt.get('best_macro_f1', 0.0):.4f}).")
                return {
                    "experiment_type": experiment_type,
                    "checkpoint_path": best_ckpt_path,
                    "history_path": history_path,
                    "skipped": True,
                }
        except Exception as e:
            print(f"Warning: Could not read existing checkpoint ({e}). Retraining...")

    # Create dataloaders
    dataloaders, datasets, class_to_idx = create_dataloaders(config, experiment_type=experiment_type)
    class_names = [datasets["train"].idx_to_class[i] for i in range(len(class_to_idx))]

    # Setup loss function
    if experiment_type == "class_weighted":
        weights = compute_class_weights(datasets["train"]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Build model
    num_classes = config.get("dataset", {}).get("num_classes", 3)
    pretrained = config.get("training", {}).get("pretrained", True)
    model = get_model(num_classes=num_classes, pretrained=pretrained).to(device)

    # Optimizer & Scheduler
    train_cfg = config.get("training", {})
    lr_head = train_cfg.get("lr_head", 1e-3)
    lr_backbone = train_cfg.get("lr_backbone", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    max_epochs = train_cfg.get("epochs", 12)
    patience = train_cfg.get("early_stopping_patience", 4)
    use_amp = train_cfg.get("mixed_precision", True)

    param_groups = get_parameter_groups(model, lr_head=lr_head, lr_backbone=lr_backbone, weight_decay=weight_decay)
    optimizer = AdamW(param_groups)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_mixed_recall": [],
        "val_mixed_precision": [],
        "val_mixed_f1": [],
        "learning_rates": [],
    }

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_time = time.time()

    print(f"\nTraining on {len(datasets['train'])} samples | Validating on {len(datasets['validation'])} samples")
    print(f"Max Epochs: {max_epochs} | Early Stopping Patience: {patience}")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        val_loss, val_metrics = validate(
            model=model,
            dataloader=dataloaders["validation"],
            criterion=criterion,
            device=device,
            class_names=class_names,
        )

        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]
        mixed_rec = val_metrics["mixed_class"]["recall"]
        mixed_prec = val_metrics["mixed_class"]["precision"]
        mixed_f1 = val_metrics["mixed_class"]["f1"]

        scheduler.step(val_macro_f1)
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_mixed_recall"].append(mixed_rec)
        history["val_mixed_precision"].append(mixed_prec)
        history["val_mixed_f1"].append(mixed_f1)
        history["learning_rates"].append(curr_lr)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch {epoch:2d}/{max_epochs:2d} ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% Macro-F1: {val_macro_f1:.4f} | "
            f"Mixed Rec: {mixed_rec*100:.2f}% Mixed F1: {mixed_f1:.4f}"
        )

        # Model selection: Highest Validation Macro F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "val_metrics": val_metrics,
                    "class_to_idx": class_to_idx,
                    "class_order": class_names,
                    "experiment_type": experiment_type,
                    "config": config,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Macro F1: {best_macro_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    total_training_time = time.time() - start_time
    history["total_training_time_seconds"] = total_training_time
    history["best_epoch"] = best_epoch
    history["best_macro_f1"] = best_macro_f1
    save_json(history, history_path)

    print(f"\nExperiment {experiment_type} finished in {total_training_time/60:.2f} mins. Best Epoch: {best_epoch} with Macro F1: {best_macro_f1:.4f}")

    return {
        "experiment_type": experiment_type,
        "checkpoint_path": best_ckpt_path,
        "history_path": history_path,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "training_time": total_training_time,
        "skipped": False,
    }
