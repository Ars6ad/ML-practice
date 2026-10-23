import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.transforms import get_transforms


CANONICAL_CLASSES = ["Good", "Bad", "Mixed"]


def normalize_label(raw_label: str) -> str:
    """Maps folder or raw label strings to canonical labels."""
    raw = str(raw_label).strip().lower()
    if "good" in raw:
        return "Good"
    elif "bad" in raw:
        return "Bad"
    elif "mix" in raw:
        return "Mixed"
    raise ValueError(f"Cannot map label '{raw_label}' to canonical classes {CANONICAL_CLASSES}")


class FruitQualityDataset(Dataset):
    """
    PyTorch Dataset for fruit quality classification.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        transform=None,
        class_order: Optional[List[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.class_order = class_order or CANONICAL_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_order)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_order)}
        
        # Validate and precompute class indices
        self.labels = [self.class_to_idx[normalize_label(l)] for l in self.df["label"]]
        self.image_paths = self.df["image_path"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rel_path = self.image_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        
        try:
            with Image.open(full_path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                label_idx = self.labels[idx]
                return img, label_idx, rel_path
        except Exception as e:
            raise IOError(f"Failed to load image at {full_path}: {e}")

    def get_class_counts(self) -> Dict[str, int]:
        """Returns sample count per canonical class."""
        counts = {c: 0 for c in self.class_order}
        for lbl in self.labels:
            counts[self.idx_to_class[lbl]] += 1
        return counts


def create_dataloaders(
    config: dict,
    experiment_type: str = "baseline",
    manifest_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, FruitQualityDataset], Dict[str, int]]:
    """
    Creates reproducible DataLoaders for train, validation, and test splits.
    
    Args:
        config (dict): Configuration dictionary.
        experiment_type (str): 'baseline', 'augmentation', or 'class_weighted'.
        manifest_df (pd.DataFrame, optional): Pre-loaded manifest DataFrame.
        
    Returns:
        dataloaders (dict): {'train': DataLoader, 'validation': DataLoader, 'test': DataLoader}
        datasets (dict): {'train': Dataset, 'validation': Dataset, 'test': Dataset}
        class_to_idx (dict): Mapping of class names to indices.
    """
    ds_cfg = config.get("dataset", {})
    root_dir = ds_cfg.get("root_dir", "Processed Images_Fruits")
    manifest_path = ds_cfg.get("manifest_path", "fruit_quality_dataset_manifest.csv")
    class_order = ds_cfg.get("class_order", CANONICAL_CLASSES)
    
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    eval_batch_size = train_cfg.get("eval_batch_size", 64)
    num_workers = train_cfg.get("num_workers", 2)
    pin_memory = train_cfg.get("pin_memory", True) and torch.cuda.is_available()

    if manifest_df is None:
        manifest_df = pd.read_csv(manifest_path)

    # Normalize split column and filter
    manifest_df["split"] = manifest_df["split"].str.strip().str.lower()
    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    val_df = manifest_df[manifest_df["split"].isin(["validation", "val"])].copy()
    test_df = manifest_df[manifest_df["split"] == "test"].copy()

    # Create dataset instances
    is_aug = (experiment_type == "augmentation")
    train_transform = get_transforms("train", config, is_augmentation_exp=is_aug)
    eval_transform = get_transforms("validation", config, is_augmentation_exp=False)

    train_dataset = FruitQualityDataset(train_df, root_dir, transform=train_transform, class_order=class_order)
    val_dataset = FruitQualityDataset(val_df, root_dir, transform=eval_transform, class_order=class_order)
    test_dataset = FruitQualityDataset(test_df, root_dir, transform=eval_transform, class_order=class_order)

    # Configure sampler for minority-focused experiment
    sampler = None
    shuffle = True
    
    if experiment_type == "augmentation":
        boost_factor = config.get("minority_sampling", {}).get("mixed_class_boost_factor", 5.0)
        mixed_idx = train_dataset.class_to_idx["Mixed"]
        
        sample_weights = []
        for lbl in train_dataset.labels:
            if lbl == mixed_idx:
                sample_weights.append(boost_factor)
            else:
                sample_weights.append(1.0)
                
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    dataloaders = {"train": train_loader, "validation": val_loader, "test": test_loader}
    datasets = {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    class_to_idx = train_dataset.class_to_idx

    return dataloaders, datasets, class_to_idx
