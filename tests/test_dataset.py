import pytest
import torch
import pandas as pd
from PIL import Image
import os

from src.dataset import FruitQualityDataset, normalize_label, create_dataloaders
from src.utils import load_config


def test_normalize_label():
    assert normalize_label("Good Quality_Fruits") == "Good"
    assert normalize_label("Apple_Good") == "Good"
    assert normalize_label("Bad") == "Bad"
    assert normalize_label("Banana_Bad") == "Bad"
    assert normalize_label("Mixed Qualit_Fruits") == "Mixed"
    assert normalize_label("Mixed") == "Mixed"


def test_dataset_item_and_shape():
    config = load_config("configs/config.yaml")
    manifest_path = config["dataset"]["manifest_path"]
    root_dir = config["dataset"]["root_dir"]

    df = pd.read_csv(manifest_path).head(10)
    from src.transforms import get_transforms
    transform = get_transforms("validation", config)

    dataset = FruitQualityDataset(df, root_dir=root_dir, transform=transform)
    assert len(dataset) == 10

    img, label_idx, rel_path = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label_idx in [0, 1, 2]
    assert isinstance(rel_path, str)


def test_dataloaders_creation():
    config = load_config("configs/config.yaml")
    dataloaders, datasets, class_to_idx = create_dataloaders(config, experiment_type="baseline")

    assert "train" in dataloaders
    assert "validation" in dataloaders
    assert "test" in dataloaders
    assert class_to_idx == {"Good": 0, "Bad": 1, "Mixed": 2}
    assert len(datasets["test"]) == 1953
