import pytest
import torch
from src.models import get_model, get_parameter_groups


def test_model_forward():
    model = get_model(num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 3)
    assert not torch.isnan(output).any()


def test_parameter_groups():
    model = get_model(num_classes=3, pretrained=False)
    groups = get_parameter_groups(model, lr_head=1e-3, lr_backbone=1e-4)

    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-3
