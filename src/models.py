import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


def get_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Constructs a DenseNet121 model with transfer learning.
    
    Args:
        num_classes (int): Number of target classes (3 for Good, Bad, Mixed).
        pretrained (bool): Whether to initialize backbone with ImageNet weights.
        
    Returns:
        nn.Module: Configured DenseNet121 model.
    """
    weights = None
    if pretrained:
        try:
            weights = DenseNet121_Weights.DEFAULT
            model = models.densenet121(weights=weights)
            print("Loaded DenseNet121 with pretrained ImageNet weights.")
        except Exception as e:
            print(f"Warning: Failed to load ImageNet weights ({e}). Initializing randomly.")
            model = models.densenet121(weights=None)
    else:
        model = models.densenet121(weights=None)
        print("Initialized DenseNet121 from scratch (no pretrained weights).")

    # Replace classifier head: DenseNet121 has in_features = 1024
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_parameter_groups(model: nn.Module, lr_head: float = 1e-3, lr_backbone: float = 1e-4, weight_decay: float = 1e-4):
    """
    Separates classifier head parameters and backbone parameters for fine-tuning.
    """
    head_params = list(model.classifier.parameters())
    head_param_ids = set(map(id, head_params))
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
