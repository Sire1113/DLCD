from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    train_mode: str = "linear_probe",
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if train_mode == "scratch":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif train_mode == "linear_probe":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif train_mode == "finetune_last_block":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif train_mode == "finetune_all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
