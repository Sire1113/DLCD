from __future__ import annotations

from .config import ExperimentConfig


def build_default_experiments() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="scratch_resnet18",
            pretrained=False,
            train_mode="scratch",
            lr=1e-3,
            epochs=30,
        ),
        ExperimentConfig(
            name="linear_probe_resnet18",
            pretrained=True,
            train_mode="linear_probe",
            lr=1e-3,
            epochs=20,
        ),
        ExperimentConfig(
            name="finetune_last_block_resnet18",
            pretrained=True,
            train_mode="finetune_last_block",
            lr=1e-4,
            epochs=20,
        ),
    ]


def build_lr_sweep(base_epochs: int = 20) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name=f"finetune_last_block_lr_{learning_rate:g}",
            pretrained=True,
            train_mode="finetune_last_block",
            lr=learning_rate,
            epochs=base_epochs,
        )
        for learning_rate in (1e-3, 1e-4, 1e-5)
    ]
