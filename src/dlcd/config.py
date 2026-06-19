from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    batch_size: int = 32
    image_size: int = 224
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    pretrained: bool
    train_mode: str
    lr: float
    epochs: int = 15
    weight_decay: float = 1e-4
    momentum: float = 0.9
