from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from .metrics import accuracy_score, confusion_matrix


@dataclass
class PredictionRecord:
    image: torch.Tensor
    target: int
    pred: int
    path: str


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    confusion: list[list[int]]
    predictions: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    records: list[PredictionRecord] = field(default_factory=list)


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


@dataclass
class TrainingResult:
    model: nn.Module
    history: History
    best_epoch: int
    best_val_accuracy: float
    best_checkpoint: Path | None = None


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found.")
    return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)


def _normalize_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str | None,
    *,
    use_amp: bool = True,
) -> tuple[float, float]:
    device = _normalize_device(device)
    model.train()
    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    for images, targets, _paths in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    average_loss = running_loss / max(1, len(loader.dataset))
    accuracy = accuracy_score(all_predictions, all_targets)
    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str | None,
) -> EvaluationResult:
    device = _normalize_device(device)
    model.eval()
    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []
    records: list[PredictionRecord] = []

    for images, targets, paths in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        cpu_images = images.detach().cpu()
        cpu_targets = targets.detach().cpu().tolist()
        cpu_predictions = predictions.detach().cpu().tolist()
        cpu_paths = list(paths)

        all_predictions.extend(cpu_predictions)
        all_targets.extend(cpu_targets)

        for index, path in enumerate(cpu_paths):
            records.append(
                PredictionRecord(
                    image=cpu_images[index],
                    target=cpu_targets[index],
                    pred=cpu_predictions[index],
                    path=str(path),
                )
            )

    average_loss = running_loss / max(1, len(loader.dataset))
    accuracy = accuracy_score(all_predictions, all_targets)
    matrix = confusion_matrix(all_predictions, all_targets, num_classes=2)
    return EvaluationResult(
        loss=average_loss,
        accuracy=accuracy,
        confusion=matrix,
        predictions=all_predictions,
        targets=all_targets,
        records=records,
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device | str | None,
    output_dir: Path | None = None,
    use_amp: bool = True,
) -> TrainingResult:
    device = _normalize_device(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=lr, momentum=momentum, weight_decay=weight_decay)
    history = History()
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    best_epoch = 0
    best_checkpoint: Path | None = None

    checkpoint_dir = None
    if output_dir is not None:
        checkpoint_dir = output_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
        )
        val_result = evaluate(model, val_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_result.loss)
        history.val_acc.append(val_result.accuracy)

        if val_result.accuracy >= best_val_accuracy:
            best_val_accuracy = val_result.accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_dir is not None:
                best_checkpoint = checkpoint_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "val_accuracy": best_val_accuracy,
                    },
                    best_checkpoint,
                )

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_checkpoint=best_checkpoint,
    )
