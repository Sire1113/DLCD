from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import DataConfig, ExperimentConfig
from .data import create_data_bundle
from .engine import evaluate, fit_model
from .experiments import build_default_experiments, build_lr_sweep
from .models import build_resnet18, trainable_parameter_count
from .visualize import plot_confusion_matrix, plot_training_history, save_prediction_examples


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_experiment(
    experiment: ExperimentConfig,
    data_config: DataConfig,
    output_dir: Path,
    device: torch.device,
    bundle=None,
) -> dict[str, object]:
    if bundle is None:
        bundle = create_data_bundle(
            data_config.data_root,
            batch_size=data_config.batch_size,
            image_size=data_config.image_size,
            val_ratio=data_config.val_ratio,
            test_ratio=data_config.test_ratio,
            num_workers=data_config.num_workers,
            seed=data_config.seed,
        )

    model = build_resnet18(pretrained=experiment.pretrained, train_mode=experiment.train_mode)
    model.to(device)
    artifact_dir = output_dir / experiment.name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    training_result = fit_model(
        model,
        bundle.train_loader,
        bundle.val_loader,
        epochs=experiment.epochs,
        lr=experiment.lr,
        momentum=experiment.momentum,
        weight_decay=experiment.weight_decay,
        device=device,
        output_dir=artifact_dir,
        use_amp=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
    test_result = evaluate(training_result.model, bundle.test_loader, criterion, device)

    plot_training_history(training_result.history, artifact_dir / "history.png")
    plot_confusion_matrix(test_result.confusion, bundle.class_names, artifact_dir / "confusion_matrix.png")
    save_prediction_examples(test_result, bundle.class_names, artifact_dir)

    summary: dict[str, object] = {
        "experiment": experiment.name,
        "pretrained": experiment.pretrained,
        "train_mode": experiment.train_mode,
        "lr": experiment.lr,
        "epochs": experiment.epochs,
        "trainable_parameters": trainable_parameter_count(training_result.model),
        "best_epoch": training_result.best_epoch,
        "best_val_accuracy": training_result.best_val_accuracy,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "confusion_matrix": test_result.confusion,
        "class_names": bundle.class_names,
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "test_size": bundle.test_size,
        "checkpoint": str(training_result.best_checkpoint) if training_result.best_checkpoint is not None else None,
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def with_epoch_override(experiment: ExperimentConfig, epochs: int | None) -> ExperimentConfig:
    if epochs is None:
        return experiment
    return replace(experiment, epochs=epochs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DLCD binary image classification experiments")

    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument("--data-root", type=Path, required=True, help="Root folder containing class subfolders")
    shared_parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Directory for experiment outputs")
    shared_parser.add_argument("--batch-size", type=int, default=32)
    shared_parser.add_argument("--image-size", type=int, default=224)
    shared_parser.add_argument("--val-ratio", type=float, default=0.15)
    shared_parser.add_argument("--test-ratio", type=float, default=0.15)
    shared_parser.add_argument("--num-workers", type=int, default=4)
    shared_parser.add_argument("--seed", type=int, default=42)
    shared_parser.add_argument("--device", type=str, default="auto")
    shared_parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for each experiment")

    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Run the three required ResNet18 comparison experiments",
        parents=[shared_parser],
    )
    compare_parser.add_argument("--lr-sweep", action="store_true", help="Also run a learning-rate sweep for fine-tuning")

    train_parser = subparsers.add_parser("train", help="Run a single experiment by name", parents=[shared_parser])
    train_parser.add_argument("--experiment", required=True, help="Experiment name, e.g. scratch_resnet18")

    return parser


def find_experiment(name: str) -> ExperimentConfig:
    for experiment in build_default_experiments():
        if experiment.name == name:
            return experiment
    for experiment in build_lr_sweep():
        if experiment.name == name:
            return experiment
    raise ValueError(f"Unknown experiment: {name}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = resolve_device(args.device)
    data_config = DataConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    summaries: list[dict[str, object]] = []
    if args.command == "compare":
        bundle = create_data_bundle(
            data_config.data_root,
            batch_size=data_config.batch_size,
            image_size=data_config.image_size,
            val_ratio=data_config.val_ratio,
            test_ratio=data_config.test_ratio,
            num_workers=data_config.num_workers,
            seed=data_config.seed,
        )
        experiments = [with_epoch_override(experiment, args.epochs) for experiment in build_default_experiments()]
        if args.lr_sweep:
            experiments.extend(
                with_epoch_override(experiment, args.epochs)
                for experiment in build_lr_sweep(base_epochs=args.epochs or 20)
            )
        for experiment in experiments:
            summary = run_experiment(experiment, data_config, args.output_dir, device, bundle=bundle)
            summaries.append(summary)
        (args.output_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    if args.command == "train":
        experiment = with_epoch_override(find_experiment(args.experiment), args.epochs)
        summary = run_experiment(experiment, data_config, args.output_dir, device)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    parser.error("Unsupported command")
