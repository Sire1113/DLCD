from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_size: int
    val_size: int
    test_size: int


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[Path],
        labels: list[int],
        class_names: list[str],
        transform=None,
    ) -> None:
        self.samples = samples
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        label = self.labels[index]
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            raise OSError(f"Failed to open image: {path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path)


def discover_samples(data_root: Path) -> tuple[list[Path], list[int], list[str]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    class_dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found under: {data_root}")

    samples: list[Path] = []
    labels: list[int] = []
    class_names: list[str] = []

    for label, class_dir in enumerate(class_dirs):
        class_names.append(class_dir.name)
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append(image_path)
                labels.append(label)

    if not samples:
        raise ValueError(f"No image files found under: {data_root}")

    return samples, labels, class_names


def _class_split_sizes(total: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        return total, 0, 0

    test_count = max(1, int(round(total * test_ratio)))
    val_count = max(1, int(round(total * val_ratio)))

    while test_count + val_count >= total:
        if test_count > val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    train_count = total - test_count - val_count
    if train_count <= 0:
        if test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        train_count = total - test_count - val_count

    return train_count, val_count, test_count


def stratified_split(
    labels: list[int],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped_indices[label].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_indices in grouped_indices.values():
        rng.shuffle(class_indices)
        train_count, val_count, test_count = _class_split_sizes(len(class_indices), val_ratio, test_ratio)
        test_indices.extend(class_indices[:test_count])
        val_indices.extend(class_indices[test_count : test_count + val_count])
        train_indices.extend(class_indices[test_count + val_count : test_count + val_count + train_count])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform, eval_transform


def create_data_bundle(
    data_root: Path,
    *,
    batch_size: int,
    image_size: int,
    val_ratio: float,
    test_ratio: float,
    num_workers: int,
    seed: int,
) -> DataBundle:
    samples, labels, class_names = discover_samples(data_root)
    train_indices, val_indices, test_indices = stratified_split(
        labels,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_transform, val_transform, test_transform = build_transforms(image_size)
    train_dataset = ImageClassificationDataset(samples, labels, class_names, transform=train_transform)
    val_dataset = ImageClassificationDataset(samples, labels, class_names, transform=val_transform)
    test_dataset = ImageClassificationDataset(samples, labels, class_names, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        train_size=len(train_indices),
        val_size=len(val_indices),
        test_size=len(test_indices),
    )
