from __future__ import annotations

from collections.abc import Sequence


def accuracy_score(predictions: Sequence[int], targets: Sequence[int]) -> float:
    if len(predictions) == 0:
        return 0.0
    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    return correct / len(predictions)


def confusion_matrix(
    predictions: Sequence[int],
    targets: Sequence[int],
    num_classes: int = 2,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for prediction, target in zip(predictions, targets):
        matrix[target][prediction] += 1
    return matrix
