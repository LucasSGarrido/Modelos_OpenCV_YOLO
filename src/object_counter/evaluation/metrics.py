from __future__ import annotations

from collections.abc import Iterable, Mapping


def normalize_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {str(label): int(value) for label, value in counts.items() if int(value) != 0}


def absolute_count_error(y_true: Mapping[str, int], y_pred: Mapping[str, int]) -> int:
    true_counts = normalize_counts(y_true)
    pred_counts = normalize_counts(y_pred)
    labels = set(true_counts) | set(pred_counts)
    return sum(abs(true_counts.get(label, 0) - pred_counts.get(label, 0)) for label in labels)


def per_class_count_errors(
    y_true: Mapping[str, int], y_pred: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    true_counts = normalize_counts(y_true)
    pred_counts = normalize_counts(y_pred)
    labels = sorted(set(true_counts) | set(pred_counts))
    return {
        label: {
            "real": true_counts.get(label, 0),
            "previsto": pred_counts.get(label, 0),
            "erro_absoluto": abs(true_counts.get(label, 0) - pred_counts.get(label, 0)),
        }
        for label in labels
    }


def mean_absolute_count_error(
    y_true_samples: Iterable[Mapping[str, int]],
    y_pred_samples: Iterable[Mapping[str, int]],
) -> float:
    errors = [
        absolute_count_error(y_true, y_pred)
        for y_true, y_pred in zip(y_true_samples, y_pred_samples, strict=False)
    ]
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def mean_absolute_percentage_error(
    y_true_samples: Iterable[Mapping[str, int]],
    y_pred_samples: Iterable[Mapping[str, int]],
) -> float:
    percentages: list[float] = []

    for y_true, y_pred in zip(y_true_samples, y_pred_samples, strict=False):
        true_total = sum(normalize_counts(y_true).values())
        if true_total == 0:
            continue
        percentages.append(absolute_count_error(y_true, y_pred) / true_total)

    if not percentages:
        return 0.0
    return sum(percentages) / len(percentages)


def exact_count_match_rate(
    y_true_samples: Iterable[Mapping[str, int]],
    y_pred_samples: Iterable[Mapping[str, int]],
) -> float:
    pairs = list(zip(y_true_samples, y_pred_samples, strict=False))
    if not pairs:
        return 0.0

    exact_matches = sum(
        normalize_counts(y_true) == normalize_counts(y_pred) for y_true, y_pred in pairs
    )
    return exact_matches / len(pairs)
