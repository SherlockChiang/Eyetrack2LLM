from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .transitions import TransitionEvents, aggregate_transitions


@dataclass(frozen=True)
class ReliabilityResult:
    value: float
    raw_half_correlation: float
    n_subjects: int
    n_features: int
    n_repeats_valid: int
    repeat_values: np.ndarray
    raw_repeat_values: np.ndarray | None = None
    feature_counts: np.ndarray | None = None
    half_sizes: tuple[int, int] | None = None


@dataclass(frozen=True)
class ExactReliabilityResult:
    value: float
    raw_half_correlation: float
    n_subjects: int
    n_features: int
    n_partitions: int
    n_partitions_valid: int
    raw_values: np.ndarray
    corrected_values: np.ndarray
    feature_counts: np.ndarray


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def exact_four_subject_split_half(
    events: TransitionEvents,
    *,
    text_id: str,
    event_type: str,
    n_words: int,
    normalize: str = "row",
) -> ExactReliabilityResult:
    """Evaluate the three unique unordered 2-vs-2 partitions of four subjects."""
    text_id = str(text_id)
    subjects = np.unique(events.subject_id[events.text_id == text_id])
    if len(subjects) != 4:
        return ExactReliabilityResult(
            float("nan"), float("nan"), len(subjects), 0, 0, 0,
            np.empty(0), np.empty(0), np.empty(0, dtype=np.int64),
        )

    partitions = []
    for first_indices in combinations(range(4), 2):
        if 0 not in first_indices:
            continue
        first = subjects[list(first_indices)]
        second = subjects[[index for index in range(4) if index not in first_indices]]
        partitions.append((first, second))

    raw_values = []
    corrected_values = []
    feature_counts = []
    for halves in partitions:
        matrices = [
            aggregate_transitions(
                events,
                n_words={text_id: n_words},
                subjects=half,
                normalize=normalize,
                event_types=(event_type,),
            )[(text_id, event_type)]
            for half in halves
        ]
        common = matrices[0].mask & matrices[1].mask
        correlation = _correlation(
            matrices[0].probability[common], matrices[1].probability[common]
        )
        if not np.isfinite(correlation):
            continue
        raw_values.append(correlation)
        feature_counts.append(int(common.sum()))
        denominator = 1 + correlation
        corrected = 2 * correlation / denominator if denominator != 0 else -1.0
        corrected_values.append(float(np.clip(corrected, -1.0, 1.0)))

    return ExactReliabilityResult(
        value=float(np.median(corrected_values)) if corrected_values else float("nan"),
        raw_half_correlation=float(np.median(raw_values)) if raw_values else float("nan"),
        n_subjects=4,
        n_features=int(np.median(feature_counts)) if feature_counts else 0,
        n_partitions=len(partitions),
        n_partitions_valid=len(raw_values),
        raw_values=np.asarray(raw_values, dtype=np.float64),
        corrected_values=np.asarray(corrected_values, dtype=np.float64),
        feature_counts=np.asarray(feature_counts, dtype=np.int64),
    )


def split_half_reliability(
    events: TransitionEvents,
    *,
    text_id: str,
    event_type: str,
    n_words: int,
    repeats: int = 100,
    seed: int | None = 0,
    normalize: str = "row",
    correction: str = "spearman_brown",
    min_subjects: int = 4,
) -> ReliabilityResult:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if correction not in {"spearman_brown", "none"}:
        raise ValueError(f"Unsupported correction: {correction}")

    text_id = str(text_id)
    subjects = np.unique(events.subject_id[events.text_id == text_id])
    if len(subjects) < min_subjects:
        return ReliabilityResult(
            value=float("nan"),
            raw_half_correlation=float("nan"),
            n_subjects=len(subjects),
            n_features=0,
            n_repeats_valid=0,
            repeat_values=np.empty(0, dtype=np.float64),
        )

    event_mask = (events.text_id == text_id) & (events.event_type == event_type)
    counts = np.zeros((len(subjects), n_words, n_words), dtype=np.float64)
    subject_indices = {subject: index for index, subject in enumerate(subjects)}
    for subject, index in subject_indices.items():
        selected = event_mask & (events.subject_id == subject)
        np.add.at(
            counts[index],
            (events.src_word[selected], events.dst_word[selected]),
            events.weight[selected],
        )

    return split_half_from_counts(
        counts, repeats=repeats, seed=seed, normalize=normalize,
        correction=correction, min_subjects=min_subjects,
    )


def split_half_from_counts(
    counts: np.ndarray,
    *,
    repeats: int = 100,
    seed: int | None = 0,
    normalize: str = "row",
    correction: str = "spearman_brown",
    min_subjects: int = 4,
) -> ReliabilityResult:
    """Split-half reliability from a precomputed subject-by-edge count array."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 3 or counts.shape[1] != counts.shape[2]:
        raise ValueError("counts must have shape (subjects, words, words)")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if correction not in {"spearman_brown", "none"}:
        raise ValueError(f"Unsupported correction: {correction}")
    if normalize not in {"row", "global", "none"}:
        raise ValueError(f"Unsupported normalization: {normalize}")
    n_subjects = counts.shape[0]
    if n_subjects < min_subjects:
        return ReliabilityResult(
            float("nan"), float("nan"), n_subjects, 0, 0, np.empty(0),
            np.empty(0), np.empty(0, dtype=np.int64), None,
        )

    rng = np.random.default_rng(seed)
    raw_values: list[float] = []
    corrected_values: list[float] = []
    feature_counts: list[int] = []
    for _ in range(repeats):
        shuffled = rng.permutation(n_subjects)
        split = len(shuffled) // 2
        halves = (shuffled[:split], shuffled[split:])
        matrices = []
        masks = []
        for half in halves:
            count = counts[half].sum(axis=0)
            if normalize == "row":
                totals = count.sum(axis=1, keepdims=True)
                valid_rows = totals[:, 0] > 0
                probability = np.zeros_like(count)
                probability[valid_rows] = count[valid_rows] / totals[valid_rows]
                mask = np.broadcast_to(valid_rows[:, None], count.shape)
            elif normalize == "global":
                total = count.sum()
                probability = count / total if total else np.zeros_like(count)
                mask = np.full(count.shape, total > 0)
            else:
                probability = count
                mask = np.ones(count.shape, dtype=bool)
            matrices.append(probability)
            masks.append(mask)

        common = masks[0] & masks[1]
        first = matrices[0][common]
        second = matrices[1][common]
        correlation = _correlation(first, second)
        if not np.isfinite(correlation):
            continue
        raw_values.append(correlation)
        feature_counts.append(int(common.sum()))
        if correction == "spearman_brown":
            denominator = 1 + correlation
            corrected = 2 * correlation / denominator if denominator != 0 else -1.0
        else:
            corrected = correlation
        corrected_values.append(float(np.clip(corrected, -1.0, 1.0)))

    if not corrected_values:
        value = raw = float("nan")
        n_features = 0
    else:
        value = float(np.median(corrected_values))
        raw = float(np.median(raw_values))
        n_features = int(np.median(feature_counts))
    return ReliabilityResult(
        value=value,
        raw_half_correlation=raw,
        n_subjects=n_subjects,
        n_features=n_features,
        n_repeats_valid=len(corrected_values),
        repeat_values=np.asarray(corrected_values, dtype=np.float64),
        raw_repeat_values=np.asarray(raw_values, dtype=np.float64),
        feature_counts=np.asarray(feature_counts, dtype=np.int64),
        half_sizes=(n_subjects // 2, n_subjects - n_subjects // 2),
    )
