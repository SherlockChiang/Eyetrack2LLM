from __future__ import annotations

import random
from collections.abc import Callable, Iterable

import numpy as np

from .baseline import PairDesign, fit_baseline, residual_vector
from .transfer import sentence_folds


def balanced_subject_subsets(subjects: Iterable[str], k: int, repeats: int, seed: int) -> list[tuple[str, ...]]:
    """Draw reproducible subsets while keeping subject inclusion counts nearly equal."""
    subjects = tuple(sorted(map(str, subjects)))
    if not 1 <= k <= len(subjects):
        raise ValueError("k must be between one and the number of subjects")
    if k == len(subjects):
        return [subjects]
    rng = random.Random(seed)
    usage = {subject: 0 for subject in subjects}
    result = []
    for _ in range(repeats):
        jitter = {subject: rng.random() for subject in subjects}
        chosen = tuple(sorted(subjects, key=lambda subject: (usage[subject], jitter[subject]))[:k])
        result.append(tuple(sorted(chosen)))
        for subject in chosen:
            usage[subject] += 1
    return result


def permute_destinations_within_source(
    counts: np.ndarray, design: PairDesign, rng: np.random.Generator
) -> np.ndarray:
    """Permute destination labels independently within every conditional risk set."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or len(counts) != len(design.features):
        raise ValueError("counts must match the PairDesign pair count")
    permuted = counts.copy()
    for start, stop in zip(design.group_start[:-1], design.group_start[1:], strict=True):
        permuted[start:stop] = counts[start:stop][rng.permutation(stop - start)]
    return permuted


def crossfit_raw_residuals(
    design: PairDesign,
    counts: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 20260711,
    min_exposure: int = 10,
    fitter: Callable = fit_baseline,
) -> tuple[dict[str, dict[str, np.ndarray]], list[tuple[list[str], list[str]]]]:
    """Fit only on training texts and return unscaled, unclipped held-out residuals."""
    targets = {}
    folds = sentence_folds(design.text_id, n_folds, seed)
    for train, test in folds:
        train_mask = np.isin(design.text_id, train)
        model = fitter(design.subset(train_mask), np.asarray(counts)[train_mask], l2=1.0)
        for text in test:
            mask = design.text_id == text
            target = design.subset(mask)
            residual, exposure = residual_vector(counts[mask], model.predict(target), target.group_start)
            targets[text] = {
                "residual": residual,
                "exposure": exposure,
                "reliable": (exposure >= min_exposure) & np.isfinite(residual),
                "src": target.src_word,
                "dst": target.dst_word,
            }
    return targets, folds


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x, y, weights = x[valid], y[valid], weights[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    weights /= weights.sum()
    dx, dy = x - np.sum(weights * x), y - np.sum(weights * y)
    denominator = np.sqrt(np.sum(weights * dx * dx) * np.sum(weights * dy * dy))
    return float(np.sum(weights * dx * dy) / denominator) if denominator > 0 else None


def source_equal_metrics(x: np.ndarray, y: np.ndarray, sources: np.ndarray) -> dict[str, object]:
    """Report ordinary edge weighting and two defensible source-equal summaries."""
    x, y, sources = np.asarray(x), np.asarray(y), np.asarray(sources)
    unique, sizes = np.unique(sources, return_counts=True)
    weights = np.zeros(len(sources), float)
    per_source = []
    for source, size in zip(unique, sizes, strict=True):
        selected = sources == source
        weights[selected] = 1.0 / size
        value = weighted_correlation(x[selected], y[selected])
        if value is not None:
            per_source.append(value)
    fisher = float(np.tanh(np.mean(np.arctanh(np.clip(per_source, -1 + 1e-7, 1 - 1e-7))))) if per_source else None
    return {
        "edge_weighted": weighted_correlation(x, y),
        "source_equal_flatten": weighted_correlation(x, y, weights),
        "per_source_fisher_equal": fisher,
        "n_edges": int(len(x)),
        "n_sources": int(len(unique)),
        "n_sources_with_defined_correlation": len(per_source),
    }


def paired_residual_metrics(first: dict[str, dict[str, np.ndarray]], second: dict[str, dict[str, np.ndarray]]) -> dict[str, object]:
    per_text = {}
    for text in sorted(first, key=lambda value: int(value.split(":")[-1])):
        a, b = first[text], second[text]
        if not (np.array_equal(a["src"], b["src"]) and np.array_equal(a["dst"], b["dst"])):
            raise ValueError(f"Candidate mismatch for {text}")
        eligible = a["reliable"] & b["reliable"] & np.isfinite(a["residual"]) & np.isfinite(b["residual"])
        per_text[text] = source_equal_metrics(a["residual"][eligible], b["residual"][eligible], a["src"][eligible])
    summary = {}
    for metric in ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal"):
        all_values = [item[metric] for item in per_text.values()]
        values = [value for value in all_values if value is not None]
        summary[metric] = {
            "valid_texts": len(values),
            "undefined_texts": len(all_values) - len(values),
            "undefined_proportion": (len(all_values) - len(values)) / len(all_values) if all_values else None,
            "negative_proportion": float(np.mean(np.asarray(values) < 0)) if values else None,
            "median": float(np.median(values)) if values else None,
            "q25": float(np.quantile(values, .25)) if values else None,
            "q75": float(np.quantile(values, .75)) if values else None,
        }
    return {
        "per_text": per_text,
        "text_summary": summary,
        "edge_source_summary": {
            "median_eligible_edges_per_text": float(np.median([x["n_edges"] for x in per_text.values()])),
            "iqr_eligible_edges_per_text": np.quantile([x["n_edges"] for x in per_text.values()], [.25, .75]).tolist(),
            "median_eligible_sources_per_text": float(np.median([x["n_sources"] for x in per_text.values()])),
            "iqr_eligible_sources_per_text": np.quantile([x["n_sources"] for x in per_text.values()], [.25, .75]).tolist(),
        },
    }
