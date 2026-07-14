from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .stability import source_equal_metrics, weighted_correlation


CATEGORIES = ("adjacent", "near_skip", "far_same_line")
METRICS = ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal")
RESIDUAL_TYPES = ("pearson", "deviance", "raw_deviation")


def separation_category(src: int, dst: int) -> str:
    """Classify a forward token separation, not visual angle or saccade amplitude."""
    distance = int(dst) - int(src)
    if distance < 1:
        raise ValueError("target-selection categories require a forward edge")
    if distance == 1:
        return "adjacent"
    if distance <= 3:
        return "near_skip"
    return "far_same_line"


def category_masks(src: np.ndarray, dst: np.ndarray) -> dict[str, np.ndarray]:
    src, dst = np.asarray(src), np.asarray(dst)
    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError("src and dst must be matching one-dimensional arrays")
    distance = dst - src
    if np.any(distance < 1):
        raise ValueError("candidate universe contains a non-forward edge")
    masks = {
        "adjacent": distance == 1,
        "near_skip": (distance >= 2) & (distance <= 3),
        "far_same_line": distance >= 4,
    }
    if not np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1):
        raise ValueError("categories do not partition the candidate universe")
    return masks


def deterministic_subject_splits(subjects, repeats: int, seed: int):
    subjects = np.asarray(sorted(map(str, subjects)))
    if len(subjects) < 2 or repeats < 1:
        raise ValueError("at least two subjects and one repeat are required")
    rng = np.random.default_rng(seed)
    split = len(subjects) // 2
    return [
        (tuple(sorted((shuffled := subjects[rng.permutation(len(subjects))])[:split].tolist())),
         tuple(sorted(shuffled[split:].tolist())))
        for _ in range(repeats)
    ]


def categorized_paired_metrics(first: Mapping, second: Mapping) -> dict[str, object]:
    """Compute category metrics per text, retaining undefined source correlations."""
    per_category = {category: {} for category in CATEGORIES}
    for text in sorted(first, key=str):
        a, b = first[text], second[text]
        if not (np.array_equal(a["src"], b["src"]) and np.array_equal(a["dst"], b["dst"])):
            raise ValueError(f"Candidate mismatch for {text}")
        eligible = np.asarray(a["reliable"]) & np.asarray(b["reliable"])
        for category, selected in category_masks(a["src"], a["dst"]).items():
            use = eligible & selected
            per_category[category][text] = source_equal_metrics(
                np.asarray(a["residual"])[use], np.asarray(b["residual"])[use],
                np.asarray(a["src"])[use],
            )
    result = {}
    for category, per_text in per_category.items():
        summary = {}
        for metric in METRICS:
            values = [row[metric] for row in per_text.values() if row[metric] is not None]
            summary[metric] = {
                "median": float(np.median(values)) if values else None,
                "q25": float(np.quantile(values, .25)) if values else None,
                "q75": float(np.quantile(values, .75)) if values else None,
                "valid_texts": len(values),
            }
        result[category] = {"per_text": per_text, "text_summary": summary}
    return result


def residual_identity(first: Mapping, second: Mapping) -> dict[str, object]:
    result = {}
    for category in CATEGORIES:
        per_text, left, right = {}, [], []
        for text in sorted(first, key=str):
            a, b = first[text], second[text]
            selected = (np.asarray(a["reliable"]) & np.asarray(b["reliable"])
                        & category_masks(a["src"], a["dst"])[category])
            value = weighted_correlation(np.asarray(a["residual"])[selected],
                                         np.asarray(b["residual"])[selected])
            per_text[text] = value
            left.extend(np.asarray(a["residual"])[selected].tolist())
            right.extend(np.asarray(b["residual"])[selected].tolist())
        valid = [value for value in per_text.values() if value is not None]
        result[category] = {
            "text_equal_median": float(np.median(valid)) if valid else None,
            "text_equal_q25": float(np.quantile(valid, .25)) if valid else None,
            "text_equal_q75": float(np.quantile(valid, .75)) if valid else None,
            "valid_texts": len(valid), "n_edges": len(left),
            "edge_overall_descriptive": weighted_correlation(np.asarray(left), np.asarray(right)),
            "per_text": per_text,
        }
    return result


def distribution(values) -> dict[str, float | list[float] | None]:
    values = np.asarray([value for value in values if value is not None], float)
    if not len(values):
        return {"median": None, "q25": None, "q75": None, "range": None}
    return {"median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)),
            "range": [float(values.min()), float(values.max())]}


def residual_arrays(counts, probability, group_start, min_exposure=5):
    """Construct conditional-multinomial cell diagnostics without dropping candidates."""
    counts, probability = np.asarray(counts, float), np.asarray(probability, float)
    lengths = np.diff(np.asarray(group_start))
    exposure = np.repeat(np.add.reduceat(counts, group_start[:-1]), lengths)
    expected = exposure * probability
    deviation = counts - expected
    variance = expected * (1.0 - probability)
    structural = np.repeat(lengths, lengths) >= 2
    pearson = np.divide(deviation, np.sqrt(variance), out=np.full_like(deviation, np.nan),
                        where=structural & (variance > 0))
    log_term = np.zeros_like(counts)
    positive = counts > 0
    valid = positive & (expected > 0)
    log_term[valid] = counts[valid] * np.log(counts[valid] / expected[valid])
    deviance_sq = 2.0 * (log_term - deviation)
    deviance = np.sign(deviation) * np.sqrt(np.maximum(deviance_sq, 0.0))
    deviance[(expected <= 0) & positive] = np.nan
    return {"pearson": pearson, "deviance": deviance, "raw_deviation": deviation,
            "probability": probability, "expected": expected, "exposure": exposure,
            "risk_size": np.repeat(lengths, lengths),
            "reliable": (exposure >= min_exposure) & structural}


def thin_subject_categories(counts, src, dst, rng, reference="far_same_line"):
    """Binomial-thin each subject/category to the reference category's mean mass."""
    counts = np.asarray(counts)
    masks = category_masks(src, dst)
    masses = {name: float(counts[mask].sum()) for name, mask in masks.items()}
    target = masses[reference]
    thinned = counts.copy()
    probabilities = {}
    for category, mask in masks.items():
        probability = min(1.0, target / masses[category]) if masses[category] > 0 else 0.0
        probabilities[category] = probability
        if category != reference:
            thinned[mask] = rng.binomial(np.rint(counts[mask]).astype(int), probability)
    return thinned, probabilities
