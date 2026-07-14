from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def analytic_leave_one_out_mean(values: Sequence[float]) -> np.ndarray:
    """Return every leave-one-out mean without refitting or repeated summation."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional array of length at least two")
    return (array.sum() - array) / (len(array) - 1)


def median_hierarchy(repeats: Sequence[Mapping[str, float]], texts: Sequence[str] | None = None) -> float:
    """Median over texts within repeat, followed by the median over repeats."""
    if not repeats:
        raise ValueError("at least one repeat is required")
    selected = tuple(texts) if texts is not None else tuple(repeats[0])
    if not selected:
        raise ValueError("at least one text is required")
    within = []
    for repeat in repeats:
        values = [repeat[text] for text in selected if text in repeat and repeat[text] is not None]
        if not values:
            raise ValueError("each repeat must have a defined selected text")
        within.append(np.median(values))
    return float(np.median(within))


def leave_one_text_out_median_hierarchy(
    repeats: Sequence[Mapping[str, float]], texts: Sequence[str] | None = None
) -> tuple[float, dict[str, float]]:
    selected = tuple(texts) if texts is not None else tuple(repeats[0])
    full = median_hierarchy(repeats, selected)
    return full, {text: median_hierarchy(repeats, [other for other in selected if other != text])
                  for text in selected}


def most_influential(full: float, estimates: Mapping[str, float]) -> tuple[str, float]:
    """Select the first maximum absolute change in mapping iteration order."""
    if not estimates:
        raise ValueError("at least one estimate is required")
    text = max(estimates, key=lambda key: abs(estimates[key] - full))
    return text, abs(float(estimates[text]) - float(full))


def jackknife_mean_interval(values: Sequence[float]) -> tuple[float, float, float]:
    """Return delete-one jackknife SE and normal 95% CI for a sample mean."""
    array = np.asarray(values, dtype=float)
    loo = analytic_leave_one_out_mean(array)
    se = float(np.sqrt((len(array) - 1) / len(array) * np.sum((loo - loo.mean()) ** 2)))
    mean = float(array.mean())
    return se, mean - 1.959963984540054 * se, mean + 1.959963984540054 * se


def resampled_mean_inference(
    values: Sequence[float], *, repeats: int = 10_000, seed: int = 20260711
) -> tuple[list[float], float]:
    """Deterministic text bootstrap CI and two-sided random sign-flip p-value."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("values must be a nonempty finite one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(array), size=(repeats, len(array)))
    bootstrap = array[indices].mean(axis=1)
    signs = rng.choice((-1, 1), size=(repeats, len(array)))
    observed = abs(float(array.mean()))
    p = (1 + np.count_nonzero(np.abs((signs * array).mean(axis=1)) >= observed)) / (repeats + 1)
    return np.quantile(bootstrap, [.025, .975]).tolist(), float(p)
