from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .baseline import PairDesign, fit_baseline, residual_vector


METHODS = ("raw", "correct", "misspecified")


@dataclass(frozen=True)
class ResidualSimulationConfig:
    subjects: tuple[int, ...] = (4, 12, 42, 84)
    latent_effects: tuple[float, ...] = (0.0, 0.55)
    concentrations: tuple[float, ...] = (120.0, 8.0)
    replicates: int = 80
    seed: int = 20260711
    n_sources: int = 30
    n_destinations: int = 6
    events_per_subject_source: int = 24
    l2: float = 0.25


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _design(features: np.ndarray, n_sources: int, n_destinations: int) -> PairDesign:
    sources = np.repeat(np.arange(n_sources), n_destinations)
    destinations = np.tile(np.arange(n_destinations), n_sources)
    return PairDesign(
        features=np.asarray(features, float), text_id=sources.astype(str), src_word=sources,
        dst_word=destinations, group_start=np.arange(0, len(sources) + 1, n_destinations, dtype=np.int64),
        feature_names=tuple(f"x{i + 1}" for i in range(features.shape[1])),
    )


def _crossfit_residuals(
    design: PairDesign, counts: np.ndarray, columns: np.ndarray, fold: np.ndarray, l2: float
) -> np.ndarray:
    result = np.empty(len(counts), float)
    selected = _design(design.features[:, columns], len(design.group_start) - 1, int(np.diff(design.group_start)[0]))
    for heldout in (0, 1):
        train_sources = np.flatnonzero(fold != heldout)
        test_sources = np.flatnonzero(fold == heldout)
        train_mask = np.isin(selected.src_word, train_sources)
        test_mask = np.isin(selected.src_word, test_sources)
        model = fit_baseline(selected.subset(train_mask), counts[train_mask], l2=l2)
        target = selected.subset(test_mask)
        residual, _ = residual_vector(counts[test_mask], model.predict(target), target.group_start)
        result[test_mask] = residual
    return result


def _generate_replicate(
    rng: np.random.Generator, config: ResidualSimulationConfig, subjects: int,
    effect: float, concentration: float,
) -> dict[str, dict[str, float]]:
    shape = (config.n_sources, config.n_destinations)
    x1, x2, z = (rng.normal(size=shape) for _ in range(3))
    for value in (x1, x2, z):
        value -= value.mean(axis=1, keepdims=True)
        value /= value.std()
    features = np.column_stack((x1.ravel(), x2.ravel()))
    design = _design(features, *shape)
    logits = 0.8 * x1 - 0.9 * x2 + effect * z
    logits -= logits.max(axis=1, keepdims=True)
    mean_probability = np.exp(logits)
    mean_probability /= mean_probability.sum(axis=1, keepdims=True)

    subject_counts = np.empty((subjects, features.shape[0]), dtype=np.int64)
    for subject in range(subjects):
        rows = []
        for probability in mean_probability:
            individual_probability = rng.dirichlet(concentration * probability)
            rows.append(rng.multinomial(config.events_per_subject_source, individual_probability))
        subject_counts[subject] = np.concatenate(rows)
    shuffled = rng.permutation(subjects)
    halves = [subject_counts[index].sum(axis=0) for index in np.array_split(shuffled, 2)]
    fold = rng.integers(0, 2, size=config.n_sources)
    if np.all(fold == fold[0]):
        fold[: config.n_sources // 2] = 1 - fold[0]

    estimates: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    for counts in halves:
        totals = np.repeat(np.add.reduceat(counts, design.group_start[:-1]), config.n_destinations)
        estimates["raw"].append(counts / totals - 1.0 / config.n_destinations)
        estimates["correct"].append(_crossfit_residuals(design, counts, np.array([True, True]), fold, config.l2))
        estimates["misspecified"].append(_crossfit_residuals(design, counts, np.array([True, False]), fold, config.l2))

    latent = z.ravel()
    return {
        method: {
            "latent_recovery_correlation": _correlation(np.mean(values, axis=0), latent),
            "split_half_residual_reliability": _correlation(values[0], values[1]),
        }
        for method, values in estimates.items()
    }


def _summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(row["subjects"], row["latent_effect"], row["concentration"], row["method"]) for row in records})
    result = []
    for subjects, effect, concentration, method in keys:
        selected = [row for row in records if (row["subjects"], row["latent_effect"], row["concentration"], row["method"]) == (subjects, effect, concentration, method)]
        item: dict[str, object] = {
            "subjects": subjects, "latent_effect": effect, "concentration": concentration,
            "overdispersion": "low" if concentration >= 50 else "high", "method": method,
            "replicates": len(selected),
        }
        for metric in ("latent_recovery_correlation", "split_half_residual_reliability"):
            values = np.asarray([row[metric] for row in selected], float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            item[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        recovery = np.asarray([row["latent_recovery_correlation"] for row in selected], float)
        item["null_abs_correlation_gt_0_2"] = float(np.mean(np.abs(recovery) > 0.2)) if effect == 0 else None
        result.append(item)
    return result


def run_residual_recovery_simulation(config: ResidualSimulationConfig = ResidualSimulationConfig()) -> dict[str, object]:
    """Run a deterministic grid; each row is one independently generated replicate."""
    if any(subjects < 4 or subjects % 2 for subjects in config.subjects):
        raise ValueError("subjects must be even and at least four")
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, object]] = []
    for subjects in config.subjects:
        for effect in config.latent_effects:
            for concentration in config.concentrations:
                for replicate in range(config.replicates):
                    metrics = _generate_replicate(rng, config, subjects, effect, concentration)
                    for method, values in metrics.items():
                        records.append({"subjects": subjects, "latent_effect": effect, "concentration": concentration,
                                        "overdispersion": "low" if concentration >= 50 else "high",
                                        "replicate": replicate, "method": method, **values})
    return {
        "status": "complete", "simulation_unit": "replicate", "config": asdict(config),
        "data_generating_process": {
            "nuisance_logits": "0.8*x1 - 0.9*x2", "latent_term": "latent_effect*z",
            "counts": "subject-level Dirichlet-multinomial",
            "correct_fit": "independent half-specific, source-cross-fitted multinomial using x1+x2",
            "misspecified_fit": "independent half-specific, source-cross-fitted multinomial omitting x2",
            "raw": "row proportion minus uniform probability",
        },
        "summary": _summary(records), "replicate_results": records,
    }


def summary_csv_rows(result: dict[str, object]) -> Iterable[dict[str, object]]:
    return result["summary"]
