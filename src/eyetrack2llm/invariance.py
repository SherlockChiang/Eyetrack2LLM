from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .baseline import PairDesign, fit_baseline, predict_probabilities, residual_vector
from .transfer import sentence_folds


def _texts(design: PairDesign) -> list[str]:
    return sorted(set(map(str, design.text_id)), key=lambda x: int(x.split(":")[-1]))


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)), "min": float(values.min()), "max": float(values.max())}


def text_summaries(design: PairDesign, counts: np.ndarray) -> tuple[list[str], np.ndarray, list[str], dict]:
    """Aggregate design and exposure diagnostics with one row per text."""
    names = ["words", "candidate_pairs", "source_groups", "candidates_per_source",
             "mean_distance", "adjacency_rate", "line_span", "events_per_source",
             "zero_destination_proportion", "destination_concentration", "destination_entropy"]
    rows, texts = [], _texts(design)
    per_text = {}
    for text in texts:
        mask = design.text_id == text
        d, y = design.subset(mask), np.asarray(counts)[mask]
        lengths = np.diff(d.group_start)
        totals = np.add.reduceat(y, d.group_start[:-1])
        probabilities = []
        entropies, concentrations = [], []
        for start, stop in zip(d.group_start[:-1], d.group_start[1:], strict=True):
            group = y[start:stop]; total = group.sum()
            p = group / total if total else np.zeros(len(group))
            probabilities.extend(p)
            entropies.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()) if total else 0.0)
            concentrations.append(float((p * p).sum()) if total else 0.0)
        distances = d.dst_word - d.src_word
        row = [len(set(d.src_word) | set(d.dst_word)), len(y), len(lengths), lengths.mean(),
               distances.mean(), np.mean(distances == 1), float(distances.max()), totals.mean(),
               np.mean(y == 0), np.mean(concentrations), np.mean(entropies)]
        rows.append(row); per_text[text] = dict(zip(names, map(float, row), strict=True))
    matrix = np.asarray(rows)
    return texts, matrix, names, {name: _summary(matrix[:, i]) for i, name in enumerate(names)} | {"per_text": per_text}


def feature_audit(a: PairDesign, b: PairDesign, repeats: int, seed: int) -> list[dict]:
    if a.feature_names != b.feature_names or a.features.shape[1] != b.features.shape[1]:
        raise ValueError("Corpus feature names/columns differ")
    a_texts, b_texts = _texts(a), _texts(b)
    am = np.asarray([a.features[a.text_id == t].mean(0) for t in a_texts])
    bm = np.asarray([b.features[b.text_id == t].mean(0) for t in b_texts])
    rng = np.random.default_rng(seed)
    ia = rng.integers(len(am), size=(repeats, len(am)))
    ib = rng.integers(len(bm), size=(repeats, len(bm)))
    out = []
    for j, name in enumerate(a.feature_names):
        pooled = np.sqrt(((len(am) - 1) * am[:, j].var(ddof=1) + (len(bm) - 1) * bm[:, j].var(ddof=1)) /
                         max(1, len(am) + len(bm) - 2))
        delta = am[:, j].mean() - bm[:, j].mean()
        boot_delta = am[ia, j].mean(1) - bm[ib, j].mean(1)
        boot = boot_delta / pooled if pooled > 0 else np.zeros(repeats)
        out.append({"feature": name, "provo_text_mean": float(am[:, j].mean()),
                    "zuco_text_mean": float(bm[:, j].mean()), "text_equal_smd": float(delta / pooled) if pooled else 0.0,
                    "bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist(),
                    "standardizer": "pooled between-text SD of text-level feature means"})
    return out


def nll(design: PairDesign, counts: np.ndarray, coefficients: np.ndarray | None) -> dict:
    probabilities = (predict_probabilities(design, coefficients) if coefficients is not None else
                     np.repeat(1 / np.diff(design.group_start), np.diff(design.group_start)))
    per_text, observations = {}, 0.0
    for text in _texts(design):
        mask = design.text_id == text; total = float(counts[mask].sum()); observations += total
        per_text[text] = float(-(counts[mask] @ np.log(np.maximum(probabilities[mask], 1e-300))) / total) if total else None
    valid = np.asarray([v for v in per_text.values() if v is not None])
    return {"text_equal_mean_nll": float(valid.mean()), "texts": len(valid), "events": observations,
            "per_text_mean_nll": per_text}


def crossfit_nll(design: PairDesign, counts: np.ndarray, seed: int, folds: int = 5) -> dict:
    per_text = {}
    for train, test in sentence_folds(design.text_id, folds, seed):
        mask = np.isin(design.text_id, train)
        model = fit_baseline(design.subset(mask), counts[mask], l2=1.0, maxiter=1000)
        for text in test:
            selected = design.text_id == text
            per_text[text] = nll(design.subset(selected), counts[selected], model.coefficients)["text_equal_mean_nll"]
    return {"text_equal_mean_nll": float(np.mean(list(per_text.values()))), "texts": len(per_text), "per_text_mean_nll": per_text}


def coefficient_audit(a: PairDesign, ac: np.ndarray, b: PairDesign, bc: np.ndarray,
                      repeats: int, seed: int) -> dict:
    ma, mb = fit_baseline(a, ac, l2=1.0, maxiter=1000), fit_baseline(b, bc, l2=1.0, maxiter=1000)
    ta, tb = _texts(a), _texts(b); rng = np.random.default_rng(seed)
    differences = np.empty((repeats, a.features.shape[1]))
    for r in range(repeats):
        wa = dict(zip(ta, np.bincount(rng.integers(len(ta), size=len(ta)), minlength=len(ta)), strict=True))
        wb = dict(zip(tb, np.bincount(rng.integers(len(tb), size=len(tb)), minlength=len(tb)), strict=True))
        ya = ac * np.asarray([wa[str(t)] for t in a.text_id]); yb = bc * np.asarray([wb[str(t)] for t in b.text_id])
        differences[r] = fit_baseline(a, ya, l2=1.0).coefficients - fit_baseline(b, yb, l2=1.0).coefficients
    rows = [{"feature": name, "provo": float(ma.coefficients[j]), "zuco": float(mb.coefficients[j]),
             "provo_minus_zuco": float(ma.coefficients[j] - mb.coefficients[j]),
             "bootstrap_95_ci": np.quantile(differences[:, j], [.025, .975]).tolist()}
            for j, name in enumerate(a.feature_names)]
    return {"l2": 1.0, "scaling": "shared raw feature units; no corpus standardization", "coefficients": rows,
            "provo_model": ma.coefficients, "zuco_model": mb.coefficients}


def residual_audit(design: PairDesign, counts: np.ndarray, coefficients: np.ndarray) -> dict:
    residual, exposure = residual_vector(counts, predict_probabilities(design, coefficients), design.group_start)
    rows = []
    for text in _texts(design):
        mask = design.text_id == text; value = residual[mask]
        rows.append([value.mean(), value.var(ddof=1), np.mean(np.abs(value) > 2),
                     np.mean(value * value), exposure[mask].mean()])
    names = ["residual_mean", "residual_variance", "absolute_gt_2", "pearson_dispersion", "mean_exposure"]
    return {name: _summary(np.asarray(rows)[:, j]) for j, name in enumerate(names)}


def domain_classification(xa: np.ndarray, xb: np.ndarray, repeats: int, seed: int) -> dict:
    x = np.vstack((xa, xb)); y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), int)
    for label in (0, 1):
        idx = np.flatnonzero(y == label); folds[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % 5

    def score(labels):
        p = np.empty(len(y))
        for fold in range(5):
            train = folds != fold
            mean, scale = x[train].mean(0), x[train].std(0)
            scale[scale == 0] = 1
            train_x = (x[train] - mean) / scale
            test_x = (x[~train] - mean) / scale
            def objective(beta):
                q = expit(beta[0] + train_x @ beta[1:])
                loss = -np.sum(labels[train] * np.log(q + 1e-12) + (1-labels[train]) * np.log(1-q + 1e-12)) + .5 * np.sum(beta[1:] ** 2)
                grad = np.r_[np.sum(q-labels[train]), train_x.T @ (q-labels[train]) + beta[1:]]
                return loss, grad
            fit = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B")
            p[~train] = expit(fit.x[0] + test_x @ fit.x[1:])
        pred = p >= .5
        balanced = .5 * (np.mean(pred[labels == 1]) + np.mean(~pred[labels == 0]))
        pairs = (p[labels == 1, None] > p[labels == 0]).mean() + .5 * (p[labels == 1, None] == p[labels == 0]).mean()
        return float(balanced), float(pairs)
    observed = score(y); null = np.asarray([score(rng.permutation(y)) for _ in range(repeats)])
    return {"unit": "text", "features": "aggregated risk-set geometry, exposure, sparsity, and observed allocation summaries", "folds": 5,
            "standardization": "fold-local training mean and SD, frozen on each held-out fold",
            "balanced_accuracy": observed[0], "auc": observed[1], "label_permutations": repeats,
            "permutation_p_balanced_accuracy": float((1 + np.sum(null[:, 0] >= observed[0])) / (repeats + 1)),
            "permutation_p_auc": float((1 + np.sum(null[:, 1] >= observed[1])) / (repeats + 1))}


def audit_corpora(a: PairDesign, ac: np.ndarray, b: PairDesign, bc: np.ndarray,
                  *, repeats: int = 1000, permutations: int = 500, seed: int = 20260711) -> dict:
    started = time.perf_counter()
    if a.feature_names != b.feature_names or a.design_rank() != b.design_rank():
        raise ValueError("Common-core names/rank do not match")
    at, ax, summary_names, ag = text_summaries(a, ac); bt, bx, _, bg = text_summaries(b, bc)
    coefficients = coefficient_audit(a, ac, b, bc, repeats, seed + 1)
    pa, pb = coefficients.pop("provo_model"), coefficients.pop("zuco_model")
    result = {"status": "complete", "seed": seed, "bootstrap_repeats": repeats,
              "definitions": {"scope": "observable differences in cross-corpus measurement conditions, not strict psychometric invariance",
                "unit": "text; edges are never bootstrap, CV, or classification units",
                "bootstrap": "corpora independently resampled with replacement over texts",
                "risk_set": "common_forward_same_sentence_same_line", "feature_scaling": "shared raw units", "l2": 1.0,
                "limitations": "Two observational corpora cannot identify task, reader, layout, or text-composition causes; summaries do not prove invariance or non-invariance."},
              "design": {"feature_names": list(a.feature_names), "provo_rank": a.design_rank(), "zuco_rank": b.design_rank(),
                         "summary_names": summary_names, "provo_texts": len(at), "zuco_texts": len(bt)},
              "risk_set_geometry_and_exposure": {"provo": ag, "zuco": bg},
              "feature_distribution": feature_audit(a, b, repeats, seed), "nuisance_fit": coefficients,
              "transport_calibration": {
                  "provo": {"within_crossfit": crossfit_nll(a, ac, seed), "zuco_coefficients": nll(a, ac, pb), "uniform": nll(a, ac, None)},
                  "zuco": {"within_crossfit": crossfit_nll(b, bc, seed), "provo_coefficients": nll(b, bc, pa), "uniform": nll(b, bc, None)}},
              "residual_distribution": {"provo": residual_audit(a, ac, pa), "zuco": residual_audit(b, bc, pb)},
              "domain_distinguishability": domain_classification(ax, bx, permutations, seed + 2)}
    result["runtime_seconds"] = time.perf_counter() - started
    return result
