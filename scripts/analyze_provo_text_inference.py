from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("mlm", "gaze", "shuffled", "position")


def text_signflip(values: np.ndarray, permutations: int, seed: int) -> float:
    """Two-sided exact sign-flip test by enumeration."""
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    statistics = (
        abs(float((values * np.asarray(signs)).mean()))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    return sum(statistic >= observed for statistic in statistics) / (2 ** len(values))


def bootstrap(values: np.ndarray, repeats: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def analyze(runs: list[dict], repeats: int, signflips: int, seed: int,
            existing: dict | None = None) -> tuple[dict, list[dict]]:
    texts = list(runs[0]["conditions"]["mlm"]["test"]["per_text"])
    if existing is not None:
        result = dict(existing)
        nll_differences = [float(np.mean([
            run["conditions"]["gaze"]["test"]["per_text"][text]["mlm_nll"]
            - run["conditions"]["mlm"]["test"]["per_text"][text]["mlm_nll"]
            for run in runs
        ])) for text in texts]
        values = np.asarray(nll_differences)
        result["comparisons"]["gaze_minus_mlm_nll"] = {
            "scale": "paired per-text mean NLL difference", "seed_aggregation": "mean before text inference",
            "texts": len(values), "mean": float(values.mean()), "ci95": bootstrap(values, repeats, seed),
            "signflip_p_two_sided": text_signflip(values, permutations=signflips, seed=seed),
            "per_text": dict(zip(texts, nll_differences, strict=True)), "direction": "negative favors gaze",
        }
        rows = [{"seed": run["seed"], "condition": condition, "text_id": text,
                 **run["conditions"][condition]["test"]["per_text"][text]}
                for run in runs for condition in CONDITIONS for text in texts]
        return result, rows
    rows = []
    conditions = {}
    for condition in CONDITIONS:
        correlations = []
        nlls = []
        pooled_y, pooled_p = [], []
        for run in runs:
            for text in texts:
                item = run["conditions"][condition]["test"]["per_text"][text]
                rows.append({"seed": run["seed"], "condition": condition, "text_id": text,
                             "valid_edges": item["valid_edges"], "correlation": item["gaze_correlation"],
                             "mlm_nll": item["mlm_nll"], "mlm_tokens": item["mlm_tokens"]})
                correlations.append(item["gaze_correlation"]); nlls.append(item["mlm_nll"])
                pooled_y.extend(edge["target"] for edge in item["edge_predictions"])
                pooled_p.extend(edge["prediction"] for edge in item["edge_predictions"])
        per_text_seed_z = np.asarray(correlations).reshape(len(runs), len(texts))
        per_text_seed_z = np.arctanh(np.clip(per_text_seed_z, -.9999999, .9999999))
        conditions[condition] = {
            "macro_text_equal_correlation_seed_averaged_fisher_z": float(np.tanh(per_text_seed_z.mean(axis=0).mean())),
            "macro_text_equal_mlm_nll": float(np.mean(nlls)),
            "pooled_edge_correlation": float(np.corrcoef(pooled_y, pooled_p)[0, 1]),
            "pooled_token_mlm_nll": float(np.average([r["mlm_nll"] for r in rows if r["condition"] == condition],
                                                      weights=[r["mlm_tokens"] for r in rows if r["condition"] == condition])),
            "valid_edges_per_seed": int(sum(r["valid_edges"] for r in rows if r["condition"] == condition) / len(runs)),
        }
    comparisons = {}
    for control in ("shuffled", "mlm", "position"):
        differences = []
        for text in texts:
            seed_differences = []
            for run in runs:
                first = run["conditions"]["gaze"]["test"]["per_text"][text]["gaze_correlation"]
                second = run["conditions"][control]["test"]["per_text"][text]["gaze_correlation"]
                seed_differences.append(np.arctanh(np.clip(first, -.9999999, .9999999)) - np.arctanh(np.clip(second, -.9999999, .9999999)))
            differences.append(float(np.mean(seed_differences)))
        values = np.asarray(differences)
        comparisons[f"gaze_minus_{control}"] = {
            "scale": "paired per-text Fisher-z difference", "seed_aggregation": "mean before text inference",
            "texts": len(values), "mean": float(values.mean()), "ci95": bootstrap(values, repeats, seed),
            "signflip_p_two_sided": text_signflip(values, permutations=signflips, seed=seed),
            "per_text": dict(zip(texts, differences, strict=True)),
        }
    nll_differences = []
    for text in texts:
        nll_differences.append(float(np.mean([
            run["conditions"]["gaze"]["test"]["per_text"][text]["mlm_nll"]
            - run["conditions"]["mlm"]["test"]["per_text"][text]["mlm_nll"]
            for run in runs
        ])))
    nll_values = np.asarray(nll_differences)
    comparisons["gaze_minus_mlm_nll"] = {
        "scale": "paired per-text mean NLL difference",
        "seed_aggregation": "mean before text inference",
        "texts": len(nll_values), "mean": float(nll_values.mean()),
        "ci95": bootstrap(nll_values, repeats, seed),
        "signflip_p_two_sided": text_signflip(nll_values, permutations=signflips, seed=seed),
        "per_text": dict(zip(texts, nll_differences, strict=True)),
        "direction": "negative favors gaze",
    }
    return {"status": "complete", "analysis_role": "fixed50 text-level inference", "seeds": [r["seed"] for r in runs],
            "test_texts": texts, "conditions": conditions, "comparisons": comparisons,
            "bootstrap_repeats": repeats, "signflip_patterns": 2 ** len(texts),
            "target_disclosure": "Pearson residuals are centered/scaled by train-only median and 1.4826*MAD, then clipped to [-5,5]."}, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+"); parser.add_argument("--output", required=True); parser.add_argument("--csv-output", required=True)
    parser.add_argument("--existing", help="Existing inference artifact whose correlation fields should be retained")
    parser.add_argument("--bootstrap", type=int, default=100000); parser.add_argument("--signflips", type=int, default=0, help="Ignored; exact sign enumeration is always used"); parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args(); runs = [json.loads(Path(path).read_text()) for path in args.inputs]
    existing = json.loads(Path(args.existing).read_text()) if args.existing else None
    result, rows = analyze(runs, args.bootstrap, args.signflips, args.seed, existing)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    with Path(args.csv_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
