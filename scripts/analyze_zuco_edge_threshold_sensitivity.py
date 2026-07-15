from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONTRASTS = {
    "gaze_vs_mlm": ("gaze", "mlm"),
    "gaze_vs_shuffled": ("gaze", "shuffled"),
    "gaze_vs_position": ("gaze", "position"),
}


def analyze(payload: dict, thresholds: tuple[int, ...] = (4, 10, 20, 30),
            bootstrap: int = 100000, seed: int = 20260713) -> tuple[dict, list[dict]]:
    results = payload["seed_results"]
    seeds = sorted(results, key=int)
    texts = sorted(next(iter(results.values()))["gaze"]["per_text"])
    rows = []
    threshold_results = {}
    for threshold in thresholds:
        comparisons = {}
        for name, (first, second) in CONTRASTS.items():
            values = {}
            raw_correlations = []
            minimum_edges = {}
            for text in texts:
                pairs = []
                edge_counts = []
                for run_seed in seeds:
                    left = results[run_seed][first]["per_text"][text]
                    right = results[run_seed][second]["per_text"][text]
                    edge_counts.extend((left["n_edges"], right["n_edges"]))
                    if left["correlation"] is not None and right["correlation"] is not None:
                        pairs.append((left["correlation"], right["correlation"]))
                minimum_edges[text] = min(edge_counts)
                if minimum_edges[text] < threshold or len(pairs) != len(seeds):
                    continue
                raw_correlations.extend(value for pair in pairs for value in pair)
                left_z = np.arctanh(np.clip([pair[0] for pair in pairs], -1 + 1e-7, 1 - 1e-7))
                right_z = np.arctanh(np.clip([pair[1] for pair in pairs], -1 + 1e-7, 1 - 1e-7))
                values[text] = float(np.mean(left_z - right_z))
            array = np.asarray(list(values.values()))
            if threshold == 4:
                interval = payload["comparisons"][name]["text_equal_fisher_z"]["descriptive_text_resampling_interval"]
                interval_source = "frozen primary transfer interval"
                interval_seed = payload["comparisons"][name]["text_equal_fisher_z"].get("bootstrap_seed")
            else:
                interval_seed = seed + threshold * 100 + list(CONTRASTS).index(name) + 1
                rng = np.random.default_rng(interval_seed)
                draws = array[rng.integers(len(array), size=(bootstrap, len(array)))].mean(axis=1)
                interval = np.quantile(draws, (0.025, 0.975)).tolist()
                interval_source = "threshold-specific descriptive text resampling"
            strata = {
                label: sum(low <= minimum_edges[text] <= high for text in texts)
                for label, low, high in (("4-9", 4, 9), ("10-19", 10, 19), ("20-29", 20, 29), ("30+", 30, float("inf")))
            }
            raw = np.asarray(raw_correlations)
            comparison = {
                "minimum_edges": threshold,
                "texts_retained": len(values),
                "mean_difference": float(array.mean()),
                "descriptive_text_resampling_interval": interval,
                "interval_source": interval_source,
                "interval_seed": interval_seed,
                "max_absolute_raw_correlation": float(np.max(np.abs(raw))),
                "max_absolute_fisher_z": float(np.max(np.abs(np.arctanh(np.clip(raw, -1 + 1e-7, 1 - 1e-7))))),
                "near_perfect_raw_correlation_count": int(np.count_nonzero(np.abs(raw) >= 0.99)),
                "max_absolute_text_mean_contribution": float(np.max(np.abs(array)) / len(array)),
                "available_text_strata_by_minimum_edges": strata,
            }
            comparisons[name] = comparison
            rows.append({"minimum_edges": threshold, "contrast": name, **comparison,
                         "descriptive_text_resampling_interval": json.dumps(interval),
                         "available_text_strata_by_minimum_edges": json.dumps(strata, sort_keys=True)})
        threshold_results[str(threshold)] = {"comparisons": comparisons}
    return {
        "status": "complete",
        "analysis_role": "descriptive edge-threshold sensitivity on frozen fixed-reader transfer results",
        "bootstrap_repeats": bootstrap,
        "seed": seed,
        "derived_seed_rule": "base seed + 100*threshold + one-based contrast index; threshold 4 reuses frozen primary intervals",
        "thresholds": list(thresholds),
        "results": threshold_results,
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Reaggregate frozen ZuCo cross-corpus scorer results across minimum-edge thresholds")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--bootstrap", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    result, rows = analyze(json.loads(Path(args.input).read_text(encoding="utf-8")), bootstrap=args.bootstrap, seed=args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with Path(args.csv_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
