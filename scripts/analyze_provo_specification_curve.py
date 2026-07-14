from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetrack2llm import extract_events, read_fixation_csv
from eyetrack2llm.baseline import (
    STRICT_LINE_SPECIFICATION_SETS, build_pair_design, count_vector,
    enrich_spacy_syntax, enrich_word_frequencies, fit_baseline,
    read_provo_word_metadata, residual_vector,
)
from eyetrack2llm.stability import (paired_residual_metrics,
                                    permute_destinations_within_source,
                                    weighted_correlation)
from eyetrack2llm.transfer import sentence_folds


SEED = 20260711
SPECIFICATIONS = tuple(STRICT_LINE_SPECIFICATION_SETS)
METRICS = ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal")


def subject_counts(design, events):
    result = {}
    for subject in sorted(set(map(str, events.subject_id))):
        selected = (events.subject_id == subject) & (events.event_type == "forward")
        edges = defaultdict(float)
        for text, src, dst, weight in zip(
            events.text_id[selected], events.src_word[selected], events.dst_word[selected],
            events.weight[selected], strict=True,
        ):
            edges[(str(text), int(src), int(dst))] += float(weight)
        result[subject] = count_vector(design, edges)
    return result


def crossfit(design, counts, seed, min_exposure):
    residuals = {}
    folds = sentence_folds(design.text_id, 5, seed)
    nll = uniform = observations = 0.0
    for train, test in folds:
        train_mask = np.isin(design.text_id, train)
        model = fit_baseline(design.subset(train_mask), counts[train_mask], l2=1.0,
                             maxiter=1000)
        for text in test:
            mask = design.text_id == text
            target = design.subset(mask)
            values = counts[mask]
            probability = model.predict(target)
            residual, exposure = residual_vector(values, probability, target.group_start)
            residuals[text] = {"residual": residual, "exposure": exposure,
                               "reliable": (exposure >= min_exposure) & np.isfinite(residual),
                               "src": target.src_word, "dst": target.dst_word}
            nll -= float(values @ np.log(np.maximum(probability, 1e-300)))
            observations += float(values.sum())
            for start, stop in zip(target.group_start[:-1], target.group_start[1:], strict=True):
                uniform += float(values[start:stop].sum() * np.log(stop - start))
    return residuals, {
        "nll": nll, "uniform_nll": uniform, "nll_minus_uniform": nll - uniform,
        "mean_nll": nll / observations if observations else None,
        "observed_transitions": observations,
    }


def reliability_summary(metrics):
    return {
        metric: metrics["text_summary"][metric]["median"] for metric in METRICS
    }


def distribution(values):
    values = np.asarray(values, float)
    return {
        "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
        "q75": float(np.quantile(values, .75)),
        "range": [float(values.min()), float(values.max())],
    }


def identity_correlations(residuals_by_spec):
    result = {}
    for left_index, left in enumerate(SPECIFICATIONS):
        for right in SPECIFICATIONS[left_index + 1:]:
            per_text, all_left, all_right = {}, [], []
            for text in sorted(residuals_by_spec[left], key=lambda value: int(value.split(":")[-1])):
                a, b = residuals_by_spec[left][text], residuals_by_spec[right][text]
                eligible = a["reliable"] & b["reliable"]
                per_text[text] = weighted_correlation(a["residual"][eligible], b["residual"][eligible])
                all_left.extend(a["residual"][eligible]); all_right.extend(b["residual"][eligible])
            valid = [value for value in per_text.values() if value is not None]
            result[f"{left}__{right}"] = {
                "text_equal_median": float(np.median(valid)) if valid else None,
                "text_equal_q25": float(np.quantile(valid, .25)) if valid else None,
                "text_equal_q75": float(np.quantile(valid, .75)) if valid else None,
                "edge_descriptive": weighted_correlation(np.asarray(all_left), np.asarray(all_right)),
                "valid_texts": len(valid), "n_edges": len(all_left), "per_text": per_text,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description="Predefined strict-line nuisance specification curve")
    parser.add_argument("--fixations", default="data/processed/provo_fixations_with_lines.csv")
    parser.add_argument("--main-csv", default="data/raw/Provo_Corpus-Eyetracking_Data.csv")
    parser.add_argument("--output", default="data/processed/provo_strictline_specification_curve.json")
    parser.add_argument("--csv-output", default="data/processed/provo_strictline_specification_curve.csv")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--null-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-exposure", type=int, default=5)
    args = parser.parse_args()

    fixations = read_fixation_csv(args.fixations)
    events = extract_events(fixations, include_self=False, require_consecutive_order=True)
    observed = defaultdict(set)
    for text, word in zip(fixations.text_id, fixations.word_index, strict=True):
        observed[str(text)].add(int(word))
    import spacy
    metadata = enrich_word_frequencies(read_provo_word_metadata(args.main_csv, dict(observed)))
    metadata, syntax_audit = enrich_spacy_syntax(metadata, spacy.load("en_core_web_sm"))
    designs = {name: build_pair_design(metadata, name, "common_forward_same_sentence_same_line")
               for name in SPECIFICATIONS}
    reference = designs[SPECIFICATIONS[0]]
    for name, design in designs.items():
        if not (np.array_equal(reference.text_id, design.text_id)
                and np.array_equal(reference.src_word, design.src_word)
                and np.array_equal(reference.dst_word, design.dst_word)):
            raise ValueError(f"Candidate mismatch in specification {name}")
        if design.group_constant_features() or design.design_rank() != design.features.shape[1]:
            raise ValueError(f"Conditionally rank-deficient specification {name}")

    by_subject = subject_counts(reference, events)
    subjects = sorted(by_subject)
    if len(subjects) != 84:
        raise ValueError(f"Expected 84 Provo subjects, found {len(subjects)}")
    rng = np.random.default_rng(args.seed)
    splits = []
    for repeat in range(args.repeats):
        shuffled = np.asarray(subjects)[rng.permutation(len(subjects))]
        splits.append((sorted(shuffled[:42].tolist()), sorted(shuffled[42:].tolist())))

    rows, real = [], {name: [] for name in SPECIFICATIONS}
    for repeat, halves in enumerate(splits):
        counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        for name, design in designs.items():
            fitted = [crossfit(design, value, args.seed, args.min_exposure) for value in counts]
            metrics = paired_residual_metrics(fitted[0][0], fitted[1][0])
            reliability = reliability_summary(metrics)
            item = {"repeat": repeat, "halves": halves, "predictive": [fitted[0][1], fitted[1][1]],
                    "reliability": reliability,
                    "per_text_metrics": {
                        text: {metric: values[metric] for metric in METRICS}
                        for text, values in metrics["per_text"].items()
                    }}
            real[name].append(item)
            rows.append({"kind": "observed", "replicate": repeat, "specification": name,
                         **reliability, "half1_nll_minus_uniform": fitted[0][1]["nll_minus_uniform"],
                         "half2_nll_minus_uniform": fitted[1][1]["nll_minus_uniform"]})

    null = {name: [] for name in SPECIFICATIONS}
    null_seeds = np.random.SeedSequence(args.seed).spawn(args.null_repeats * 2)
    for replicate in range(args.null_repeats):
        halves = splits[replicate % len(splits)]
        counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        permuted = [permute_destinations_within_source(counts[index], reference,
                    np.random.default_rng(null_seeds[2 * replicate + index])) for index in range(2)]
        for name, design in designs.items():
            fitted = [crossfit(design, value, args.seed, args.min_exposure)[0] for value in permuted]
            reliability = reliability_summary(paired_residual_metrics(*fitted))
            null[name].append({"control_replicate": replicate, "reliability": reliability})
            rows.append({"kind": "destination_destruction", "replicate": replicate,
                         "specification": name, **reliability,
                         "half1_nll_minus_uniform": "", "half2_nll_minus_uniform": ""})

    full_counts = np.sum(list(by_subject.values()), axis=0)
    full_residuals = {name: crossfit(design, full_counts, args.seed, 2 * args.min_exposure)[0]
                      for name, design in designs.items()}
    summary = {}
    for name in SPECIFICATIONS:
        summary[name] = {
            "predictive_nll_minus_uniform": distribution([
                np.mean([half["nll_minus_uniform"] for half in item["predictive"]])
                for item in real[name]
            ]),
            "reliability": {metric: distribution([item["reliability"][metric] for item in real[name]])
                            for metric in METRICS},
            "negative_control": {
                metric: {
                    **distribution([item["reliability"][metric] for item in null[name]]),
                    "observed_median_above_all_controls": float(np.median([
                        item["reliability"][metric] for item in real[name]
                    ])) > max(item["reliability"][metric] for item in null[name]),
                }
                for metric in METRICS
            },
        }
    output = {
        "status": "complete" if args.repeats == 100 and args.null_repeats == 500 else "pilot", "seed": args.seed,
        "repeats": args.repeats, "null_repeats": args.null_repeats, "subjects": 84,
        "split": "42/42", "risk_set": "common_forward_same_sentence_same_line",
        "text_crossfit_folds": 5, "min_half_source_exposure": args.min_exposure,
        "specifications": {name: {"features": list(design.feature_names),
            "columns": design.features.shape[1], "conditional_rank": design.design_rank(),
            "group_constant_features": list(design.group_constant_features())}
            for name, design in designs.items()},
        "negative_control": "500 one-split destination-label destruction controls balanced across the 100-split bank; independent derived RNG per half; each specification fitted separately; controls are descriptive because they do not reproduce the observed 100-split summary statistic",
        "syntax_audit": {key: value for key, value in syntax_audit.items() if key != "sentence_reports"},
        "summary": summary, "full_sample_residual_identity": identity_correlations(full_residuals),
        "repeat_results": real, "null_results": null,
    }
    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = Path(args.csv_output); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
