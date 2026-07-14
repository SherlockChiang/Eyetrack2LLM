from __future__ import annotations

import argparse
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
from eyetrack2llm.stability import paired_residual_metrics
from eyetrack2llm.transfer import sentence_folds


SEED = 20260711
SPECS = tuple(STRICT_LINE_SPECIFICATION_SETS)
METRICS = ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal")


def subject_counts(design, events, verified):
    result = {}
    affected = 0.0
    for subject in sorted(set(map(str, events.subject_id))):
        selected = (events.subject_id == subject) & (events.event_type == "forward")
        edges = defaultdict(float)
        for text, src, dst, weight in zip(events.text_id[selected], events.src_word[selected],
                                           events.dst_word[selected], events.weight[selected], strict=True):
            key = str(text), int(src), int(dst)
            if (key[0], key[1]) not in verified or (key[0], key[2]) not in verified:
                affected += float(weight)
                continue
            edges[key] += float(weight)
        result[subject] = count_vector(design, edges)
    return result, affected


def crossfit(design, counts, seed, min_exposure):
    residuals = {}
    nll = uniform = observations = 0.0
    for train, test in sentence_folds(design.text_id, 5, seed):
        train_mask = np.isin(design.text_id, train)
        model = fit_baseline(design.subset(train_mask), counts[train_mask], l2=1.0, maxiter=1000)
        for text in test:
            mask = design.text_id == text
            target, values = design.subset(mask), counts[mask]
            probability = model.predict(target)
            residual, exposure = residual_vector(values, probability, target.group_start)
            residuals[text] = {"residual": residual, "exposure": exposure,
                               "reliable": exposure >= min_exposure,
                               "src": target.src_word, "dst": target.dst_word}
            nll -= float(values @ np.log(np.maximum(probability, 1e-300)))
            observations += float(values.sum())
            for start, stop in zip(target.group_start[:-1], target.group_start[1:], strict=True):
                uniform += float(values[start:stop].sum() * np.log(stop - start))
    return residuals, {"nll": nll, "uniform_nll": uniform, "nll_minus_uniform": nll - uniform,
                       "mean_nll": nll / observations if observations else None,
                       "observed_transitions": observations}


def distribution(values):
    values = np.asarray(values, float)
    return {"median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)), "range": [float(values.min()), float(values.max())]}


def main():
    parser = argparse.ArgumentParser(description="Provo verified-position 100-split four-spec sensitivity")
    parser.add_argument("--fixations", default="data/processed/provo_fixations_with_lines.csv")
    parser.add_argument("--main-csv", default="data/raw/Provo_Corpus-Eyetracking_Data.csv")
    parser.add_argument("--reconciliation", default="data/processed/provo_word_position_reconciliation.json")
    parser.add_argument("--original", default="data/processed/provo_strictline_specification_curve.json")
    parser.add_argument("--output", default="data/processed/provo_reconciled_sensitivity.json")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-exposure", type=int, default=5)
    args = parser.parse_args()

    audit = json.loads(Path(args.reconciliation).read_text(encoding="utf-8"))
    verified = {(str(text), int(word)) for text, word in audit["verified_mask"]}
    fixations = read_fixation_csv(args.fixations)
    all_positions = {(str(text), int(word)) for text, word in zip(fixations.text_id, fixations.word_index, strict=True)}
    observed = defaultdict(set)
    excluded_fixations = 0
    for text, word in zip(fixations.text_id, fixations.word_index, strict=True):
        key = str(text), int(word)
        if key in verified:
            observed[key[0]].add(key[1])
        else:
            excluded_fixations += 1
    events = extract_events(fixations, include_self=False, require_consecutive_order=True)

    import spacy
    metadata = enrich_word_frequencies(read_provo_word_metadata(args.main_csv, dict(observed)))
    metadata, syntax_audit = enrich_spacy_syntax(metadata, spacy.load("en_core_web_sm"))
    designs = {name: build_pair_design(metadata, name, "common_forward_same_sentence_same_line") for name in SPECS}
    reference = designs[SPECS[0]]
    for name, design in designs.items():
        if not (np.array_equal(reference.text_id, design.text_id)
                and np.array_equal(reference.src_word, design.src_word)
                and np.array_equal(reference.dst_word, design.dst_word)):
            raise ValueError(f"Candidate mismatch in specification {name}")

    by_subject, excluded_forward_events = subject_counts(reference, events, verified)
    subjects = sorted(by_subject)
    if len(subjects) != 84:
        raise ValueError(f"Expected 84 subjects, found {len(subjects)}")
    rng = np.random.default_rng(args.seed)
    splits = []
    for _ in range(args.repeats):
        shuffled = np.asarray(subjects)[rng.permutation(len(subjects))]
        splits.append((sorted(shuffled[:42].tolist()), sorted(shuffled[42:].tolist())))

    results = {name: [] for name in SPECS}
    for repeat, halves in enumerate(splits):
        counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        for name, design in designs.items():
            fitted = [crossfit(design, value, args.seed, args.min_exposure) for value in counts]
            metrics = paired_residual_metrics(fitted[0][0], fitted[1][0])
            results[name].append({
                "repeat": repeat,
                "reliability": {metric: metrics["text_summary"][metric]["median"] for metric in METRICS},
                "predictive": [item[1] for item in fitted],
            })

    original = json.loads(Path(args.original).read_text(encoding="utf-8"))
    summary = {}
    for name in SPECS:
        reliability = {metric: distribution([row["reliability"][metric] for row in results[name]]) for metric in METRICS}
        nll = distribution([np.mean([half["nll_minus_uniform"] for half in row["predictive"]]) for row in results[name]])
        summary[name] = {
            "reliability": reliability, "predictive_nll_minus_uniform": nll,
            "original": {"reliability_median": {metric: original["summary"][name]["reliability"][metric]["median"] for metric in METRICS},
                         "predictive_nll_minus_uniform_median": original["summary"][name]["predictive_nll_minus_uniform"]["median"]},
            "delta_median": {"reliability": {metric: reliability[metric]["median"] - original["summary"][name]["reliability"][metric]["median"] for metric in METRICS},
                             "predictive_nll_minus_uniform": nll["median"] - original["summary"][name]["predictive_nll_minus_uniform"]["median"]},
        }
    ordering = sorted(SPECS, key=lambda name: summary[name]["predictive_nll_minus_uniform"]["median"])
    output = {
        "status": "complete" if args.repeats == 100 else "pilot", "seed": args.seed,
        "repeats": args.repeats, "specifications": list(SPECS), "subjects": len(subjects),
        "verified_mask_positions": len(verified), "processed_observed_positions": len(all_positions),
        "processed_observed_verified_positions": len(all_positions & verified),
        "processed_observed_unverified_positions": sorted([list(value) for value in all_positions - verified]),
        "excluded_fixation_rows": excluded_fixations,
        "excluded_forward_event_weight": excluded_forward_events,
        "candidate_edges_after_mask": len(reference.text_id),
        "candidate_source_groups_after_mask": len(reference.group_start) - 1,
        "nll_order_best_to_worst": ordering,
        "nll_order_unchanged": ordering == sorted(SPECS, key=lambda name: original["summary"][name]["predictive_nll_minus_uniform"]["median"]),
        "syntax_audit": {key: value for key, value in syntax_audit.items() if key != "sentence_reports"},
        "summary": summary, "repeat_results": results,
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key not in {"repeat_results", "syntax_audit"}}, indent=2))


if __name__ == "__main__":
    main()
