from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetrack2llm import extract_events, read_fixation_csv
from eyetrack2llm.baseline import build_pair_design, count_vector, enrich_spacy_syntax, enrich_word_frequencies, read_provo_word_metadata
from eyetrack2llm.stability import crossfit_raw_residuals, paired_residual_metrics


SEED = 20260711


def subject_counts(design, events):
    result = {}
    for subject in sorted(set(map(str, events.subject_id))):
        selected = (events.subject_id == subject) & (events.event_type == "forward")
        edges = defaultdict(float)
        for text, src, dst, weight in zip(events.text_id[selected], events.src_word[selected], events.dst_word[selected], events.weight[selected], strict=True):
            edges[(str(text), int(src), int(dst))] += float(weight)
        result[subject] = count_vector(design, edges)
    return result


def summarize_repeats(repeats, condition, metric):
    values = [row[condition]["text_summary"][metric]["median"] for row in repeats]
    return {"median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
             "q75": float(np.quantile(values, .75)), "range": [float(np.min(values)), float(np.max(values))]}


def reader_bootstrap_draws(subjects, repeats, seed):
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    draws = []
    for _ in range(repeats):
        sampled = subjects[rng.integers(len(subjects), size=len(subjects))]
        sampled = sampled[rng.permutation(len(sampled))]
        draws.append((sampled[:42].tolist(), sampled[42:].tolist()))
    return draws


def main():
    parser = argparse.ArgumentParser(description="Provo common-core fixed-sample agreement and reader-resampling sensitivity")
    parser.add_argument("--fixations", default="data/processed/provo_fixations_with_lines.csv")
    parser.add_argument("--main-csv", default="data/raw/Provo_Corpus-Eyetracking_Data.csv")
    parser.add_argument("--output", default="data/processed/provo_commoncore_independent_reliability.json")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--reader-bootstraps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    fixations = read_fixation_csv(args.fixations)
    events = extract_events(fixations, include_self=False, require_consecutive_order=True)
    observed = defaultdict(set)
    for text, word in zip(fixations.text_id, fixations.word_index, strict=True):
        observed[str(text)].add(int(word))
    import spacy
    metadata = enrich_word_frequencies(read_provo_word_metadata(args.main_csv, dict(observed)))
    metadata, syntax_audit = enrich_spacy_syntax(metadata, spacy.load("en_core_web_sm"))
    design = build_pair_design(metadata, "common_core", "common_forward_same_sentence_same_line")
    by_subject = subject_counts(design, events)
    subjects = sorted(by_subject)
    if len(subjects) != 84:
        raise ValueError(f"Expected 84 Provo subjects, found {len(subjects)}")
    rng = np.random.default_rng(args.seed)
    repeats = []
    for repeat in range(args.repeats):
        shuffled = np.asarray(subjects)[rng.permutation(len(subjects))]
        halves = [sorted(shuffled[:42].tolist()), sorted(shuffled[42:].tolist())]
        counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        independent = [crossfit_raw_residuals(design, value, seed=args.seed, min_exposure=5)[0] for value in counts]
        # Shared nuisance means one pooled fit, but each half retains its own counts/exposure.
        from eyetrack2llm.baseline import fit_baseline, residual_vector
        from eyetrack2llm.transfer import sentence_folds
        shared = [{}, {}]
        for train, test in sentence_folds(design.text_id, 5, args.seed):
            train_mask = np.isin(design.text_id, train)
            model = fit_baseline(design.subset(train_mask), (counts[0] + counts[1])[train_mask], l2=1.0)
            for text in test:
                mask = design.text_id == text; target = design.subset(mask); probability = model.predict(target)
                for index in range(2):
                    residual, exposure = residual_vector(counts[index][mask], probability, target.group_start)
                    shared[index][text] = {"residual": residual, "exposure": exposure, "reliable": exposure >= 5,
                                           "src": target.src_word, "dst": target.dst_word}
        repeats.append({"repeat": repeat, "halves": halves,
                        "independent": paired_residual_metrics(*independent),
                        "shared": paired_residual_metrics(*shared)})
    metrics = ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal")
    summary = {condition: {metric: summarize_repeats(repeats, condition, metric) for metric in metrics}
               for condition in ("independent", "shared")}
    summary["shared_minus_independent_median"] = {
        metric: summary["shared"][metric]["median"] - summary["independent"][metric]["median"] for metric in metrics}
    bootstrap_records = []
    for repeat, halves in enumerate(reader_bootstrap_draws(subjects, args.reader_bootstraps, args.seed + 1)):
        counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        fitted = [crossfit_raw_residuals(design, value, seed=args.seed, min_exposure=5)[0]
                  for value in counts]
        bootstrap_records.append({
            "repeat": repeat,
            "unique_readers": [len(set(half)) for half in halves],
            "shared_reader_identities": len(set(halves[0]) & set(halves[1])),
            "agreement": paired_residual_metrics(*fitted),
        })
    bootstrap_summary = {
        metric: summarize_repeats(bootstrap_records, "agreement", metric)
        for metric in metrics
    } if bootstrap_records else {}
    output = {"status": "complete" if args.repeats == 100 else "pilot", "seed": args.seed, "repeats": args.repeats,
              "subjects": len(subjects), "split": "42/42", "feature_set": "common_core",
              "risk_set": "common_forward_same_sentence_same_line", "text_crossfit_folds": 5, "min_half_source_exposure": 5,
              "target": "raw unclipped Pearson residual", "shared_control": "pooled 84-subject nuisance fit; half-specific counts/exposure",
              "reader_resampling_sensitivity": {
                  "role": "fixed-55-text outer reader-resampling sensitivity; not a joint reader-text confidence interval",
                  "design": "sample 84 reader positions with replacement, randomly divide positions 42/42, preserve multiplicity, independently refit both halves",
                  "seed": args.seed + 1, "repeats": args.reader_bootstraps,
                  "summary": bootstrap_summary, "records": bootstrap_records,
              },
              "syntax_audit": {key: value for key, value in syntax_audit.items() if key != "sentence_reports"},
              "summary": summary, "repeat_results": repeats}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
