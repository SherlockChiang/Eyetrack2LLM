from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np

from eyetrack2llm import extract_events, read_fixation_csv
from eyetrack2llm.baseline import (STRICT_LINE_SPECIFICATION_SETS, build_pair_design,
                                   count_vector, enrich_spacy_syntax,
                                   enrich_word_frequencies, fit_baseline,
                                   read_provo_word_metadata)
from eyetrack2llm.stability import crossfit_raw_residuals, permute_destinations_within_source
from eyetrack2llm.target_selection import (CATEGORIES, METRICS, categorized_paired_metrics,
                                           category_masks, deterministic_subject_splits,
                                           distribution, residual_identity)


SEED = 20260711
SPECIFICATIONS = tuple(STRICT_LINE_SPECIFICATION_SETS)
SPEC_CURVE_FITTER = partial(fit_baseline, maxiter=1000)


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


def candidate_audit(design, counts, min_exposure):
    audits = {}
    source_exposure = np.repeat(np.add.reduceat(counts, design.group_start[:-1]),
                                np.diff(design.group_start))
    for category, selected in category_masks(design.src_word, design.dst_word).items():
        keys = set(zip(design.text_id[selected], design.src_word[selected], strict=True))
        eligible = selected & (source_exposure >= min_exposure)
        audits[category] = {
            "candidate_edges": int(selected.sum()),
            "candidate_sources": len(keys),
            "candidate_texts": len(set(design.text_id[selected].tolist())),
            "observed_transition_count": int(np.count_nonzero(counts[selected])),
            "observed_transition_mass": float(counts[selected].sum()),
            "eligible_edges": int(eligible.sum()),
            "eligible_sources": len(set(zip(design.text_id[eligible], design.src_word[eligible], strict=True))),
            "eligible_exposure_sum_over_edges": float(source_exposure[eligible].sum()),
            "eligible_exposure_definition": "full-sample source transition total, repeated for each eligible candidate edge; threshold >= 10",
        }
    return audits


def full_residual_summary(residuals):
    result = {}
    for category in CATEGORIES:
        values, texts = [], 0
        for item in residuals.values():
            selected = np.asarray(item["reliable"]) & category_masks(item["src"], item["dst"])[category]
            current = np.asarray(item["residual"])[selected]
            if len(current):
                texts += 1
                values.extend(current.tolist())
        values = np.asarray(values, float)
        result[category] = {
            "n_edges": len(values), "texts": texts,
            "variance": float(np.var(values)) if len(values) else None,
            "quantiles": ({str(q): float(np.quantile(values, q)) for q in (0, .05, .25, .5, .75, .95, 1)}
                          if len(values) else {}),
        }
    return result


def summarize_replicates(records, null_records):
    output = {}
    for specification in SPECIFICATIONS:
        output[specification] = {}
        for category in CATEGORIES:
            metrics = {}
            for metric in METRICS:
                observed = [row["categories"][category]["text_summary"][metric]["median"]
                            for row in records[specification]]
                null = [row["categories"][category]["text_summary"][metric]["median"]
                        for row in null_records[specification]]
                observed_median = distribution(observed)["median"]
                defined_null = [value for value in null if value is not None]
                metrics[metric] = {
                    "observed_100_split": distribution(observed),
                    "destination_permutation_25": distribution(null),
                    "empirical_exceedance": ((1 + sum(value >= observed_median for value in defined_null))
                                               / (len(defined_null) + 1)
                                               if observed_median is not None and defined_null else None),
                    "defined_split_replicates": sum(value is not None for value in observed),
                    "defined_null_replicates": sum(value is not None for value in null),
                }
            output[specification][category] = metrics
    return output


def main():
    parser = argparse.ArgumentParser(description="Secondary theory-guided forward target-selection decomposition")
    parser.add_argument("--fixations", default="data/processed/provo_fixations_with_lines.csv")
    parser.add_argument("--main-csv", default="data/raw/Provo_Corpus-Eyetracking_Data.csv")
    parser.add_argument("--output", default="data/processed/provo_target_selection_decomposition.json")
    parser.add_argument("--csv-output", default="data/processed/provo_target_selection_decomposition.csv")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--null-repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    fixations = read_fixation_csv(args.fixations)
    events = extract_events(fixations, include_self=False, require_consecutive_order=True)
    observed = defaultdict(set)
    for text, word in zip(fixations.text_id, fixations.word_index, strict=True):
        observed[str(text)].add(int(word))
    import spacy
    metadata = enrich_word_frequencies(read_provo_word_metadata(args.main_csv, dict(observed)))
    metadata, _ = enrich_spacy_syntax(metadata, spacy.load("en_core_web_sm"))
    designs = {name: build_pair_design(metadata, name, "common_forward_same_sentence_same_line")
               for name in SPECIFICATIONS}
    reference = designs[SPECIFICATIONS[0]]
    for design in designs.values():
        if not (np.array_equal(reference.text_id, design.text_id)
                and np.array_equal(reference.src_word, design.src_word)
                and np.array_equal(reference.dst_word, design.dst_word)):
            raise ValueError("Specification candidate universes differ")

    by_subject = subject_counts(reference, events)
    subjects = sorted(by_subject)
    if len(subjects) != 84:
        raise ValueError(f"Expected 84 Provo subjects, found {len(subjects)}")
    splits = deterministic_subject_splits(subjects, args.repeats, args.seed)
    real = {name: [] for name in SPECIFICATIONS}
    for repeat, halves in enumerate(splits):
        counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        for name, design in designs.items():
            fitted = [crossfit_raw_residuals(design, value, seed=args.seed, min_exposure=5,
                                             fitter=SPEC_CURVE_FITTER)[0]
                      for value in counts]
            real[name].append({"repeat": repeat, "categories": categorized_paired_metrics(*fitted)})

    null = {name: [] for name in SPECIFICATIONS}
    null_seeds = np.random.SeedSequence(args.seed).spawn(args.null_repeats * 2)
    for replicate in range(args.null_repeats):
        halves = splits[replicate % len(splits)]
        counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        permuted = [permute_destinations_within_source(counts[index], reference,
                    np.random.default_rng(null_seeds[2 * replicate + index])) for index in range(2)]
        for name, design in designs.items():
            fitted = [crossfit_raw_residuals(design, value, seed=args.seed, min_exposure=5,
                                             fitter=SPEC_CURVE_FITTER)[0]
                      for value in permuted]
            null[name].append({"permutation_replicate": replicate,
                               "categories": categorized_paired_metrics(*fitted)})

    full_counts = np.sum(list(by_subject.values()), axis=0)
    full = {name: crossfit_raw_residuals(design, full_counts, seed=args.seed,
                                         min_exposure=10, fitter=SPEC_CURVE_FITTER)[0]
            for name, design in designs.items()}
    identity = {f"{left}__{right}": residual_identity(full[left], full[right])
                for left, right in combinations(SPECIFICATIONS, 2)}
    summary = summarize_replicates(real, null)
    output = {
        "status": "complete" if args.repeats == 100 and args.null_repeats == 25 else "pilot",
        "analysis_role": "secondary", "selection": "theory_guided",
        "primary_pipeline": "frozen and unchanged", "seed": args.seed,
        "repeats": args.repeats, "null_repeats": args.null_repeats, "subjects": 84,
        "risk_set": "common_forward_same_sentence_same_line",
        "category_definition": {
            "construct": "forward token separation; not visual angle or true saccade amplitude",
            "adjacent": "d = dst token index - src token index = 1",
            "near_skip": "d = 2-3", "far_same_line": "d >= 4",
        },
        "inference_unit": "text; 100 reader splits are summarized by text-equal medians",
        "edge_overall_role": "descriptive only; no edge-level category significance tests",
        "not_allowed_inferences": [
            "This secondary decomposition is not a new primary analysis or a basis for post-hoc selection.",
            "Token separation is not visual angle or true saccade amplitude.",
            "The categories do not confirm parafoveal processing or semantic integration.",
            "Category differences are descriptive/theory-guided and are not edge-level significance tests.",
        ],
        "candidate_and_observation_audit": candidate_audit(reference, full_counts, 10),
        "summary": summary, "full_sample_residuals": {name: full_residual_summary(value)
                                                        for name, value in full.items()},
        "full_sample_spec_pair_identity_within_category": identity,
        "auxiliary_learnability": {
            "status": "not_estimable_from_artifacts",
            "reason": "Frozen Provo seed JSONs retain aggregate/per-text correlations but no identifiable per-edge test predictions, residuals, and scores; no approximation or model retraining was used.",
        },
        "zuco_transfer": {
            "status": "not_category_stratified",
            "reason": "Frozen seed_results do not retain per-edge predictions; the Provo construct is the target and ZuCo models were not retrained.",
        },
        "repeat_results": real, "null_results": null,
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for specification in SPECIFICATIONS:
        for category in CATEGORIES:
            audit = output["candidate_and_observation_audit"][category]
            for metric in METRICS:
                item = summary[specification][category][metric]
                rows.append({"specification": specification, "category": category, "metric": metric,
                             **audit,
                             **{f"observed_{key}": value for key, value in item["observed_100_split"].items()},
                             **{f"null_{key}": value for key, value in item["destination_permutation_25"].items()},
                             "empirical_exceedance": item["empirical_exceedance"],
                             "defined_split_replicates": item["defined_split_replicates"],
                             "defined_null_replicates": item["defined_null_replicates"]})
    csv_path = Path(args.csv_output); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
