from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetrack2llm import extract_events, read_fixation_csv
from eyetrack2llm.baseline import (build_pair_design, count_vector, enrich_spacy_syntax,
                                   enrich_word_frequencies, fit_baseline,
                                   read_provo_word_metadata)
from eyetrack2llm.stability import paired_residual_metrics
from eyetrack2llm.target_selection import (CATEGORIES, RESIDUAL_TYPES, category_masks,
                                            deterministic_subject_splits, distribution,
                                            residual_arrays, thin_subject_categories)


SEED = 20260711
SPECS = ("syntax", "flexible")
EXPECTED_STRATA = ((0, .5), (.5, 1), (1, 5), (5, np.inf))
EXPOSURE_STRATA = ((0, 5), (5, 10), (10, 25), (25, np.inf))
HALF_ELIGIBILITY_THRESHOLDS = (1, 3, 5, 10)


def fit_predictions(design, counts, folds):
    probability = np.empty(len(counts), float)
    coefficients = []
    for train, test in folds:
        train_mask = np.isin(design.text_id, train)
        model = fit_baseline(design.subset(train_mask), counts[train_mask], l2=1.0,
                             maxiter=1000)
        test_mask = np.isin(design.text_id, test)
        probability[test_mask] = model.predict(design.subset(test_mask))
        coefficients.append(model.coefficients)
    return probability, np.asarray(coefficients)


def records(design, bundle, value, selected=None):
    selected = np.ones(len(design.text_id), bool) if selected is None else selected
    output = {}
    for text in sorted(set(design.text_id), key=int):
        mask = (design.text_id == text) & selected
        output[text] = {"residual": bundle[value][mask], "reliable": bundle["reliable"][mask],
                        "src": design.src_word[mask], "dst": design.dst_word[mask]}
    return output


def reliability(design, left, right, value, selected=None):
    result = paired_residual_metrics(records(design, left, value, selected),
                                     records(design, right, value, selected))
    return {metric: {"value": item["median"], "defined_texts": item["valid_texts"],
                     "undefined_texts": item["undefined_texts"]}
            for metric, item in result["text_summary"].items()}


def summarize(values):
    values = np.asarray(values, float)
    return {"n": int(len(values)), "mean": float(np.mean(values)) if len(values) else None,
            "sd": float(np.std(values)) if len(values) else None,
            "quantiles": ({str(q): float(np.quantile(values, q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)} if len(values) else {}),
            "abs_gt_2": float(np.mean(np.abs(values) > 2)) if len(values) else None,
            "abs_gt_3": float(np.mean(np.abs(values) > 3)) if len(values) else None}


def stratum_label(lower, upper):
    return f"{lower:g}-{upper:g}" if np.isfinite(upper) else f">={lower:g}"


def full_diagnostics(design, bundle):
    output = {value: summarize(bundle[value][bundle["reliable"] & np.isfinite(bundle[value])])
              for value in RESIDUAL_TYPES}
    strata = {}
    for field, bins in (("probability", ((0, .05), (.05, .1), (.1, .25), (.25, .5), (.5, np.inf))),
                        ("expected", EXPECTED_STRATA), ("exposure", EXPOSURE_STRATA),
                        ("risk_size", ((0, 2), (2, 4), (4, 8), (8, np.inf)))):
        strata[field] = {}
        for lower, upper in bins:
            mask = bundle["reliable"] & (bundle[field] >= lower) & (bundle[field] < upper)
            strata[field][stratum_label(lower, upper)] = {
                value: summarize(bundle[value][mask & np.isfinite(bundle[value])]) for value in RESIDUAL_TYPES}
    output["strata"] = strata
    return output


def category_audit(design, counts, bundle):
    result = {}
    for category, mask in category_masks(design.src_word, design.dst_word).items():
        result[category] = {"candidate_edges": int(mask.sum()),
            "mean_expected_count": float(np.mean(bundle["expected"][mask])),
            "median_expected_count": float(np.median(bundle["expected"][mask])),
            "nonzero_rate": float(np.mean(counts[mask] > 0)), "observed_mass": float(counts[mask].sum()),
            "source_exposure_mean": float(np.mean(bundle["exposure"][mask])),
            "source_exposure_median": float(np.median(bundle["exposure"][mask])),
            "risk_size_mean": float(np.mean(bundle["risk_size"][mask])),
            "risk_size_median": float(np.median(bundle["risk_size"][mask]))}
    return result


def aggregate(rows):
    result = defaultdict(lambda: defaultdict(list))
    for row in rows:
        result[(row["specification"], row["analysis"], row["category"], row["stratum"], row["residual_type"])][row["metric"]].append(row)
    output = []
    for key, metrics in result.items():
        for metric, items in metrics.items():
            values = [item["value"] for item in items if item["value"] is not None]
            output.append({"specification": key[0], "analysis": key[1], "category": key[2],
                           "stratum": key[3], "residual_type": key[4], "metric": metric,
                           **distribution(values), "defined_repeats": len(values),
                           "median_defined_texts": float(np.median([x["defined_texts"] for x in items])),
                           "median_defined_edges": float(np.median([x["defined_edges"] for x in items]))})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixations", default="data/processed/provo_fixations_with_lines.csv")
    parser.add_argument("--main-csv", default="data/raw/Provo_Corpus-Eyetracking_Data.csv")
    parser.add_argument("--output", default="data/processed/provo_residual_exposure_diagnostics.json")
    parser.add_argument("--csv-output", default="data/processed/provo_residual_exposure_diagnostics.csv")
    parser.add_argument("--repeats", type=int, default=100); parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    fixations = read_fixation_csv(args.fixations)
    events = extract_events(fixations, include_self=False, require_consecutive_order=True)
    observed = defaultdict(set)
    for text, word in zip(fixations.text_id, fixations.word_index, strict=True): observed[str(text)].add(int(word))
    import spacy
    metadata = enrich_word_frequencies(read_provo_word_metadata(args.main_csv, dict(observed)))
    metadata, _ = enrich_spacy_syntax(metadata, spacy.load("en_core_web_sm"))
    designs = {name: build_pair_design(metadata, name, "common_forward_same_sentence_same_line") for name in SPECS}
    reference = designs["syntax"]
    by_subject = {}
    for subject in sorted(set(map(str, events.subject_id))):
        use = (events.subject_id == subject) & (events.event_type == "forward"); edges = defaultdict(float)
        for text, src, dst, weight in zip(events.text_id[use], events.src_word[use], events.dst_word[use], events.weight[use], strict=True):
            edges[(str(text), int(src), int(dst))] += float(weight)
        by_subject[subject] = count_vector(reference, edges)
    subjects = sorted(by_subject); splits = deterministic_subject_splits(subjects, args.repeats, args.seed)
    full_counts = np.sum(list(by_subject.values()), axis=0); full, category = {}, {}
    from eyetrack2llm.transfer import sentence_folds
    folds = sentence_folds(reference.text_id, 5, args.seed)
    full_bundles = {}
    for name, design in designs.items():
        probability, _ = fit_predictions(design, full_counts, folds)
        bundle = residual_arrays(full_counts, probability, design.group_start, 10)
        full_bundles[name] = bundle
        full[name] = full_diagnostics(design, bundle); category[name] = category_audit(design, full_counts, bundle)
    rows = []; thinning_audit = []
    seed_sequences = np.random.SeedSequence(args.seed).spawn(args.repeats)
    for repeat, halves in enumerate(splits):
        original_counts = [np.sum([by_subject[s] for s in half], axis=0) for half in halves]
        thinned_subjects = {}; rng = np.random.default_rng(seed_sequences[repeat])
        for subject in subjects:
            thinned_subjects[subject], probabilities = thin_subject_categories(
                by_subject[subject], reference.src_word, reference.dst_word, rng)
            thinning_audit.append({"repeat": repeat, "subject": subject, **probabilities})
        thinned_counts = [np.sum([thinned_subjects[s] for s in half], axis=0) for half in halves]
        for name, design in designs.items():
            probabilities = [fit_predictions(design, counts, folds)[0] for counts in original_counts]
            original = [residual_arrays(counts, probabilities[i], design.group_start, 5) for i, counts in enumerate(original_counts)]
            threshold_bundles = {
                threshold: [residual_arrays(counts, probabilities[i], design.group_start, threshold)
                            for i, counts in enumerate(original_counts)]
                for threshold in HALF_ELIGIBILITY_THRESHOLDS
            }
            thinned = [residual_arrays(thinned_counts[i], probabilities[i], design.group_start, 5) for i in range(2)]
            masks = category_masks(design.src_word, design.dst_word)
            selections = [("all", "all", None)] + [("category", c, m) for c, m in masks.items()]
            selections += [("expected_stratum", stratum_label(lo, hi),
                            (original[0]["expected"] >= lo) & (original[0]["expected"] < hi) &
                            (original[1]["expected"] >= lo) & (original[1]["expected"] < hi)) for lo, hi in EXPECTED_STRATA]
            selections += [("exposure_stratum", stratum_label(lo, hi),
                            (original[0]["exposure"] >= lo) & (original[0]["exposure"] < hi) &
                            (original[1]["exposure"] >= lo) & (original[1]["exposure"] < hi)) for lo, hi in EXPOSURE_STRATA]
            for analysis, label, selected in selections:
                for value in RESIDUAL_TYPES:
                    stats = reliability(design, original[0], original[1], value, selected)
                    edge_mask = (original[0]["reliable"] & original[1]["reliable"]
                                 & np.isfinite(original[0][value]) & np.isfinite(original[1][value])
                                 & (True if selected is None else selected))
                    for metric in ("edge_weighted", "source_equal_flatten"):
                        rows.append({"repeat": repeat, "specification": name, "analysis": analysis,
                                     "category": label if analysis == "category" else "all", "stratum": label,
                                     "residual_type": value, "metric": metric, **stats[metric],
                                     "defined_edges": int(edge_mask.sum())})
            for threshold, bundles in threshold_bundles.items():
                for value in RESIDUAL_TYPES:
                    stats = reliability(design, bundles[0], bundles[1], value)
                    edge_mask = (bundles[0]["reliable"] & bundles[1]["reliable"]
                                 & np.isfinite(bundles[0][value]) & np.isfinite(bundles[1][value]))
                    for metric in ("edge_weighted", "source_equal_flatten"):
                        rows.append({"repeat": repeat, "specification": name,
                                     "analysis": "half_eligibility_threshold", "category": "all",
                                     "stratum": f">={threshold}_each_half", "residual_type": value,
                                     "metric": metric, **stats[metric], "defined_edges": int(edge_mask.sum())})
            half_without_threshold = [residual_arrays(counts, probabilities[i], design.group_start, 0)
                                      for i, counts in enumerate(original_counts)]
            for value in RESIDUAL_TYPES:
                stats = reliability(design, half_without_threshold[0], half_without_threshold[1], value,
                                    full_bundles[name]["reliable"])
                edge_mask = (full_bundles[name]["reliable"] & np.isfinite(half_without_threshold[0][value])
                             & np.isfinite(half_without_threshold[1][value]))
                for metric in ("edge_weighted", "source_equal_flatten"):
                    rows.append({"repeat": repeat, "specification": name,
                                 "analysis": "full84_eligibility_mask", "category": "all",
                                 "stratum": ">=10_full84_only", "residual_type": value,
                                 "metric": metric, **stats[metric], "defined_edges": int(edge_mask.sum())})
            for category_name, selected in masks.items():
                for value in RESIDUAL_TYPES:
                    stats = reliability(design, thinned[0], thinned[1], value, selected)
                    edge_mask = (thinned[0]["reliable"] & thinned[1]["reliable"] & selected
                                 & np.isfinite(thinned[0][value]) & np.isfinite(thinned[1][value]))
                    for metric in ("edge_weighted", "source_equal_flatten"):
                        rows.append({"repeat": repeat, "specification": name, "analysis": "exposure_matched",
                                     "category": category_name, "stratum": "subject_category_far_rate",
                                     "residual_type": value, "metric": metric, **stats[metric],
                                     "defined_edges": int(edge_mask.sum())})
    summary = aggregate(rows)
    output = {"status": "complete" if args.repeats >= 100 else "pilot", "seed": args.seed,
        "repeats": args.repeats, "subjects": len(subjects), "specifications": list(SPECS),
        "risk_set": "common_forward_same_sentence_same_line", "candidate_universe_changed": False,
        "residual_definitions": {"pearson": "(y-E)/sqrt(E(1-p)); undefined for singleton risk sets or nonpositive variance",
            "deviance": "signed sqrt of 2[y log(y/E)-(y-E)], with 0 log(0/E)=0",
            "raw_deviation": "y-E", "common_support": "All residual-scale comparisons exclude singleton risk sets, which contain no within-source destination allocation contrast.",
            "limitation": "Cellwise signed deviance is a descriptive allocation of conditional multinomial group deviance; cells are dependent and it is not an independent-Poisson residual or a unique one-dimensional multinomial residual."},
        "thinning_design": "Within each subject and category, independently binomial-thin adjacent and near counts to that subject's far-category total mass when possible; retain every candidate edge and far counts. One seeded thinning draw per each of 100 independently randomized reader partitions. Original half-fitted nuisance probabilities are held fixed, so this isolates measurement mass rather than refitting consequences.",
        "eligibility_sensitivity": {"primary_rule": "source exposure >=5 in each half",
            "partitions": args.repeats,
            "half_thresholds": list(HALF_ELIGIBILITY_THRESHOLDS),
            "full84_rule": "source exposure >=10 in the fixed full-84 aggregate, applied to both half residual vectors without an additional half threshold",
            "conditioning_note": "Exposure is an observed outcome count. Threshold sensitivity reuses each half's fitted probabilities and changes only the source eligibility rule."},
        "guardrail": "Lower far-category reproducibility indicates observed reproducibility under sparse measurement, not weaker latent structure. No edge-level significance tests are performed.",
        "full_sample": full, "target_category_exposure": category, "reliability_summary": summary,
        "repeat_rows": rows, "thinning_probabilities": thinning_audit}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = Path(args.csv_output); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows)))); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
