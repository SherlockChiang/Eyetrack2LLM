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
    read_provo_word_metadata,
)
from eyetrack2llm.stability import paired_residual_metrics, weighted_correlation
from eyetrack2llm.transfer import sentence_folds


SEED = 20260711
SPECIFICATIONS = tuple(STRICT_LINE_SPECIFICATION_SETS)
METRICS = ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal")
IDENTITY_PAIRS = (("lexical", "syntax"), ("lexical", "flexible"),
                  ("syntax", "flexible"))
PROBABILITY_BINS = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0000001)
EXPECTED_BINS = (0.0, 0.5, 1.0, 2.0, 5.0, float("inf"))


def distribution(values):
    values = np.asarray(values, float)
    return {"median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)),
            "range": [float(values.min()), float(values.max())]}


def residual_bundle(counts, probability, group_start, min_exposure):
    from eyetrack2llm.baseline import residual_vector

    lengths = np.diff(group_start)
    exposure = np.repeat(np.add.reduceat(counts, group_start[:-1]), lengths)
    expected = exposure * probability
    deviation = counts - expected
    residual, _ = residual_vector(counts, probability, group_start)
    return {"residual": residual, "deviation": deviation, "exposure": exposure,
            "expected": expected, "probability": probability,
            "reliable": (exposure >= min_exposure) & np.isfinite(residual)}


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


def text_records(design, bundle):
    records = {}
    for text in sorted(set(design.text_id), key=int):
        mask = design.text_id == text
        records[text] = {key: value[mask] for key, value in bundle.items()
                         if key in {"residual", "exposure", "reliable"}}
        records[text]["src"] = design.src_word[mask]
        records[text]["dst"] = design.dst_word[mask]
    return records


def reliability(design, left, right, value="residual"):
    converted = []
    for bundle in (left, right):
        records = {}
        for text in sorted(set(design.text_id), key=int):
            mask = design.text_id == text
            records[text] = {"residual": bundle[value][mask],
                             "exposure": bundle["exposure"][mask],
                             "reliable": bundle["reliable"][mask],
                             "src": design.src_word[mask], "dst": design.dst_word[mask]}
        converted.append(records)
    metrics = paired_residual_metrics(*converted)
    return {name: metrics["text_summary"][name]["median"] for name in METRICS}


def identity(design, left, right, value="residual"):
    per_text, pooled_left, pooled_right = [], [], []
    for text in sorted(set(design.text_id), key=int):
        mask = design.text_id == text
        eligible = mask & left["reliable"] & right["reliable"]
        correlation = weighted_correlation(left[value][eligible], right[value][eligible])
        if correlation is not None:
            per_text.append(correlation)
            pooled_left.extend(left[value][eligible]); pooled_right.extend(right[value][eligible])
    return {"text_equal_median": float(np.median(per_text)) if per_text else None,
            "edge_descriptive": weighted_correlation(np.asarray(pooled_left), np.asarray(pooled_right)),
            "valid_texts": len(per_text), "n_edges": len(pooled_left)}


def stratified_rows(bundle, repeat, half, specification, mode):
    eligible = bundle["reliable"]
    rows = []
    for variable, bins in (("probability", PROBABILITY_BINS), ("expected", EXPECTED_BINS)):
        values = bundle[variable]
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            selected = eligible & (values >= lower) & (values < upper)
            if np.any(selected):
                rows.append({"record_type": "stratum", "repeat": repeat, "half": half,
                             "specification": specification, "mode": mode, "quantity": variable,
                             "lower": lower, "upper": upper, "n_edges": int(selected.sum()),
                             "residual_variance": float(np.var(bundle["residual"][selected])),
                             "deviation_variance": float(np.var(bundle["deviation"][selected])),
                             "mean_probability": float(np.mean(bundle["probability"][selected])),
                             "mean_expected": float(np.mean(bundle["expected"][selected]))})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Half-specific identity and baseline-noise audit")
    parser.add_argument("--fixations", default="data/processed/provo_fixations_with_lines.csv")
    parser.add_argument("--main-csv", default="data/raw/Provo_Corpus-Eyetracking_Data.csv")
    parser.add_argument("--output", default="data/processed/provo_half_specific_baseline_audit.json")
    parser.add_argument("--csv-output", default="data/processed/provo_half_specific_baseline_audit.csv")
    parser.add_argument("--repeats", type=int, default=100)
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
    reference = designs["position_only"]
    by_subject = {}
    for subject in sorted(set(map(str, events.subject_id))):
        selected = (events.subject_id == subject) & (events.event_type == "forward")
        edges = defaultdict(float)
        for text, src, dst, weight in zip(events.text_id[selected], events.src_word[selected],
                                          events.dst_word[selected], events.weight[selected], strict=True):
            edges[(str(text), int(src), int(dst))] += float(weight)
        by_subject[subject] = count_vector(reference, edges)
    subjects = sorted(by_subject)
    if len(subjects) != 84:
        raise ValueError(f"Expected 84 subjects, found {len(subjects)}")
    folds = sentence_folds(reference.text_id, 5, args.seed)
    full_counts = np.sum(list(by_subject.values()), axis=0)
    shared, full_coefficients = {}, {}
    for name, design in designs.items():
        shared[name], full_coefficients[name] = fit_predictions(design, full_counts, folds)

    rng = np.random.default_rng(args.seed)
    repeat_records, identity_records, csv_rows = [], [], []
    coefficient_values = {name: [] for name in SPECIFICATIONS}
    for repeat in range(args.repeats):
        shuffled = np.asarray(subjects)[rng.permutation(84)]
        halves = (shuffled[:42], shuffled[42:])
        half_counts = [np.sum([by_subject[subject] for subject in half], axis=0) for half in halves]
        bundles = {"half_specific": [{}, {}], "shared_full84": [{}, {}]}
        for half_index, counts in enumerate(half_counts):
            for name, design in designs.items():
                probability, coefficients = fit_predictions(design, counts, folds)
                coefficient_values[name].append(coefficients)
                bundles["half_specific"][half_index][name] = residual_bundle(
                    counts, probability, design.group_start, args.min_exposure)
                bundles["shared_full84"][half_index][name] = residual_bundle(
                    counts, shared[name], design.group_start, args.min_exposure)
                for mode in bundles:
                    bundle = bundles[mode][half_index][name]
                    selected = bundle["reliable"]
                    repeat_records.append({"repeat": repeat, "half": half_index + 1,
                        "specification": name, "mode": mode,
                        "residual_variance": float(np.var(bundle["residual"][selected])),
                        "deviation_variance": float(np.var(bundle["deviation"][selected]))})
                    csv_rows.extend(stratified_rows(bundle, repeat, half_index + 1, name, mode))
        for mode in bundles:
            for name, design in designs.items():
                for scale in ("residual", "deviation"):
                    values = reliability(design, bundles[mode][0][name], bundles[mode][1][name], scale)
                    for metric, value in values.items():
                        csv_rows.append({"record_type": "reliability", "repeat": repeat,
                            "half": "both", "specification": name, "mode": mode,
                            "quantity": scale, "metric": metric, "value": value})
            for half_index in range(2):
                for left, right in IDENTITY_PAIRS:
                    for scale in ("residual", "deviation"):
                        values = identity(reference, bundles[mode][half_index][left],
                                          bundles[mode][half_index][right], scale)
                        record = {"repeat": repeat, "half": half_index + 1, "mode": mode,
                                  "pair": f"{left}__{right}", "quantity": scale, **values}
                        identity_records.append(record)
                        csv_rows.append({"record_type": "identity", "repeat": repeat,
                            "half": half_index + 1, "specification": record["pair"], "mode": mode,
                            "quantity": scale, "value": values["text_equal_median"],
                            "edge_descriptive": values["edge_descriptive"],
                            "n_edges": values["n_edges"]})

    reliability_summary = {}
    for mode in ("half_specific", "shared_full84"):
        reliability_summary[mode] = {}
        for name in SPECIFICATIONS:
            reliability_summary[mode][name] = {}
            for scale in ("residual", "deviation"):
                reliability_summary[mode][name][scale] = {}
                for metric in METRICS:
                    selected = [row["value"] for row in csv_rows if row.get("record_type") == "reliability"
                                and row["mode"] == mode and row["specification"] == name
                                and row["quantity"] == scale and row["metric"] == metric]
                    reliability_summary[mode][name][scale][metric] = distribution(selected)
    identity_summary = {}
    for mode in ("half_specific", "shared_full84"):
        identity_summary[mode] = {}
        for pair in (f"{a}__{b}" for a, b in IDENTITY_PAIRS):
            identity_summary[mode][pair] = {}
            for scale in ("residual", "deviation"):
                selected = [row for row in identity_records if row["mode"] == mode
                            and row["pair"] == pair and row["quantity"] == scale]
                identity_summary[mode][pair][scale] = {
                    "text_equal_median": distribution([row["text_equal_median"] for row in selected]),
                    "edge_descriptive": distribution([row["edge_descriptive"] for row in selected])}
    variance_summary = {}
    strata_summary = {}
    for mode in ("half_specific", "shared_full84"):
        variance_summary[mode] = {}
        strata_summary[mode] = {}
        for name in SPECIFICATIONS:
            selected = [row for row in repeat_records if row["mode"] == mode
                        and row["specification"] == name]
            variance_summary[mode][name] = {
                "residual_variance": distribution([row["residual_variance"] for row in selected]),
                "deviation_variance": distribution([row["deviation_variance"] for row in selected])}
            strata_summary[mode][name] = {}
            for quantity in ("probability", "expected"):
                groups = defaultdict(list)
                for row in csv_rows:
                    if (row.get("record_type") == "stratum" and row["mode"] == mode
                            and row["specification"] == name and row["quantity"] == quantity):
                        groups[(row["lower"], row["upper"])].append(row)
                strata_summary[mode][name][quantity] = [{
                    "lower": lower, "upper": upper if np.isfinite(upper) else None,
                    "median_n_edges": float(np.median([row["n_edges"] for row in rows])),
                    "residual_variance": distribution([row["residual_variance"] for row in rows]),
                    "deviation_variance": distribution([row["deviation_variance"] for row in rows])}
                    for (lower, upper), rows in sorted(groups.items())]
    coefficient_summary = {}
    for name, arrays in coefficient_values.items():
        values = np.concatenate(arrays, axis=0)
        coefficient_summary[name] = [{"feature": feature,
            "half_fit_sd": float(np.std(values[:, column], ddof=1)),
            "half_fit_range": [float(values[:, column].min()), float(values[:, column].max())],
            "full84_fold_coefficients": full_coefficients[name][:, column].tolist()}
            for column, feature in enumerate(designs[name].feature_names)]
    mode_difference = {"reliability_half_specific_minus_shared_full84": {},
                       "identity_half_specific_minus_shared_full84": {},
                       "variance_half_specific_minus_shared_full84": {}}
    for name in SPECIFICATIONS:
        mode_difference["reliability_half_specific_minus_shared_full84"][name] = {}
        for scale in ("residual", "deviation"):
            mode_difference["reliability_half_specific_minus_shared_full84"][name][scale] = {}
            for metric in METRICS:
                values = {mode: [row["value"] for row in csv_rows
                    if row.get("record_type") == "reliability" and row["mode"] == mode
                    and row["specification"] == name and row["quantity"] == scale
                    and row["metric"] == metric]
                    for mode in ("half_specific", "shared_full84")}
                mode_difference["reliability_half_specific_minus_shared_full84"][name][scale][metric] = distribution(
                    np.asarray(values["half_specific"]) - np.asarray(values["shared_full84"]))
        values = {mode: [row for row in repeat_records if row["mode"] == mode
                  and row["specification"] == name] for mode in ("half_specific", "shared_full84")}
        mode_difference["variance_half_specific_minus_shared_full84"][name] = {
            quantity: distribution(np.asarray([row[quantity] for row in values["half_specific"]])
                                   - np.asarray([row[quantity] for row in values["shared_full84"]]))
            for quantity in ("residual_variance", "deviation_variance")}
    for pair in (f"{a}__{b}" for a, b in IDENTITY_PAIRS):
        mode_difference["identity_half_specific_minus_shared_full84"][pair] = {}
        for scale in ("residual", "deviation"):
            mode_difference["identity_half_specific_minus_shared_full84"][pair][scale] = {}
            for statistic in ("text_equal_median", "edge_descriptive"):
                values = {mode: [row[statistic] for row in identity_records if row["mode"] == mode
                    and row["pair"] == pair and row["quantity"] == scale]
                    for mode in ("half_specific", "shared_full84")}
                mode_difference["identity_half_specific_minus_shared_full84"][pair][scale][statistic] = distribution(
                    np.asarray(values["half_specific"]) - np.asarray(values["shared_full84"]))
    output = {"status": "complete" if args.repeats == 100 else "pilot", "seed": args.seed,
        "repeats": args.repeats, "subjects": 84, "split": "42/42", "halves_per_repeat": 2,
        "specifications": list(SPECIFICATIONS), "identity_focus": [f"{a}__{b}" for a, b in IDENTITY_PAIRS],
        "baseline_modes": {"half_specific": "fit on each half's counts within each training-text fold",
            "shared_full84": "fixed full-84 crossfit nuisance probabilities applied to each half's counts"},
        "interpretation_guardrail": "diagnostic contrasts separate stable removal, fit noise, and Pearson scaling descriptively; they are not a complete variance decomposition",
        "syntax_audit": {key: value for key, value in syntax_audit.items() if key != "sentence_reports"},
        "reliability_summary": reliability_summary, "identity_summary": identity_summary,
        "variance_summary": variance_summary, "strata_summary": strata_summary,
        "mode_difference": mode_difference,
        "coefficient_variability": coefficient_summary, "half_spec_records": repeat_records,
        "identity_records": identity_records}
    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = Path(args.csv_output); csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in csv_rows)))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
