from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eyetrack2llm import extract_events, read_fixation_csv
from eyetrack2llm.auxiliary import load_trainable_state_dict
from eyetrack2llm.baseline import (build_pair_design, count_vector, enrich_spacy_syntax,
                                   enrich_word_frequencies, read_provo_word_metadata)
from eyetrack2llm.stability import (balanced_subject_subsets, crossfit_raw_residuals,
                                    paired_residual_metrics, weighted_correlation)
from run_zuco_transfer import (SEED, SUBJECTS, TEXTS, TransferModel, build_metadata,
                               cache_hidden, relation_scores)

METRICS = ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal")
CONTRASTS = (("gaze", "mlm"), ("gaze", "shuffled"), ("gaze", "position"))


def counts_by_subject(design, events, subjects=None):
    names = sorted(set(map(str, events.subject_id))) if subjects is None else list(subjects)
    result = {}
    for subject in names:
        selected = (events.subject_id == subject) & (events.event_type == "forward")
        edges = defaultdict(float)
        for text, src, dst, weight in zip(events.text_id[selected], events.src_word[selected],
                                          events.dst_word[selected], events.weight[selected], strict=True):
            edges[(str(text), int(src), int(dst))] += float(weight)
        result[subject] = count_vector(design, edges)
    return result


def unique_six_splits(subjects):
    subjects = tuple(sorted(subjects)); anchor = subjects[0]
    return [(tuple((anchor,) + chosen), tuple(x for x in subjects if x not in (anchor,) + chosen))
            for chosen in itertools.combinations(subjects[1:], 5)]


def distribution(values):
    values = np.asarray([x for x in values if x is not None], float)
    return {"n": int(len(values)), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "negative_proportion": float(np.mean(values < 0))}


def partition_audit(design, by_subject, partitions, seed):
    rows = []
    for index, halves in enumerate(partitions):
        targets = [crossfit_raw_residuals(design, np.sum([by_subject[x] for x in half], axis=0),
                                          seed=seed, min_exposure=5)[0] for half in halves]
        metrics = paired_residual_metrics(*targets)
        rows.append({"partition": index, "halves": [list(x) for x in halves], "metrics": metrics})
    summary = {}
    for metric in METRICS:
        summary[metric] = distribution([row["metrics"]["text_summary"][metric]["median"] for row in rows])
        summary[metric]["text_negative_proportion"] = distribution([
            row["metrics"]["text_summary"][metric]["negative_proportion"] for row in rows])
        summary[metric]["text_undefined_proportion"] = distribution([
            row["metrics"]["text_summary"][metric]["undefined_proportion"] for row in rows])
    return rows, summary


def frozen_scores(metadata, cache, checkpoint_dir):
    paths = {}
    for path in checkpoint_dir.glob("provo_auxiliary_seed*_*.pt"):
        match = re.fullmatch(r"provo_auxiliary_seed(\d+)_(mlm|gaze|shuffled|position)\.pt", path.name)
        if match: paths[(int(match.group(1)), match.group(2))] = path
    seeds = sorted({seed for seed, condition in paths if condition == "gaze"})
    if len(seeds) != 5 or any((seed, c) not in paths for seed in seeds for c in ("gaze", "mlm", "shuffled", "position")):
        raise ValueError("Expected five complete frozen checkpoint sets")
    scores = {c: {text: [] for text in TEXTS} for c in ("gaze", "mlm", "shuffled", "position")}
    for seed in seeds:
        for condition in scores:
            checkpoint = torch.load(paths[(seed, condition)], map_location="cpu", weights_only=False)
            model = TransferModel(checkpoint["hidden_size"], checkpoint["rank"])
            load_trainable_state_dict(model, checkpoint["state_dict"]); model.eval()
            with torch.no_grad():
                for text in TEXTS:
                    item = cache["texts"][text]
                    scores[condition][text].append(relation_scores(
                        model.adapter, model.gaze_head, item["hidden"].float(), item["word_ids"]).numpy())
    return {c: {text: np.asarray(value) for text, value in texts.items()} for c, texts in scores.items()}, seeds


def criterion_model_values(targets, scores):
    per_text = {f"{a}_vs_{b}": {} for a, b in CONTRASTS}
    correlations = {condition: {} for condition in scores}
    for text, target in targets.items():
        mask = target["reliable"]
        if mask.sum() < 4: continue
        y = target["residual"][mask]
        for condition in scores:
            correlations[condition][text] = [weighted_correlation(
                y, seed_scores[target["src"][mask], target["dst"][mask]])
                for seed_scores in scores[condition][text]]
        for a, b in CONTRASTS:
            if all(value is not None for value in correlations[a][text] + correlations[b][text]):
                az = np.arctanh(np.clip(correlations[a][text], -.9999999, .9999999))
                bz = np.arctanh(np.clip(correlations[b][text], -.9999999, .9999999))
                per_text[f"{a}_vs_{b}"][text] = float(np.mean(az) - np.mean(bz))
    return per_text, correlations


def bootstrap_audit(design, by_subject, scores, repeats, text_repeats, seed):
    rng = np.random.default_rng(seed); subjects = np.asarray(sorted(by_subject)); records = []
    for repeat in range(repeats):
        sampled = subjects[rng.integers(len(subjects), size=len(subjects))]
        targets = crossfit_raw_residuals(design, np.sum([by_subject[x] for x in sampled], axis=0),
                                         seed=seed, min_exposure=10)[0]
        values, _ = criterion_model_values(targets, scores)
        record = {"reader_bootstrap": repeat, "readers": sampled.tolist(), "contrasts": {}}
        for name, text_values in values.items():
            array = np.asarray(list(text_values.values()), float)
            nested = np.mean(array[rng.integers(len(array), size=(text_repeats, len(array)))], axis=1)
            record["contrasts"][name] = {"fixed_text_mean": float(array.mean()), "valid_texts": len(array),
                                           "nested_text_means": nested.tolist()}
        records.append(record)
    summary = {}
    for name in records[0]["contrasts"]:
        fixed = [x["contrasts"][name]["fixed_text_mean"] for x in records]
        nested = [v for x in records for v in x["contrasts"][name]["nested_text_means"]]
        summary[name] = {"reader_generalization_fixed_texts": {**distribution(fixed), "95_ci": np.quantile(fixed, [.025, .975]).tolist()},
                         "joint_reader_and_text": {**distribution(nested), "95_ci": np.quantile(nested, [.025, .975]).tolist()}}
    return records, summary


def main():
    parser = argparse.ArgumentParser(description="ZuCo strict-line criterion reliability and reader uncertainty audit")
    parser.add_argument("--output", default="data/processed/zuco_strictline_criterion_uncertainty.json")
    parser.add_argument("--csv-output", default="data/processed/zuco_strictline_criterion_uncertainty.csv")
    parser.add_argument("--checkpoint-dir", default="data/processed/strictline_fixed50")
    parser.add_argument("--cache", default="data/processed/zuco_transfer_bert.pt")
    parser.add_argument("--reader-bootstraps", type=int, default=200)
    parser.add_argument("--text-bootstraps", type=int, default=200)
    parser.add_argument("--provo-subsets", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(); processed = Path("data/processed")
    metadata_rows, event_items = {}, []
    for subject in SUBJECTS:
        stem = f"zuco_{subject.lower()}_nr"
        rows = json.loads((processed / f"{stem}_words.json").read_text(encoding="utf-8")); by_id = {x["text_id"]: x for x in rows}
        if not metadata_rows: metadata_rows = {text: by_id[text] for text in TEXTS}
        event_items.append(extract_events(read_fixation_csv(processed / f"{stem}_fixations.csv"), include_self=False, require_consecutive_order=True))
    class Events: pass
    events = Events()
    for field in ("subject_id", "text_id", "src_word", "dst_word", "event_type", "weight"):
        setattr(events, field, np.concatenate([getattr(x, field) for x in event_items]))
    import spacy
    nlp = spacy.load("en_core_web_sm"); metadata, syntax = build_metadata(metadata_rows, nlp)
    design = build_pair_design(metadata, "common_core", "common_forward_same_sentence_same_line")
    by_subject = counts_by_subject(design, events, SUBJECTS)
    partitions = unique_six_splits(SUBJECTS); partition_rows, reliability = partition_audit(design, by_subject, partitions, args.seed)
    full_targets = crossfit_raw_residuals(design, np.sum(list(by_subject.values()), axis=0), seed=args.seed, min_exposure=10)[0]
    cache = cache_hidden(metadata, Path(args.cache)); scores, checkpoint_seeds = frozen_scores(metadata, cache, Path(args.checkpoint_dir))
    full_values, full_correlations = criterion_model_values(full_targets, scores)
    fixed12 = {name: {"mean": float(np.mean(list(values.values()))), "valid_texts": len(values)} for name, values in full_values.items()}
    ceiling = {}
    for metric in METRICS:
        half_r = reliability[metric]["median"]
        full_sensitivity = 2 * half_r / (1 + half_r) if half_r > -1 else None
        ceiling[metric] = {"half_pattern_reliability_primary": half_r,
                           "spearman_brown_full_reliability_sensitivity_only": full_sensitivity,
                           "sqrt_attenuation_bound_sensitivity_only": float(np.sqrt(max(0, full_sensitivity))) if full_sensitivity is not None else None,
                           "caution": "Descriptive upper-bound interpretation requires classical parallel-form, independent-error, and criterion/model-error assumptions; these are not established."}
    bootstrap_records, bootstrap_summary = bootstrap_audit(design, by_subject, scores, args.reader_bootstraps, args.text_bootstraps, args.seed + 31)

    pf = read_fixation_csv(processed / "provo_fixations_with_lines.csv"); pe = extract_events(pf, include_self=False, require_consecutive_order=True)
    observed = defaultdict(set)
    for text, word in zip(pf.text_id, pf.word_index, strict=True): observed[str(text)].add(int(word))
    pm = enrich_word_frequencies(read_provo_word_metadata("data/raw/Provo_Corpus-Eyetracking_Data.csv", dict(observed)))
    pm, _ = enrich_spacy_syntax(pm, nlp); pd = build_pair_design(pm, "common_core", "common_forward_same_sentence_same_line")
    pc = counts_by_subject(pd, pe); subsets = balanced_subject_subsets(pc, 12, args.provo_subsets, args.seed + 71)
    provo_partitions = [(tuple(x[:6]), tuple(x[6:])) for x in subsets]
    provo_rows, provo_summary = partition_audit(pd, pc, provo_partitions, args.seed)
    output = {"status": "complete" if args.reader_bootstraps >= 200 and args.provo_subsets >= 200 else "pilot",
              "seed": args.seed, "design": {"subjects": 12, "texts": 200, "partitions": len(partitions), "split": "6/6",
              "risk_set": "common_forward_same_sentence_same_line", "nuisance": "half-independent corpus-local 5-fold common-core fit",
              "target": "raw unclipped Pearson residual", "full_min_exposure": 10, "half_min_exposure": 5},
              "reliability": {"summary": reliability, "partitions": partition_rows}, "noise_ceiling": ceiling,
              "frozen_model": {"checkpoint_seeds": checkpoint_seeds, "fixed12_text_conditioned": fixed12,
                               "per_condition_text_correlations": full_correlations},
              "reader_bootstrap": {"repeats": args.reader_bootstraps, "nested_text_repeats": args.text_bootstraps,
                                   "summary": bootstrap_summary, "records": bootstrap_records,
                                   "distinction": "fixed12 text intervals condition on the observed readers; reader bootstrap targets reader-population generalization and nested resampling additionally varies texts."},
              "provo_12_reader_sensitivity": {"subsets": args.provo_subsets, "selection": "balanced 12-reader subsets; deterministic 6/6 halves; exposure matched at 5 per half", "summary": provo_summary, "records": provo_rows},
              "syntax_audit": {k: v for k, v in syntax.items() if k != "sentence_reports"}}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    with Path(args.csv_output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["analysis", "replicate", "metric", "value"])
        for row in partition_rows:
            for metric in METRICS: writer.writerow(["zuco_partition", row["partition"], metric, row["metrics"]["text_summary"][metric]["median"]])
        for row in provo_rows:
            for metric in METRICS: writer.writerow(["provo_12_reader", row["partition"], metric, row["metrics"]["text_summary"][metric]["median"]])
        for row in bootstrap_records:
            for name, value in row["contrasts"].items(): writer.writerow(["zuco_reader_bootstrap", row["reader_bootstrap"], name, value["fixed_text_mean"]])


if __name__ == "__main__": main()
