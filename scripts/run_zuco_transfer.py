from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eyetrack2llm import Fixations, extract_events, read_fixation_csv
from eyetrack2llm.auxiliary import (CACHE_FORMAT_VERSION, MODEL_ID, MODEL_REVISION, RESIDUAL_SUPPORT_POLICY, canonical_sha256,
    encoding_identity, file_sha256, load_trainable_state_dict, pretrained_provenance, text_sha256,
    validate_checkpoint)
from eyetrack2llm.baseline import WordMetadata, build_pair_design, count_vector, enrich_spacy_syntax, enrich_word_frequencies
from eyetrack2llm.torch import LowRankDirectedHead, ResidualBottleneckAdapter, word_pool
from eyetrack2llm.transfer import crossfit_residual_targets, relation_scores
from eyetrack2llm.provo import cluster_vertical_intervals
from eyetrack2llm.zuco import validate_subject_line_partitions


SUBJECTS = ("ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH")
TEXTS = tuple(f"NR:{index}" for index in range(101, 301))
SEED = 20260711


class TransferModel(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.adapter = ResidualBottleneckAdapter(hidden_size, rank)
        self.gaze_head = LowRankDirectedHead(hidden_size, rank)


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else None


def build_metadata(rows, nlp):
    items = []
    for text in TEXTS:
        words = rows[text]["words"]
        bounds = rows[text]["bounds"]
        lines = cluster_vertical_intervals([(float(bound[1]), float(bound[3])) for bound in bounds])
        for index, surface in enumerate(words):
            length = max(1, len(re.sub(r"[^A-Za-z0-9']", "", surface)))
            items.append((text, index, surface, np.log(float(length)), bool(re.search(r"[.!?][\"')\]]*$", surface)), lines[index]))
    metadata = WordMetadata(
        text_id=np.asarray([x[0] for x in items]), word_index=np.asarray([x[1] for x in items]),
        log_length=np.asarray([x[3] for x in items]), cloze_logit=np.zeros(len(items)),
        cloze_missing=np.ones(len(items), bool), terminal_punctuation=np.asarray([x[4] for x in items]),
        sentence_number=np.ones(len(items), np.int64), surface=np.asarray([x[2] for x in items]),
        line_id=np.asarray([x[5] for x in items], np.int64),
    )
    metadata = enrich_word_frequencies(metadata)
    return enrich_spacy_syntax(metadata, nlp)


def cache_hidden(metadata, path: Path):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    provenance = pretrained_provenance()
    source_words, encodings, identities = {}, {}, {}
    for text in TEXTS:
        positions = np.flatnonzero(metadata.text_id == text)
        words = metadata.surface[positions].tolist()
        encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt")
        ids = encoded.word_ids(0)
        source_words[text] = words
        encodings[text] = (encoded, ids)
        identities[text] = encoding_identity(encoded, ids)
    input_identity = {"text_sha256": text_sha256(source_words), "per_text": identities,
                      "tokenization_sha256": canonical_sha256(identities)}
    fingerprint = canonical_sha256({"format_version": CACHE_FORMAT_VERSION, "provenance": provenance,
                                    "inputs": input_identity, "texts": list(TEXTS)})
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("format_version") == CACHE_FORMAT_VERSION and cached.get("fingerprint") == fingerprint:
            return cached
    encoder = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).eval()
    cached = {"format_version": CACHE_FORMAT_VERSION, "artifact_type": "zuco_transfer_bert_inputs",
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "provenance": provenance,
              "inputs": input_identity, "fingerprint": fingerprint, "hidden_layer": "last", "texts": {}}
    with torch.inference_mode():
        for text in TEXTS:
            words = source_words[text]
            encoded, ids = encodings[text]
            aligned = len(words) == len(set(x for x in ids if x is not None)) and all(ids.count(i) > 0 for i in range(len(words)))
            if not aligned or encoded.input_ids.shape[1] >= 512:
                raise ValueError(f"BERT word alignment failed for {text}")
            hidden = encoder(**encoded, return_dict=True).last_hidden_state[0].to(torch.float16).cpu()
            cached["texts"][text] = {"hidden": hidden, "word_ids": torch.tensor([-1 if x is None else x for x in ids]),
                                            "words": words, "tokens": int(encoded.input_ids.shape[1]), "aligned": aligned}
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(cached, path)
    return cached


def model_result(model, cache, targets):
    all_y, all_p, per_text = [], [], {}
    model.eval()
    with torch.no_grad():
        for text in TEXTS:
            item, target = cache["texts"][text], targets[text]
            scores = relation_scores(model.adapter, model.gaze_head, item["hidden"].float(), item["word_ids"])
            mask = target["reliable"]; y = target["residual"][mask]
            p = scores[torch.from_numpy(target["src"][mask]), torch.from_numpy(target["dst"][mask])].numpy()
            all_y.extend(y.tolist()); all_p.extend(p.tolist())
            per_text[text] = {"correlation": correlation(y, p), "n_edges": int(mask.sum())}
    return {"overall_correlation": correlation(all_y, all_p), "n_edges": len(all_y), "per_text": per_text}


def cosine_result(cache, targets):
    all_y, all_p, per_text = [], [], {}
    for text in TEXTS:
        item, target = cache["texts"][text], targets[text]
        words, _ = word_pool(item["hidden"].float().unsqueeze(0), item["word_ids"].unsqueeze(0)); words = words[0]
        words = torch.nn.functional.normalize(words, dim=-1); scores = words @ words.T
        mask = target["reliable"]; y = target["residual"][mask]
        p = scores[torch.from_numpy(target["src"][mask]), torch.from_numpy(target["dst"][mask])].numpy()
        all_y.extend(y.tolist()); all_p.extend(p.tolist()); per_text[text] = {"correlation": correlation(y, p), "n_edges": int(mask.sum())}
    return {"overall_correlation": correlation(all_y, all_p), "n_edges": len(all_y), "per_text": per_text}


def paired_summary(results, first, second, seed=SEED, repeats=10000):
    seed_differences = [results[str(s)][first]["overall_correlation"] - results[str(s)][second]["overall_correlation"] for s in sorted(map(int, results))]
    text_values = {}
    for text in TEXTS:
        a = [results[str(s)][first]["per_text"][text]["correlation"] for s in sorted(map(int, results))]
        b = [results[str(s)][second]["per_text"][text]["correlation"] for s in sorted(map(int, results))]
        enough_edges = all(results[str(s)][condition]["per_text"][text]["n_edges"] >= 4
                           for s in sorted(map(int, results)) for condition in (first, second))
        if enough_edges and all(value is not None for value in a + b):
            a_z = np.arctanh(np.clip(a, -1 + 1e-7, 1 - 1e-7))
            b_z = np.arctanh(np.clip(b, -1 + 1e-7, 1 - 1e-7))
            text_values[text] = float(np.mean(a_z) - np.mean(b_z))
    values = np.asarray(list(text_values.values())); rng = np.random.default_rng(seed)
    bootstrap = np.mean(values[rng.integers(len(values), size=(repeats, len(values)))], axis=1)
    return {"overall_raw_descriptive": {"per_seed_differences": seed_differences,
                                         "mean_seed_difference": float(np.mean(seed_differences))},
            "text_equal_fisher_z": {"minimum_edges": 4, "texts_valid": len(values), "bootstrap_seed": seed,
              "per_text_seed_averaged_differences": text_values, "mean_difference": float(values.mean()),
              "descriptive_text_resampling_interval": np.quantile(bootstrap, [.025, .975]).tolist()}}


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed Provo relation checkpoints on all-subject ZuCo NR101-300")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--cache", default="data/processed/zuco_transfer_bert.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); processed = Path("data/processed")
    metadata_rows, fixation_items, subject_metadata = {}, [], {}
    for subject in SUBJECTS:
        stem = f"zuco_{subject.lower()}_nr"
        rows = json.loads((processed / f"{stem}_words.json").read_text(encoding="utf-8")); by_id = {row["text_id"]: row for row in rows}
        subject_metadata[subject] = by_id
        if not all(text in by_id for text in TEXTS): raise ValueError(f"{subject} lacks a common target text")
        if not metadata_rows: metadata_rows = {text: by_id[text] for text in TEXTS}
        elif any(by_id[text]["words"] != metadata_rows[text]["words"] for text in TEXTS): raise ValueError(f"Metadata mismatch for {subject}")
        fixation_items.append(read_fixation_csv(processed / f"{stem}_fixations.csv"))
    line_partition_audit = validate_subject_line_partitions(subject_metadata, TEXTS)
    fixations = Fixations(*[
        np.concatenate([getattr(item, field) for item in fixation_items])
        for field in ("subject_id", "text_id", "word_index", "fixation_order", "duration_ms", "line_id")
    ])
    events = extract_events(fixations, include_self=False, require_consecutive_order=True)
    selected = np.isin(events.text_id, TEXTS) & (events.event_type == "forward")
    edge_counts = defaultdict(float)
    for text, src, dst, weight in zip(events.text_id[selected], events.src_word[selected], events.dst_word[selected], events.weight[selected], strict=True):
        edge_counts[(str(text), int(src), int(dst))] += float(weight)
    import spacy
    metadata, syntax_audit = build_metadata(metadata_rows, spacy.load("en_core_web_sm"))
    design = build_pair_design(metadata, "common_core", "common_forward_same_sentence_same_line")
    counts = count_vector(design, edge_counts)
    targets, scaling, folds = crossfit_residual_targets(design, counts, seed=SEED, scale_clip=False)
    cache = cache_hidden(metadata, Path(args.cache))
    checkpoints = {}
    for path in Path(args.checkpoint_dir).glob("provo_auxiliary_seed*_*.pt"):
        match = re.fullmatch(r"provo_auxiliary_seed(\d+)_(mlm|gaze|shuffled|position)\.pt", path.name)
        if match: checkpoints[(int(match.group(1)), match.group(2))] = path
    seeds = sorted({seed for seed, condition in checkpoints if condition == "gaze"})
    conditions = ("mlm", "gaze", "shuffled", "position")
    if len(seeds) != 5 or any((seed, condition) not in checkpoints for seed in seeds for condition in conditions):
        raise ValueError("Expected five complete mlm/gaze/shuffled/position checkpoint sets")
    results = {}
    checkpoint_metadata = {}
    for seed in seeds:
        results[str(seed)] = {}; checkpoint_metadata[str(seed)] = {}
        for condition in conditions:
            checkpoint = torch.load(checkpoints[(seed, condition)], map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, cache["provenance"], condition=condition, seed=seed)
            model = TransferModel(checkpoint["hidden_size"], checkpoint["rank"]); load_trainable_state_dict(model, checkpoint["state_dict"])
            results[str(seed)][condition] = model_result(model, cache, targets)
            checkpoint_metadata[str(seed)][condition] = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    cosine = cosine_result(cache, targets)
    checkpoint_set_sha256 = canonical_sha256({seed: {condition: values[condition]["state_dict_sha256"]
        for condition in conditions} for seed, values in checkpoint_metadata.items()})
    output = {"status": "complete", "schema_version": 3, "model": MODEL_ID, "model_revision": MODEL_REVISION,
              "pretrained_provenance": cache["provenance"], "cache_fingerprint": cache["fingerprint"],
              "cache_file_sha256": file_sha256(args.cache), "checkpoint_set_sha256": checkpoint_set_sha256,
              "design": {"subjects": list(SUBJECTS), "texts": [TEXTS[0], TEXTS[-1]], "n_texts": len(TEXTS),
              "event": "forward-only same-sentence same-line candidate risk set and counts", "aggregation": "sum counts over all 12 subjects", "min_source_exposure": 10,
              "baseline": "fixed-seed 5-fold sentence cross-fitting (160 train, 40 held out)", "folds": [{"train": train, "test": test} for train, test in folds],
              "feature_names": list(design.feature_names), "design_rank": design.design_rank(),
              "group_constant_features": list(design.group_constant_features()), "cloze_all_missing": True,
              "candidate_pairs": len(design.features), "source_groups": len(design.group_start) - 1,
               "target_residual": "unclipped raw cross-fitted Pearson residual",
               "residual_support_policy": RESIDUAL_SUPPORT_POLICY, "scaling": scaling},
              "audit": {"line_partition_identity": line_partition_audit,
                         "bert_word_alignment": float(np.mean([item["aligned"] for item in cache["texts"].values()])),
                        "bert_hidden_layer": cache["hidden_layer"], "max_bert_tokens": max(item["tokens"] for item in cache["texts"].values()),
                        "syntax": {key: value for key, value in syntax_audit.items() if key != "sentence_reports"}},
              "checkpoint_metadata": checkpoint_metadata, "seed_results": results, "unadapted_bert_cosine": cosine,
               "comparisons": {"gaze_vs_mlm": paired_summary(results, "gaze", "mlm"),
                              "gaze_vs_shuffled": paired_summary(results, "gaze", "shuffled", SEED + 1),
                              "gaze_vs_position": paired_summary(results, "gaze", "position", SEED + 2)},
               "interpretation_rule": "Fixed-reader contrasts describe cross-corpus scorer evaluation against a corpus-locally recalibrated constructed-residual criterion; they do not define population transport success or gaze-specific information beyond residual geometry."}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"seed_overall": {seed: {condition: value["overall_correlation"] for condition, value in conditions.items()} for seed, conditions in results.items()},
                      "cosine": cosine["overall_correlation"], "comparisons": output["comparisons"]}, indent=2))


if __name__ == "__main__": main()
