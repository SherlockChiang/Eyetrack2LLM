from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eyetrack2llm import extract_events, read_fixation_csv
from eyetrack2llm.auxiliary import (CHECKPOINT_FORMAT_VERSION, MODEL_ID, MODEL_REVISION, AuxiliaryModel,
    build_gaze_targets, cache_bert_inputs, file_sha256, load_trainable_state_dict, make_split_manifest,
    signed_huber_source_balanced, source_preserving_shuffle, state_dict_sha256, trainable_state_dict,
    validate_checkpoint)
from eyetrack2llm.baseline import (build_pair_design, count_vector, enrich_spacy_syntax,
    enrich_word_frequencies, read_provo_word_metadata)
from eyetrack2llm.torch import word_pool


def evaluate(model, texts, cache, targets, *, retain_predictions=False):
    model.eval(); total_loss = correct = tokens = 0; gaze_y = []; gaze_p = []; gaze_losses = []; per_text = {}
    with torch.no_grad():
        for text in texts:
            item = cache["texts"][text]
            text_loss = text_correct = text_tokens = 0
            for variant in item["variants"]:
                logits = model.mlm_logits(variant["hidden"].float().clone().unsqueeze(0))[0]
                mask = variant["mask"]
                loss_sum = float(F.cross_entropy(logits[mask], variant["labels"][mask], reduction="sum"))
                variant_tokens = int(mask.sum())
                variant_correct = int((logits[mask].argmax(-1) == variant["labels"][mask]).sum())
                total_loss += loss_sum; tokens += variant_tokens; correct += variant_correct
                text_loss += loss_sum; text_tokens += variant_tokens; text_correct += variant_correct
            words, _ = word_pool(model.adapter(item["clean"].float().clone().unsqueeze(0)), item["word_ids"].unsqueeze(0))
            prediction = model.gaze_head(words)[0]; target = targets[text]; reliable = target["reliable"]
            local = {int(word): index for index, word in enumerate(item["word_indices"])}
            src = torch.tensor([local[int(word)] for word in target["src"][reliable]]); dst = torch.tensor([local[int(word)] for word in target["dst"][reliable]]); y = torch.from_numpy(target["residual"][reliable])
            p = prediction[src, dst]; gaze_losses.append(float(signed_huber_source_balanced(p, y, src))); gaze_y.extend(y.tolist()); gaze_p.extend(p.tolist())
            y_values = y.tolist(); p_values = p.tolist()
            text_correlation = float(np.corrcoef(y_values, p_values)[0, 1]) if np.std(y_values) and np.std(p_values) else None
            per_text[text] = {"mlm_nll": text_loss / text_tokens, "mlm_accuracy": text_correct / text_tokens,
                              "mlm_tokens": text_tokens, "gaze_correlation": text_correlation,
                              "valid_edges": len(y_values)}
            if retain_predictions:
                per_text[text]["edge_predictions"] = [
                    {"src_word": int(source), "dst_word": int(destination), "target": float(target_value),
                     "prediction": float(prediction_value)}
                    for source, destination, target_value, prediction_value in zip(
                        target["src"][reliable], target["dst"][reliable], y_values, p_values, strict=True)
                ]
    correlation = float(np.corrcoef(gaze_y, gaze_p)[0, 1]) if np.std(gaze_p) and np.std(gaze_y) else None
    return {"mlm_nll": total_loss / tokens, "mlm_accuracy": correct / tokens, "mlm_tokens": tokens,
            "gaze_loss": float(np.mean(gaze_losses)), "gaze_correlation": correlation, "per_text": per_text}


def main():
    parser = argparse.ArgumentParser(description="Fixed Provo gaze auxiliary smoke experiment")
    parser.add_argument("--steps", type=int, default=300); parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260711, help="Optimization and shuffle seed")
    parser.add_argument("--split-seed", type=int, default=20260711, help="Fixed split and mask seed")
    parser.add_argument("--output", default="data/processed/provo_auxiliary_smoke.json")
    parser.add_argument("--cache", default="data/processed/provo_auxiliary_bert.pt")
    parser.add_argument("--checkpoint-dir", help="Optional directory for selected adapter and gaze-head checkpoints")
    parser.add_argument("--load-checkpoint-dir", help="Evaluate existing checkpoints without training")
    parser.add_argument("--gaze-weight", type=float, default=0.1)
    parser.add_argument("--retain-predictions", action="store_true")
    parser.add_argument("--fixed-step", type=int, default=None,
                        help="Pre-registered evaluation step loaded for every condition instead of validation selection")
    parser.add_argument("--feature-set", choices=("basic", "lexical", "syntax", "common_core"), default="syntax")
    parser.add_argument("--risk-set", choices=("all", "common_forward_same_sentence", "common_forward_same_sentence_same_line"), default="all")
    args = parser.parse_args(); torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if args.gaze_weight < 0:
        parser.error("--gaze-weight must be nonnegative")
    if args.fixed_step is not None and (args.fixed_step < 1 or args.fixed_step > args.steps or args.fixed_step % args.eval_every):
        parser.error("--fixed-step must be a positive evaluated step no greater than --steps")
    fixation_path = "data/processed/provo_fixations_with_lines.csv"; main_path = "data/raw/Provo_Corpus-Eyetracking_Data.csv"
    fixations = read_fixation_csv(fixation_path); events = extract_events(fixations, include_self=False, require_consecutive_order=True)
    observed = defaultdict(set)
    for text, word in zip(fixations.text_id, fixations.word_index, strict=True): observed[str(text)].add(int(word))
    metadata = read_provo_word_metadata(main_path, dict(observed)); metadata = enrich_word_frequencies(metadata)
    import spacy
    nlp = spacy.load("en_core_web_sm"); metadata, syntax_audit = enrich_spacy_syntax(metadata, nlp)
    edge_counts = defaultdict(float)
    for text, src, dst in zip(events.text_id, events.src_word, events.dst_word, strict=True): edge_counts[(str(text), int(src), int(dst))] += 1
    stats = {text: (len(words), int(np.count_nonzero(events.text_id == text))) for text, words in observed.items()}
    manifest = make_split_manifest(stats, args.split_seed); manifest_path = Path(args.output).with_name("provo_auxiliary_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    design = build_pair_design(metadata, args.feature_set, args.risk_set); counts = count_vector(design, edge_counts)
    targets, scaling = build_gaze_targets(design, counts, manifest["splits"]); cache = cache_bert_inputs(metadata, manifest, Path(args.cache), args.split_seed)
    position_train = np.concatenate([
        -np.log1p(targets[text]["dst"][targets[text]["reliable"]] - targets[text]["src"][targets[text]["reliable"]])
        for text in manifest["splits"]["train"]
    ])
    position_median = float(np.median(position_train))
    position_scale = max(1.4826 * float(np.median(np.abs(position_train - position_median))), 1e-8)
    from transformers import AutoModelForMaskedLM
    pretrained = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    torch.manual_seed(args.seed); template = AuxiliaryModel(pretrained.config.hidden_size, copy.deepcopy(pretrained.cls), 16); initial = copy.deepcopy(template.state_dict())
    schedules = [(manifest["splits"]["train"][step % 35], step % 8) for step in range(args.steps)]
    results = {}; experiment_start = time.perf_counter()
    conditions = ("mlm", "gaze", "shuffled", "position") if args.feature_set == "common_core" else ("mlm", "gaze", "shuffled")
    for condition in conditions:
        started = time.perf_counter()
        model = AuxiliaryModel(pretrained.config.hidden_size, copy.deepcopy(pretrained.cls), 16); model.load_state_dict(initial); model.train()
        if args.load_checkpoint_dir:
            checkpoint_path = Path(args.load_checkpoint_dir) / f"provo_auxiliary_seed{args.seed}_{condition}.pt"
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint_data, cache["provenance"], condition=condition, seed=args.seed)
            load_trainable_state_dict(model, checkpoint_data["state_dict"])
            results[condition] = {
                "lambda_gaze": 0.0 if condition == "mlm" else args.gaze_weight,
                "selected_step": checkpoint_data["selected_step"], "checkpoint": str(checkpoint_path),
                "checkpoint_state_sha256": checkpoint_data["state_dict_sha256"], "curves": [],
                "val": evaluate(model, manifest["splits"]["val"], cache, targets, retain_predictions=args.retain_predictions),
                "test": evaluate(model, manifest["splits"]["test"], cache, targets, retain_predictions=args.retain_predictions),
                "wall_seconds": time.perf_counter() - started,
            }
            continue
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3); curves = []; best = None; fixed_state = None
        for step, (text, variant_index) in enumerate(schedules, 1):
            item = cache["texts"][text]; variant = item["variants"][variant_index]; optimizer.zero_grad(set_to_none=True)
            logits = model.mlm_logits(variant["hidden"].float().clone().unsqueeze(0))[0]; mask = variant["mask"]
            mlm_loss = F.cross_entropy(logits[mask], variant["labels"][mask]); words, _ = word_pool(model.adapter(item["clean"].float().clone().unsqueeze(0)), item["word_ids"].unsqueeze(0))
            prediction = model.gaze_head(words)[0]; target = targets[text]; dense = np.full((len(item["words"]), len(item["words"])), np.nan, np.float32)
            local = {int(word): index for index, word in enumerate(item["word_indices"])}
            local_src = np.asarray([local[int(word)] for word in target["src"]]); local_dst = np.asarray([local[int(word)] for word in target["dst"]])
            values = target["residual"]
            if condition == "shuffled":
                values = source_preserving_shuffle(values, target["src"], target["dst"], args.seed + int(text))
            elif condition == "position":
                values = -np.log1p(target["dst"] - target["src"]).astype(np.float32)
                values = ((values - position_median) / position_scale).clip(-5, 5)
            dense[local_src, local_dst] = values
            reliable = np.zeros_like(dense, bool); reliable[local_src, local_dst] = target["reliable"]
            src, dst = np.nonzero(reliable); gaze_loss = signed_huber_source_balanced(prediction[src, dst], torch.from_numpy(dense[src, dst]), torch.from_numpy(src))
            loss = mlm_loss + (0.0 if condition == "mlm" else args.gaze_weight) * gaze_loss; loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0); optimizer.step()
            if step % args.eval_every == 0 or step == args.steps:
                val = evaluate(model, manifest["splits"]["val"], cache, targets); curves.append({"step": step, "train_mlm": float(mlm_loss.detach()), "train_gaze": float(gaze_loss.detach()), "val": val})
                if best is None or val["mlm_nll"] < best[0]: best = (val["mlm_nll"], step, copy.deepcopy(model.state_dict()))
                if step == args.fixed_step: fixed_state = copy.deepcopy(model.state_dict())
                model.train()
        selected_step = args.fixed_step if args.fixed_step is not None else best[1]
        selected_state = fixed_state if args.fixed_step is not None else best[2]
        if selected_state is None:
            raise RuntimeError(f"Fixed step {args.fixed_step} was not evaluated")
        model.load_state_dict(selected_state)
        checkpoint = None
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir) / f"provo_auxiliary_seed{args.seed}_{condition}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            state = trainable_state_dict(model)
            torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "artifact_type": "provo_auxiliary_trainable_checkpoint",
                        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "base_provenance": cache["provenance"],
                        "cache_fingerprint": cache["fingerprint"], "manifest_sha256": file_sha256(manifest_path),
                        "hidden_size": pretrained.config.hidden_size,
                        "rank": 16, "condition": condition, "seed": args.seed, "split_seed": args.split_seed,
                        "selected_step": selected_step,
                        "selection_metric": "fixed pre-registered step" if args.fixed_step is not None else "Provo validation MLM NLL",
                        "fixed_step": args.fixed_step,
                        "state_dict_sha256": state_dict_sha256(state), "state_dict": state}, checkpoint_path)
            checkpoint = str(checkpoint_path)
        results[condition] = {"lambda_gaze": 0.0 if condition == "mlm" else args.gaze_weight, "selected_step": selected_step, "checkpoint": checkpoint,
            "checkpoint_state_sha256": state_dict_sha256(trainable_state_dict(model)), "curves": curves,
            "val": evaluate(model, manifest["splits"]["val"], cache, targets, retain_predictions=args.retain_predictions), "test": evaluate(model, manifest["splits"]["test"], cache, targets, retain_predictions=args.retain_predictions), "wall_seconds": time.perf_counter() - started}
    output = {"status": "complete", "schema_version": 2, "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "pretrained_provenance": cache["provenance"], "cache_fingerprint": cache["fingerprint"],
        "cache_file_sha256": file_sha256(args.cache),
        "manifest_sha256": file_sha256(manifest_path), "seed": args.seed, "split_seed": args.split_seed,
        "steps": args.steps, "eval_every": args.eval_every, "fixed_step": args.fixed_step,
        "manifest": str(manifest_path), "cache": str(args.cache), "feature_set": args.feature_set, "risk_set": args.risk_set,
        "feature_names": list(design.feature_names), "design_rank": design.design_rank(),
        "group_constant_features": list(design.group_constant_features()), "min_source_exposure": 10,
        "scaling": scaling, "position_scaling": {"median": position_median, "normal_consistency_scale": position_scale, "clip": 5.0},
        "trainable_parameters": sum(p.numel() for p in template.parameters() if p.requires_grad), "total_wall_seconds": time.perf_counter() - experiment_start,
        "peak_rss_bytes": None, "peak_rss_note": "not recorded: psutil is not installed and Windows resource module is unavailable",
        "bert_audit": {"texts": len(cache["texts"]), "max_tokens": max(item["tokens"] for item in cache["texts"].values()),
                       "all_below_512": all(item["tokens"] < 512 for item in cache["texts"].values()), "hidden_layer": cache["hidden_layer"]},
        "versions": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "spacy": spacy.__version__, "wordfreq": importlib.metadata.version("wordfreq")},
        "syntax_audit_summary": {k: v for k, v in syntax_audit.items() if k != "sentence_reports"}, "conditions": results}
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"); print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
