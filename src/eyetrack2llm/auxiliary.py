from __future__ import annotations

import json
import hashlib
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline import PairDesign, count_vector, fit_baseline, residual_vector
from .torch import LowRankDirectedHead, ResidualBottleneckAdapter, word_pool

MODEL_ID = "google-bert/bert-base-uncased"


def make_split_manifest(text_stats: dict[str, tuple[int, int]], seed: int = 20260711) -> dict[str, object]:
    """Create the fixed 35/10/10 split without consulting any model outcome."""
    if len(text_stats) != 55:
        raise ValueError(f"Expected 55 texts, found {len(text_stats)}")
    ids = sorted(text_stats, key=lambda value: int(value))
    random.Random(seed).shuffle(ids)
    split = {"train": ids[:35], "val": ids[35:45], "test": ids[45:]}
    return {
        "seed": seed,
        "method": "fixed-seed permutation of numeric text IDs; outcomes unavailable",
        "splits": split,
        "observables": {text: {"words": text_stats[text][0], "transitions": text_stats[text][1]} for text in sorted(ids, key=int)},
    }


def whole_word_mask(word_ids: list[int | None], probability: float, seed: int) -> np.ndarray:
    """Select complete tokenizer words using a local deterministic RNG."""
    words = sorted({word for word in word_ids if word is not None})
    if not words:
        raise ValueError("No words available to mask")
    rng = np.random.default_rng(seed)
    selected = [word for word in words if rng.random() < probability]
    if not selected:
        selected = [words[int(rng.integers(len(words)))]]
    return np.asarray([word in selected if word is not None else False for word in word_ids], dtype=bool)


def complementary_masks(word_ids: list[int | None], variants: int, seed: int) -> list[np.ndarray]:
    """Deterministically partition words so fixed evaluation variants are complementary."""
    words = sorted({word for word in word_ids if word is not None})
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(len(words)) % variants
    return [np.asarray([word is not None and assignment[word] == variant for word in word_ids], bool) for variant in range(variants)]


def source_preserving_shuffle(target: np.ndarray, src: np.ndarray, dst: np.ndarray, seed: int) -> np.ndarray:
    """Permute labels only among the existing destinations of each source."""
    target, src, dst = np.asarray(target), np.asarray(src), np.asarray(dst)
    if not (target.ndim == src.ndim == dst.ndim == 1 and len(target) == len(src) == len(dst)):
        raise ValueError("target, src, and dst must be equal-length vectors")
    rng = np.random.default_rng(seed)
    shuffled = target.copy()
    for source in np.unique(src):
        positions = np.flatnonzero(src == source)
        shuffled[positions] = target[positions][rng.permutation(len(positions))]
    return shuffled


def subset_design(design: PairDesign, text: str) -> PairDesign:
    return design.subset(design.text_id == text)


def build_gaze_targets(
    design: PairDesign, counts: np.ndarray, splits: dict[str, list[str]], *, min_exposure: int = 10, clip: float = 5.0
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Cross-fit train labels and fit robust scaling only on train residuals."""
    train = set(splits["train"])
    targets: dict[str, dict[str, np.ndarray]] = {}
    scaling_values = []
    for text in sum((splits[name] for name in ("train", "val", "test")), []):
        target = subset_design(design, text)
        fit_counts = counts.copy()
        allowed = train - {text} if text in train else train
        fit_counts[~np.isin(design.text_id, list(allowed))] = 0
        probability = fit_baseline(design, fit_counts, l2=1.0).predict(target)
        text_counts = counts[design.text_id == text]
        residual, exposure = residual_vector(text_counts, probability, target.group_start)
        reliable = exposure >= min_exposure
        targets[text] = {"residual": residual, "reliable": reliable, "src": target.src_word, "dst": target.dst_word}
        if text in train:
            scaling_values.append(residual[reliable])
    values = np.concatenate(scaling_values)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-8)
    for item in targets.values():
        item["residual"] = np.clip((item["residual"] - median) / scale, -clip, clip).astype(np.float32)
    return targets, {"median": median, "mad": mad, "normal_consistency_scale": scale, "clip": clip}


class AuxiliaryModel(nn.Module):
    def __init__(self, hidden_size: int, mlm_head: nn.Module, rank: int = 16) -> None:
        super().__init__()
        self.adapter = ResidualBottleneckAdapter(hidden_size, rank)
        self.gaze_head = LowRankDirectedHead(hidden_size, rank)
        self.mlm_head = mlm_head
        for parameter in self.mlm_head.parameters():
            parameter.requires_grad_(False)

    def mlm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # The frozen head stays in the autograd graph so gradients reach the adapter.
        return self.mlm_head(self.adapter(hidden))


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU checkpoint containing only parameters optimized by this experiment."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in trainable}


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError(f"Trainable checkpoint keys differ: missing={sorted(expected - set(state))}, extra={sorted(set(state) - expected)}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError(f"Invalid trainable checkpoint: {result}")


def signed_huber_source_balanced(prediction: torch.Tensor, target: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    return torch.stack([losses[src == source].mean() for source in torch.unique(src)]).mean()


def cache_bert_inputs(metadata, manifest: dict[str, object], cache_path: Path, seed: int) -> dict[str, object]:
    """Cache final-layer whole-text encoder states for clean and fixed masked inputs."""
    fingerprint = hashlib.sha256(
        json.dumps({"manifest": manifest, "mask_seed": seed}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (cached.get("fingerprint") == fingerprint and cached.get("texts")
                and all("word_indices" in item for item in cached["texts"].values())):
            return cached
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    bert_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_ID, local_files_only=True).eval()
    encoder = bert_mlm.bert
    cache: dict[str, object] = {
        "model_id": MODEL_ID,
        "hidden_layer": "last",
        "fingerprint": fingerprint,
        "texts": {},
    }
    split_by_text = {text: name for name, texts in manifest["splits"].items() for text in texts}
    with torch.no_grad():
        for ordinal, text in enumerate(sorted(split_by_text, key=int)):
            positions = np.flatnonzero(metadata.text_id == text)
            positions = positions[np.argsort(metadata.word_index[positions])]
            words = metadata.surface[positions].tolist()
            encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt")
            word_ids = encoded.word_ids(0)
            if encoded.input_ids.shape[1] >= 512 or any(word_ids.count(i) == 0 for i in range(len(words))):
                raise ValueError(f"Whole-text BERT audit failed for text {text}: {encoded.input_ids.shape[1]} tokens")
            clean = encoder(**encoded, return_dict=True).last_hidden_state[0].to(torch.float16).cpu()
            count = 8 if split_by_text[text] == "train" else 3
            masks = ([whole_word_mask(word_ids, 0.15, seed + ordinal * 100 + i) for i in range(count)]
                     if split_by_text[text] == "train" else complementary_masks(word_ids, count, seed + ordinal * 100))
            variants = []
            for mask in masks:
                ids = encoded.input_ids.clone()
                ids[0, torch.from_numpy(mask)] = tokenizer.mask_token_id
                hidden = encoder(input_ids=ids, attention_mask=encoded.attention_mask, return_dict=True).last_hidden_state[0]
                variants.append({"hidden": hidden.to(torch.float16).cpu(), "mask": torch.from_numpy(mask),
                                 "labels": encoded.input_ids[0].cpu()})
            cache["texts"][text] = {"clean": clean, "word_ids": torch.tensor([-1 if x is None else x for x in word_ids]),
                                    "word_indices": torch.from_numpy(metadata.word_index[positions].copy()),
                                    "variants": variants, "words": words, "tokens": int(encoded.input_ids.shape[1])}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache
