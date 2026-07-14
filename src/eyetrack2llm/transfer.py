from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .auxiliary import subset_design
from .baseline import PairDesign, fit_baseline, residual_vector


def sentence_folds(text_ids, n_folds: int = 5, seed: int = 20260711) -> list[tuple[list[str], list[str]]]:
    ids = sorted(set(map(str, text_ids)), key=lambda value: int(value.split(":")[-1]))
    random.Random(seed).shuffle(ids)
    buckets = [ids[index::n_folds] for index in range(n_folds)]
    return [([text for index, bucket in enumerate(buckets) if index != fold for text in bucket], buckets[fold]) for fold in range(n_folds)]


def crossfit_residual_targets(design: PairDesign, counts: np.ndarray, *, n_folds: int = 5,
                               seed: int = 20260711, min_exposure: int = 10, clip: float = 5.0,
                               scale_clip: bool = True):
    targets = {}
    folds = sentence_folds(design.text_id, n_folds, seed)
    for train, test in folds:
        fit_counts = counts.copy()
        fit_counts[~np.isin(design.text_id, train)] = 0
        model = fit_baseline(design, fit_counts, l2=1.0)
        for text in test:
            target = subset_design(design, text)
            selected = design.text_id == text
            residual, exposure = residual_vector(counts[selected], model.predict(target), target.group_start)
            reliable = (exposure >= min_exposure) & np.isfinite(residual)
            targets[text] = {"residual": residual, "exposure": exposure, "reliable": reliable,
                             "src": target.src_word, "dst": target.dst_word}
    values = np.concatenate([item["residual"][item["reliable"]] for item in targets.values()])
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-8)
    if scale_clip:
        for item in targets.values():
            item["residual"] = np.clip((item["residual"] - median) / scale, -clip, clip).astype(np.float32)
    else:
        for item in targets.values():
            item["residual"] = item["residual"].astype(np.float32)
    return targets, {"median": median, "mad": mad, "normal_consistency_scale": scale,
                     "clip": clip if scale_clip else None, "scale_clip": scale_clip}, folds


def relation_scores(adapter, gaze_head, hidden: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
    from .torch import word_pool
    words, _ = word_pool(adapter(hidden.unsqueeze(0)), word_ids.unsqueeze(0))
    return gaze_head(words)[0]


def pair_sensitivity_mask(src: np.ndarray, dst: np.ndarray, *, max_distance: int | None = None,
                          n_words: int | None = None, exclude_edge_last2: bool = False,
                          bounds: np.ndarray | None = None, within_line: bool = False) -> np.ndarray:
    """Select target-side edges without changing the fitted baseline or transferred model."""
    src, dst = np.asarray(src), np.asarray(dst)
    mask = dst > src
    if max_distance is not None:
        mask &= (dst - src) <= max_distance
    if exclude_edge_last2:
        if n_words is None:
            raise ValueError("n_words is required for the last-two exclusion")
        mask &= (src < n_words - 2) & (dst < n_words - 2)
    if within_line:
        if bounds is None:
            raise ValueError("bounds are required for the within-line mask")
        bounds = np.asarray(bounds, float)
        if bounds.ndim != 2 or bounds.shape[0] != n_words or bounds.shape[1] != 4:
            raise ValueError("bounds must have shape [n_words, 4]")
        # Glyph boxes on one rendered line can have different top/bottom values.
        mask &= (bounds[src, 1] <= bounds[dst, 3]) & (bounds[dst, 1] <= bounds[src, 3])
    return mask


def fit_fresh_probe(word_states: dict[str, torch.Tensor], targets: dict[str, dict], train_texts,
                    *, rank: int = 8, seed: int = 0, epochs: int = 200,
                    lr: float = 1e-2, l2: float = 1e-4):
    """Fit a new directed probe using only explicitly listed training sentences."""
    train_texts = tuple(map(str, train_texts))
    if not train_texts or any(text not in word_states or text not in targets for text in train_texts):
        raise ValueError("Every train text must have states and targets")
    values = np.concatenate([targets[text]["residual"][targets[text]["reliable"]] for text in train_texts])
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-8)
    torch.manual_seed(seed)
    probe = __import__("eyetrack2llm.torch", fromlist=["LowRankDirectedHead"]).LowRankDirectedHead(
        word_states[train_texts[0]].shape[-1], rank
    )
    source_states, target_states, responses = [], [], []
    for text in train_texts:
        target = targets[text]; selected = target["reliable"]
        source_states.append(word_states[text][torch.from_numpy(target["src"][selected])])
        target_states.append(word_states[text][torch.from_numpy(target["dst"][selected])])
        responses.append(torch.from_numpy(((target["residual"][selected] - median) / scale).astype(np.float32)))
    source_states, target_states, response = map(torch.cat, (source_states, target_states, responses))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=l2)
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = (probe.source(source_states) * probe.target(target_states)).sum(-1) / np.sqrt(rank) + probe.bias
        nn.functional.mse_loss(prediction, response).backward()
        optimizer.step()
    return probe, {"median": median, "mad": mad, "normal_consistency_scale": scale,
                   "train_texts": list(train_texts), "n_train_edges": int(sum(targets[t]["reliable"].sum() for t in train_texts))}
