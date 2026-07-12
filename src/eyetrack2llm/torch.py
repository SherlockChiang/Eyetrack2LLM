from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


def word_pool(
    token_states: torch.Tensor,
    word_index: torch.Tensor,
    *,
    n_words: int | None = None,
    reduce: Literal["mean", "sum"] = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_states.ndim != 3 or word_index.shape != token_states.shape[:2]:
        raise ValueError("Expected token_states [B,T,D] and word_index [B,T]")
    if reduce not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduce}")
    valid = word_index >= 0
    inferred = int(word_index[valid].max().item()) + 1 if valid.any() else 0
    n_words = inferred if n_words is None else n_words
    if n_words < inferred:
        raise ValueError("n_words is smaller than an observed word_index")

    batch, _, dimension = token_states.shape
    output = token_states.new_zeros((batch, n_words, dimension))
    counts = token_states.new_zeros((batch, n_words, 1))
    safe_index = word_index.clamp_min(0)
    output.scatter_add_(
        1,
        safe_index.unsqueeze(-1).expand(-1, -1, dimension),
        token_states * valid.unsqueeze(-1),
    )
    counts.scatter_add_(1, safe_index.unsqueeze(-1), valid.unsqueeze(-1).to(token_states.dtype))
    if reduce == "mean":
        output = output / counts.clamp_min(1)
    return output, counts.squeeze(-1) > 0


class DirectedBilinearHead(nn.Module):
    def __init__(self, input_dim: int, n_relations: int = 1, *, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_relations, input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(n_relations)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, word_states: torch.Tensor, word_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if word_states.ndim != 3:
            raise ValueError("word_states must have shape [B,W,D]")
        logits = torch.einsum("bid,rde,bje->brij", word_states, self.weight, word_states)
        logits = logits / math.sqrt(word_states.shape[-1])
        if self.bias is not None:
            logits = logits + self.bias[None, :, None, None]
        if word_mask is None:
            word_mask = torch.ones(
                word_states.shape[:2], dtype=torch.bool, device=word_states.device
            )
        pair_mask = word_mask[:, None, :, None] & word_mask[:, None, None, :]
        return logits, pair_mask


class ResidualBottleneckAdapter(nn.Module):
    """A zero-initialized residual adapter that is exactly identity at initialization."""

    def __init__(self, input_dim: int, rank: int = 16) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states + self.up(torch.nn.functional.gelu(self.down(states)))


class LowRankDirectedHead(nn.Module):
    """Asymmetric rank-constrained word-pair regression head."""

    def __init__(self, input_dim: int, rank: int = 16) -> None:
        super().__init__()
        self.source = nn.Linear(input_dim, rank, bias=False)
        self.target = nn.Linear(input_dim, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, word_states: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bir,bjr->bij", self.source(word_states), self.target(word_states)) / math.sqrt(
            self.source.out_features
        ) + self.bias


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, *, dim: int = -1
) -> torch.Tensor:
    mask = torch.broadcast_to(mask, logits.shape)
    compute = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    masked = compute.masked_fill(~mask, -torch.inf)
    has_value = mask.any(dim=dim, keepdim=True)
    safe = torch.where(has_value, masked, torch.zeros_like(masked))
    probability = torch.softmax(safe, dim=dim)
    probability = torch.where(mask & has_value, probability, torch.zeros_like(probability))
    return probability


def _masked_probability(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
    is_logits: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_logits:
        probability = masked_softmax(value, mask, dim=dim)
    else:
        probability = torch.where(mask, value.float(), torch.zeros_like(value.float()))
        probability = probability.clamp_min(0)
        total = probability.sum(dim=dim, keepdim=True)
        probability = torch.where(total > eps, probability / total.clamp_min(eps), probability)
    valid = mask.any(dim=dim) & (probability.sum(dim=dim) > eps)
    return probability, valid


def masked_jsd(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int = -1,
    input_is_logits: bool = True,
    target_is_logits: bool = False,
    reduction: Literal["none", "mean", "sum"] = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    input, target = torch.broadcast_tensors(input, target)
    mask = torch.broadcast_to(mask.bool(), input.shape)
    p, p_valid = _masked_probability(
        input, mask, dim=dim, is_logits=input_is_logits, eps=eps
    )
    q, q_valid = _masked_probability(
        target, mask, dim=dim, is_logits=target_is_logits, eps=eps
    )
    valid = p_valid & q_valid
    midpoint = 0.5 * (p + q)
    p_term = torch.where(p > 0, p * (p.clamp_min(eps).log() - midpoint.clamp_min(eps).log()), 0)
    q_term = torch.where(q > 0, q * (q.clamp_min(eps).log() - midpoint.clamp_min(eps).log()), 0)
    loss = 0.5 * (p_term + q_term).sum(dim=dim)
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / valid.sum().clamp_min(1)
    raise ValueError(f"Unsupported reduction: {reduction}")
