import numpy as np
import torch
from torch import nn

from eyetrack2llm.auxiliary import (
    CHECKPOINT_FORMAT_VERSION,
    AuxiliaryModel,
    canonical_sha256,
    make_split_manifest,
    signed_huber_source_balanced,
    source_preserving_shuffle,
    trainable_state_dict,
    state_dict_sha256,
    text_sha256,
    validate_checkpoint,
    load_trainable_state_dict,
    whole_word_mask,
)
from eyetrack2llm.torch import ResidualBottleneckAdapter
from scripts.analyze_provo_text_inference import analyze


def test_split_is_disjoint_and_fixed_size():
    stats = {str(index): (index + 10, index * 3) for index in range(1, 56)}
    first = make_split_manifest(stats, seed=7)
    second = make_split_manifest(stats, seed=7)
    assert first == second
    train, val, test = (set(first["splits"][key]) for key in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (35, 10, 10)
    assert not (train & val or train & test or val & test)


def test_whole_word_mask_is_deterministic_and_complete():
    word_ids = [None, 0, 0, 1, 2, 2, None]
    first = whole_word_mask(word_ids, 0.4, 19)
    assert np.array_equal(first, whole_word_mask(word_ids, 0.4, 19))
    assert first[1] == first[2]
    assert first[4] == first[5]
    assert not first[0] and not first[-1] and first.any()


def test_adapter_is_identity_at_initialization():
    adapter = ResidualBottleneckAdapter(8, 3)
    states = torch.randn(2, 4, 8)
    torch.testing.assert_close(adapter(states), states)


def test_frozen_mlm_head_still_passes_gradient_to_adapter():
    head = nn.Linear(6, 11)
    model = AuxiliaryModel(6, head, rank=3)
    hidden = torch.randn(1, 4, 6)
    model.mlm_logits(hidden).sum().backward()
    assert all(parameter.grad is None for parameter in model.mlm_head.parameters())
    assert model.adapter.up.weight.grad is not None
    assert torch.isfinite(model.adapter.up.weight.grad).all()


def test_shuffled_target_preserves_each_source_multiset_and_direction():
    target = np.arange(6)
    src = np.array([0, 0, 0, 1, 1, 2]); dst = np.array([1, 2, 3, 2, 3, 3])
    shuffled = source_preserving_shuffle(target, src, dst, 31)
    assert shuffled.shape == target.shape
    for source in np.unique(src):
        assert sorted(shuffled[src == source]) == sorted(target[src == source])
    assert np.all(dst > src)
    assert np.array_equal(shuffled, source_preserving_shuffle(target, src, dst, 31))


def test_tiny_synthetic_training_is_finite():
    torch.manual_seed(3)
    model = AuxiliaryModel(6, nn.Linear(6, 13), rank=3)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    hidden = torch.randn(1, 7, 6)
    for _ in range(3):
        optimizer.zero_grad()
        mlm = torch.nn.functional.cross_entropy(model.mlm_logits(hidden)[0, :3], torch.tensor([1, 2, 3]))
        prediction = model.gaze_head(model.adapter(hidden[:, :4]))[0]
        gaze = signed_huber_source_balanced(
            prediction[torch.tensor([0, 0, 1]), torch.tensor([1, 2, 3])],
            torch.tensor([-1.0, 0.5, 2.0]),
            torch.tensor([0, 0, 1]),
        )
        loss = mlm + 0.1 * gaze
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()


def test_split_seed_can_remain_fixed_across_optimization_seeds():
    stats = {str(index): (index + 10, index * 3) for index in range(1, 56)}
    manifest = make_split_manifest(stats, seed=20260711)
    assert manifest == make_split_manifest(stats, seed=20260711)


def test_trainable_checkpoint_filter_and_load_equality():
    model = AuxiliaryModel(6, nn.Linear(6, 13), rank=3)
    state = trainable_state_dict(model)
    assert state and all(name.startswith(("adapter.", "gaze_head.")) for name in state)
    clone = AuxiliaryModel(6, nn.Linear(6, 13), rank=3)
    load_trainable_state_dict(clone, state)
    for name, value in state.items():
        torch.testing.assert_close(clone.state_dict()[name], value)


def test_provenance_hashes_are_canonical_and_boundary_sensitive():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
    assert text_sha256({"1": ["ab", "c"]}) != text_sha256({"1": ["a", "bc"]})
    assert text_sha256({"1": ["a"], "2": ["b"]}) != text_sha256({"1": ["a", "b"]})


def test_checkpoint_validation_rejects_legacy_and_tampering():
    state = {"weight": torch.tensor([[1.0, 2.0]])}
    provenance = {"model": {"revision": "abc"}, "tokenizer": {"revision": "abc"}}
    from eyetrack2llm.auxiliary import RESIDUAL_SUPPORT_POLICY
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "base_provenance": provenance,
                  "residual_support_policy": RESIDUAL_SUPPORT_POLICY,
                  "condition": "gaze", "seed": 7, "state_dict": state,
                  "state_dict_sha256": state_dict_sha256(state)}
    validate_checkpoint(checkpoint, provenance, condition="gaze", seed=7)
    with np.testing.assert_raises_regex(ValueError, "schema-v3"):
        validate_checkpoint({**checkpoint, "format_version": 1}, provenance)
    with np.testing.assert_raises_regex(ValueError, "provenance"):
        validate_checkpoint(checkpoint, {"model": {"revision": "other"}})
    with np.testing.assert_raises_regex(ValueError, "hash mismatch"):
        validate_checkpoint({**checkpoint, "state_dict": {"weight": torch.tensor([[9.0]])}}, provenance)


def test_text_inference_averages_seeds_before_inference():
    runs = []
    for seed, shift in ((1, 0.0), (2, 0.1)):
        conditions = {}
        for condition in ("mlm", "gaze", "shuffled", "position"):
            per_text = {}
            for text, correlation in (("a", .2 + shift), ("b", -.1 + shift)):
                value = correlation + (.1 if condition == "gaze" else 0)
                per_text[text] = {"gaze_correlation": value, "valid_edges": 2, "mlm_nll": 1.0,
                                  "mlm_tokens": 2, "edge_predictions": [{"target": 0, "prediction": 0}, {"target": 1, "prediction": 1}]}
            conditions[condition] = {"test": {"per_text": per_text}}
        runs.append({"seed": seed, "conditions": conditions})
    result, rows = analyze(runs, 100, 7)
    assert result["comparisons"]["gaze_minus_mlm"]["texts"] == 2
    assert result["comparisons"]["gaze_minus_mlm_nll"]["texts"] == 2
    assert result["comparisons"]["gaze_minus_mlm_nll"]["mean"] == 0
    assert len(rows) == 16
