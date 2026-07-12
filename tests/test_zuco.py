import csv

import numpy as np
import torch
from scipy.io import savemat

from eyetrack2llm import extract_events, read_fixation_csv
from eyetrack2llm.zuco import convert_zuco_mat
from eyetrack2llm.transfer import fit_fresh_probe, pair_sensitivity_mask, relation_scores, sentence_folds
from eyetrack2llm.torch import LowRankDirectedHead, ResidualBottleneckAdapter


def test_zuco_mapping_preserves_sequence_gaps_and_rejects_conflicts(tmp_path):
    words = np.empty(3, dtype=object)
    words[0] = {"content": "a", "fixPositions": np.array([1, 4]), "nFixations": 2}
    words[1] = {"content": "b", "fixPositions": np.array([3, 4]), "nFixations": 2}
    words[2] = {"content": "c", "fixPositions": np.array([5]), "nFixations": 1}
    sentence = {
        "content": "a b c", "word": words,
        "wordbounds": np.array([[0, 0, 9, 9], [10, 0, 19, 9], [20, 0, 29, 9]]),
        "allFixations": {"x": [5, 50, 15, 12, 25], "y": [5] * 5, "duration": [10, 20, 30, 40, 50]},
    }
    mat = tmp_path / "resultsS1_NR.mat"
    output = tmp_path / "out.csv"
    savemat(mat, {"sentenceData": np.array([sentence], dtype=object)})
    report = convert_zuco_mat(mat, output)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["word_index"], r["fixation_order"], r["duration_ms"]) for r in rows] == [
        ("0", "1", "20.0"), ("1", "3", "60.0"), ("2", "5", "100.0")]
    assert report.outside_fixations == 1
    assert report.conflict_fixations == 1
    events = extract_events(read_fixation_csv(output), require_consecutive_order=True)
    assert len(events) == 0


def test_crossfit_sentence_folds_have_no_overlap():
    folds = sentence_folds([f"NR:{i}" for i in range(20)], seed=4)
    assert all(not (set(train) & set(test)) for train, test in folds)
    assert sorted(text for _, test in folds for text in test) == sorted(f"NR:{i}" for i in range(20))


def test_synthetic_transfer_recovers_known_relation():
    adapter = ResidualBottleneckAdapter(2, 1)
    head = LowRankDirectedHead(2, 1)
    with torch.no_grad():
        head.source.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.target.weight.copy_(torch.tensor([[0.0, 1.0]]))
        head.bias.zero_()
    hidden = torch.tensor([[1., 3.], [2., 4.], [0., 0.]])
    scores = relation_scores(adapter, head, hidden, torch.tensor([0, 1, -1]))
    torch.testing.assert_close(scores, torch.tensor([[3., 4.], [6., 8.]]))


def test_distance_and_within_line_masks():
    src = np.array([0, 0, 1, 2]); dst = np.array([1, 2, 3, 3])
    bounds = np.array([[0, 0, 5, 10], [6, 1, 10, 9], [0, 20, 5, 30], [6, 21, 10, 29]])
    np.testing.assert_array_equal(pair_sensitivity_mask(src, dst, max_distance=1), [True, False, False, True])
    np.testing.assert_array_equal(
        pair_sensitivity_mask(src, dst, n_words=4, bounds=bounds, within_line=True),
        [True, False, False, True],
    )


def test_fresh_probe_ignores_heldout_targets():
    states = {"train": torch.tensor([[1., 0.], [0., 1.]]), "test": torch.tensor([[1., 1.], [2., 1.]])}
    base = {"src": np.array([0]), "dst": np.array([1]), "reliable": np.array([True]), "residual": np.array([1.], np.float32)}
    targets = {"train": base, "test": {**base, "residual": np.array([2.], np.float32)}}
    first, audit = fit_fresh_probe(states, targets, ["train"], rank=1, seed=9, epochs=3)
    targets["test"]["residual"][:] = 1e6
    second, _ = fit_fresh_probe(states, targets, ["train"], rank=1, seed=9, epochs=3)
    assert audit["train_texts"] == ["train"]
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right)
