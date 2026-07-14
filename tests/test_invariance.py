import numpy as np

from eyetrack2llm.baseline import PairDesign
from eyetrack2llm.invariance import domain_classification, feature_audit, text_summaries


def design():
    texts = np.repeat(["T:1", "T:2", "T:3"], 4)
    return PairDesign(np.tile([[0., 1.], [1., 0.]], (6, 1)), texts,
                      np.tile([0, 0, 1, 1], 3), np.tile([1, 2, 2, 3], 3),
                      np.arange(0, 13, 2), ("a", "b"))


def test_identical_feature_audit_is_zero_and_deterministic():
    d = design(); first = feature_audit(d, d, 30, 7); second = feature_audit(d, d, 30, 7)
    assert first == second
    assert all(item["text_equal_smd"] == 0 for item in first)


def test_text_summaries_use_text_rows_and_correct_zero_rate():
    d = design(); counts = np.tile([2., 0., 1., 1.], 3)
    texts, matrix, names, summary = text_summaries(d, counts)
    assert len(texts) == matrix.shape[0] == 3
    assert summary["per_text"]["T:1"]["zero_destination_proportion"] == .25
    assert summary["per_text"]["T:1"]["source_groups"] == 2


def test_domain_classification_declares_fold_local_standardization():
    result = domain_classification(np.arange(60, dtype=float).reshape(10, 6),
                                   np.arange(60, 120, dtype=float).reshape(10, 6), 2, 7)
    assert result["standardization"].startswith("fold-local")
