import numpy as np

from eyetrack2llm.baseline import PairDesign
from eyetrack2llm.stability import (balanced_subject_subsets, crossfit_raw_residuals,
                                    paired_residual_metrics, permute_destinations_within_source, source_equal_metrics)


def synthetic_design():
    text = np.repeat(["1", "2", "3", "4", "5"], 2)
    return PairDesign(np.tile([[0.], [1.]], (5, 1)), text, np.zeros(10, int), np.tile([1, 2], 5),
                      np.arange(0, 11, 2), ("x",))


def test_crossfit_fitter_never_receives_heldout_counts():
    seen = []
    class Model:
        def predict(self, design):
            return np.tile([.5, .5], len(design.features) // 2)
    def spy(design, counts, **kwargs):
        seen.append((set(design.text_id), counts.copy()))
        return Model()
    crossfit_raw_residuals(synthetic_design(), np.arange(1, 11), n_folds=5, fitter=spy)
    assert len(seen) == 5
    assert all(len(texts) == 4 and np.all(counts > 0) for texts, counts in seen)


def test_balanced_subsets_are_reproducible_and_balanced():
    first = balanced_subject_subsets("ABCDEFGHIJKL", 4, 50, 7)
    assert first == balanced_subject_subsets("ABCDEFGHIJKL", 4, 50, 7)
    counts = [sum(subject in subset for subset in first) for subject in "ABCDEFGHIJKL"]
    assert max(counts) - min(counts) <= 1


def test_source_equal_metric_does_not_overweight_large_source():
    x = np.array([0., 1., 2., 3., 0., 1.]); y = np.array([0., 1., 2., 3., 1., 0.]); sources = np.array([0, 0, 0, 0, 1, 1])
    result = source_equal_metrics(x, y, sources)
    assert result["edge_weighted"] > result["source_equal_flatten"]
    assert abs(result["per_source_fisher_equal"]) < 1e-6


def test_destination_permutation_is_deterministic_independent_and_preserves_groups():
    design = synthetic_design()
    counts = np.arange(10)
    first = permute_destinations_within_source(counts, design, np.random.default_rng(19))
    again = permute_destinations_within_source(counts, design, np.random.default_rng(19))
    second_half = permute_destinations_within_source(counts, design, np.random.default_rng(20))
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, second_half)
    for start, stop in zip(design.group_start[:-1], design.group_start[1:], strict=True):
        np.testing.assert_array_equal(np.sort(first[start:stop]), np.sort(counts[start:stop]))
    assert len(first) == len(counts)


def test_paired_metrics_report_negative_and_undefined_texts():
    item = {"residual": np.array([0., 1.]), "exposure": np.array([5., 5.]),
            "reliable": np.array([True, True]), "src": np.array([0, 0]), "dst": np.array([1, 2])}
    opposite = {**item, "residual": np.array([1., 0.])}
    result = paired_residual_metrics({"1": item}, {"1": opposite})
    summary = result["text_summary"]["edge_weighted"]
    assert summary["negative_proportion"] == 1
    assert summary["undefined_proportion"] == 0


def test_crossfit_excludes_singleton_groups_from_residual_support():
    design = PairDesign(np.zeros((5, 1)), np.array(["1", "2", "3", "4", "5"]),
                        np.arange(5), np.arange(1, 6), np.arange(6), ("x",))
    class Model:
        def predict(self, target):
            return np.ones(len(target.features))
    targets, _ = crossfit_raw_residuals(design, np.full(5, 10), n_folds=5,
                                        fitter=lambda *args, **kwargs: Model())
    assert all(not item["reliable"].any() and np.isnan(item["residual"]).all()
               for item in targets.values())
