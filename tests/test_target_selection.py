import numpy as np

from eyetrack2llm.target_selection import (CATEGORIES, categorized_paired_metrics,
                                             category_masks, deterministic_subject_splits,
                                             residual_arrays, separation_category,
                                             thin_subject_categories)


def test_category_boundaries_and_preservation():
    assert [separation_category(4, dst) for dst in (5, 6, 7, 8)] == [
        "adjacent", "near_skip", "near_skip", "far_same_line"]
    masks = category_masks(np.array([0, 0, 0, 0]), np.array([1, 2, 3, 4]))
    assert tuple(masks) == CATEGORIES
    assert sum(int(mask.sum()) for mask in masks.values()) == 4
    assert np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1)


def test_adjacent_single_candidate_source_correlation_is_undefined():
    half = {"1": {"src": np.array([0, 1]), "dst": np.array([1, 2]),
                  "residual": np.array([.1, .2]), "reliable": np.array([True, True])}}
    result = categorized_paired_metrics(half, half)["adjacent"]
    assert result["per_text"]["1"]["edge_weighted"] == 1.0
    assert result["per_text"]["1"]["n_sources_with_defined_correlation"] == 0
    assert result["per_text"]["1"]["per_source_fisher_equal"] is None


def test_subject_splits_are_seed_deterministic():
    first = deterministic_subject_splits(["4", "1", "3", "2"], 5, 17)
    assert first == deterministic_subject_splits(["1", "2", "3", "4"], 5, 17)
    assert first != deterministic_subject_splits(["1", "2", "3", "4"], 5, 18)


def test_residual_arrays_zero_cell_and_multinomial_variance():
    result = residual_arrays(np.array([0., 4.]), np.array([.25, .75]),
                             np.array([0, 2]), 1)
    np.testing.assert_allclose(result["expected"], [1, 3])
    np.testing.assert_allclose(result["pearson"], [-1 / np.sqrt(.75), 1 / np.sqrt(.75)])
    assert result["deviance"][0] < 0 and np.isfinite(result["deviance"]).all()


def test_category_thinning_preserves_universe_and_far_counts():
    counts = np.array([10, 4, 2, 1])
    src, dst = np.zeros(4, int), np.array([1, 2, 3, 4])
    thinned, probabilities = thin_subject_categories(counts, src, dst,
                                                       np.random.default_rng(2))
    assert thinned.shape == counts.shape
    assert thinned[-1] == counts[-1]
    assert probabilities["adjacent"] == .1
    assert probabilities["near_skip"] == 1 / 6
