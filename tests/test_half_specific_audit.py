import numpy as np

from scripts.analyze_half_specific_baseline_audit import residual_bundle


def test_residual_bundle_uses_half_exposure_with_fixed_probabilities():
    counts = np.array([3.0, 1.0, 0.0, 2.0])
    probabilities = np.array([0.75, 0.25, 0.4, 0.6])
    result = residual_bundle(counts, probabilities, np.array([0, 2, 4]), 3)
    np.testing.assert_allclose(result["exposure"], [4, 4, 2, 2])
    np.testing.assert_allclose(result["expected"], [3, 1, .8, 1.2])
    np.testing.assert_allclose(result["deviation"], [0, 0, -.8, .8])
    assert result["reliable"].tolist() == [True, True, False, False]


def test_residual_bundle_changes_only_scaling_for_equal_expected_deviation():
    counts = np.array([2.0, 2.0])
    result = residual_bundle(counts, np.array([.25, .75]), np.array([0, 2]), 1)
    np.testing.assert_allclose(result["deviation"], [1, -1])
    np.testing.assert_allclose(np.abs(result["residual"]), np.abs(result["residual"][0]))
