import numpy as np

from eyetrack2llm.influence import (analytic_leave_one_out_mean,
                                    leave_one_text_out_median_hierarchy,
                                    most_influential, resampled_mean_inference)


def test_analytic_leave_one_out_mean():
    np.testing.assert_allclose(analytic_leave_one_out_mean([1, 2, 6]), [4, 3.5, 1.5])


def test_median_hierarchy_preserves_repeat_then_text_estimand():
    repeats = [{"a": 1, "b": 3, "c": 100}, {"a": 2, "b": 4, "c": 6}]
    full, loto = leave_one_text_out_median_hierarchy(repeats)
    assert full == 3.5
    assert loto == {"a": 28.25, "b": 27.25, "c": 2.5}


def test_most_influential_index_is_first_absolute_maximum():
    assert most_influential(2.0, {"a": 1.0, "b": 3.0, "c": 2.5}) == ("a", 1.0)


def test_resampling_is_seed_deterministic():
    first = resampled_mean_inference([-.2, .1, .3, .4], repeats=200, seed=9)
    assert first == resampled_mean_inference([-.2, .1, .3, .4], repeats=200, seed=9)
