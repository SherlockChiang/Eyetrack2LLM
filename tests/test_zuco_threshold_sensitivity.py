from scripts.analyze_zuco_edge_threshold_sensitivity import analyze


def test_thresholds_filter_texts_and_report_descriptive_diagnostics():
    seed_results = {}
    for seed in ("1", "2"):
        seed_results[seed] = {}
        for condition, shift in (("gaze", .2), ("mlm", .1), ("shuffled", 0), ("position", -.1)):
            seed_results[seed][condition] = {"per_text": {
                "a": {"correlation": shift, "n_edges": 5},
                "b": {"correlation": shift / 2, "n_edges": 25},
                "c": {"correlation": -.2 + shift, "n_edges": 35},
            }}
    primary = {
        name: {"text_equal_fisher_z": {"descriptive_text_resampling_interval": [0.01, 0.2], "bootstrap_seed": 7}}
        for name in ("gaze_vs_mlm", "gaze_vs_shuffled", "gaze_vs_position")
    }
    result, rows = analyze({"seed_results": seed_results, "comparisons": primary}, thresholds=(4, 20, 30), bootstrap=200, seed=3)
    assert result["results"]["4"]["comparisons"]["gaze_vs_mlm"]["texts_retained"] == 3
    assert result["results"]["20"]["comparisons"]["gaze_vs_mlm"]["texts_retained"] == 2
    assert result["results"]["30"]["comparisons"]["gaze_vs_mlm"]["texts_retained"] == 1
    assert result["results"]["4"]["comparisons"]["gaze_vs_mlm"]["available_text_strata_by_minimum_edges"]["4-9"] == 1
    assert result["results"]["4"]["comparisons"]["gaze_vs_mlm"]["descriptive_text_resampling_interval"] == [0.01, 0.2]
    assert result["results"]["4"]["comparisons"]["gaze_vs_mlm"]["interval_source"] == "frozen primary transfer interval"
    assert result["results"]["20"]["comparisons"]["gaze_vs_mlm"]["interval_seed"] == 2004
    assert "signflip" not in str(result).lower() and "p_value" not in str(result).lower()
    assert len(rows) == 9
