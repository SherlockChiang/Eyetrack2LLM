from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "provo_specification_curve", ROOT / "scripts" / "analyze_provo_specification_curve.py"
)
curve = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(curve)


def test_shared_permutation_max_statistics_are_deterministic() -> None:
    real = {name: [{"reliability": {metric: 0.5 for metric in curve.METRICS}}]
            for name in curve.SPECIFICATIONS}
    null = {}
    for spec_index, name in enumerate(curve.SPECIFICATIONS):
        null[name] = [
            {"permutation_replicate": replicate,
             "reliability": {metric: 0.1 * replicate + 0.01 * spec_index + 0.001 * metric_index
                             for metric_index, metric in enumerate(curve.METRICS)}}
            for replicate in range(3)
        ]

    first = curve.permutation_inference(real, null)
    second = curve.permutation_inference(real, null)
    assert first == second
    assert first["primary_null_maxima"] == [0.03, 0.13, 0.23]
    assert first["secondary_null_maxima"] == [0.032, 0.132, 0.232]
    assert first["cells"]["position_only"]["edge_weighted"]["raw_add_one_p"] == 0.25
    assert first["cells"]["position_only"]["edge_weighted"]["primary_edge_weighted_fwer_p"] == 0.25
    assert first["cells"]["position_only"]["source_equal_flatten"]["primary_edge_weighted_fwer_p"] is None


def test_shared_permutation_indices_must_align() -> None:
    real = {name: [{"reliability": {metric: 0.5 for metric in curve.METRICS}}]
            for name in curve.SPECIFICATIONS}
    null = {name: [{"permutation_replicate": 0,
                    "reliability": {metric: 0.0 for metric in curve.METRICS}}]
            for name in curve.SPECIFICATIONS}
    null["flexible"][0]["permutation_replicate"] = 1
    try:
        curve.permutation_inference(real, null)
    except ValueError as error:
        assert "not aligned" in str(error)
    else:
        raise AssertionError("misaligned shared replicate was accepted")
