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


def test_distribution_is_descriptive() -> None:
    summary = curve.distribution([0.1, 0.2, 0.3])
    assert summary == {"median": 0.2, "q25": 0.15000000000000002, "q75": 0.25, "range": [0.1, 0.3]}
    assert not any(token in str(summary).lower() for token in ("p_value", "fwer", "familywise"))
