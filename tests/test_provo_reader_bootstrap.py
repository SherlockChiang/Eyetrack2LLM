import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_provo_independent_reliability.py"
SPEC = importlib.util.spec_from_file_location("provo_independent", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reader_bootstrap_draws_are_seeded_and_preserve_multiplicity():
    subjects = [f"s{i}" for i in range(84)]
    first = MODULE.reader_bootstrap_draws(subjects, 3, 17)
    second = MODULE.reader_bootstrap_draws(subjects, 3, 17)

    assert first == second
    assert all(len(left) == len(right) == 42 for left, right in first)
    assert any(len(set(left + right)) < 84 for left, right in first)
