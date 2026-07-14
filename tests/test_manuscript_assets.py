from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manuscript_assets", ROOT / "scripts" / "generate_manuscript_assets.py")
assets = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(assets)


def test_artifact_extraction_matches_frozen_values() -> None:
    data, paths = assets.load_primary()
    extracted = assets.extract_assets(data)
    assert len(extracted["figure2"]) == 48
    assert len(extracted["figure3"]) == 4 * (100 + 500)
    assert len(extracted["figure4"]) == 5 * 2 + 3 + 1
    cell = next(row for row in extracted["figure2"] if row["subjects"] == 4 and row["latent_effect"] == 0 and row["method"] == "misspecified" and row["overdispersion"] == "high")
    frozen = next(row for row in data["simulation"]["summary"] if row["subjects"] == 4 and row["latent_effect"] == 0 and row["method"] == "misspecified" and row["overdispersion"] == "high")
    assert cell["split_half_residual_reliability_mean"] == frozen["split_half_residual_reliability_mean"]
    assert not ({path.name for path in paths.values()} & assets.SUPERSEDED)


def test_expected_generated_schema_and_hashes() -> None:
    provenance = json.loads((ROOT / "manuscript" / "artifact_provenance.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 1
    assert provenance["superseded_inputs"] == []
    assert {record["asset"] for record in provenance["assets"]} == {"figure1", "figure2", "figure3", "figure4", "table1", "table2", "table3", "table4", "table_s1", "table_s2", "table_s3", "table_s4", "table_s5", "table_s6"}
    for record in provenance["assets"]:
        assert record["inputs"] and record["fields_and_transforms"] and record["outputs"]
        for output in record["outputs"]:
            path = ROOT / output["path"]
            assert path.stat().st_size > 0
            assert assets.sha256(path) == output["sha256"]


def test_generated_primary_table_labels_and_nll_scale(tmp_path: Path) -> None:
    data, _ = assets.load_primary()
    extracted = assets.extract_assets(data)
    assets.make_tables(data, extracted, tmp_path)
    table1 = (tmp_path / "table1_corpus_pipeline_audit.csv").read_text(encoding="utf-8")
    table3 = (tmp_path / "table3_specification_results.csv").read_text(encoding="utf-8")
    assert "mapped_positions_or_tokens" in table1 and ",words," not in table1
    assert "-46279.02" in table3 and "-23139.51" not in table3


def test_source_data_numeric_correspondence() -> None:
    data, _ = assets.load_primary()
    with (ROOT / "manuscript" / "source_data" / "figure4_source_data.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    gaze_101 = next(row for row in rows if row["panel"] == "provo" and row["seed"] == "101" and row["condition"] == "gaze")
    assert float(gaze_101["mlm_nll"]) == data["seed_101"]["conditions"]["gaze"]["test"]["mlm_nll"]
    zuco = next(row for row in rows if row["panel"] == "zuco" and row["contrast"] == "gaze_vs_shuffled")
    fixed = data["transfer"]["comparisons"]["gaze_vs_shuffled"]["text_equal_fisher_z"]["descriptive_text_resampling_interval"]
    nested = data["criterion"]["reader_bootstrap"]["summary"]["gaze_vs_shuffled"]["joint_reader_and_text"]["95_ci"]
    assert [float(zuco["fixed_ci_low"]), float(zuco["fixed_ci_high"])] == fixed
    assert [float(zuco["nested_ci_low"]), float(zuco["nested_ci_high"])] == nested


def test_figure_smoke_in_temporary_directory(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    data, _ = assets.load_primary()
    assets.data_global = data
    outputs = assets.make_figures(assets.extract_assets(data), tmp_path)
    assert len(outputs) == 12
    assert all(path.stat().st_size > 0 for path in outputs)


def test_compact_archive_does_not_overwrite_repository_readme() -> None:
    from scripts.build_compact_artifact_bundle import build

    result = build()
    import zipfile
    with zipfile.ZipFile(ROOT / result["archive"]) as archive:
        assert "README.md" not in archive.namelist()
        assert "release/README.md" in archive.namelist()
