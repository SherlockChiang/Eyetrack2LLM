from __future__ import annotations

import json
import sys
import csv
import hashlib
import argparse
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEEDS = (101, 202, 303, 404, 505)
CONTROLS = {"mlm", "gaze", "shuffled", "position"}
RISK_SET = "common_forward_same_sentence_same_line"
SUPERSEDED = {
    "provo_auxiliary_commoncore_fixed50_seed101.json", "provo_auxiliary_commoncore_fixed50_seed202.json",
    "provo_auxiliary_commoncore_fixed50_seed303.json", "provo_auxiliary_commoncore_fixed50_seed404.json",
    "provo_auxiliary_commoncore_fixed50_seed505.json", "zuco_transfer_commoncore_fixed50.json",
    "provo_commoncore_independent_reliability.json", "zuco_zero_shot_transfer.json",
    "zuco_zero_shot_transfer_fixed50_forwardrisk.json", "fresh_probe_representation.json", "zuco_transfer_sensitivity.json",
}


def load(name: str, *, require_complete: bool = True) -> dict | list:
    path = PROCESSED / name
    if not path.is_file():
        raise ValueError(f"missing required result: {path.relative_to(ROOT)}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if require_complete and (not isinstance(data, dict) or data.get("status") != "complete"):
        raise ValueError(f"result is not complete: {path.relative_to(ROOT)}")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_csv(name: str) -> None:
    path = PROCESSED / name
    require(path.is_file(), f"missing required CSV companion: {path.relative_to(ROOT)}")
    require(path.stat().st_size > 0, f"CSV companion is empty: {path.relative_to(ROOT)}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manuscript_assets(*, verify_inputs: bool = False) -> None:
    manuscript = ROOT / "manuscript"
    provenance_path = manuscript / "artifact_provenance.json"
    manifest_path = manuscript / "artifact_manifest.csv"
    require(provenance_path.is_file() and provenance_path.stat().st_size > 0, "missing manuscript provenance")
    require(manifest_path.is_file() and manifest_path.stat().st_size > 0, "missing manuscript artifact manifest")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require(provenance.get("superseded_inputs") == [], "manuscript provenance contains superseded inputs")
    declared_inputs = []
    for asset in provenance.get("assets", []):
        for record in asset.get("inputs", []):
            path = ROOT / record["path"]
            declared_inputs.append(path.name)
            if verify_inputs:
                require(path.is_file() and file_hash(path) == record["sha256"], f"input hash mismatch: {record['path']}")
        for record in asset.get("outputs", []):
            path = ROOT / record["path"]
            if path.is_file():
                require(path.stat().st_size > 0 and file_hash(path) == record["sha256"], f"output hash mismatch: {record['path']}")
    require(not (set(declared_inputs) & SUPERSEDED), "manuscript assets use a superseded result")
    require(file_hash(manifest_path) == provenance["manifest"]["sha256"], "artifact manifest hash mismatch")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    require(bool(manifest_rows), "artifact manifest has no primary artifacts")
    for record in manifest_rows:
        path = ROOT / record["path"]
        if record["kind"] == "manuscript_asset" and path.is_file():
            require(file_hash(path) == record["sha256"], f"manifest hash mismatch: {record['path']}")


def verify_arxiv_artifacts(*, require_bundle: bool = False) -> None:
    arxiv = ROOT / "arxiv"
    required = ("main.tex", "main.bbl", "references.bib", "README.md", "Makefile")
    for name in required:
        path = arxiv / name
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty arXiv source: arxiv/{name}")
        print(f"OK: arxiv/{name} sha256={file_hash(path)}")
    main = (arxiv / "main.tex").read_text(encoding="utf-8")
    for name in re.findall(r"\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}", main):
        require((arxiv / name).is_file(), f"missing arXiv source dependency: arxiv/{name}")
    if not require_bundle:
        return
    manifest_path = ROOT / "dist" / "arxiv_bundle_manifest.json"
    require(manifest_path.is_file(), "missing arXiv bundle manifest; run scripts/build_arxiv_bundle.py")
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = ROOT / bundle["archive"]
    require(archive.is_file() and archive.stat().st_size > 0, "missing or empty arXiv bundle")
    require(file_hash(archive) == bundle["archive_sha256"], "arXiv bundle hash mismatch")
    print(f"OK: {bundle['archive']} sha256={bundle['archive_sha256']}")
    pdf = ROOT / "dist" / "Eyetrack2LLM-arxiv-draft.pdf"
    if pdf.exists():
        require(pdf.stat().st_size > 0, "compiled arXiv PDF is empty")
        print(f"OK: compiled arXiv PDF sha256={file_hash(pdf)}")
    else:
        print("OK: compiled arXiv PDF absent; source-only verification remains valid")


def verify_compact_bundle() -> None:
    manifest_path = ROOT / "dist" / "compact_artifact_bundle_manifest.json"
    require(manifest_path.is_file(), "missing compact artifact bundle manifest")
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = ROOT / bundle["archive"]
    require(archive.is_file() and file_hash(archive) == bundle["archive_sha256"], "compact bundle hash mismatch")
    require(bundle.get("uncompressed_size", 0) <= 200 * 1024 * 1024, "compact bundle exceeds 200 MiB")
    declared = {item["path"]: item for item in bundle["files"]}
    with (ROOT / "manuscript/artifact_manifest.csv").open(encoding="utf-8", newline="") as handle:
        compact = [row for row in csv.DictReader(handle) if row["kind"] == "compact_artifact" and row["role"] in {"primary", "secondary"}]
    for row in compact:
        require(row["path"] in declared, f"compact bundle omitted manifest artifact: {row['path']}")
        require(declared[row["path"]]["sha256"] == row["sha256"], f"compact bundle manifest hash mismatch: {row['path']}")
        require(file_hash(ROOT / row["path"]) == row["sha256"], f"compact source hash mismatch: {row['path']}")
    print(f"OK: {bundle['archive']} sha256={bundle['archive_sha256']}")


def verify_full_local_results() -> None:
    for seed in SEEDS:
        data = load(f"provo_auxiliary_strictline_fixed50_seed{seed}.json")
        require(data.get("feature_set") == "common_core", f"seed {seed}: expected common_core features")
        require(data.get("risk_set") == RISK_SET, f"seed {seed}: expected strict-line forward risk set")
        require(data.get("fixed_step") == 50 and data.get("steps") == 50, f"seed {seed}: expected fixed step 50")
        require(set(data.get("conditions", {})) == CONTROLS, f"seed {seed}: expected exactly four controls")
        require(data.get("design_rank") == 12, f"seed {seed}: expected rank-12 design")
        require(data.get("group_constant_features") == [], f"seed {seed}: group constants are not allowed")

    text_inference = load("provo_auxiliary_strictline_fixed50_text_inference.json")
    require(text_inference.get("seeds") == list(SEEDS) and len(text_inference.get("test_texts", [])) == 10, "Provo text inference must average five seeds over 10 test texts")
    require(set(text_inference.get("conditions", {})) == CONTROLS, "Provo text inference conditions are incomplete")
    require(all(value.get("valid_edges_per_seed") == 2838 for value in text_inference["conditions"].values()), "Provo text inference edge counts are incorrect")
    require(set(text_inference.get("comparisons", {})) == {"gaze_minus_mlm", "gaze_minus_shuffled", "gaze_minus_position"}, "Provo text inference comparisons are incomplete")
    require(all(value.get("texts") == 10 and value.get("seed_aggregation") == "mean before text inference" for value in text_inference["comparisons"].values()), "Provo inference must not treat seeds as observations")
    budget = load("provo_auxiliary_strictline_budget_sensitivity.json")
    require(budget.get("analysis_role") == "exploratory budget sensitivity", "budget sensitivity must remain exploratory")
    require(budget.get("pre_fixed_grid") == {"steps": [50, 100, 200], "gaze_weight": [0.1], "seeds": [101, 202, 303]}, "budget grid differs from the pre-fixed controlled grid")
    require(set(budget.get("summary", {})) == {"50", "100", "200"} and all(set(value) == CONTROLS for value in budget["summary"].values()), "budget summary is incomplete")

    transfer = load("zuco_transfer_strictline_fixed50.json")
    design = transfer.get("design", {})
    require(design.get("event") == "forward-only same-sentence same-line candidate risk set and counts", "transfer event is not strict-line forward risk")
    require(design.get("n_texts") == 200 and len(design.get("subjects", [])) == 12, "transfer must contain 200 texts and 12 subjects")
    require(design.get("design_rank") == 12 and len(design.get("feature_names", [])) == 12, "transfer must use the rank-12, 12-feature design")
    require(design.get("group_constant_features") == [], "transfer design contains group constants")
    require(design.get("candidate_pairs") == 18247 and design.get("source_groups") == 3880, "transfer pair/group counts are incorrect")
    comparisons = transfer.get("comparisons", {})
    required_comparisons = {"gaze_vs_mlm", "gaze_vs_shuffled", "gaze_vs_position"}
    require(set(comparisons) == required_comparisons, "primary transfer comparisons are incomplete")
    require(all(value.get("text_equal_fisher_z", {}).get("texts_valid") == 193 for value in comparisons.values()), "transfer comparisons must each have 193 primary valid texts")

    uncertainty = load("zuco_strictline_criterion_uncertainty.json")
    metrics = {"edge_weighted", "source_equal_flatten", "per_source_fisher_equal"}
    udesign = uncertainty.get("design", {})
    require(udesign.get("partitions") == 462 and udesign.get("split") == "6/6", "ZuCo criterion audit requires all 462 unique 6/6 partitions")
    require(udesign.get("half_min_exposure") == 5 and udesign.get("full_min_exposure") == 10, "ZuCo criterion exposure thresholds are incorrect")
    require(uncertainty.get("reader_bootstrap", {}).get("repeats", 0) >= 200, "ZuCo audit requires at least 200 reader bootstraps")
    require(uncertainty.get("provo_12_reader_sensitivity", {}).get("subsets", 0) >= 200, "Provo sensitivity requires at least 200 balanced 12-reader subsets")
    require(set(uncertainty.get("reliability", {}).get("summary", {})) == metrics, "ZuCo reliability summaries are incomplete")
    require_csv("zuco_strictline_criterion_uncertainty.csv")

    reliability = load("provo_commoncore_strictline_independent_reliability.json")
    require(reliability.get("repeats") == 100 and reliability.get("subjects") == 84 and reliability.get("split") == "42/42", "independent reliability repeat/subject/split design is incorrect")
    require(reliability.get("feature_set") == "common_core" and reliability.get("risk_set") == RISK_SET, "independent reliability has the wrong strict-line design")
    require(len(reliability.get("repeat_results", [])) == 100, "independent reliability must contain 100 repeat results")
    summary = reliability.get("summary", {})
    require(metrics <= set(summary.get("independent", {})) and metrics <= set(summary.get("shared", {})), "independent reliability summary metrics are incomplete")

    simulation = load("residual_recovery_simulation.json")
    config = simulation.get("config", {})
    require(config.get("subjects") == [4, 12, 42, 84], "simulation subject grid is incorrect")
    require(config.get("latent_effects") == [0.0, 0.55], "simulation effect grid is incorrect")
    require(config.get("concentrations") == [120.0, 8.0], "simulation concentration grid is incorrect")
    require(config.get("replicates") == 80, "simulation must use 80 replicates")
    require(len(simulation.get("summary", [])) == 48, "simulation summary must contain 48 cells")
    require(len(simulation.get("replicate_results", [])) == 4 * 2 * 2 * 80 * 3, "simulation replicate-level record count is incorrect")

    curve = load("provo_strictline_specification_curve.json")
    require(curve.get("repeats") == 100 and curve.get("null_repeats") == 500, "specification curve repeat counts are incorrect")
    require(curve.get("risk_set") == RISK_SET, "specification curve has the wrong risk set")
    expected_ranks = {"position_only": 5, "lexical": 9, "syntax": 12, "flexible": 15}
    specifications = curve.get("specifications", {})
    require(set(specifications) == set(expected_ranks), "specification curve must contain exactly four specifications")
    require(all(specifications[name].get("conditional_rank") == rank and specifications[name].get("group_constant_features") == [] for name, rank in expected_ranks.items()), "specification ranks or group constants are incorrect")
    require(set(curve.get("summary", {})) == set(expected_ranks), "specification summary is incomplete")
    require(bool(curve.get("full_sample_residual_identity")), "pairwise residual identity controls are missing")
    require(all(len(curve.get("repeat_results", {}).get(name, [])) == 100 for name in expected_ranks), "specification repeat results are incomplete")
    require(all(all(len(item.get("per_text_metrics", {})) == 55 for item in curve["repeat_results"][name]) for name in expected_ranks), "specification repeats must retain 55 per-text metric records")
    require(all(len(curve.get("null_results", {}).get(name, [])) == 500 for name in expected_ranks), "specification negative controls are incomplete")
    require(all([item.get("permutation_replicate") for item in curve["null_results"][name]] == list(range(500)) for name in expected_ranks), "shared permutation replicate indices are not aligned")
    inference = curve.get("permutation_inference", {})
    require(inference.get("shared_replicate_index") is True, "specification nulls must share replicate indices")
    require(len(inference.get("primary_null_maxima", [])) == 500 and len(inference.get("secondary_null_maxima", [])) == 500, "max-statistic null distributions are incomplete")
    require(all(set(inference.get("cells", {}).get(name, {})) == metrics for name in expected_ranks), "max-statistic cells are incomplete")
    with (PROCESSED / "provo_strictline_specification_curve.csv").open(encoding="utf-8", newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    require(len(curve_rows) == 4 * 100 + 4 * 500, "specification curve CSV must contain 2,400 data rows")

    half_audit = load("provo_half_specific_baseline_audit.json")
    require(half_audit.get("repeats") == 100 and half_audit.get("halves_per_repeat") == 2, "half-specific audit must contain 100 two-half partitions")
    require(set(half_audit.get("specifications", [])) == set(expected_ranks), "half-specific audit specifications are incomplete")
    require(set(half_audit.get("baseline_modes", {})) == {"half_specific", "shared_full84"}, "half-specific audit baseline controls are incomplete")
    require(len(half_audit.get("half_spec_records", [])) == 100 * 2 * 4 * 2, "half-specific residual-variance records are incomplete")
    require(len(half_audit.get("identity_records", [])) == 100 * 2 * 2 * 3 * 2, "half-specific identity records are incomplete")
    require(set(half_audit.get("variance_summary", {})) == {"half_specific", "shared_full84"}, "half-specific residual variance summaries are incomplete")
    require(set(half_audit.get("strata_summary", {})) == {"half_specific", "shared_full84"}, "half-specific probability/count strata are incomplete")
    require(bool(half_audit.get("coefficient_variability")) and bool(half_audit.get("mode_difference")), "half-specific coefficient or paired-difference diagnostics are missing")
    require("not a complete variance decomposition" in half_audit.get("interpretation_guardrail", ""), "half-specific audit is missing its variance-decomposition guardrail")

    decomposition = load("provo_target_selection_decomposition.json")
    require(decomposition.get("analysis_role") == "secondary" and decomposition.get("selection") == "theory_guided", "target-selection decomposition must remain secondary and theory-guided")
    require(decomposition.get("primary_pipeline") == "frozen and unchanged", "target-selection analysis must not replace the primary pipeline")
    require(decomposition.get("risk_set") == RISK_SET, "target-selection decomposition has the wrong candidate universe")
    require(decomposition.get("repeats") == 100 and decomposition.get("null_repeats") == 25, "target-selection repeat counts are incorrect")
    categories = {"adjacent", "near_skip", "far_same_line"}
    require(set(decomposition.get("candidate_and_observation_audit", {})) == categories, "target-selection categories are incomplete")
    require(set(decomposition.get("summary", {})) == set(expected_ranks), "target-selection specifications are incomplete")
    require(all(set(decomposition["summary"][name]) == categories for name in expected_ranks), "target-selection category summaries are incomplete")
    require(all(set(decomposition["summary"][name][category]) == metrics for name in expected_ranks for category in categories), "target-selection estimands are incomplete")
    require(all(decomposition["summary"][name]["adjacent"]["per_source_fisher_equal"]["defined_split_replicates"] == 0 for name in expected_ranks), "adjacent within-source correlations must remain undefined")

    diagnostics = load("provo_residual_exposure_diagnostics.json")
    require(diagnostics.get("repeats", 0) >= 100 and diagnostics.get("subjects") == 84, "residual/exposure diagnostics require 100 partitions and 84 subjects")
    require(set(diagnostics.get("specifications", [])) == {"syntax", "flexible"}, "residual diagnostics must contain syntax and flexible specifications")
    require(diagnostics.get("candidate_universe_changed") is False, "exposure matching must preserve the candidate universe")
    require(set(diagnostics.get("residual_definitions", {})) >= {"pearson", "deviance", "raw_deviation", "limitation"}, "residual definitions are incomplete")
    require(any(row.get("analysis") == "exposure_matched" for row in diagnostics.get("reliability_summary", [])), "exposure-matched target sensitivity is missing")
    require_csv("provo_residual_exposure_diagnostics.csv")

    invariance = load("cross_corpus_measurement_invariance.json")
    require(invariance.get("bootstrap_repeats") >= 1000, "cross-corpus audit requires at least 1000 text bootstraps")
    inv_design = invariance.get("design", {})
    require(inv_design.get("provo_texts") == 55 and inv_design.get("zuco_texts") == 200, "cross-corpus audit text counts are incorrect")
    require(inv_design.get("provo_rank") == inv_design.get("zuco_rank") == 12 and len(inv_design.get("feature_names", [])) == 12, "cross-corpus common-core names/rank are incorrect")
    require(invariance.get("definitions", {}).get("risk_set") == RISK_SET, "cross-corpus audit risk set is incorrect")
    require(len(invariance.get("feature_distribution", [])) == 12 and len(invariance.get("nuisance_fit", {}).get("coefficients", [])) == 12, "cross-corpus feature/coefficient audits are incomplete")
    require(set(invariance.get("transport_calibration", {})) == {"provo", "zuco"}, "cross-corpus transport calibration is incomplete")
    require(invariance.get("domain_distinguishability", {}).get("unit") == "text" and invariance.get("domain_distinguishability", {}).get("label_permutations", 0) >= 500, "domain classification must use texts and at least 500 permutations")

    influence = load("text_influence_diagnostics.json")
    require(influence.get("provo_reliability", {}).get("texts") == 55 and influence.get("provo_reliability", {}).get("repeats") == 100, "influence diagnostics require 55 Provo texts and 100 repeats")
    influence_specs = influence.get("provo_reliability", {}).get("specifications", {})
    require(set(influence_specs) == set(expected_ranks), "influence diagnostics require all four specifications")
    require(all(set(influence_specs[name]) == metrics for name in expected_ranks), "influence diagnostics require all three reliability metrics")
    require(all(len(result.get("leave_one_text_out", {})) == 55 for values in influence_specs.values() for result in values.values()), "each Provo influence result requires 55 deletions")
    transfer_influence = influence.get("zuco_transfer", {})
    require(transfer_influence.get("texts") == 193 and set(transfer_influence.get("comparisons", {})) == required_comparisons, "influence diagnostics require three 193-text transfer comparisons")
    require(all(len(result.get("leave_one_text_out", {})) == 193 for result in transfer_influence["comparisons"].values()), "each transfer influence result requires 193 deletions")
    require(len(transfer_influence.get("joint_transfer", {}).get("supported_by_left_out_text", {})) == 193, "joint transfer influence rule requires 193 deletions")
    influence_csv = PROCESSED / "text_influence_diagnostics.csv"
    require(influence_csv.is_file(), "missing text influence CSV")
    with influence_csv.open(encoding="utf-8", newline="") as handle:
        influence_rows = list(csv.DictReader(handle))
    require(len(influence_rows) == 4 * 3 * 55 + 3 * 193, "text influence CSV row count is incorrect")

    conversion = load("provo_conversion_strictline_report.json", require_complete=False)
    audit = load("provo_word_line_audit.json", require_complete=False)
    require(isinstance(conversion, dict) and conversion.get("written_rows", 0) > 0, "strict-line conversion report is incomplete")
    require(isinstance(audit, list) and len(audit) == conversion.get("line_mapped_words", 0) + conversion.get("line_missing_words", 0), "word-line layout audit is incomplete")
    audit_statuses = [row.get("status") for row in audit]
    require(audit_statuses.count("mapped") == conversion.get("line_mapped_words"), "word-line layout audit mapped count is incorrect")
    require(audit_statuses.count("ambiguous") == conversion.get("line_missing_words") and set(audit_statuses) <= {"mapped", "ambiguous"}, "word-line layout audit unresolved count is incorrect")

    for csv_name in (
        "residual_recovery_simulation.csv",
        "provo_strictline_specification_curve.csv",
        "provo_half_specific_baseline_audit.csv",
        "provo_fixations_with_lines.csv",
        "provo_word_line_map.csv",
        "cross_corpus_measurement_invariance.csv",
        "text_influence_diagnostics.csv",
        "provo_target_selection_decomposition.csv",
        "provo_residual_exposure_diagnostics.csv",
        "provo_auxiliary_strictline_fixed50_per_text.csv",
        "provo_auxiliary_strictline_budget_learning_curves.csv",
    ):
        require_csv(csv_name)

    legacy_names = (
        "provo_auxiliary_commoncore_fixed50_seed101.json",
        "provo_auxiliary_commoncore_fixed50_seed202.json",
        "provo_auxiliary_commoncore_fixed50_seed303.json",
        "provo_auxiliary_commoncore_fixed50_seed404.json",
        "provo_auxiliary_commoncore_fixed50_seed505.json",
        "zuco_transfer_commoncore_fixed50.json",
        "provo_commoncore_independent_reliability.json",
        "zuco_zero_shot_transfer.json",
        "zuco_zero_shot_transfer_fixed50_forwardrisk.json",
    )
    for name in legacy_names:
        if (PROCESSED / name).exists():
            print(f"OK: legacy {name} exists but is explicitly superseded")
    verify_manuscript_assets(verify_inputs=True)
    verify_arxiv_artifacts(require_bundle=True)
    verify_compact_bundle()


def verify_release_archive(path: Path) -> None:
    require(path.is_file(), f"release archive is missing: {path}")
    with (ROOT / "manuscript/artifact_manifest.csv").open(encoding="utf-8", newline="") as handle:
        records = [row for row in csv.DictReader(handle) if row["kind"] == "compact_artifact"]
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        for record in records:
            require(record["path"] in members, f"release archive omitted: {record['path']}")
            require(hashlib.sha256(archive.read(record["path"])).hexdigest() == record["sha256"],
                    f"release archive hash mismatch: {record['path']}")


def verify_public() -> None:
    verify_manuscript_assets()
    verify_arxiv_artifacts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full-local-results", action="store_true")
    mode.add_argument("--release-archive", type=Path)
    args = parser.parse_args()
    try:
        if args.full_local_results:
            verify_full_local_results()
        elif args.release_archive:
            verify_public()
            verify_release_archive(args.release_archive)
        else:
            verify_public()
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"RESULT VERIFICATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("OK: frozen public artifacts and source closure are complete")
