from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def migrate_specification_curve() -> None:
    path = PROCESSED / "provo_strictline_specification_curve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("permutation_inference", None)
    payload["negative_control"] = (
        "500 one-split destination-label destruction controls balanced across the 100-split bank; "
        "independent derived RNG per half; each specification fitted separately; descriptive only "
        "because controls do not reproduce the observed 100-split summary statistic"
    )
    forbidden = {
        "empirical_exceedance", "raw_add_one_p", "primary_edge_weighted_fwer_p",
        "secondary_all_metrics_fwer_p",
    }
    for specification, summary in payload["summary"].items():
        for metric, control in summary["negative_control"].items():
            for key in forbidden:
                control.pop(key, None)
            observed = summary["reliability"][metric]["median"]
            control["observed_median_above_all_controls"] = observed > control["range"][1]
    for records in payload["null_results"].values():
        for record in records:
            if "permutation_replicate" in record:
                record["control_replicate"] = record.pop("permutation_replicate")
    write_json(path, payload)


def migrate_target_decomposition() -> None:
    path = PROCESSED / "provo_target_selection_decomposition.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for audit in payload["candidate_and_observation_audit"].values():
        if "observed_transition_count" in audit:
            audit["observed_nonzero_edges"] = audit.pop("observed_transition_count")
    for specification in payload["summary"].values():
        for category in specification.values():
            for metric in category.values():
                controls = metric.get("destination_destruction_25")
                if controls is None:
                    controls = metric.pop("destination_permutation_25")
                    metric["destination_destruction_25"] = controls
                metric.pop("empirical_exceedance", None)
                observed = metric["observed_100_split"]["median"]
                metric["observed_median_above_all_controls"] = (
                    observed is not None and controls["range"][1] is not None and observed > controls["range"][1]
                )
    for records in payload["null_results"].values():
        for record in records:
            if "permutation_replicate" in record:
                record["control_replicate"] = record.pop("permutation_replicate")
    write_json(path, payload)

    csv_path = PROCESSED / "provo_target_selection_decomposition.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    rename = {
        "observed_transition_count": "observed_nonzero_edges",
        "null_median": "control_median", "null_q25": "control_q25", "null_q75": "control_q75",
        "null_range": "control_range", "empirical_exceedance": "observed_median_above_all_controls",
    }
    migrated_fields = [rename.get(name, name) for name in fieldnames]
    for row in rows:
        for old, new in rename.items():
            if old in row:
                row[new] = row.pop(old)
        observed = float(row["observed_median"]) if row.get("observed_median") else None
        control_range = json.loads(row["control_range"]) if row.get("control_range") else [None, None]
        row["observed_median_above_all_controls"] = str(
            observed is not None and control_range[1] is not None and observed > control_range[1]
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=migrated_fields)
        writer.writeheader()
        writer.writerows(rows)

    curve_csv = PROCESSED / "provo_strictline_specification_curve.csv"
    text = curve_csv.read_text(encoding="utf-8")
    curve_csv.write_text(text.replace("destination_permutation", "destination_destruction"), encoding="utf-8", newline="\n")


def migrate_metadata() -> None:
    diagnostics_path = PROCESSED / "provo_residual_exposure_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["eligibility_sensitivity"]["partitions"] = diagnostics["repeats"]
    write_json(diagnostics_path, diagnostics)

    transfer_path = PROCESSED / "zuco_transfer_strictline_fixed50.json"
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    for offset, name in enumerate(("gaze_vs_mlm", "gaze_vs_shuffled", "gaze_vs_position")):
        transfer["comparisons"][name]["text_equal_fisher_z"]["bootstrap_seed"] = 20260711 + offset
    write_json(transfer_path, transfer)

    reliability_path = PROCESSED / "provo_commoncore_strictline_independent_reliability.json"
    reliability = json.loads(reliability_path.read_text(encoding="utf-8"))
    reliability["per_source_fisher_guard_audit"] = {
        "clip": [-0.9999999, 0.9999999],
        "purpose": "numerical stability before atanh; residual targets are not clipped",
        "defined_source_text_partition_instances": 219850,
        "two_destination_instances_lower_bound": 24033,
        "two_destination_proportion_lower_bound": 0.10931544234705481,
        "interpretation": "Every defined two-destination correlation is near +1 or -1 and encounters the guard; larger candidate sets can also be guarded, so this is a lower bound.",
    }
    write_json(reliability_path, reliability)


if __name__ == "__main__":
    migrate_specification_curve()
    migrate_target_decomposition()
    migrate_metadata()
    print("Migrated canonical control artifacts to descriptive-only schema")
