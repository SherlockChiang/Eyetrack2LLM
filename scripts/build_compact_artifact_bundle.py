from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
SOURCE_MANIFEST = ROOT / "manuscript" / "artifact_manifest.csv"
PROVENANCE = ROOT / "manuscript" / "artifact_provenance.json"
ZIP_PATH = DIST / "Eyetrack2LLM-compact-artifacts.zip"
MANIFEST_PATH = DIST / "compact_artifact_bundle_manifest.json"
MAX_BYTES = 200 * 1024 * 1024
CURRENT_ROLES = {"primary", "secondary"}
FORBIDDEN = re.compile(r"(?:^|/)(?:raw|checkpoints?|cache|caches)(?:/|$)", re.I)
AGGREGATE_CSV = (
    "data/processed/residual_recovery_simulation.csv",
    "data/processed/provo_strictline_specification_curve.csv",
    "data/processed/provo_half_specific_baseline_audit.csv",
    "data/processed/cross_corpus_measurement_invariance.csv",
    "data/processed/text_influence_diagnostics.csv",
    "data/processed/provo_target_selection_decomposition.csv",
    "data/processed/provo_residual_exposure_diagnostics.csv",
    "data/processed/provo_auxiliary_strictline_fixed50_text_inference.csv",
    "data/processed/provo_auxiliary_strictline_budget_learning_curves.csv",
    "data/processed/zuco_strictline_criterion_uncertainty.csv",
    "data/processed/zuco_edge_threshold_sensitivity.csv",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_source(name: str) -> Path:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\\" in name or FORBIDDEN.search(name):
        raise ValueError(f"unsafe or forbidden compact artifact path: {name}")
    path = (ROOT / Path(*member.parts)).resolve()
    if ROOT.resolve() not in path.parents or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing, empty, or escaped compact artifact: {name}")
    return path


def build() -> dict:
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] in CURRENT_ROLES]
    if not rows:
        raise ValueError("artifact manifest declares no current compact artifacts")

    files = []
    for row in rows:
        path = safe_source(row["path"])
        digest = sha256(path)
        if digest != row["sha256"]:
            raise ValueError(f"artifact manifest hash mismatch: {row['path']}")
        files.append({"path": row["path"], "size": path.stat().st_size, "sha256": digest})
    for name in AGGREGATE_CSV:
        path = safe_source(name)
        files.append({"path": name, "size": path.stat().st_size, "sha256": sha256(path)})
    unique = {item["path"]: item for item in files}
    if len(unique) != len(files):
        raise ValueError("duplicate compact artifact path")
    files = [unique[name] for name in sorted(unique)]
    if sum(item["size"] for item in files) > MAX_BYTES:
        raise ValueError("declared compact artifacts exceed the 200 MiB uncompressed limit")

    readme = (
        "# Eyetrack2LLM compact artifacts\n\n"
        "This archive contains every current aggregate result and generated manuscript asset declared in "
        "manuscript/artifact_manifest.csv, aggregate CSV companions, the source manifest, and field-level provenance. "
        "Raw corpora, checkpoints, generated fixation files, and caches are excluded.\n"
    ).encode("utf-8")
    extras = {
        "release/README.md": readme,
        "manuscript/artifact_manifest.csv": SOURCE_MANIFEST.read_bytes(),
        "manuscript/artifact_provenance.json": PROVENANCE.read_bytes(),
    }
    all_files = files + [{"path": name, "size": len(data), "sha256": sha256_bytes(data)} for name, data in extras.items()]

    DIST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in files:
            data = safe_source(item["path"]).read_bytes()
            info = zipfile.ZipInfo(item["path"], date_time=(2026, 7, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
        for name, data in extras.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)

    result = {
        "archive": ZIP_PATH.relative_to(ROOT).as_posix(),
        "archive_size": ZIP_PATH.stat().st_size,
        "archive_sha256": sha256(ZIP_PATH),
        "uncompressed_size": sum(item["size"] for item in all_files),
        "files": all_files,
    }
    MANIFEST_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    result = build()
    print(f"Wrote {result['archive']}")
    print(f"SHA256 {result['archive_sha256']}")
