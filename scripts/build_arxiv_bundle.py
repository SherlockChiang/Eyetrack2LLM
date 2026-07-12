from __future__ import annotations

import hashlib
import argparse
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ARXIV = ROOT / "arxiv"
DIST = ROOT / "dist"
ZIP_PATH = DIST / "Eyetrack2LLM-arxiv-draft.zip"
MANIFEST_PATH = DIST / "arxiv_bundle_manifest.json"

ALLOWLIST = (
    "main.tex", "main.bbl", "references.bib", "README.md", "Makefile",
    "figures/figure1_evidence_ladder.pdf",
    "figures/figure2_reliability_paradox.pdf",
    "figures/figure3_specification_curve.pdf",
    "figures/figure4_functional_evidence.pdf",
    "tables/table1.tex", "tables/table2_endpoints.tex", "tables/table2_full.tex",
    "tables/table3.tex", "tables/table4.tex", "tables/table_s1.tex",
    "tables/table_s2_target_selection.tex", "tables/table_s3_criterion_uncertainty.tex",
    "tables/table_s4_residual_diagnostics.tex",
)
FORBIDDEN = re.compile(r"(?:^|/)(?:data|raw|checkpoints?|cache|caches)(?:/|$)|\.json$", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_member(name: str) -> Path:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\\" in name or FORBIDDEN.search(name):
        raise ValueError(f"unsafe or forbidden bundle path: {name}")
    path = (ARXIV / Path(*member.parts)).resolve()
    if ARXIV.resolve() not in path.parents or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing, empty, or escaped allowlisted source: {name}")
    return path


def build(output_dir: Path = DIST) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / ZIP_PATH.name
    manifest_path = output_dir / MANIFEST_PATH.name
    files = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ALLOWLIST:
            path = validate_member(name)
            digest = sha256(path)
            files.append({"path": name, "size": path.stat().st_size, "sha256": digest})
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    manifest = {
        "archive": zip_path.name,
        "archive_size": zip_path.stat().st_size,
        "archive_sha256": sha256(zip_path),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DIST)
    result = build(parser.parse_args().output_dir)
    print(f"Wrote {result['archive']}")
    print(f"SHA256 {result['archive_sha256']}")
