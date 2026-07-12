from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ARXIV = ROOT / "arxiv"


def keys(pattern: str, text: str) -> set[str]:
    return {key.strip() for group in re.findall(pattern, text) for key in group.split(",")}


def test_latex_sources_are_complete_and_resolved() -> None:
    main = (ARXIV / "main.tex").read_text(encoding="utf-8")
    bbl = (ARXIV / "main.bbl").read_text(encoding="utf-8")
    bib = (ARXIV / "references.bib").read_text(encoding="utf-8")
    assert r"\documentclass[11pt]{article}" in main
    assert r"\input{main.bbl}" in main
    assert len(re.findall(r"\\includegraphics[^{}]*{([^}]+)}", main)) == 4
    for name in re.findall(r"\\includegraphics[^{}]*{([^}]+)}", main):
        assert (ARXIV / name).is_file()
    for name in re.findall(r"\\input{([^}]+)}", main):
        assert (ARXIV / name).is_file()
    cited = keys(r"\\cite[pt]?\{([^}]+)\}", main)
    bbl_keys = keys(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl)
    bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
    assert cited <= bbl_keys
    assert bbl_keys == bib_keys
    assert len(re.findall(r"\\label\{fig:[^}]+\}", main)) == 4
    assert r"\input{tables/table_s2_target_selection.tex}" in main
    assert r"\input{tables/table_s3_criterion_uncertainty.tex}" in main
    assert r"\input{tables/table_s4_residual_diagnostics.tex}" in main
    assert len(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl)) == 46
    stripped = re.sub(r"(?m)(?<!\\)%.*$", "", main)
    depth = 0
    for char in re.sub(r"\\[{}]", "", stripped):
        depth += char == "{"
        depth -= char == "}"
        assert depth >= 0
    assert depth == 0


def test_no_markdown_or_superseded_primary_result() -> None:
    main = (ARXIV / "main.tex").read_text(encoding="utf-8")
    assert r"\title{The Reliability Paradox in Forward Target Selection: Measurement-Conditioned Cognitive Supervision in Natural Reading}" in main
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    assert abstract is not None and len(abstract.group(1).split()) <= 200
    assert "Reliability is necessary but insufficient for behavioral signals" not in main
    assert not re.search(r"(?m)^#{1,6}\s|!\[|\]\(|Figure [1-4] about here|\*\*|`", main)
    assert "194 valid text" not in main
    assert "194-text" not in main
    assert "193 valid" in main
    assert "194-text" not in main
    assert "not prespecified" in main
    assert "parafoveal processing is confirmed" not in main.lower()
    assert "../" not in main
    assert "https://github.com/SherlockChiang/Eyetrack2LLM" in main


def test_bundle_allowlist_hashes_and_safe_paths(tmp_path) -> None:
    subprocess.run([sys.executable, "scripts/build_arxiv_bundle.py", "--output-dir", str(tmp_path)], cwd=ROOT, check=True)
    manifest = json.loads((tmp_path / "arxiv_bundle_manifest.json").read_text(encoding="utf-8"))
    archive_path = tmp_path / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["archive_sha256"]
    declared = {item["path"]: item for item in manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(declared)
        assert "main.tex" in archive.namelist()
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
            assert not re.search(r"(?:^|/)(?:data|raw|checkpoints?|cache|caches)(?:/|$)|\.json$", info.filename, re.I)
            content = archive.read(info)
            assert hashlib.sha256(content).hexdigest() == declared[info.filename]["sha256"]
            assert len(content) == declared[info.filename]["size"]
