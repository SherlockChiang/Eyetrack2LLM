from pathlib import Path

import pytest

from scripts.check_release import Finding, audit_release, public_files


def _minimal_release(root: Path) -> None:
    names = (".gitignore", "README.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md", "CITATION.cff", "pyproject.toml", ".github/workflows/ci.yml")
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("release metadata\n", encoding="utf-8")
    (root / "PUBLIC_FILES.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def test_public_files_expand_globs_and_exclude_internal(tmp_path):
    _minimal_release(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/public.md").write_text("public\n", encoding="utf-8")
    (tmp_path / "docs/internal.txt").write_text("internal\n", encoding="utf-8")
    with (tmp_path / "PUBLIC_FILES.txt").open("a", encoding="utf-8") as handle:
        handle.write("docs/*.md\n")
    names = {path.relative_to(tmp_path).as_posix() for path in public_files(tmp_path)}
    assert "docs/public.md" in names and "docs/internal.txt" not in names


def test_manifest_rejects_missing_and_parent_paths(tmp_path):
    _minimal_release(tmp_path)
    (tmp_path / "PUBLIC_FILES.txt").write_text("../secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe public path"):
        public_files(tmp_path)


def test_audit_detects_secret(tmp_path):
    _minimal_release(tmp_path)
    secret = tmp_path / "credentials.txt"
    secret.write_text("api_" + 'key = "123456789-secret"\n', encoding="utf-8")
    with (tmp_path / "PUBLIC_FILES.txt").open("a", encoding="utf-8") as handle:
        handle.write("credentials.txt\n")
    assert Finding("ERROR", "credentials.txt", "possible assigned secret") in audit_release(tmp_path)


def test_current_public_release_has_no_findings():
    root = Path(__file__).resolve().parents[1]
    assert audit_release(root) == []
