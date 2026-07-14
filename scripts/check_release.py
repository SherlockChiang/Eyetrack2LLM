from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
REQUIRED_FILES = ("PUBLIC_FILES.txt", ".gitignore", "README.md", "LICENSE", "CITATION.cff", "pyproject.toml", ".github/workflows/ci.yml")
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "assigned secret": re.compile(rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
}


@dataclass(frozen=True)
class Finding:
    level: str
    path: str
    message: str


def public_files(root: Path) -> list[Path]:
    manifest = root / "PUBLIC_FILES.txt"
    if not manifest.is_file():
        raise ValueError("PUBLIC_FILES.txt is missing")
    files: set[Path] = {manifest}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        posix = PurePosixPath(pattern)
        if posix.is_absolute() or ".." in posix.parts or "\\" in pattern:
            raise ValueError(f"unsafe public path: {pattern}")
        matches = [path for path in root.glob(pattern) if path.is_file()]
        if not matches:
            raise ValueError(f"public path has no match: {pattern}")
        files.update(matches)
    return sorted(files)


def candidate_files(root: Path, mode: str = "public") -> list[Path]:
    if mode in {"public", "staged"}:
        return public_files(root)
    raise ValueError(f"unknown release audit mode: {mode}")


def staged_files(root: Path) -> list[Path]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only", "-z"], cwd=root, capture_output=True, check=True)
    return [root / name.decode() for name in result.stdout.split(b"\0") if name]


def _python_import_findings(root: Path, files: list[Path]) -> list[Finding]:
    available = {path.relative_to(root).as_posix() for path in files}
    findings = []
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            findings.append(Finding("ERROR", path.relative_to(root).as_posix(), f"invalid Python: {error}"))
            continue
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and module.startswith("eyetrack2llm"):
                target = "src/" + module.replace(".", "/") + ".py"
                package = "src/" + module.replace(".", "/") + "/__init__.py"
                if target not in available and package not in available:
                    findings.append(Finding("ERROR", path.relative_to(root).as_posix(), f"public import is missing: {module}"))
    return findings


def _closure_findings(root: Path, files: list[Path]) -> list[Finding]:
    available = {path.relative_to(root).as_posix() for path in files}
    findings = []
    markdown_link = re.compile(r"\[[^]]*\]\(([^)#]+)(?:#[^)]*)?\)")
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            for link in markdown_link.findall(text):
                if "://" in link or link.startswith("mailto:"):
                    continue
                target = (path.parent / link).resolve()
                try:
                    target_name = target.relative_to(root).as_posix()
                except ValueError:
                    target_name = link
                directory_prefix = target_name.rstrip("/") + "/"
                if target_name not in available and not any(name.startswith(directory_prefix) for name in available):
                    findings.append(Finding("ERROR", relative, f"link target is not public: {link}"))
        if relative == "arxiv/main.tex":
            text = path.read_text(encoding="utf-8")
            refs = re.findall(r"\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}", text)
            for ref in refs:
                target = f"arxiv/{ref}"
                if target not in available:
                    findings.append(Finding("ERROR", relative, f"TeX dependency is not public: {target}"))
    return findings


def audit_release(root: Path, mode: str = "public") -> list[Finding]:
    findings = []
    try:
        files = candidate_files(root, mode)
        public = {path.resolve() for path in public_files(root)}
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        return [Finding("ERROR", "PUBLIC_FILES.txt", str(error))]
    if mode == "staged":
        try:
            staged = staged_files(root)
        except subprocess.CalledProcessError as error:
            return [Finding("ERROR", "git index", str(error))]
        for path in staged:
            if path.resolve() not in public:
                findings.append(Finding("ERROR", path.relative_to(root).as_posix(), "staged file is not public"))
    names = {path.relative_to(root).as_posix() for path in files}
    for name in REQUIRED_FILES:
        if name not in names:
            findings.append(Finding("ERROR", name, "required release file is missing"))
    for path in files:
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            findings.append(Finding("ERROR", relative, "public file is missing"))
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append(Finding("ERROR", relative, "file exceeds 10 MiB"))
            continue
        content = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(Finding("ERROR", relative, f"possible {label}"))
    findings.extend(_python_import_findings(root, files))
    findings.extend(_closure_findings(root, files))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--public", action="store_true")
    group.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    findings = audit_release(ROOT, "staged" if args.staged else "public")
    for finding in findings:
        print(f"{finding.level}: {finding.path}: {finding.message}")
    errors = sum(item.level == "ERROR" for item in findings)
    warnings = sum(item.level == "WARNING" for item in findings)
    if errors:
        print(f"FAIL: {errors} error(s), {warnings} warning(s)")
        return 1
    print(f"PASS: {len(candidate_files(ROOT, 'staged' if args.staged else 'public'))} files, {warnings} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
