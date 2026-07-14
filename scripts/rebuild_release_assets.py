from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from verify_results import ROOT, verify_release_archive


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a compact release, rebuild manuscript assets, and verify hashes")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    verify_release_archive(archive)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.filename.startswith(("data/processed/", "manuscript/")):
                handle.extract(member, ROOT)
    run("scripts/generate_manuscript_assets.py", "--timestamp", "2026-07-12T00:00:00+00:00")
    run("scripts/verify_results.py", "--manuscript-assets")
    run("scripts/build_compact_artifact_bundle.py")


if __name__ == "__main__":
    main()
