from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from check_release import MAX_FILE_BYTES, ROOT, public_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely stage only PUBLIC_FILES.txt entries.")
    parser.add_argument("--stage", action="store_true", help="stage files; default is dry-run")
    args = parser.parse_args()
    files = public_files(ROOT)
    relative = [path.relative_to(ROOT).as_posix() for path in files]
    for path, name in zip(files, relative, strict=True):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SystemExit(f"refusing file over 10 MiB: {name}")
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", name], cwd=ROOT).returncode == 0
        if ignored:
            raise SystemExit(f"refusing ignored file: {name}")
    print("\n".join(relative))
    print(f"{'STAGE' if args.stage else 'DRY RUN'}: {len(relative)} public files")
    if args.stage:
        subprocess.run(["git", "add", "--", *relative], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
