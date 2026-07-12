from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from eyetrack2llm.zuco import convert_zuco_mat


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservatively convert a ZuCo 1.0 subject MAT file")
    parser.add_argument("mat")
    parser.add_argument("output")
    parser.add_argument("report")
    parser.add_argument("metadata")
    parser.add_argument("--subject")
    parser.add_argument("--task")
    args = parser.parse_args()
    report = convert_zuco_mat(args.mat, args.output, report_path=args.report,
                              metadata_path=args.metadata, subject=args.subject, task=args.task)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
