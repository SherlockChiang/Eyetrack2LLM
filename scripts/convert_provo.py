from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from eyetrack2llm.provo import convert_provo_fixations


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official Provo fixation data")
    parser.add_argument("main_csv")
    parser.add_argument("fixation_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--report")
    parser.add_argument("--line-map")
    parser.add_argument("--line-audit")
    args = parser.parse_args()
    report = convert_provo_fixations(
        args.main_csv,
        args.fixation_csv,
        args.output_csv,
        report_path=args.report,
        line_map_path=args.line_map,
        line_audit_path=args.line_audit,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
