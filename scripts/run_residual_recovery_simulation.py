from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eyetrack2llm.simulation import ResidualSimulationConfig, run_residual_recovery_simulation, summary_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted multinomial gaze residual recovery simulation")
    parser.add_argument("--output-json", default="data/processed/residual_recovery_simulation.json")
    parser.add_argument("--output-csv", default="data/processed/residual_recovery_simulation.csv")
    parser.add_argument("--replicates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    result = run_residual_recovery_simulation(ResidualSimulationConfig(replicates=args.replicates, seed=args.seed))
    json_path, csv_path = Path(args.output_json), Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = list(summary_csv_rows(result))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
