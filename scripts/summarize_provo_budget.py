from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("mlm", "gaze", "shuffled", "position")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True); parser.add_argument("--csv-output", required=True)
    args = parser.parse_args(); runs = [json.loads(Path(path).read_text()) for path in args.inputs]
    rows = []; summary = {}
    for run in runs:
        for condition in CONDITIONS:
            result = run["conditions"][condition]
            for point in result["curves"]:
                rows.append({"budget_steps": run["steps"], "seed": run["seed"], "condition": condition,
                             "curve_step": point["step"], "train_mlm": point["train_mlm"], "train_gaze": point["train_gaze"],
                             "val_mlm_nll": point["val"]["mlm_nll"], "val_gaze_correlation": point["val"]["gaze_correlation"],
                             "test_mlm_nll": result["test"]["mlm_nll"] if point["step"] == run["steps"] else "",
                             "test_gaze_correlation": result["test"]["gaze_correlation"] if point["step"] == run["steps"] else ""})
    for steps in sorted({run["steps"] for run in runs}):
        selected = [run for run in runs if run["steps"] == steps]
        summary[str(steps)] = {}
        for condition in CONDITIONS:
            test_z = []; val_z = []
            for run in selected:
                test_z.extend(np.arctanh(np.clip([v["gaze_correlation"] for v in run["conditions"][condition]["test"]["per_text"].values()], -.9999999, .9999999)))
                val_z.extend(np.arctanh(np.clip([v["gaze_correlation"] for v in run["conditions"][condition]["val"]["per_text"].values()], -.9999999, .9999999)))
            summary[str(steps)][condition] = {
                "val_macro_text_equal_correlation": float(np.tanh(np.mean(val_z))),
                "test_macro_text_equal_correlation": float(np.tanh(np.mean(test_z))),
                "val_pooled_mlm_nll": float(np.mean([r["conditions"][condition]["val"]["mlm_nll"] for r in selected])),
                "test_pooled_mlm_nll": float(np.mean([r["conditions"][condition]["test"]["mlm_nll"] for r in selected])),
            }
    output = {"status": "complete", "analysis_role": "exploratory budget sensitivity", "pre_fixed_grid": {"steps": [50, 100, 200], "gaze_weight": [0.1], "seeds": [101, 202, 303]},
              "selection": "No test-based selection; every pre-fixed budget is reported.", "fairness": "All four conditions share initialization, optimizer, text/mask schedule, and fixed endpoint within seed/budget.",
              "summary": summary, "split_variability": "Not run: alternative splits require new BERT targets and checkpoint retraining; compute was prioritized for the fixed budget grid."}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    with Path(args.csv_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
