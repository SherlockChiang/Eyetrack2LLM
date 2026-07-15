from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eyetrack2llm.influence import (analytic_leave_one_out_mean,
                                    jackknife_mean_interval,
                                    leave_one_text_out_median_hierarchy,
                                    most_influential)


SPECIFICATIONS = ("position_only", "lexical", "syntax", "flexible")
METRICS = ("edge_weighted", "source_equal_flatten", "per_source_fisher_equal")
COMPARISONS = ("gaze_vs_mlm", "gaze_vs_shuffled", "gaze_vs_position")
SEED = 20260711


def ordered_texts(values):
    return sorted(values, key=lambda value: int(value.split(":")[-1]))


def change_flags(full, loto, threshold=0.0):
    return {
        "sign_reversal": bool((full > 0 and any(value <= 0 for value in loto.values()))
                              or (full < 0 and any(value >= 0 for value in loto.values()))),
        "threshold": threshold,
        "threshold_crossing": bool((full > threshold and any(value <= threshold for value in loto.values()))
                                   or (full <= threshold and any(value > threshold for value in loto.values()))),
    }


def main():
    parser = argparse.ArgumentParser(description="Frozen leave-one-text-out influence diagnostics")
    parser.add_argument("--provo", default="data/processed/provo_strictline_specification_curve.json")
    parser.add_argument("--zuco", default="data/processed/zuco_transfer_strictline_fixed50.json")
    parser.add_argument("--output", default="data/processed/text_influence_diagnostics.json")
    parser.add_argument("--csv-output", default="data/processed/text_influence_diagnostics.csv")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    provo = json.loads(Path(args.provo).read_text(encoding="utf-8"))
    zuco = json.loads(Path(args.zuco).read_text(encoding="utf-8"))

    rows, provo_results = [], {}
    for specification in SPECIFICATIONS:
        repeat_results = provo["repeat_results"][specification]
        texts = ordered_texts(repeat_results[0]["per_text_metrics"])
        if len(repeat_results) != 100 or len(texts) != 55:
            raise ValueError(f"{specification}: expected 100 repeats and 55 texts")
        provo_results[specification] = {}
        for metric in METRICS:
            repeats = [{text: item["per_text_metrics"][text][metric] for text in texts}
                       for item in repeat_results]
            full, loto = leave_one_text_out_median_hierarchy(repeats, texts)
            expected = float(provo["summary"][specification]["reliability"][metric]["median"])
            if not np.isclose(full, expected, rtol=0, atol=1e-12):
                raise ValueError(f"{specification}/{metric}: full estimand does not reproduce summary")
            influential, max_change = most_influential(full, loto)
            flags = change_flags(full, loto)
            result = {"full": full, "loto_range": [min(loto.values()), max(loto.values())],
                      "max_absolute_change": max_change, "most_influential_text": influential,
                      **flags, "leave_one_text_out": loto}
            provo_results[specification][metric] = result
            for text, estimate in loto.items():
                rows.append({"corpus": "Provo", "analysis": specification, "metric": metric,
                             "left_out_text": text, "full": full, "loto": estimate,
                             "change": estimate - full, "bootstrap_ci_low": "", "bootstrap_ci_high": "",
                             "signflip_p": "", "jackknife_ci_low": "", "jackknife_ci_high": "",
                             "conclusion_flip": flags["threshold_crossing"]})

    zuco_results = {}
    comparison_loto = {}
    comparison_ci_excludes = {}
    expected_texts = None
    for index, comparison in enumerate(COMPARISONS):
        source = zuco["comparisons"][comparison]["text_equal_fisher_z"]
        per_text = source["per_text_seed_averaged_differences"]
        texts = ordered_texts(per_text)
        values = np.asarray([per_text[text] for text in texts], float)
        if len(values) != source["texts_valid"]:
            raise ValueError(f"{comparison}: per-text values differ from declared valid-text count")
        expected_texts = len(values) if expected_texts is None else expected_texts
        if len(values) != expected_texts:
            raise ValueError(f"{comparison}: valid-text support differs across comparisons")
        full = float(values.mean())
        if not np.isclose(full, source["mean_difference"], rtol=0, atol=1e-15):
            raise ValueError(f"{comparison}: mean does not reproduce transfer artifact")
        loto_values = analytic_leave_one_out_mean(values)
        loto = dict(zip(texts, map(float, loto_values), strict=True))
        influential, max_change = most_influential(full, loto)
        se, low, high = jackknife_mean_interval(values)
        records, ci_excludes = {}, {}
        for text_index, (text, estimate) in enumerate(loto.items()):
            retained = np.delete(values, text_index)
            rng = np.random.default_rng(SEED + index * 1000 + text_index)
            means = retained[rng.integers(len(retained), size=(args.resamples, len(retained)))].mean(axis=1)
            ci = np.quantile(means, (.025, .975)).tolist()
            excludes = bool(ci[0] > 0 or ci[1] < 0)
            ci_excludes[text] = excludes
            records[text] = {"mean": estimate, "descriptive_text_resampling_interval": ci,
                             "ci_excludes_zero": excludes}
            rows.append({"corpus": "ZuCo", "analysis": comparison, "metric": "text_equal_fisher_z_difference",
                         "left_out_text": text, "full": full, "loto": estimate, "change": estimate - full,
                         "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1], "signflip_p": "",
                         "jackknife_ci_low": low, "jackknife_ci_high": high,
                         "conclusion_flip": excludes != bool(source["descriptive_text_resampling_interval"][0] > 0 or source["descriptive_text_resampling_interval"][1] < 0)})
        flags = change_flags(full, loto)
        zuco_results[comparison] = {"full": full, "full_descriptive_text_resampling_interval": source["descriptive_text_resampling_interval"],
            "loto_range": [float(loto_values.min()), float(loto_values.max())],
            "max_absolute_change": max_change, "most_influential_text": influential, **flags,
            "jackknife_se": se, "jackknife_95_ci": [low, high],
            "any_bootstrap_conclusion_flip": any(row["conclusion_flip"] for row in rows if row["analysis"] == comparison),
            "leave_one_text_out": records}
        comparison_loto[comparison] = loto
        comparison_ci_excludes[comparison] = ci_excludes

    output = {"status": "complete", "seed": SEED, "resamples_per_zuco_loto": args.resamples,
        "sources": {"provo": args.provo, "zuco": args.zuco},
        "definitions": {"selection_rule": "All texts are deleted one at a time; no exclusion, refit, or result-dependent selection.",
            "provo_unit": "text (55); within each frozen subject split take the median over retained texts, then the median over 100 splits",
            "zuco_unit": f"text ({expected_texts} valid paired texts); analytic delete-one mean of frozen per-text seed-averaged Fisher-z differences",
            "influential_rule": "text with largest absolute LOTO-minus-full change; numeric text order breaks exact ties",
            "provo_threshold": "zero reliability", "zuco_threshold": "zero paired difference",
            "zuco_uncertainty": "delete-one jackknife SE/normal interval plus deterministic descriptive text resampling for every retained set; no sign-flip p values or joint decision rule"},
        "provo_reliability": {"texts": 55, "repeats": 100, "specifications": provo_results},
        "zuco_transfer": {"texts": expected_texts, "comparisons": zuco_results}}
    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = Path(args.csv_output); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
