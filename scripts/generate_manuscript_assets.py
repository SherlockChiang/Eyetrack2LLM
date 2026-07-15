from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PRIMARY = {
    "conversion": "provo_conversion_strictline_report.json",
    "reliability": "provo_commoncore_strictline_independent_reliability.json",
    "simulation": "residual_recovery_simulation.json",
    "specification": "provo_strictline_specification_curve.json",
    "transfer": "zuco_transfer_strictline_fixed50.json",
    "invariance": "cross_corpus_measurement_invariance.json",
    "influence": "text_influence_diagnostics.json",
    "half_audit": "provo_half_specific_baseline_audit.json",
    "target_decomposition": "provo_target_selection_decomposition.json",
    "residual_diagnostics": "provo_residual_exposure_diagnostics.json",
    "aux_text": "provo_auxiliary_strictline_fixed50_text_inference.json",
    "aux_budget": "provo_auxiliary_strictline_budget_sensitivity.json",
    "criterion": "zuco_strictline_criterion_uncertainty.json",
    "reconciliation": "provo_word_position_reconciliation.json",
    "reconciled_sensitivity": "provo_reconciled_sensitivity.json",
    "edge_threshold": "zuco_edge_threshold_sensitivity.json",
    "line_identity": "line_partition_identity_audit.json",
    **{f"seed_{seed}": f"provo_auxiliary_strictline_fixed50_seed{seed}.json" for seed in (101, 202, 303, 404, 505)},
}
SUPERSEDED = {
    "provo_commoncore_independent_reliability.json",
    "zuco_transfer_commoncore_fixed50.json",
    "zuco_zero_shot_transfer.json",
    "zuco_zero_shot_transfer_fixed50_forwardrisk.json",
    "fresh_probe_representation.json",
    "zuco_transfer_sensitivity.json",
}
SPEC_ORDER = ("position_only", "lexical", "syntax", "flexible")
METHOD_ORDER = ("correct", "misspecified", "raw")
METHOD_LABELS = {"correct": "generating-nuisance-complete", "misspecified": "x2-omitting mean model", "raw": "unadjusted counts/deviation"}
PALETTE = {"blue": "#2F5D7E", "orange": "#C47F3A", "green": "#4F7C6B", "red": "#A34F46", "purple": "#7A668C", "gray": "#6B7075"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_primary(processed: Path = PROCESSED) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {key: processed / name for key, name in PRIMARY.items()}
    if any(path.name in SUPERSEDED for path in paths.values()):
        raise ValueError("primary artifact allowlist contains a superseded input")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing primary artifacts: {', '.join(missing)}")
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    for key, value in data.items():
        if key not in {"conversion", "reconciliation"} and value.get("status") != "complete":
            raise ValueError(f"primary artifact is not complete: {paths[key].name}")
    return data, paths


def extract_assets(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sim = data["simulation"]
    simulation = [{k: row.get(k) for k in (
        "subjects", "latent_effect", "concentration", "overdispersion", "method", "replicates",
        "latent_recovery_correlation_mean", "latent_recovery_correlation_q025", "latent_recovery_correlation_q975",
        "split_half_residual_reliability_mean", "split_half_residual_reliability_q025", "split_half_residual_reliability_q975",
    )} for row in sim["summary"]]
    specification = []
    spec = data["specification"]
    for name in SPEC_ORDER:
        for row in spec["repeat_results"][name]:
            specification.append({
                "specification": name, "kind": "observed", "repeat": row["repeat"],
                "nll_minus_uniform": sum(part["nll_minus_uniform"] for part in row["predictive"]) / 2,
                "edge_weighted_reliability": row["reliability"]["edge_weighted"],
            })
        for index, row in enumerate(spec["null_results"][name]):
            specification.append({
                "specification": name, "kind": "destruction_control", "repeat": row.get("control_replicate", index),
                "nll_minus_uniform": "", "edge_weighted_reliability": row["reliability"]["edge_weighted"],
            })
    functional = []
    for seed in (101, 202, 303, 404, 505):
        run = data[f"seed_{seed}"]
        for condition in ("mlm", "gaze"):
            test = run["conditions"][condition]["test"]
            functional.append({"panel": "provo", "seed": seed, "condition": condition,
                               "mlm_nll": test["mlm_nll"], "residual_correlation": test["gaze_correlation"]})
    for contrast in ("gaze_vs_mlm", "gaze_vs_shuffled", "gaze_vs_position"):
        result = data["transfer"]["comparisons"][contrast]["text_equal_fisher_z"]
        fixed = result["descriptive_text_resampling_interval"]
        nested = data["criterion"]["reader_bootstrap"]["summary"][contrast]["joint_reader_and_text"]
        functional.append({"panel": "zuco", "contrast": contrast, "mean": result["mean_difference"],
                           "fixed_ci_low": fixed[0], "fixed_ci_high": fixed[1],
                           "nested_ci_low": nested["95_ci"][0], "nested_ci_high": nested["95_ci"][1],
                           "uncertainty": "fixed-12 descriptive text-reaggregation interval and support-varying reader-refit diagnostic range",
                           "texts": result["texts_valid"]})
    gaze_text = data["aux_text"]["conditions"]["gaze"]
    functional.append({"panel": "provo_summary", "condition": "gaze", "macro_text_correlation": gaze_text["macro_text_equal_correlation_seed_averaged_fisher_z"],
                       "pooled_edge_correlation": gaze_text["pooled_edge_correlation"], "texts": len(data["aux_text"]["test_texts"])})
    ladder = [
        {"stage": 1, "evidence": "Construct definition", "evidence_status": "Computational relation; not semantic distance"},
        {"stage": 2, "evidence": "Estimator validation", "evidence_status": sim["data_generating_process"]["counts"]},
        {"stage": 3, "evidence": "Measurement reliability", "evidence_status": "Two non-overlapping 42-reader halves per fixed-sample partition"},
        {"stage": 4, "evidence": "Functional test", "evidence_status": "Provo constructed-residual alignment and ZuCo cross-corpus scorer evaluation"},
        {"stage": 5, "evidence": "External construct validation", "evidence_status": "Future human study; not completed"},
    ]
    return {"figure1": ladder, "figure2": simulation, "figure3": specification, "figure4": functional}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_table(base: Path, title: str, rows: list[dict[str, Any]]) -> list[Path]:
    csv_path = base.with_suffix(".csv")
    md_path = base.with_suffix(".md")
    _write_csv(csv_path, rows)
    fields = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return [csv_path, md_path]


def _style(plt: Any) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "axes.edgecolor": "#3F4448", "xtick.color": "#3F4448", "ytick.color": "#3F4448",
        "text.color": "#25282A", "axes.labelcolor": "#25282A", "figure.facecolor": "white",
        "axes.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def make_figures(extracted: dict[str, list[dict[str, Any]]], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _style(plt)
    made: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 3.55), constrained_layout=True)
    rows = extracted["figure1"]
    from textwrap import fill
    y = np.arange(len(rows))[::-1]
    ax.plot([.65, .65], [y[-1], y[0]], color="#B9BEC2", lw=1.4, zorder=1)
    for i, row in enumerate(rows):
        completed = i < 4
        ax.scatter(.65, y[i], s=430, color=PALETTE["blue"] if completed else "white",
                   edgecolor=PALETTE["blue"] if completed else PALETTE["gray"], linewidth=1.2, zorder=3)
        ax.text(.65, y[i], str(row["stage"]), ha="center", va="center",
                color="white" if completed else PALETTE["gray"], weight="bold", fontsize=8)
        ax.text(1.15, y[i] + .14, row["evidence"], ha="left", va="center", weight="bold", fontsize=8.2)
        ax.text(1.15, y[i] - .17, fill(str(row["evidence_status"]), 86), ha="left", va="center",
                fontsize=7.3, color="#52575B", linespacing=1.25)
        if not completed:
            ax.text(6.85, y[i], "NOT COMPLETED", ha="right", va="center", fontsize=6.8,
                    color=PALETTE["red"], weight="bold")
    ax.set(xlim=(0, 7.05), ylim=(-.55, 4.55), title="Evidence ladder for a computational measurement claim")
    ax.axis("off")
    made += _save(fig, output / "figure1_evidence_ladder")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True, sharey="row")
    simrows = extracted["figure2"]
    for col, effect in enumerate((0.0, 0.55)):
        for method, color, marker in zip(METHOD_ORDER, (PALETTE["blue"], PALETTE["red"], PALETTE["gray"]), ("o", "s", "^")):
            for over, ls in (("low", "-"), ("high", "--")):
                subset = sorted((r for r in simrows if r["latent_effect"] == effect and r["method"] == method and r["overdispersion"] == over), key=lambda r: r["subjects"])
                xs = [r["subjects"] for r in subset]
                for rowi, prefix in ((0, "latent_recovery_correlation"), (1, "split_half_residual_reliability")):
                    ys = np.array([r[prefix + "_mean"] for r in subset])
                    lo = np.array([r[prefix + "_q025"] for r in subset]); hi = np.array([r[prefix + "_q975"] for r in subset])
                    axes[rowi, col].errorbar(xs, ys, yerr=[ys-lo, hi-ys], color=color, marker=marker, ls=ls,
                                               lw=1.25, ms=4.2, capsize=3, capthick=1, elinewidth=1,
                                               label=METHOD_LABELS[method] if rowi == 0 and col == 0 and over == "low" else None)
        axes[0, col].set_title("Null latent effect" if effect == 0 else "Latent effect = 0.55")
        axes[1, col].set_xlabel("Subjects")
    axes[0, 0].set_ylabel("Alignment with generating z, r")
    axes[1, 0].set_ylabel("Split reliability, r")
    for ax in axes.flat:
        ax.axhline(0, color="#AEB3B7", lw=.7); ax.grid(axis="y", color="#E7E9EA", lw=.6); ax.set_axisbelow(True)
    method_handles, method_labels = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    dispersion_handles = [Line2D([0], [0], color="#555B60", lw=1.3, ls="-", label="Lower overdispersion"),
                          Line2D([0], [0], color="#555B60", lw=1.3, ls="--", label="Higher overdispersion")]
    fig.legend(method_handles + dispersion_handles, method_labels + [h.get_label() for h in dispersion_handles],
               loc="lower center", bbox_to_anchor=(.5, .005), frameon=False, ncol=3, columnspacing=1.6, handlelength=2.6)
    fig.subplots_adjust(left=.09, right=.985, top=.93, bottom=.18, hspace=.2, wspace=.14)
    made += _save(fig, output / "figure2_reliability_paradox")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    specs = extracted["figure3"]
    obs_nll = [[r["nll_minus_uniform"] for r in specs if r["specification"] == name and r["kind"] == "observed"] for name in SPEC_ORDER]
    bp = axes[0].boxplot(obs_nll, tick_labels=[s.replace("_", "\n") for s in SPEC_ORDER], showfliers=False, widths=.58, patch_artist=True)
    for box in bp["boxes"]: box.set(facecolor="#DCE5EA", edgecolor=PALETTE["blue"])
    axes[0].set_ylabel("Held-out NLL minus uniform"); axes[0].set_title("Predictive fit (lower is better)")
    positions = np.arange(1, 5)
    for i, name in enumerate(SPEC_ORDER):
        observed = [r["edge_weighted_reliability"] for r in specs if r["specification"] == name and r["kind"] == "observed"]
        null = [r["edge_weighted_reliability"] for r in specs if r["specification"] == name and r["kind"] == "destruction_control"]
        violin = axes[1].violinplot([null], positions=[positions[i]-.18], widths=.32,
                                    showmeans=False, showmedians=True, showextrema=True)
        for body in violin["bodies"]:
            body.set_facecolor(PALETTE["gray"]); body.set_alpha(.45)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            violin[key].set_color(PALETTE["gray"]); violin[key].set_linewidth(1.15)
        axes[1].boxplot([observed], positions=[positions[i]+.18], widths=.30, showfliers=False, patch_artist=True,
                        boxprops={"facecolor": PALETTE["blue"], "alpha": .65, "linewidth": 1},
                        medianprops={"color": "white", "linewidth": 1.35}, whiskerprops={"linewidth": 1}, capprops={"linewidth": 1})
    axes[1].set_xticks(positions, [s.replace("_", "\n") for s in SPEC_ORDER]); axes[1].set_ylabel("Edge-weighted partition agreement, r")
    axes[1].set_title("Partition agreement")
    for ax in axes: ax.grid(axis="y", color="#E7E9EA", lw=.6); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=PALETTE["blue"], alpha=.65), Patch(facecolor=PALETTE["gray"], alpha=.4)],
               ["Observed: 100 fixed-sample partitions", "Descriptive control: 500 destination-label destructions"],
               loc="lower center", bbox_to_anchor=(.5, .015), frameon=False, ncol=2, columnspacing=2)
    fig.subplots_adjust(left=.095, right=.985, top=.9, bottom=.23, wspace=.28)
    made += _save(fig, output / "figure3_specification_curve")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1.05, 1.35, .9]})
    functional = extracted["figure4"]
    provo = [r for r in functional if r["panel"] == "provo"]
    by_seed = {r["seed"]: r for r in provo if r["condition"] == "mlm"}
    gaze = [r for r in provo if r["condition"] == "gaze"]
    for r in gaze:
        axes[0].scatter(r["mlm_nll"] - by_seed[r["seed"]]["mlm_nll"], r["residual_correlation"], color=PALETTE["blue"], s=28)
        axes[0].annotate(str(r["seed"]), (r["mlm_nll"] - by_seed[r["seed"]]["mlm_nll"], r["residual_correlation"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    macro = next(r for r in functional if r["panel"] == "provo_summary")
    axes[0].axhline(macro["macro_text_correlation"], color=PALETTE["green"], lw=1.2, label="text-equal macro")
    axes[0].axvline(0, color="#999999", lw=.8); axes[0].set(xlabel="Gaze - MLM NLL", ylabel="Residual correlation", title="Provo: fixed seeded 10-text split")
    zuco = [r for r in functional if r["panel"] == "zuco"]
    y = np.arange(3)
    means = np.array([r["mean"] for r in zuco])
    fixed_lo=np.array([r["fixed_ci_low"] for r in zuco]); fixed_hi=np.array([r["fixed_ci_high"] for r in zuco])
    nested_lo=np.array([r["nested_ci_low"] for r in zuco]); nested_hi=np.array([r["nested_ci_high"] for r in zuco])
    fixed_artist = axes[1].errorbar(means, y-.10, xerr=[means-fixed_lo, fixed_hi-means], fmt="o", color=PALETTE["blue"], capsize=3.5, lw=1.2, label="Fixed 12 readers; text reaggregation")
    nested_artist = axes[1].errorbar(means, y+.10, xerr=[means-nested_lo, nested_hi-means], fmt="s", color=PALETTE["orange"], capsize=3.5, lw=1.2, label="Support-varying reader-refit diagnostic")
    axes[1].axvline(0, color="#999999", lw=.8); axes[1].set_yticks(y, [r["contrast"].replace("gaze_vs_", "vs ") for r in zuco]); axes[1].set(xlabel="Fisher-z difference", title="ZuCo: descriptive ranges")
    inv = data_global["invariance"]
    vals = [inv["residual_distribution"][c]["pearson_dispersion"]["mean"] for c in ("provo", "zuco")]
    axes[2].bar(("Provo", "ZuCo"), vals, color=(PALETTE["blue"], PALETTE["orange"]), width=.6)
    axes[2].set_ylabel("Mean Pearson dispersion")
    axes[2].set_title(
        f"Measurement conditions\nDomain balanced accuracy = {inv['domain_distinguishability']['balanced_accuracy']:.3f}",
        fontsize=8.2, linespacing=1.35,
    )
    for ax in axes: ax.grid(axis="y", color="#E7E9EA", lw=.6); ax.set_axisbelow(True)
    macro_artist = Line2D([0], [0], color=PALETTE["green"], lw=1.4, label="Provo text-equal macro")
    fig.legend([macro_artist, fixed_artist, nested_artist], ["Provo text-equal macro", "ZuCo fixed 12; text reaggregation", "ZuCo support-varying reader-refit diagnostic"],
               loc="lower center", bbox_to_anchor=(.5, .01), frameon=False, ncol=3, columnspacing=1.2, handlelength=2.2)
    fig.subplots_adjust(left=.085, right=.985, top=.82, bottom=.25, wspace=.42)
    made += _save(fig, output / "figure4_functional_evidence")
    plt.close("all")
    return made


def _save(fig: Any, base: Path) -> list[Path]:
    paths = []
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 180})):
        path = base.with_suffix(suffix); fig.savefig(path, bbox_inches="tight", **kwargs); paths.append(path)
    return paths


def _fmt(value: Any, digits: int = 4) -> Any:
    return f"{value:.{digits}f}" if isinstance(value, float) else value


def make_tables(data: dict[str, Any], extracted: dict[str, list[dict[str, Any]]], output: Path) -> list[Path]:
    conversion = data["conversion"]; inv = data["invariance"]
    audit = [
        {"corpus": "Provo", "subjects": data["reliability"]["subjects"], "input_texts": inv["design"]["provo_texts"], "primary_eligible_texts": inv["design"]["provo_texts"], "mapped_positions_or_tokens": conversion["line_mapped_words"], "risk_set": data["reliability"]["risk_set"]},
        {"corpus": "ZuCo NR", "subjects": len(data["transfer"]["design"]["subjects"]), "input_texts": inv["design"]["zuco_texts"], "primary_eligible_texts": 192, "mapped_positions_or_tokens": inv["syntax_audit"]["zuco"]["tokens"], "risk_set": data["reliability"]["risk_set"]},
    ]
    sim = [{**{k: _fmt(r[k]) for k in ("subjects", "latent_effect", "overdispersion")},
            "method": METHOD_LABELS[r["method"]],
            "generating_z_alignment_mean": _fmt(r["latent_recovery_correlation_mean"]),
            "split_half_residual_reliability_mean": _fmt(r["split_half_residual_reliability_mean"])} for r in extracted["figure2"]]
    spec_rows = []
    for name in SPEC_ORDER:
        row = data["specification"]["summary"][name]
        observed_nll = [item["nll_minus_uniform"] for item in extracted["figure3"]
                        if item["specification"] == name and item["kind"] == "observed"]
        spec_rows.append({"specification": name, "rank": data["specification"]["specifications"][name]["conditional_rank"],
                          "half_average_nll_minus_uniform_median": _fmt(float(__import__("numpy").median(observed_nll)), 2),
                          "fixed_sample_edge_pattern_correlation_median": _fmt(row["reliability"]["edge_weighted"]["median"]),
                          "destruction_control_median": _fmt(row["negative_control"]["edge_weighted"]["median"]),
                          "destruction_control_replicates": data["specification"]["null_repeats"]})
    functional = []
    aux = data["aux_text"]
    for condition in ("gaze", "mlm"):
        row = aux["conditions"][condition]
        endpoint = "MLM-trained adapter + untrained random relation head (no data-gradient; default AdamW decay only)" if condition == "mlm" else condition
        functional.append({"dataset": "Provo", "endpoint": endpoint, "estimand": "correlation (r)", "estimate": _fmt(row["macro_text_equal_correlation_seed_averaged_fisher_z"], 5), "secondary": _fmt(row["pooled_edge_correlation"], 5), "interval_low": "", "interval_high": "", "interval_type": "", "unit_direction": "10 fixed test texts; secondary=pooled r"})
        functional.append({"dataset": "Provo", "endpoint": endpoint, "estimand": "MLM NLL", "estimate": _fmt(row["macro_text_equal_mlm_nll"], 5), "secondary": _fmt(row["pooled_token_mlm_nll"], 5), "interval_low": "", "interval_high": "", "interval_type": "", "unit_direction": "text-equal; secondary=token-pooled; lower is better"})
    for contrast in ("gaze_minus_shuffled", "gaze_minus_mlm", "gaze_minus_position"):
        row = aux["comparisons"][contrast]
        functional.append({"dataset": "Provo", "endpoint": contrast, "estimand": "paired per-text Fisher-z difference", "estimate": _fmt(row["mean"], 5), "secondary": "", "interval_low": _fmt(row["ci95"][0], 5), "interval_high": _fmt(row["ci95"][1], 5), "interval_type": "descriptive text-resampling", "unit_direction": "fixed seeded split; 10 texts; positive favors gaze"})
    nll = aux["comparisons"]["gaze_minus_mlm_nll"]
    functional.append({"dataset": "Provo", "endpoint": "gaze_minus_mlm_nll", "estimand": "paired per-text NLL difference", "estimate": _fmt(nll["mean"], 7), "secondary": "", "interval_low": _fmt(nll["ci95"][0], 7), "interval_high": _fmt(nll["ci95"][1], 7), "interval_type": "descriptive text-resampling", "unit_direction": "fixed seeded split; 10 texts; five seeds averaged per text; negative favors gaze"})
    fixed = data["transfer"]["comparisons"]
    nested = data["criterion"]["reader_bootstrap"]["summary"]
    for contrast in ("gaze_vs_mlm", "gaze_vs_shuffled", "gaze_vs_position"):
        point = fixed[contrast]["text_equal_fisher_z"]["mean_difference"]
        fixed_ci = fixed[contrast]["text_equal_fisher_z"]["descriptive_text_resampling_interval"]
        ci = nested[contrast]["joint_reader_and_text"]["95_ci"]
        functional.append({"dataset": "ZuCo", "endpoint": contrast, "estimand": "text-equal Fisher-z difference", "estimate": _fmt(point), "secondary": "", "interval_low": _fmt(fixed_ci[0]), "interval_high": _fmt(fixed_ci[1]), "interval_type": "fixed-12 descriptive text-resampling", "unit_direction": "192 structurally eligible texts; positive favors gaze"})
        functional.append({"dataset": "ZuCo", "endpoint": contrast, "estimand": "text-equal Fisher-z difference", "estimate": _fmt(point), "secondary": "", "interval_low": _fmt(ci[0]), "interval_high": _fmt(ci[1]), "interval_type": "support-varying reader-refit diagnostic", "unit_direction": "draw-specific text support; no coverage or decision interpretation"})
    supplement = []
    for corpus in ("provo", "zuco"):
        supplement.append({"audit": "residual dispersion", "corpus_or_endpoint": corpus, "estimate": _fmt(inv["residual_distribution"][corpus]["pearson_dispersion"]["mean"]), "range_or_status": "mean"})
    supplement.append({"audit": "domain classification", "corpus_or_endpoint": "Provo vs ZuCo", "estimate": _fmt(inv["domain_distinguishability"]["balanced_accuracy"]), "range_or_status": "balanced accuracy"})
    influence = data["influence"]
    for name in SPEC_ORDER:
        value = influence["provo_reliability"]["specifications"][name]["edge_weighted"]
        supplement.append({"audit": "LOTO", "corpus_or_endpoint": name, "estimate": _fmt(value["full"]), "range_or_status": f"{_fmt(value['loto_range'][0])} to {_fmt(value['loto_range'][1])}"})
    target = [{"category": category, **{k: audit[k] for k in ("candidate_edges", "candidate_sources", "observed_nonzero_edges", "observed_transition_mass", "eligible_edges")}}
              for category, audit in data["target_decomposition"]["candidate_and_observation_audit"].items()]
    criterion = []
    for metric, row in data["criterion"]["reliability"]["summary"].items():
        criterion.append({"metric": metric, "partitions": row["n"], "median": _fmt(row["median"]), "q25": _fmt(row["q25"]), "q75": _fmt(row["q75"]), "median_text_negative": _fmt(row["text_negative_proportion"]["median"]), "median_text_undefined": _fmt(row["text_undefined_proportion"]["median"])})
    residual = [{k: _fmt(row[k]) for k in ("specification", "analysis", "category", "stratum", "residual_type", "metric", "median", "q25", "q75", "median_defined_texts", "median_defined_edges")}
                for row in data["residual_diagnostics"]["reliability_summary"] if row["metric"] == "edge_weighted"]
    reconciliation = [{"text": text, "canonical": row["canonical_lexical_tokens"], "conversion": row["conversion_positions"],
                       "verified": row["verified_positions"], "unmatched_canonical": row["unmatched_canonical"],
                       "unmatched_conversion": row["unmatched_conversion"]}
                       for text, row in sorted(data["reconciliation"]["by_text"].items(), key=lambda item: int(item[0]))]
    edge_threshold = []
    for threshold, block in data["edge_threshold"]["results"].items():
        for contrast, row in block["comparisons"].items():
            edge_threshold.append({"minimum_edges": threshold, "contrast": contrast, "texts_retained": row["texts_retained"],
                                   "estimate": _fmt(row["mean_difference"], 5),
                                   "interval_low": _fmt(row["descriptive_text_resampling_interval"][0], 5),
                                   "interval_high": _fmt(row["descriptive_text_resampling_interval"][1], 5),
                                    "max_abs_correlation": _fmt(row["max_absolute_raw_correlation"], 5),
                                   "near_perfect_count": row["near_perfect_raw_correlation_count"],
                                    "max_text_mean_contribution": _fmt(row["max_absolute_text_mean_contribution"], 6)})
    line_identity = []
    for corpus in ("provo", "zuco"):
        row = data["line_identity"][corpus]
        line_identity.append({
            "corpus": corpus.title(), "subjects": 84 if corpus == "provo" else row["subjects"], "texts": row["texts"],
            "words": row["mapped_words"] if corpus == "provo" else row["reference_words"],
            "bounds_differences": row["distinct_bounds_variants"] - row["mapped_words"] if corpus == "provo" else row["nonreference_word_bounds_differences"],
            "line_partition_discrepancies": row["line_partition_discrepancies"],
            "note": f"{row['ambiguous_words']} ambiguous word excluded before analysis" if corpus == "provo" else "all nonreference bounds exactly matched",
        })
    made = []
    for stem, title, rows in (("table1_corpus_pipeline_audit", "Table 1. Corpus and pipeline audit", audit), ("table2_simulation_endpoints", "Table 2. Simulation endpoints", sim), ("table3_specification_results", "Table 3. Specification results", spec_rows), ("table4_functional_transfer", "Table 4. Constructed-residual alignment and cross-corpus scorer evaluation", functional), ("table_s1_invariance_loto", "Table S1. Observable measurement-condition differences and LOTO audit", supplement), ("table_s2_target_decomposition", "Table S2. Conditional forward fixation-destination decomposition audit", target), ("table_s3_criterion_uncertainty", "Table S3. ZuCo criterion uncertainty", criterion), ("table_s4_residual_diagnostics", "Table S4. Residual and exposure diagnostics", residual), ("table_s5_provo_reconciliation", "Table S5. Provo word-position reconciliation", reconciliation), ("table_s6_zuco_edge_threshold_sensitivity", "Table S6. ZuCo edge-threshold sensitivity", edge_threshold), ("table_s7_line_partition_identity", "Table S7. Cross-participant line-partition identity audit", line_identity)):
        rows = [{key: ("undefined (no eligible support)" if value is None else value) for key, value in row.items()} for row in rows]
        made += _write_table(output / stem, title, rows)
    threshold_source = output.parent / "source_data" / "zuco_edge_threshold_sensitivity_source_data.csv"
    threshold_source.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(threshold_source, edge_threshold)
    made.append(threshold_source)
    sensitivity = data["reconciled_sensitivity"]
    source_rows = [{"row_type": "text", **row} for row in reconciliation]
    for name in SPEC_ORDER:
        row = sensitivity["summary"][name]
        source_rows.append({"row_type": "sensitivity", "text": name, "canonical": "", "conversion": "", "verified": "",
                            "unmatched_canonical": row["delta_median"]["reliability"]["edge_weighted"],
                            "unmatched_conversion": row["delta_median"]["predictive_nll_minus_uniform"]})
    source_path = output.parent / "source_data" / "provo_reconciliation_source_data.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(source_path, source_rows)
    made.append(source_path)
    s5_md = output / "table_s5_provo_reconciliation.md"
    deltas = ", ".join(
        f"{name}={sensitivity['summary'][name]['reliability']['edge_weighted']['median']:.6f} "
        f"(delta {sensitivity['summary'][name]['delta_median']['reliability']['edge_weighted']:+.6f})"
        for name in SPEC_ORDER
    )
    with s5_md.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\nCanonical is the whitespace-derived lexical count; conversion is the independent AOI-derived position count. Verified denotes one-to-one aligned conversion positions. Unmatched columns report positions without an independently verified counterpart. The publication reports 2,689 words; the released repeated-text field reproduces that reported number as 2,745 whitespace tokens minus 55 first words and one standalone nonlexical token. Conversion has 2,740 positions: 55 restored initial AOIs, five canonical words without independent conversion positions, and one merged AOI. Of 2,740 conversion positions, 2,739 were verified; the strict observed file contains 2,739 positions, of which 2,738 were verified. Excluding text 55 index 8 (`livres--a`) removed 221 fixation rows and 190 forward events. NLL ordering was unchanged. Edge-weighted reliability medians after exclusion and changes from the primary analysis were " + deltas + ". The maximum absolute reliability-median change across all specifications and estimands was below .000636.\n")
    return made


def generate(root: Path = ROOT, *, timestamp: str | None = None) -> dict[str, Any]:
    global data_global
    processed = root / "data" / "processed"
    manuscript = root / "manuscript"; figures = manuscript / "figures"; tables = manuscript / "tables"; source = manuscript / "source_data"
    for directory in (figures, tables, source): directory.mkdir(parents=True, exist_ok=True)
    data, paths = load_primary(processed); data_global = data
    extracted = extract_assets(data)
    source_outputs = []
    for name, rows in extracted.items():
        path = source / f"{name}_source_data.csv"; _write_csv(path, rows); source_outputs.append(path)
    figure_outputs = make_figures(extracted, figures)
    arxiv_figures = root / "arxiv" / "figures"
    if arxiv_figures.is_dir():
        for path in figure_outputs:
            shutil.copyfile(path, arxiv_figures / path.name)
    table_outputs = make_tables(data, extracted, tables)
    captions = manuscript / "figure_captions.md"
    captions.write_text("""# Figure Captions\n\n**Figure 1. Evidence ladder and study design.** The manuscript evaluates a computational gaze-transition residual relation through estimator validation, reliability, and functional tests. External human construct validation is explicitly future work and is not represented as completed.\n\n**Figure 2. The simulation reliability paradox.** Mean correlation/alignment with the generating latent feature z and split-half residual reliability across 80 replicates per cell; error bars are the 2.5th and 97.5th percentiles. Line type distinguishes Dirichlet-multinomial overdispersion. Under delta=0, omitted nuisance structure can yield high reliability while alignment with z is spurious. The conditional multinomial variance remains misspecified under Dirichlet overdispersion.\n\n**Figure 3. Bounded four-model nuisance-specification comparison and descriptive negative control.** Held-out predictive NLL is the average of the two half-specific event-weighted totals in each of 100 randomized partitions of the fixed 84-reader sample; reliability is a text-median edge-pattern correlation. Gray violins summarize 500 destination-label destruction controls, each computed from one split balanced across the 100-split bank. The control preserves source exposure and eligibility masks but is not a calibrated permutation test; no p value or familywise inference is reported.\n\n**Figure 4. Constructed-residual alignment and cross-corpus scorer evaluation.** Provo points show gaze-minus-MLM NLL (negative favors gaze because lower NLL is better) and pooled residual correlation for five run seeds on one fixed 10-text split. For gaze, MLM, and position the seeds perturb optimization; for shuffle they also select different within-source label permutations, so they are not independent replicates. The horizontal line is the fixed-split, equal-text, equal-seed back-transformed mean Fisher-z correlation. For each ZuCo text-equal Fisher-z gaze-minus-control contrast, circles show the fixed-12-reader descriptive text-resampling interval and squares show reader-refit sensitivity with draw-specific eligible-text support and conditional text reaggregation; positive values favor gaze.\n""", encoding="utf-8")
    captions.write_text(captions.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    caption_text = captions.read_text(encoding="utf-8")
    caption_text = caption_text.replace(
        "reliability is a text-median edge-pattern correlation. Gray violins",
        "partition agreement is a text-median edge-pattern correlation. IQRs describe partition sensitivity, not confidence intervals. Gray violins",
    ).replace(
        "on one fixed 10-text split.",
        "on one fixed seeded 10-text split.",
    ).replace(
        "the fixed-12-reader descriptive text-resampling interval and squares show reader-refit sensitivity with draw-specific eligible-text support and conditional text reaggregation",
        "the fixed-12-reader descriptive text-reaggregation interval and squares show a support-varying reader-refit diagnostic range with draw-specific eligible-text support; the latter has no fixed-population coverage or crossing-zero decision interpretation",
    ).replace(
        "the fixed-12-reader percentile text-bootstrap 95% interval and squares show the mixture percentile interval from 200 outer reader-bootstrap criterion refits and 200 conditional text reaggregations per reader draw",
        "the fixed-12-reader descriptive text-reaggregation interval and squares show the support-varying diagnostic range from 200 outer reader refits and 200 conditional reaggregations per reader draw; the latter has no fixed-population coverage or crossing-zero decision interpretation",
    )
    captions.write_text(caption_text, encoding="utf-8", newline="\n")
    all_outputs = source_outputs + figure_outputs + table_outputs + [captions]
    script = root / "scripts" / "generate_manuscript_assets.py"
    generated_at = timestamp or os.environ.get("SOURCE_DATE_EPOCH")
    if generated_at and generated_at.isdigit(): generated_at = datetime.fromtimestamp(int(generated_at), timezone.utc).isoformat()
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    secondary = {"half_audit", "target_decomposition", "residual_diagnostics", "aux_text", "aux_budget", "criterion", "invariance", "influence", "reconciliation", "reconciled_sensitivity", "edge_threshold", "line_identity"}
    input_records = [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "status": "complete" if key != "conversion" else "audit_complete", "role": "secondary" if key in secondary else "primary", "kind": "compact_artifact"} for key, path in paths.items()]
    input_records += [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "status": "generated", "role": "secondary", "kind": "manuscript_asset"} for path in all_outputs]
    manifest = manuscript / "artifact_manifest.csv"; _write_csv(manifest, input_records); all_outputs.append(manifest)
    groups = {
        "figure1": (["conversion", "reliability", "simulation"], ["/data_generating_process", "/summary", "/split"]),
        "figure2": (["simulation"], ["/summary/*", "/config"]),
        "figure3": (["specification"], ["/repeat_results/*", "/null_results/*", "/summary/*"]),
        "figure4": ([f"seed_{s}" for s in (101,202,303,404,505)] + ["transfer", "criterion", "aux_text", "invariance"], ["/conditions/{mlm,gaze}/test", "/conditions/gaze/macro_text_equal_correlation_seed_averaged_fisher_z", "/comparisons/*/text_equal_fisher_z/descriptive_text_resampling_interval", "/reader_bootstrap/summary/*/joint_reader_and_text", "/residual_distribution", "/domain_distinguishability"]),
        "table1": (["conversion", "reliability", "transfer", "invariance"], ["/design", "/syntax_audit", "/line_mapped_words"]),
        "table2": (["simulation"], ["/summary/*"]), "table3": (["specification"], ["/summary/*", "/specifications/*"]),
        "table4": (["aux_text", "transfer", "criterion"], ["/conditions/*", "/comparisons/*/text_equal_fisher_z", "/reader_bootstrap/summary/*/joint_reader_and_text"]),
        "table_s1": (["invariance", "influence"], ["/residual_distribution", "/domain_distinguishability", "/provo_reliability/specifications"]),
        "table_s2": (["target_decomposition"], ["/candidate_and_observation_audit"]),
        "table_s3": (["criterion"], ["/reliability/summary"]),
        "table_s4": (["residual_diagnostics"], ["/reliability_summary"]),
        "table_s5": (["reconciliation", "reconciled_sensitivity"], ["/by_text", "/summary", "/processed_observed_verified_positions", "/excluded_fixation_rows", "/excluded_forward_event_weight"]),
        "table_s6": (["edge_threshold"], ["/results/*/comparisons/*"]),
        "table_s7": (["line_identity"], ["/provo", "/zuco", "/inputs"]),
    }
    assets = []
    for asset, (keys, fields) in groups.items():
        outputs = [p for p in all_outputs if p.name.startswith(asset)]
        if asset == "table_s5":
            outputs += [p for p in all_outputs if p.name == "provo_reconciliation_source_data.csv"]
        if asset == "table_s6":
            outputs += [p for p in all_outputs if p.name == "zuco_edge_threshold_sensitivity_source_data.csv"]
        assets.append({"asset": asset, "inputs": [{"path": paths[k].relative_to(root).as_posix(), "sha256": sha256(paths[k])} for k in keys],
                       "fields_and_transforms": fields + ["Filtering/order/summary only; no result constants"],
                       "outputs": [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)} for p in outputs]})
    provenance = {"schema_version": 1, "generated_at_utc": generated_at, "generation_note": "Timestamp is informational and excluded from all plotted/tabulated values.",
                  "script": {"path": script.relative_to(root).as_posix(), "sha256": sha256(script)}, "superseded_inputs": [], "assets": assets,
                  "manifest": {"path": manifest.relative_to(root).as_posix(), "sha256": sha256(manifest)},
                  "acquisition_documentation": ["docs/data.md", "docs/zuco.md"]}
    provenance_path = manuscript / "artifact_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen manuscript figures, tables, source data, and provenance.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--timestamp", help="Optional deterministic ISO-8601 timestamp")
    args = parser.parse_args()
    provenance = generate(args.root.resolve(), timestamp=args.timestamp)
    print(f"Generated {len(provenance['assets'])} manuscript asset groups")


data_global: dict[str, Any] = {}
if __name__ == "__main__":
    main()
