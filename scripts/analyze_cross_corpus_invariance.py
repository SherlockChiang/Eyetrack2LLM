from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetrack2llm import Fixations, extract_events, read_fixation_csv
from eyetrack2llm.baseline import build_pair_design, count_vector, enrich_spacy_syntax, enrich_word_frequencies, read_provo_word_metadata
from eyetrack2llm.invariance import audit_corpora
from run_zuco_transfer import SUBJECTS, TEXTS, build_metadata


def event_counts(design, events):
    selected = events.event_type == "forward"; edges = defaultdict(float)
    for text, src, dst, weight in zip(events.text_id[selected], events.src_word[selected], events.dst_word[selected], events.weight[selected], strict=True):
        edges[(str(text), int(src), int(dst))] += float(weight)
    return count_vector(design, edges)


def main():
    parser = argparse.ArgumentParser(description="Audit observable differences in Provo-vs-ZuCo measurement conditions")
    parser.add_argument("--output", default="data/processed/cross_corpus_measurement_invariance.json")
    parser.add_argument("--csv-output", default="data/processed/cross_corpus_measurement_invariance.csv")
    parser.add_argument("--bootstrap", type=int, default=1000); parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260711); args = parser.parse_args()
    processed = Path("data/processed")
    provo_fix = read_fixation_csv(processed / "provo_fixations_with_lines.csv")
    observed = defaultdict(set)
    for text, word in zip(provo_fix.text_id, provo_fix.word_index, strict=True): observed[str(text)].add(int(word))
    import spacy
    nlp = spacy.load("en_core_web_sm")
    pm, pa = enrich_spacy_syntax(enrich_word_frequencies(read_provo_word_metadata("data/raw/Provo_Corpus-Eyetracking_Data.csv", dict(observed))), nlp)
    pd = build_pair_design(pm, "common_core", "common_forward_same_sentence_same_line")
    pc = event_counts(pd, extract_events(provo_fix, include_self=False, require_consecutive_order=True))
    metadata_rows, items = {}, []
    for subject in SUBJECTS:
        stem = f"zuco_{subject.lower()}_nr"; rows = json.loads((processed / f"{stem}_words.json").read_text(encoding="utf-8")); by_id = {r["text_id"]: r for r in rows}
        if not metadata_rows: metadata_rows = {text: by_id[text] for text in TEXTS}
        items.append(read_fixation_csv(processed / f"{stem}_fixations.csv"))
    zf = Fixations(*[np.concatenate([getattr(item, field) for item in items]) for field in ("subject_id", "text_id", "word_index", "fixation_order", "duration_ms", "line_id")])
    zm, za = build_metadata(metadata_rows, nlp); zd = build_pair_design(zm, "common_core", "common_forward_same_sentence_same_line")
    zc = event_counts(zd, extract_events(zf, include_self=False, require_consecutive_order=True))
    result = audit_corpora(pd, pc, zd, zc, repeats=args.bootstrap, permutations=args.permutations, seed=args.seed)
    result["syntax_audit"] = {"provo": {k: v for k, v in pa.items() if k != "sentence_reports"}, "zuco": {k: v for k, v in za.items() if k != "sentence_reports"}}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for item in result["feature_distribution"]: rows.append({"section": "feature", "metric": item["feature"], "provo": item["provo_text_mean"], "zuco": item["zuco_text_mean"], "difference": item["text_equal_smd"], "ci_low": item["bootstrap_95_ci"][0], "ci_high": item["bootstrap_95_ci"][1]})
    for item in result["nuisance_fit"]["coefficients"]: rows.append({"section": "coefficient", "metric": item["feature"], "provo": item["provo"], "zuco": item["zuco"], "difference": item["provo_minus_zuco"], "ci_low": item["bootstrap_95_ci"][0], "ci_high": item["bootstrap_95_ci"][1]})
    for corpus, values in result["transport_calibration"].items():
        for model, value in values.items(): rows.append({"section": "transport_nll", "metric": f"{corpus}:{model}", "provo": value["text_equal_mean_nll"], "zuco": "", "difference": "", "ci_low": "", "ci_high": ""})
    with Path(args.csv_output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"runtime_seconds": result["runtime_seconds"], "domain": result["domain_distinguishability"], "transport": {c: {m: v["text_equal_mean_nll"] for m, v in x.items()} for c, x in result["transport_calibration"].items()}}, indent=2))


if __name__ == "__main__": main()
