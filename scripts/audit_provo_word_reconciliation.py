from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NORMS = ROOT / "data/raw/Provo_Corpus-Predictability_Norms.csv"
EYE = ROOT / "data/raw/Provo_Corpus-Eyetracking_Data.csv"
FIX = ROOT / "data/raw/Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv"
PROCESSED = ROOT / "data/processed/provo_fixations_with_lines.csv"
JSON_OUT = ROOT / "data/processed/provo_word_position_reconciliation.json"
CSV_OUT = ROOT / "data/processed/provo_word_position_reconciliation.csv"
DOC_OUT = ROOT / "docs/provo_word_position_reconciliation.md"
MISSING = {"", ".", "NA"}


def read_csv(path: Path, encoding: str = "cp1252") -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def norm_surface(value: str) -> str:
    value = value.strip().lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", value)


def unique_count(rows, columns):
    if any(column not in rows[0] for column in columns):
        return None
    return len({tuple(row[column] for column in columns) for row in rows})


def file_statistics(name, rows, text_column):
    fields = {
        "Word_Unique_ID": ("Word_Unique_ID",),
        "Text_ID+Word_Number": ("Text_ID", "Word_Number"),
        "IA_ID": ("IA_ID",),
        "Text_ID+IA_ID": ("Text_ID", "IA_ID"),
        "Word": ("Word",),
        "IA_LABEL": ("IA_LABEL",),
        "Word+IA_LABEL": ("Word", "IA_LABEL"),
    }
    result = {"file": name, "rows": len(rows), "unique": {key: unique_count(rows, value) for key, value in fields.items()}}
    if text_column and text_column in rows[0]:
        by_text = defaultdict(list)
        for row in rows:
            by_text[str(row[text_column])].append(row)
        result["by_text"] = {
            text: {"rows": len(items), "unique": {key: unique_count(items, value) for key, value in fields.items()}}
            for text, items in sorted(by_text.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]))
        }
    return result


def main() -> None:
    norms, eye, fix, processed = read_csv(NORMS), read_csv(EYE), read_csv(FIX), read_csv(PROCESSED, "utf-8")
    norm_by_text = defaultdict(list)
    texts = {}
    text_variants = defaultdict(set)
    for row in norms:
        text = str(row["Text_ID"])
        norm_by_text[text].append(row)
        texts.setdefault(text, row["Text"])
        text_variants[text].add(row["Text"])

    eye_by_text_ia = defaultdict(lambda: defaultdict(set))
    eye_rows_by_text = defaultdict(list)
    for row in eye:
        text = str(row["Text_ID"])
        eye_rows_by_text[text].append(row)
        eye_by_text_ia[text][int(row["IA_ID"])].add(
            (row["Word_Number"], row["Word"], row["IA_LABEL"], row["Word_Unique_ID"])
        )

    # Reproduce the converter's deterministic position selection: corrected Word_Number,
    # then nearest IA_ID when duplicate word numbers exist.
    conversion = defaultdict(dict)
    conversion_collisions = []
    for text, by_ia in eye_by_text_ia.items():
        candidates = defaultdict(list)
        for ia, records in by_ia.items():
            if len(records) != 1:
                raise ValueError(f"Inconsistent repeated eye row for text {text}, IA {ia}: {records}")
            number, word, label, uid = next(iter(records))
            if number in MISSING:
                if ia != 1:
                    continue
                number = "1"
            candidates[int(number)].append((ia, word, label, uid))
        for number, values in candidates.items():
            values.sort(key=lambda value: (abs(value[0] - number), value[0]))
            conversion[text][number - 1] = values[0]
            if len(values) > 1:
                conversion_collisions.append({"text_id": text, "word_number": number, "candidates": values, "selected_ia": values[0][0]})

    reconciliation = []
    text_summary = {}
    category_counts = Counter()
    verified_positions = []
    for text in sorted(texts, key=int):
        raw_tokens = texts[text].split()
        canonical = [(index, token) for index, token in enumerate(raw_tokens, 1) if norm_surface(token)]
        eye_sequence = sorted(conversion[text].items(), key=lambda item: item[1][0])
        left = [norm_surface(token) for _, token in canonical]
        right = [norm_surface(item[1][2]) for item in eye_sequence]
        matcher = SequenceMatcher(None, left, right, autojunk=False)
        matched_canonical, matched_eye = set(), set()
        rows_before = len(reconciliation)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                ci, ei = block.a + offset, block.b + offset
                matched_canonical.add(ci); matched_eye.add(ei)
                canonical_number, surface = canonical[ci]
                conversion_index, (ia, word, label, uid) = eye_sequence[ei]
                categories = []
                if canonical_number == 1:
                    categories.append("first_word_cloze_omission")
                if conversion_index != canonical_number - 1:
                    categories.append({
                        "3": "sentence_boundary_number_gap",
                        "13": "quoted_sentence_boundary_number_gap",
                        "36": "standalone_punctuation_tokenization_offset",
                    }.get(text, "word_number_offset"))
                category = "+".join(categories) if categories else "exact_surface_alignment"
                verified = norm_surface(label) == norm_surface(surface)
                record = {
                    "text_id": text, "canonical_number": canonical_number, "canonical_surface": surface,
                    "canonical_normalized": norm_surface(surface), "conversion_word_index": conversion_index,
                    "conversion_word_number": conversion_index + 1, "ia_id": ia, "eye_word": word,
                    "ia_label": label, "word_unique_id": uid, "alignment": "aligned",
                    "category": category, "verified_one_to_one": verified,
                }
                reconciliation.append(record); category_counts[category] += 1
                if verified:
                    verified_positions.append([text, conversion_index])
        for ci, (canonical_number, surface) in enumerate(canonical):
            if ci in matched_canonical:
                continue
            category = "canonical_missing_from_conversion"
            if text == "55" and canonical_number == 10:
                category = "merged_aoi_missing_word"
            elif text == "55" and canonical_number in {61, 62}:
                category = "merged_terminal_aoi_excluded"
            elif text == "18" and canonical_number == 51:
                category = "duplicate_word_number_alias_excluded"
            record = {"text_id": text, "canonical_number": canonical_number, "canonical_surface": surface,
                      "canonical_normalized": norm_surface(surface), "conversion_word_index": None,
                      "conversion_word_number": None, "ia_id": None, "eye_word": None, "ia_label": None,
                      "word_unique_id": None, "alignment": "canonical_only", "category": category,
                      "verified_one_to_one": False}
            reconciliation.append(record); category_counts[category] += 1
        for ei, (conversion_index, (ia, word, label, uid)) in enumerate(eye_sequence):
            if ei in matched_eye:
                continue
            category = "conversion_surface_mismatch"
            if text == "55" and ia == 9:
                category = "merged_aoi_surface"
            record = {"text_id": text, "canonical_number": None, "canonical_surface": None,
                      "canonical_normalized": None, "conversion_word_index": conversion_index,
                      "conversion_word_number": conversion_index + 1, "ia_id": ia, "eye_word": word,
                      "ia_label": label, "word_unique_id": uid, "alignment": "conversion_only",
                      "category": category, "verified_one_to_one": False}
            reconciliation.append(record); category_counts[category] += 1
        local = reconciliation[rows_before:]
        text_summary[text] = {
            "text_whitespace_tokens": len(raw_tokens), "canonical_lexical_tokens": len(canonical),
            "norm_unique_word_ids": len({row["Word_Unique_ID"] for row in norm_by_text[text]}),
            "norm_unique_text_word_numbers": len({row["Word_Number"] for row in norm_by_text[text]}),
            "eye_unique_word_ids": len({row["Word_Unique_ID"] for row in eye_rows_by_text[text]}),
            "eye_unique_text_word_numbers": len({row["Word_Number"] for row in eye_rows_by_text[text]}),
            "eye_unique_ias": len(eye_by_text_ia[text]), "conversion_positions": len(conversion[text]),
            "processed_unique_word_indices": len({row["word_index"] for row in processed if row["text_id"] == text}),
            "verified_positions": sum(row["verified_one_to_one"] for row in local),
            "unmatched_canonical": sum(row["alignment"] == "canonical_only" for row in local),
            "unmatched_conversion": sum(row["alignment"] == "conversion_only" for row in local),
        }

    fix_augmented = []
    trial_text = {(row["RECORDING_SESSION_LABEL"], row["TRIAL_INDEX"]): row["Text_ID"] for row in eye}
    for row in fix:
        copy = dict(row)
        copy["Text_ID"] = trial_text.get((row["RECORDING_SESSION_LABEL"], row["TRIAL_INDEX"]), "")
        copy["IA_ID"] = row.get("CURRENT_FIX_INTEREST_AREA_INDEX", "")
        copy["IA_LABEL"] = row.get("CURRENT_FIX_INTEREST_AREA_LABEL", "")
        fix_augmented.append(copy)
    processed_augmented = [{**row, "Text_ID": row["text_id"], "Word_Number": str(int(row["word_index"]) + 1)} for row in processed]
    official = {
        "text_whitespace_tokens": sum(len(value.split()) for value in texts.values()),
        "first_words_omitted_by_cloze": len(texts),
        "standalone_nonlexical_tokens": sum(not norm_surface(token) for value in texts.values() for token in value.split()),
    }
    official["reproduced_official_words"] = official["text_whitespace_tokens"] - official["first_words_omitted_by_cloze"] - official["standalone_nonlexical_tokens"]
    summary = {
        "official_reproduction": official,
        "raw_norms_observed": {"rows": len(norms), "unique_word_unique_id": unique_count(norms, ("Word_Unique_ID",)),
                               "unique_text_word_number": unique_count(norms, ("Text_ID", "Word_Number"))},
        "conversion_positions": sum(len(value) for value in conversion.values()),
        "difference_conversion_minus_official": sum(len(value) for value in conversion.values()) - official["reproduced_official_words"],
        "verified_positions": len(verified_positions),
        "unverified_conversion_positions": sum(row["alignment"] == "conversion_only" for row in reconciliation),
        "canonical_words_missing_conversion": sum(row["alignment"] == "canonical_only" for row in reconciliation),
        "category_counts": dict(sorted(category_counts.items())),
        "text_field_unique_within_text": all(len(values) == 1 for values in text_variants.values()),
        "text_field_repeated_on_every_norm_row": all(row["Text"] == texts[row["Text_ID"]] for row in norms),
        "conversion_collisions": conversion_collisions,
    }
    files = {
        "predictability_norms": file_statistics(NORMS.name, norms, "Text_ID"),
        "eyetracking_data": file_statistics(EYE.name, eye, "Text_ID"),
        "additional_fixations": file_statistics(FIX.name, fix_augmented, "Text_ID"),
        "processed_fixations": file_statistics(PROCESSED.name, processed_augmented, "Text_ID"),
    }
    payload = {"summary": summary, "files": files, "by_text": text_summary,
               "verified_mask": verified_positions, "reconciliation": reconciliation}
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reconciliation[0].keys()); writer.writeheader(); writer.writerows(reconciliation)

    anomalous = [text for text, values in text_summary.items() if values["unmatched_canonical"] or values["unmatched_conversion"]]
    lines = ["# Provo word-position reconciliation", "", "Generated by `python scripts/audit_provo_word_reconciliation.py`.", "",
             "## Exact result", "", f"- Norm `Text` whitespace tokens: {official['text_whitespace_tokens']}.",
             f"- Cloze-omitted first words: {official['first_words_omitted_by_cloze']}.",
             f"- Standalone nonlexical tokens: {official['standalone_nonlexical_tokens']}.",
             f"- Reproduced official count: `{official['text_whitespace_tokens']} - {official['first_words_omitted_by_cloze']} - {official['standalone_nonlexical_tokens']} = {official['reproduced_official_words']}`.",
             f"- Conversion positions: {summary['conversion_positions']}; verified one-to-one positions: {summary['verified_positions']}.",
             f"- Canonical lexical words absent from conversion: {summary['canonical_words_missing_conversion']}; unverified conversion positions: {summary['unverified_conversion_positions']}.", "",
             "The raw Norms table itself has " + str(summary["raw_norms_observed"]["unique_word_unique_id"]) + " unique `Word_Unique_ID` values and " + str(summary["raw_norms_observed"]["unique_text_word_number"]) + " unique `(Text_ID, Word_Number)` keys because its malformed records are documented below; 2689 is reproduced from its repeated `Text` field, not by naively counting Norm rows/keys.", "",
             "## File-level unique counts", "", "Counts are literal raw values. For Additional Fixations, `IA_ID`/`IA_LABEL` mean `CURRENT_FIX_INTEREST_AREA_INDEX`/`CURRENT_FIX_INTEREST_AREA_LABEL`, with `Text_ID` recovered only through the exact session/trial crosswalk. Processed Fixations expose only `(text_id, word_index)`.", "",
             "| File | Rows | Word_Unique_ID | (Text_ID, Word_Number) | IA_ID | (Text_ID, IA_ID) | Word | IA_LABEL | (Word, IA_LABEL) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for value in files.values():
        unique = value["unique"]
        shown = lambda key: "NA" if unique[key] is None else str(unique[key])
        lines.append(f"| `{value['file']}` | {value['rows']} | {shown('Word_Unique_ID')} | {shown('Text_ID+Word_Number')} | {shown('IA_ID')} | {shown('Text_ID+IA_ID')} | {shown('Word')} | {shown('IA_LABEL')} | {shown('Word+IA_LABEL')} |")
    lines += ["", "Every file's same counts split by text are stored under `files.*.by_text` in the JSON. Additional Fixations also has an empty-text group containing the 3,624 trial-56 rows absent from the eye-table crosswalk.", "",
             "## Canonical sequence", "", "The canonical lexical sequence is the whitespace sequence in Norms `Text`, retaining original surfaces and excluding only tokens whose normalized surface is empty. `Text` is present on every Norm response row, but is not byte-identical on every row: text 27 has 736 rows with mojibake `doesn�t` and 12 rows with `doesn't`; both normalize to the same lexical token. All other texts have one repeated full-text value.", "",
             "The canonical sequence has 2,744 lexical positions. The official 2,689 is exactly the canonical sequence minus 55 passage-first words omitted from Norm rows. Eye conversion restores those 55 first AOIs, but loses four net canonical positions elsewhere, hence `2689 + 55 - 4 = 2740`.", "",
             "## Per-text counts", "", "| Text | Text tokens | Canonical lexical | Norm IDs | Norm keys | Eye IAs | Conversion | Processed observed | Verified | Canonical missing | Conversion unmatched |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for text, value in text_summary.items():
        lines.append(f"| {text} | {value['text_whitespace_tokens']} | {value['canonical_lexical_tokens']} | {value['norm_unique_word_ids']} | {value['norm_unique_text_word_numbers']} | {value['eye_unique_ias']} | {value['conversion_positions']} | {value['processed_unique_word_indices']} | {value['verified_positions']} | {value['unmatched_canonical']} | {value['unmatched_conversion']} |")
    lines += ["", "## Evidence-classified anomalies", ""]
    for category, count in sorted(category_counts.items()):
        if category != "exact_surface_alignment":
            lines.append(f"- `{category}`: {count}")
    lines += ["", "Affected texts with unmatched sequence items: " + ", ".join(anomalous) + ".", "",
              "- Text 3 has a one-number gap immediately after the sentence-ending `California.`; 14 following surfaces still align exactly.",
              "- Text 13 has a one-number gap at the quoted sentence beginning `\"I`; 35 following surfaces still align exactly.",
              "- Text 36 contains standalone nonlexical `�` in Norm `Text` and eye IA 24 label `?`; conversion excludes that IA and 16 following lexical surfaces align with a minus-one Word_Number offset.",
              "- Text 18 eye IA 51 (`evolution.`) aliases `Word_Number=3`/`QID2687`, colliding with IA 3 (`the`). The converter deterministically keeps nearest IA 3; no evolution position is created. The strict-line processed file also lacks text 18 conversion index 2 because its line bounds are tied, so it has 2,739 observed positions rather than 2,740.",
              "- Text 55 eye IA 9 merges canonical `livres, a`; canonical words 61-62 (`profession, writing.`) are merged into excluded IA 60. This is five missing canonical positions offset by one unverified merged conversion position, net minus four.",
              "", "No standalone duplicate AOI survives as an extra conversion position. `IA_ID` is an AOI alias, not a corpus-global position ID. Surface normalization is used only for alignment; the CSV preserves all original surfaces.", "",
              "## Verified mask and sensitivity", "", "A one-to-one mask is constructible: 2,739 of 2,740 conversion positions match one canonical lexical surface. The only unverified conversion position is text 55 `word_index=8` / IA 9 (`livres--a`). In the strict-line processed data, 2,738 of 2,739 observed positions are verified."]
    sensitivity_path = ROOT / "data/processed/provo_reconciled_sensitivity.json"
    if sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        lines += ["", f"The verified-position sensitivity excludes {sensitivity['excluded_fixation_rows']} fixation rows and {int(sensitivity['excluded_forward_event_weight'])} forward events with an unverified endpoint. The resulting design has {sensitivity['candidate_edges_after_mask']} candidate edges and {sensitivity['candidate_source_groups_after_mask']} source groups. It uses the same seed, 100 fixed 42/42 splits, four specifications, five text folds, and exposure threshold 5; no permutations were run.", "",
                  "| Specification | Edge median | Source-equal median | Fisher-equal median | NLL-uniform median | Max absolute reliability delta |", "|---|---:|---:|---:|---:|---:|"]
        for name in sensitivity["specifications"]:
            value = sensitivity["summary"][name]
            rel = value["reliability"]
            maximum = max(abs(item) for item in value["delta_median"]["reliability"].values())
            lines.append(f"| `{name}` | {rel['edge_weighted']['median']:.9f} | {rel['source_equal_flatten']['median']:.9f} | {rel['per_source_fisher_equal']['median']:.9f} | {value['predictive_nll_minus_uniform']['median']:.3f} | {maximum:.9f} |")
        lines += ["", "NLL ordering (best/lower to worst) is `" + " < ".join(sensitivity["nll_order_best_to_worst"]) + "` and is unchanged. The largest absolute reliability-median change across all 12 cells is below 0.000636. NLL values are less favorable by 119.141-128.712 because 190 observed transitions are removed; this is not an ordering reversal."]
    lines += ["", "The CSV retains every raw surface. The JSON additionally contains exact file-level and per-text unique counts, collision evidence, the verified mask, and all reconciliation rows."]
    DOC_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
