from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class ProvoConversionReport:
    main_sessions: int
    main_trials: int
    word_mappings: int
    fixation_rows: int
    written_rows: int
    excluded_missing_trial: int
    excluded_page_mismatch: int
    excluded_outside_aoi: int
    excluded_unmapped_aoi: int
    excluded_invalid_value: int
    excluded_missing_line: int
    line_mapped_words: int
    line_missing_words: int
    exclusion_by_text_aoi: dict[str, int]


def cluster_vertical_intervals(intervals: list[tuple[float, float]]) -> list[int]:
    """Cluster overlapping vertical AOI intervals, tolerating small coordinate shifts."""
    order = sorted(range(len(intervals)), key=lambda i: (intervals[i][0], intervals[i][1]))
    labels = [-1] * len(intervals)
    clusters: list[tuple[float, float]] = []
    for index in order:
        top, bottom = intervals[index]
        matches = [i for i, (low, high) in enumerate(clusters) if top < high and low < bottom]
        if len(matches) > 1:
            raise ValueError(f"Vertical interval bridges multiple lines: {(top, bottom)}")
        if matches:
            label = matches[0]
            low, high = clusters[label]
            clusters[label] = (min(low, top), max(high, bottom))
        else:
            label = len(clusters)
            clusters.append((top, bottom))
        labels[index] = label
    return labels


def build_provo_line_map(main_path, *, encoding="cp1252"):
    """Return conservative word lines and a row-level audit from repeated Provo AOIs."""
    bounds: dict[tuple[str, int], Counter[tuple[float, float]]] = defaultdict(Counter)
    with open(main_path, encoding=encoding, newline="") as handle:
        for row in csv.DictReader(handle):
            number = row.get("Word_Number")
            if number in {None, "", ".", "NA"}:
                if row.get("IA_ID") != "1":
                    continue
                number = "1"
            try:
                interval = (float(row["IA_TOP"]), float(row["IA_BOTTOM"]))
                word = int(float(number))
            except (KeyError, TypeError, ValueError):
                continue
            if interval[0] > interval[1]:
                continue
            bounds[(str(row["Text_ID"]), word)][interval] += 1
    consensus = {}
    audit = []
    for key, counts in sorted(bounds.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        ranked = counts.most_common()
        accepted = len(ranked) == 1 or ranked[0][1] > sum(counts.values()) / 2
        interval = ranked[0][0] if accepted else None
        consensus[key] = interval
        audit.append({"text_id": key[0], "word_number": key[1], "ia_top": interval[0] if interval else None,
                      "ia_bottom": interval[1] if interval else None, "status": "mapped" if accepted else "ambiguous",
                      "observations": sum(counts.values()),
                      "bounds_counts": [{"top": pair[0], "bottom": pair[1], "count": count} for pair, count in ranked]})
    line_map = {}
    for text in sorted({key[0] for key in consensus}, key=int):
        keys = [key for key, interval in consensus.items() if key[0] == text and interval is not None]
        labels = cluster_vertical_intervals([consensus[key] for key in keys])
        for key, label in zip(keys, labels, strict=True):
            line_map[key] = label
    for row in audit:
        row["line_id"] = line_map.get((row["text_id"], row["word_number"]))
    for text in sorted({key[0] for key in line_map}, key=int):
        line_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for key, line in line_map.items():
            if key[0] == text:
                line_intervals[line].append(consensus[key])
        envelopes = {
            line: (min(interval[0] for interval in intervals), max(interval[1] for interval in intervals))
            for line, intervals in line_intervals.items()
        }
        for row in (item for item in audit if item["text_id"] == text and item["status"] == "mapped"):
            expected = row["line_id"]
            for variant in row["bounds_counts"]:
                top, bottom = variant["top"], variant["bottom"]
                matches = [line for line, (low, high) in envelopes.items() if top < high and low < bottom]
                if matches != [expected]:
                    raise ValueError(
                        f"Provo line-partition variant mismatch at text {text}, word {row['word_number']}: "
                        f"bounds {(top, bottom)} match consensus lines {matches}, expected {expected}"
                    )
            row["all_bounds_variants_match_consensus_line"] = True
    return line_map, audit


def convert_provo_fixations(
    main_path: str | PathLike[str],
    fixation_path: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    report_path: str | PathLike[str] | None = None,
    line_map_path: str | PathLike[str] | None = None,
    line_audit_path: str | PathLike[str] | None = None,
    encoding: str = "cp1252",
) -> ProvoConversionReport:
    """Convert official Provo files without guessing ambiguous AOI mappings."""
    session_participant: dict[str, str] = {}
    trial_text: dict[tuple[str, str], str] = {}
    word_map_values: dict[tuple[str, int], set[int]] = defaultdict(set)
    with open(main_path, encoding=encoding, newline="") as handle:
        for row in csv.DictReader(handle):
            session = row["RECORDING_SESSION_LABEL"]
            participant = row["Participant_ID"]
            previous_participant = session_participant.setdefault(session, participant)
            if previous_participant != participant:
                raise ValueError(f"Session {session!r} maps to multiple participants")

            trial = (session, row["TRIAL_INDEX"])
            text = row["Text_ID"]
            previous_text = trial_text.setdefault(trial, text)
            if previous_text != text:
                raise ValueError(f"Trial {trial!r} maps to multiple texts")

            if row["Word_Number"] not in {"", ".", "NA"}:
                word_map_values[(text, int(row["IA_ID"]))].add(int(row["Word_Number"]))

    ambiguous = {key: values for key, values in word_map_values.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"Ambiguous Provo AOI mappings: {ambiguous}")
    word_map = {key: next(iter(values)) for key, values in word_map_values.items()}

    # Cloze collection omitted every passage's first word, but the fixation AOI is stable.
    texts = set(trial_text.values())
    for text in texts:
        if (text, 1) in word_map and word_map[(text, 1)] != 1:
            raise ValueError(f"Unexpected first-word mapping for text {text}")
        word_map[(text, 1)] = 1
    line_map, line_audit = build_provo_line_map(main_path, encoding=encoding)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    with (
        open(fixation_path, encoding=encoding, newline="") as source,
        output.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            destination,
            fieldnames=(
                "subject_id",
                "text_id",
                "word_index",
                "fixation_order",
                "duration_ms",
                "line_id",
            ),
        )
        writer.writeheader()
        for row in reader:
            counts["fixation_rows"] += 1
            trial = (row["RECORDING_SESSION_LABEL"], row["TRIAL_INDEX"])
            text = trial_text.get(trial)
            if text is None:
                counts["excluded_missing_trial"] += 1
                continue
            if row["page"] != text:
                counts["excluded_page_mismatch"] += 1
                continue
            aoi = row["CURRENT_FIX_INTEREST_AREA_INDEX"]
            if aoi in {"", "."}:
                counts["excluded_outside_aoi"] += 1
                continue
            mapping = word_map.get((text, int(aoi)))
            if mapping is None:
                counts["excluded_unmapped_aoi"] += 1
                exclusions[f"{text}:{aoi}"] += 1
                continue
            line = line_map.get((text, mapping))
            if line is None:
                counts["excluded_missing_line"] += 1
                exclusions[f"{text}:{aoi}"] += 1
                continue
            try:
                order = int(row["CURRENT_FIX_INDEX"])
                duration = float(row["CURRENT_FIX_DURATION"])
            except ValueError:
                counts["excluded_invalid_value"] += 1
                continue
            if duration < 0:
                counts["excluded_invalid_value"] += 1
                continue
            writer.writerow(
                {
                    "subject_id": session_participant[trial[0]],
                    "text_id": text,
                    "word_index": mapping - 1,
                    "fixation_order": order,
                    "duration_ms": duration,
                    "line_id": line,
                }
            )
            counts["written_rows"] += 1

    report = ProvoConversionReport(
        main_sessions=len(session_participant),
        main_trials=len(trial_text),
        word_mappings=len(word_map),
        fixation_rows=counts["fixation_rows"],
        written_rows=counts["written_rows"],
        excluded_missing_trial=counts["excluded_missing_trial"],
        excluded_page_mismatch=counts["excluded_page_mismatch"],
        excluded_outside_aoi=counts["excluded_outside_aoi"],
        excluded_unmapped_aoi=counts["excluded_unmapped_aoi"],
        excluded_invalid_value=counts["excluded_invalid_value"],
        excluded_missing_line=counts["excluded_missing_line"],
        line_mapped_words=len(line_map),
        line_missing_words=sum(row["status"] != "mapped" for row in line_audit),
        exclusion_by_text_aoi=dict(sorted(exclusions.items())),
    )
    if report_path is not None:
        report_output = Path(report_path)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    if line_map_path is not None:
        path = Path(line_map_path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("text_id", "word_index", "line_id")); writer.writeheader()
            writer.writerows({"text_id": text, "word_index": word - 1, "line_id": line}
                             for (text, word), line in sorted(line_map.items(), key=lambda x: (int(x[0][0]), x[0][1])))
    if line_audit_path is not None:
        path = Path(line_audit_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(line_audit, indent=2), encoding="utf-8")
    return report
