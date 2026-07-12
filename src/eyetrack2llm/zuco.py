from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from .provo import cluster_vertical_intervals


@dataclass(frozen=True)
class ZuCoConversionReport:
    subject_id: str
    task: str
    slots: int
    valid_sentences: int
    missing_sentences: int
    all_fixations: int
    mapped_fixations: int
    outside_fixations: int
    conflict_fixations: int
    out_of_range_positions: int
    bounds_mismatches: int
    nfix_mismatches: int
    duration_invalid: int
    geometry_unique: int
    geometry_outside: int
    geometry_conflicts: int


def _identity(path: str | PathLike[str], subject: str | None, task: str | None) -> tuple[str, str]:
    match = re.fullmatch(r"results([A-Za-z0-9]+)_([A-Za-z0-9]+)\.mat", Path(path).name)
    if subject is None or task is None:
        if match is None:
            raise ValueError("MAT filename must be results<SUBJECT>_<TASK>.mat when identity arguments are omitted")
    parsed_subject, parsed_task = match.groups() if match else (None, None)
    subject = subject or parsed_subject
    task = task or parsed_task
    if not subject or re.fullmatch(r"[A-Za-z0-9]+", subject) is None:
        raise ValueError(f"Invalid subject: {subject!r}")
    if not task or re.fullmatch(r"[A-Za-z0-9]+", task) is None:
        raise ValueError(f"Invalid task: {task!r}")
    return subject, task.upper()


def _vector(value, dtype=float) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def map_sentence_fixations(words, bounds, x, y, durations):
    """Map MATLAB fixation positions conservatively; returned orders remain 1-based."""
    x, y, durations = (_vector(value) for value in (x, y, durations))
    if not (len(x) == len(y) == len(durations)):
        raise ValueError("allFixations x/y/duration lengths differ")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (len(words), 4):
        raise ValueError("wordbounds must have one [left, top, right, bottom] row per word")

    claims: dict[int, list[int]] = {}
    counts: Counter[str] = Counter(all_fixations=len(x))
    for word_index, word in enumerate(words):
        positions = _vector(getattr(word, "fixPositions"), dtype=float)
        positions = positions[np.isfinite(positions)]
        declared_value = _vector(getattr(word, "nFixations"), dtype=float)
        declared = int(declared_value[0]) if declared_value.size else 0
        if declared != len(positions):
            counts["nfix_mismatches"] += 1
        for position in positions:
            if position != int(position) or not 1 <= position <= len(x):
                counts["out_of_range_positions"] += 1
                continue
            claims.setdefault(int(position), []).append(word_index)

    rows = []
    for order in range(1, len(x) + 1):
        candidates = claims.get(order, [])
        geometric = []
        if np.isfinite(x[order - 1]) and np.isfinite(y[order - 1]):
            # Closed bounds match MATLAB AOI convention; shared edges can therefore conflict.
            geometric = np.flatnonzero(
                (bounds[:, 0] <= x[order - 1]) & (x[order - 1] <= bounds[:, 2])
                & (bounds[:, 1] <= y[order - 1]) & (y[order - 1] <= bounds[:, 3])
            ).tolist()
        counts["geometry_unique" if len(geometric) == 1 else "geometry_outside" if not geometric else "geometry_conflicts"] += 1
        if not candidates:
            counts["outside_fixations"] += 1
            continue
        if len(set(candidates)) != 1:
            counts["conflict_fixations"] += 1
            continue
        word_index = candidates[0]
        if word_index not in geometric:
            counts["bounds_mismatches"] += 1
        duration = durations[order - 1]
        if not np.isfinite(duration) or duration < 0:
            counts["duration_invalid"] += 1
            continue
        rows.append((word_index, order, float(duration) * 2.0))
        counts["mapped_fixations"] += 1
    return rows, counts


def convert_zuco_mat(
    mat_path: str | PathLike[str], output_path: str | PathLike[str], *,
    report_path: str | PathLike[str] | None = None,
    metadata_path: str | PathLike[str] | None = None,
    subject: str | None = None, task: str | None = None,
) -> ZuCoConversionReport:
    subject_id, task_id = _identity(mat_path, subject, task)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "sentenceData" not in data:
        raise ValueError("MAT file has no sentenceData")
    sentences = np.asarray(data["sentenceData"], dtype=object).reshape(-1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter(slots=len(sentences))
    metadata = []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "text_id", "word_index", "fixation_order", "duration_ms", "line_id"))
        for slot, sentence in enumerate(sentences, start=1):
            text_id = f"{task_id}:{slot}"
            words_value = getattr(sentence, "word", None)
            fixations = getattr(sentence, "allFixations", None)
            if not hasattr(fixations, "x") or not (
                hasattr(words_value, "content")
                or isinstance(words_value, np.ndarray) and words_value.size and hasattr(words_value.reshape(-1)[0], "content")
            ):
                totals["missing_sentences"] += 1
                continue
            words = np.asarray(words_value, dtype=object).reshape(-1)
            bounds = np.asarray(sentence.wordbounds, dtype=float)
            rows, counts = map_sentence_fixations(
                words, bounds, fixations.x, fixations.y, fixations.duration
            )
            lines = cluster_vertical_intervals([(float(bound[1]), float(bound[3])) for bound in bounds])
            totals.update(counts)
            totals["valid_sentences"] += 1
            metadata.append({
                "text_id": text_id,
                "content": str(sentence.content),
                "words": [str(word.content) for word in words],
                "bounds": bounds.tolist(),
            })
            writer.writerows((subject_id, text_id, word, order, duration, lines[word]) for word, order, duration in rows)

    report = ZuCoConversionReport(subject_id, task_id, **{
        field: totals[field] for field in ZuCoConversionReport.__dataclass_fields__
        if field not in {"subject_id", "task"}
    })
    if report_path:
        path = Path(report_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    if metadata_path:
        path = Path(metadata_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
