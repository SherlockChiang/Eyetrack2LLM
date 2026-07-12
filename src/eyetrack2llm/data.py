from __future__ import annotations

import csv
from dataclasses import dataclass
from os import PathLike

import numpy as np


REQUIRED_COLUMNS = (
    "subject_id",
    "text_id",
    "word_index",
    "fixation_order",
    "duration_ms",
)


@dataclass(frozen=True)
class Fixations:
    subject_id: np.ndarray
    text_id: np.ndarray
    word_index: np.ndarray
    fixation_order: np.ndarray
    duration_ms: np.ndarray
    line_id: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.subject_id)


def validate_fixations(data: Fixations) -> None:
    lengths = {
        len(data.subject_id),
        len(data.text_id),
        len(data.word_index),
        len(data.fixation_order),
        len(data.duration_ms),
    }
    if data.line_id is not None:
        lengths.add(len(data.line_id))
    if len(lengths) != 1:
        raise ValueError("All fixation columns must have the same length")
    if np.any(data.word_index < 0):
        raise ValueError("word_index must contain non-negative integers")
    if not np.all(np.isfinite(data.duration_ms)) or np.any(data.duration_ms < 0):
        raise ValueError("duration_ms must contain finite non-negative values")

    seen: set[tuple[str, str, int]] = set()
    for subject, text, order in zip(
        data.subject_id, data.text_id, data.fixation_order, strict=True
    ):
        key = (str(subject), str(text), int(order))
        if key in seen:
            raise ValueError(
                "fixation_order must be unique within each subject_id/text_id group"
            )
        seen.add(key)

    if data.line_id is not None:
        word_lines: dict[tuple[str, int], int] = {}
        for text, word, line in zip(
            data.text_id, data.word_index, data.line_id, strict=True
        ):
            key = (str(text), int(word))
            previous = word_lines.setdefault(key, int(line))
            if previous != int(line):
                raise ValueError("A word_index cannot map to multiple line_id values")


def read_fixation_csv(
    path: str | PathLike[str], *, validate: bool = True
) -> Fixations:
    columns: dict[str, list[str]] = {name: [] for name in REQUIRED_COLUMNS}
    line_values: list[str] = []

    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(REQUIRED_COLUMNS) - fieldnames
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        has_line_id = "line_id" in fieldnames
        for row_number, row in enumerate(reader, start=2):
            try:
                for name in REQUIRED_COLUMNS:
                    columns[name].append(row[name])
                if has_line_id:
                    line_values.append(row["line_id"])
            except (KeyError, TypeError) as error:
                raise ValueError(f"Invalid CSV row {row_number}") from error

    try:
        data = Fixations(
            subject_id=np.asarray(columns["subject_id"], dtype=str),
            text_id=np.asarray(columns["text_id"], dtype=str),
            word_index=np.asarray(columns["word_index"], dtype=np.int64),
            fixation_order=np.asarray(columns["fixation_order"], dtype=np.int64),
            duration_ms=np.asarray(columns["duration_ms"], dtype=np.float64),
            line_id=np.asarray(line_values, dtype=np.int64) if has_line_id else None,
        )
    except ValueError as error:
        raise ValueError("Numeric fixation columns contain invalid values") from error

    if validate:
        validate_fixations(data)

    order = np.lexsort((data.fixation_order, data.text_id, data.subject_id))
    sorted_data = Fixations(
        subject_id=data.subject_id[order],
        text_id=data.text_id[order],
        word_index=data.word_index[order],
        fixation_order=data.fixation_order[order],
        duration_ms=data.duration_ms[order],
        line_id=data.line_id[order] if data.line_id is not None else None,
    )
    return sorted_data
