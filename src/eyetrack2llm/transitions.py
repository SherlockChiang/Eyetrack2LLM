from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import Fixations, validate_fixations

EventType = Literal["refixation", "forward", "regression", "line_return"]
EVENT_TYPES: tuple[EventType, ...] = (
    "refixation",
    "forward",
    "regression",
    "line_return",
)


@dataclass(frozen=True)
class TransitionEvents:
    subject_id: np.ndarray
    text_id: np.ndarray
    src_word: np.ndarray
    dst_word: np.ndarray
    event_type: np.ndarray
    weight: np.ndarray

    def __len__(self) -> int:
        return len(self.subject_id)


@dataclass(frozen=True)
class TransitionMatrix:
    text_id: str
    event_type: str
    count: np.ndarray
    probability: np.ndarray
    mask: np.ndarray


def extract_events(
    data: Fixations,
    *,
    weight: Literal["count", "source_duration", "target_duration"] = "count",
    include_self: bool = True,
    require_consecutive_order: bool = False,
) -> TransitionEvents:
    validate_fixations(data)
    if weight not in {"count", "source_duration", "target_duration"}:
        raise ValueError(f"Unsupported weight: {weight}")

    order = np.lexsort((data.fixation_order, data.text_id, data.subject_id))
    subjects = data.subject_id[order]
    texts = data.text_id[order]
    words = data.word_index[order]
    fixation_orders = data.fixation_order[order]
    durations = data.duration_ms[order]
    lines = data.line_id[order] if data.line_id is not None else None

    output_subjects: list[str] = []
    output_texts: list[str] = []
    sources: list[int] = []
    destinations: list[int] = []
    types: list[str] = []
    weights: list[float] = []

    for index in range(max(0, len(words) - 1)):
        if subjects[index] != subjects[index + 1] or texts[index] != texts[index + 1]:
            continue
        if require_consecutive_order and fixation_orders[index + 1] != fixation_orders[index] + 1:
            continue
        src = int(words[index])
        dst = int(words[index + 1])
        if src == dst:
            if not include_self:
                continue
            event_type = "refixation"
        elif lines is not None and lines[index + 1] > lines[index]:
            event_type = "line_return"
        elif dst > src:
            event_type = "forward"
        else:
            event_type = "regression"

        if weight == "source_duration":
            event_weight = float(durations[index])
        elif weight == "target_duration":
            event_weight = float(durations[index + 1])
        else:
            event_weight = 1.0

        output_subjects.append(str(subjects[index]))
        output_texts.append(str(texts[index]))
        sources.append(src)
        destinations.append(dst)
        types.append(event_type)
        weights.append(event_weight)

    return TransitionEvents(
        subject_id=np.asarray(output_subjects, dtype=str),
        text_id=np.asarray(output_texts, dtype=str),
        src_word=np.asarray(sources, dtype=np.int64),
        dst_word=np.asarray(destinations, dtype=np.int64),
        event_type=np.asarray(types, dtype=str),
        weight=np.asarray(weights, dtype=np.float64),
    )


def _normalize(count: np.ndarray, normalize: str) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros_like(count, dtype=np.float64)
    mask = np.zeros_like(count, dtype=bool)
    if normalize == "row":
        totals = count.sum(axis=1, keepdims=True)
        valid_rows = totals[:, 0] > 0
        probability[valid_rows] = count[valid_rows] / totals[valid_rows]
        mask[valid_rows, :] = True
    elif normalize == "global":
        total = count.sum()
        if total > 0:
            probability = count / total
            mask[:, :] = True
    elif normalize == "none":
        probability = count.copy()
        mask[:, :] = True
    else:
        raise ValueError(f"Unsupported normalization: {normalize}")
    return probability, mask


def aggregate_transitions(
    events: TransitionEvents,
    *,
    n_words: Mapping[str, int] | None = None,
    subjects: Collection[str] | None = None,
    normalize: Literal["row", "global", "none"] = "row",
    event_types: Sequence[str] = EVENT_TYPES,
) -> dict[tuple[str, str], TransitionMatrix]:
    selected_subjects = set(map(str, subjects)) if subjects is not None else None
    texts = set(map(str, events.text_id))
    if n_words is not None:
        texts.update(map(str, n_words))

    result: dict[tuple[str, str], TransitionMatrix] = {}
    for text in sorted(texts):
        text_mask = events.text_id == text
        if selected_subjects is not None:
            text_mask &= np.isin(events.subject_id, list(selected_subjects))
        observed = np.concatenate(
            (events.src_word[text_mask], events.dst_word[text_mask])
        )
        inferred_words = int(observed.max()) + 1 if observed.size else 0
        word_count = int(n_words[text]) if n_words is not None and text in n_words else inferred_words
        if word_count < inferred_words:
            raise ValueError(f"n_words[{text!r}] is smaller than an observed word index")

        for event_type in event_types:
            count = np.zeros((word_count, word_count), dtype=np.float64)
            mask = text_mask & (events.event_type == event_type)
            np.add.at(
                count,
                (events.src_word[mask], events.dst_word[mask]),
                events.weight[mask],
            )
            probability, valid_mask = _normalize(count, normalize)
            result[(text, event_type)] = TransitionMatrix(
                text_id=text,
                event_type=event_type,
                count=count,
                probability=probability,
                mask=valid_mask,
            )
    return result
