from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eyetrack2llm import (
    TransitionEvents,
    extract_events,
    read_fixation_csv,
    split_half_reliability,
)


def subset_events(events: TransitionEvents, mask: np.ndarray) -> TransitionEvents:
    return TransitionEvents(
        subject_id=events.subject_id[mask],
        text_id=events.text_id[mask],
        src_word=events.src_word[mask],
        dst_word=events.dst_word[mask],
        event_type=events.event_type[mask],
        weight=events.weight[mask],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze normalized Provo transitions")
    parser.add_argument("fixation_csv")
    parser.add_argument("output_json")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixations = read_fixation_csv(args.fixation_csv)
    events = extract_events(fixations, require_consecutive_order=True)
    event_counts = Counter(map(str, events.event_type))
    results: list[dict[str, object]] = []
    for text_id in sorted(set(map(str, events.text_id)), key=int):
        text_mask = events.text_id == text_id
        text_events = subset_events(events, text_mask)
        n_words = int(max(text_events.src_word.max(), text_events.dst_word.max())) + 1
        for event_type in ("forward", "regression", "refixation"):
            for normalization in ("row", "global"):
                result = split_half_reliability(
                    text_events,
                    text_id=text_id,
                    event_type=event_type,
                    n_words=n_words,
                    repeats=args.repeats,
                    seed=args.seed,
                    normalize=normalization,
                )
                row = asdict(result)
                row.pop("repeat_values")
                row.update(
                    text_id=text_id,
                    event_type=event_type,
                    normalization=normalization,
                    n_words=n_words,
                    n_events=int(np.sum(text_events.event_type == event_type)),
                )
                results.append(row)

    summary: dict[str, object] = {
        "n_fixations": len(fixations),
        "n_transitions": len(events),
        "n_subjects": len(set(map(str, fixations.subject_id))),
        "n_texts": len(set(map(str, fixations.text_id))),
        "event_counts": dict(sorted(event_counts.items())),
        "repeats": args.repeats,
        "seed": args.seed,
        "reliability": results,
    }
    for event_type in ("forward", "regression", "refixation"):
        for normalization in ("row", "global"):
            values = np.asarray(
                [
                    row["value"]
                    for row in results
                    if row["event_type"] == event_type
                    and row["normalization"] == normalization
                    and np.isfinite(row["value"])
                ]
            )
            summary[f"{event_type}_{normalization}_summary"] = {
                "valid_texts": len(values),
                "median": float(np.median(values)) if len(values) else None,
                "q25": float(np.quantile(values, 0.25)) if len(values) else None,
                "q75": float(np.quantile(values, 0.75)) if len(values) else None,
            }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "reliability"}, indent=2))


if __name__ == "__main__":
    main()
