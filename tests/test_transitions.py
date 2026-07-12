import csv

import numpy as np
import pytest

from eyetrack2llm import (
    Fixations,
    aggregate_transitions,
    exact_four_subject_split_half,
    extract_events,
    read_fixation_csv,
    split_half_reliability,
    split_half_from_counts,
)


def test_read_extract_and_classify(tmp_path):
    path = tmp_path / "fixations.csv"
    rows = [
        ["s1", "t1", 0, 2, 110, 0],
        ["s1", "t1", 0, 1, 100, 0],
        ["s1", "t1", 2, 3, 120, 0],
        ["s1", "t1", 1, 4, 130, 0],
        ["s1", "t1", 3, 5, 140, 1],
        ["s2", "t1", 1, 1, 100, 0],
        ["s2", "t2", 0, 1, 100, 0],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["subject_id", "text_id", "word_index", "fixation_order", "duration_ms", "line_id"]
        )
        writer.writerows(rows)

    events = extract_events(read_fixation_csv(path))
    assert events.event_type.tolist() == [
        "refixation",
        "forward",
        "regression",
        "line_return",
    ]
    assert list(zip(events.src_word, events.dst_word, strict=True)) == [
        (0, 0),
        (0, 2),
        (2, 1),
        (1, 3),
    ]


def test_aggregation_uses_full_row_mask():
    data = Fixations(
        subject_id=np.array(["s1", "s1", "s1", "s1"]),
        text_id=np.array(["t", "t", "t", "t"]),
        word_index=np.array([0, 1, 0, 2]),
        fixation_order=np.arange(4),
        duration_ms=np.ones(4),
    )
    matrix = aggregate_transitions(
        extract_events(data), n_words={"t": 4}, event_types=("forward",)
    )[("t", "forward")]
    np.testing.assert_array_equal(matrix.count[0], [0, 1, 1, 0])
    np.testing.assert_allclose(matrix.probability[0], [0, 0.5, 0.5, 0])
    assert matrix.mask[0].all()
    assert not matrix.mask[1:].any()


def test_split_half_identical_subject_patterns_are_reliable():
    subjects = []
    texts = []
    words = []
    orders = []
    for subject in range(6):
        subjects.extend([f"s{subject}"] * 5)
        texts.extend(["t"] * 5)
        words.extend([0, 1, 0, 2, 0])
        orders.extend(range(5))
    events = extract_events(
        Fixations(
            subject_id=np.array(subjects),
            text_id=np.array(texts),
            word_index=np.array(words),
            fixation_order=np.array(orders),
            duration_ms=np.ones(len(words)),
        )
    )
    result = split_half_reliability(
        events, text_id="t", event_type="forward", n_words=3, repeats=10
    )
    assert result.value == pytest.approx(1.0)
    assert result.n_repeats_valid == 10


def test_generic_split_half_is_reproducible_and_supports_unequal_halves():
    counts = np.zeros((11, 3, 3))
    for subject in range(11):
        counts[subject, 0, 1] = subject + 1
        counts[subject, 0, 2] = 12 - subject
        counts[subject, 1, 2] = subject % 3 + 1
    first = split_half_from_counts(counts, repeats=25, seed=17, correction="none")
    second = split_half_from_counts(counts, repeats=25, seed=17, correction="none")
    np.testing.assert_array_equal(first.repeat_values, second.repeat_values)
    assert first.half_sizes == (5, 6)
    assert first.n_repeats_valid == 25


def test_generic_split_half_identical_counts_equal_one():
    pattern = np.array([[0, 2, 1], [0, 0, 3], [0, 0, 0]], dtype=float)
    result = split_half_from_counts(np.repeat(pattern[None], 10, axis=0), repeats=20)
    np.testing.assert_allclose(result.repeat_values, 1.0)


def test_exact_four_subject_partitions_and_identical_patterns():
    subjects, texts, words, orders = [], [], [], []
    for subject in range(4):
        subjects.extend([f"s{subject}"] * 5)
        texts.extend(["t"] * 5)
        words.extend([0, 1, 0, 2, 0])
        orders.extend(range(5))
    events = extract_events(Fixations(
        np.array(subjects), np.array(texts), np.array(words), np.array(orders),
        np.ones(len(words)),
    ))
    result = exact_four_subject_split_half(
        events, text_id="t", event_type="forward", n_words=3
    )
    assert result.n_partitions == 3
    assert result.n_partitions_valid == 3
    assert result.value == pytest.approx(1.0)


def test_exact_four_subject_sparse_patterns_are_nan():
    events = extract_events(Fixations(
        np.repeat(["s0", "s1", "s2", "s3"], 2),
        np.repeat("t", 8),
        np.tile([0, 1], 4),
        np.tile([0, 1], 4),
        np.ones(8),
    ))
    result = exact_four_subject_split_half(
        events, text_id="t", event_type="regression", n_words=2
    )
    assert result.n_partitions == 3
    assert result.n_partitions_valid == 0
    assert np.isnan(result.value)


def test_duplicate_order_is_rejected():
    data = Fixations(
        subject_id=np.array(["s", "s"]),
        text_id=np.array(["t", "t"]),
        word_index=np.array([0, 1]),
        fixation_order=np.array([1, 1]),
        duration_ms=np.ones(2),
    )
    with pytest.raises(ValueError, match="unique"):
        extract_events(data)


def test_excluded_fixation_is_not_bridged():
    data = Fixations(
        subject_id=np.array(["s", "s"]),
        text_id=np.array(["t", "t"]),
        word_index=np.array([0, 2]),
        fixation_order=np.array([1, 3]),
        duration_ms=np.ones(2),
    )
    assert len(extract_events(data)) == 1
    assert len(extract_events(data, require_consecutive_order=True)) == 0


def test_line_return_precedes_global_word_direction():
    data = Fixations(np.array(["s", "s"]), np.array(["t", "t"]), np.array([10, 2]),
                     np.array([1, 2]), np.ones(2), np.array([0, 1]))
    events = extract_events(data)
    assert events.event_type.tolist() == ["line_return"]
