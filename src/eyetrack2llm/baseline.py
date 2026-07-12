from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from os import PathLike

import numpy as np
from scipy.optimize import OptimizeResult, minimize


FEATURE_NAMES = (
    "forward",
    "log1p_abs_distance",
    "forward_x_log_distance",
    "adjacent",
    "source_log_length",
    "target_log_length",
    "source_cloze_logit",
    "target_cloze_logit",
    "source_cloze_missing",
    "target_cloze_missing",
    "source_terminal_punctuation",
    "target_terminal_punctuation",
    "same_sentence",
    "cross_sentence_direction",
    "source_zipf_frequency",
    "target_zipf_frequency",
    "source_frequency_oov",
    "target_frequency_oov",
    "src_head_of_dst",
    "dst_head_of_src",
    "dependency_tree_distance",
    "dependency_missing",
)

FEATURE_SETS = {
    "basic": FEATURE_NAMES[:4],
    "lexical": FEATURE_NAMES[:18],
    "syntax": FEATURE_NAMES,
}

COMMON_CORE_FEATURE_NAMES = (
    "log1p_abs_distance",
    "adjacent",
    "target_log_length",
    "target_zipf_frequency",
    "target_frequency_oov",
    "target_terminal_punctuation",
    "src_head_of_dst",
    "dst_head_of_src",
    "dependency_tree_distance",
    "target_relative_position",
    "target_first2",
    "target_last2",
)
FEATURE_SETS["common_core"] = COMMON_CORE_FEATURE_NAMES

STRICT_LINE_SPECIFICATION_SETS = {
    "position_only": (
        "log1p_abs_distance", "adjacent", "target_relative_position",
        "target_first2", "target_last2",
    ),
    "lexical": (
        "log1p_abs_distance", "adjacent", "target_relative_position",
        "target_first2", "target_last2", "target_log_length",
        "target_zipf_frequency", "target_frequency_oov",
        "target_terminal_punctuation",
    ),
    "syntax": COMMON_CORE_FEATURE_NAMES,
    "flexible": COMMON_CORE_FEATURE_NAMES + (
        "log1p_abs_distance_squared", "log1p_abs_distance_cubed",
        "distance_x_target_relative_position",
    ),
}
FEATURE_SETS.update({name: features for name, features in STRICT_LINE_SPECIFICATION_SETS.items()
                     if name not in FEATURE_SETS})


def _text_sort_key(value: str) -> tuple[str, int | str]:
    match = re.fullmatch(r"(.*?)(\d+)", str(value))
    return (match.group(1), int(match.group(2))) if match else (str(value), str(value))


@dataclass(frozen=True)
class WordMetadata:
    text_id: np.ndarray
    word_index: np.ndarray
    log_length: np.ndarray
    cloze_logit: np.ndarray
    cloze_missing: np.ndarray
    terminal_punctuation: np.ndarray
    sentence_number: np.ndarray
    surface: np.ndarray | None = None
    zipf_frequency: np.ndarray | None = None
    frequency_oov: np.ndarray | None = None
    head_word_index: np.ndarray | None = None
    dependency_label: np.ndarray | None = None
    syntax_missing: np.ndarray | None = None
    line_id: np.ndarray | None = None


@dataclass(frozen=True)
class PairDesign:
    features: np.ndarray
    text_id: np.ndarray
    src_word: np.ndarray
    dst_word: np.ndarray
    group_start: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def subset(self, mask: np.ndarray) -> PairDesign:
        """Select pairs and rebuild groups, dropping sources with no candidates."""
        selected = np.asarray(mask, dtype=bool)
        if selected.ndim != 1 or len(selected) != len(self.features):
            raise ValueError("PairDesign subset mask must be one-dimensional and match the pair count")
        indices = np.flatnonzero(selected)
        if not len(indices):
            raise ValueError("PairDesign subset has no candidate pairs")
        group_changes = (
            (self.text_id[indices][1:] != self.text_id[indices][:-1])
            | (self.src_word[indices][1:] != self.src_word[indices][:-1])
        )
        starts = np.r_[0, np.flatnonzero(group_changes) + 1, len(indices)].astype(np.int64)
        return PairDesign(
            self.features[indices], self.text_id[indices], self.src_word[indices],
            self.dst_word[indices], starts, self.feature_names,
        )

    def filter(self, mask: np.ndarray) -> PairDesign:
        return self.subset(mask)

    def group_constant_features(self) -> tuple[str, ...]:
        """Features unidentifiable after a conditional-softmax group intercept."""
        constant = []
        for column, name in enumerate(self.feature_names):
            if all(np.ptp(self.features[start:stop, column]) == 0
                   for start, stop in zip(self.group_start[:-1], self.group_start[1:], strict=True)):
                constant.append(name)
        return tuple(constant)

    def design_rank(self) -> int:
        """Rank of the design after removing source-group means."""
        centered = self.features.copy()
        for start, stop in zip(self.group_start[:-1], self.group_start[1:], strict=True):
            centered[start:stop] -= centered[start:stop].mean(axis=0)
        return int(np.linalg.matrix_rank(centered))


@dataclass(frozen=True)
class BaselineModel:
    coefficients: np.ndarray
    l2: float
    result: OptimizeResult

    def predict(self, design: PairDesign) -> np.ndarray:
        return predict_probabilities(design, self.coefficients)


def _number(value: str) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def _clean_length(label: str) -> int:
    cleaned = re.sub(r"[^A-Za-z0-9']", "", label.strip())
    return max(1, len(cleaned))


def read_provo_word_metadata(
    main_path: str | PathLike[str],
    observed_words: dict[str, set[int]],
    *,
    encoding: str = "cp1252",
) -> WordMetadata:
    """Read one conservative metadata record for every observed Provo token."""
    values: dict[tuple[str, int], dict[int, set[tuple[str, str, str, str, str]]]] = {}
    with open(main_path, encoding=encoding, newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row["Text_ID"])
            number = _number(row["Word_Number"])
            if number is None:
                if row["IA_ID"] != "1":
                    continue
                number = 1
            ia = _number(row["IA_ID"])
            if ia is None:
                continue
            index = number - 1
            if text not in observed_words or index not in observed_words[text]:
                continue
            record = (
                row.get("IA_LABEL", ""),
                row.get("Word_Length", ""),
                row.get("OrthographicMatch", ""),
                row.get("Total_Response_Count", ""),
                row.get("Sentence_Number", ""),
            )
            values.setdefault((text, index), {}).setdefault(ia, set()).add(record)

    metadata: list[tuple[str, int, str, float, float, bool, bool, int | None]] = []
    for text, words in observed_words.items():
        if not words or max(words) + 1 <= 0:
            raise ValueError(f"No observed words for text {text}")
        for index in sorted(words):
            candidates = values.get((text, index))
            if not candidates:
                raise ValueError(f"Missing Provo metadata for text {text}, word {index}")
            ranked_ias = sorted(candidates, key=lambda ia: abs(ia - (index + 1)))
            if len(ranked_ias) > 1 and abs(ranked_ias[0] - (index + 1)) == abs(ranked_ias[1] - (index + 1)):
                raise ValueError(f"Ambiguous Provo AOI for text {text}, word {index}")
            records = candidates[ranked_ias[0]]
            if len(records) != 1:
                raise ValueError(f"Inconsistent Provo metadata for text {text}, word {index}")
            label, length_value, cloze_value, total_value, sentence_value = next(iter(records))
            try:
                length = int(float(length_value))
            except ValueError:
                length = _clean_length(label)
            if length < 1:
                length = _clean_length(label)
            try:
                cloze = float(cloze_value)
                total = float(total_value)
                if not (np.isfinite(cloze) and 0 <= cloze <= 1 and total > 0):
                    raise ValueError
                # OrthographicMatch is a proportion; Jeffreys smoothing avoids infinite logits.
                probability = (cloze * total + 0.5) / (total + 1.0)
                cloze_logit = float(np.log(probability / (1.0 - probability)))
                cloze_missing = False
            except ValueError:
                cloze_logit = 0.0
                cloze_missing = True
            sentence = _number(sentence_value)
            terminal = bool(re.search(r"[.!?][\"')\]]*\s*$", label))
            metadata.append(
                (text, index, label.strip(), np.log(float(length)), cloze_logit, cloze_missing, terminal, sentence)
            )

    # Fill isolated corpus omissions from local sentence boundaries, never from gaze outcomes.
    by_text = {text: [] for text in observed_words}
    for item in metadata:
        by_text[item[0]].append(item)
    fixed = []
    for text, items in by_text.items():
        items.sort(key=lambda item: item[1])
        for position, item in enumerate(items):
            if item[7] is None:
                previous = items[position - 1] if position > 0 else None
                following = next((candidate for candidate in items[position + 1:] if candidate[7] is not None), None)
                if previous is not None and previous[7] is not None and not previous[6]:
                    sentence = previous[7]
                elif following is not None:
                    sentence = following[7]
                elif previous is not None:
                    sentence = previous[7]
                else:
                    sentence = None
                item = (*item[:7], sentence)
            if item[7] is None:
                raise ValueError(f"Missing sentence number for text {text}, word {item[1]}")
            fixed.append(item)
    fixed.sort(key=lambda item: (int(item[0]), item[1]))
    from .provo import build_provo_line_map
    line_map, _ = build_provo_line_map(main_path, encoding=encoding)
    return WordMetadata(
        text_id=np.asarray([item[0] for item in fixed], dtype=str),
        word_index=np.asarray([item[1] for item in fixed], dtype=np.int64),
        log_length=np.asarray([item[3] for item in fixed]),
        cloze_logit=np.asarray([item[4] for item in fixed]),
        cloze_missing=np.asarray([item[5] for item in fixed], dtype=bool),
        terminal_punctuation=np.asarray([item[6] for item in fixed], dtype=bool),
        sentence_number=np.asarray([item[7] for item in fixed], dtype=np.int64),
        surface=np.asarray([item[2] for item in fixed], dtype=str),
        line_id=np.asarray([line_map.get((item[0], item[1] + 1), -1) for item in fixed], dtype=np.int64),
    )


def enrich_word_frequencies(metadata: WordMetadata, *, language: str = "en") -> WordMetadata:
    """Return metadata with independent wordfreq Zipf values (optional dependency)."""
    from dataclasses import replace
    from wordfreq import zipf_frequency

    if metadata.surface is None:
        raise ValueError("surface forms are required for frequency enrichment")
    frequencies = np.asarray([zipf_frequency(word, language) for word in metadata.surface], dtype=float)
    if not np.isfinite(frequencies).all():
        raise ValueError("wordfreq returned a non-finite Zipf frequency")
    return replace(metadata, zipf_frequency=frequencies, frequency_oov=frequencies <= 0.0)


def enrich_spacy_syntax(metadata: WordMetadata, nlp) -> tuple[WordMetadata, dict[str, object]]:
    """Parse each Provo sentence with forced one-word-per-token alignment."""
    from dataclasses import replace
    from spacy.tokens import Doc

    if metadata.surface is None:
        raise ValueError("surface forms are required for syntax enrichment")
    heads = np.full(len(metadata.word_index), -1, dtype=np.int64)
    labels = np.full(len(metadata.word_index), "", dtype=object)
    missing = np.zeros(len(metadata.word_index), dtype=bool)
    sentence_reports = []
    keys = sorted(
        set(zip(metadata.text_id, metadata.sentence_number, strict=True)),
        key=lambda x: (_text_sort_key(str(x[0])), int(x[1])),
    )
    for text, sentence in keys:
        positions = np.flatnonzero((metadata.text_id == text) & (metadata.sentence_number == sentence))
        positions = positions[np.argsort(metadata.word_index[positions])]
        words = metadata.surface[positions].tolist()
        doc = Doc(nlp.vocab, words=words, spaces=[True] * (len(words) - 1) + [False])
        for _, component in nlp.pipeline:
            doc = component(doc)
        if len(doc) != len(words) or [token.text for token in doc] != words:
            raise ValueError(f"spaCy token alignment failure in text {text}, sentence {sentence}")
        roots = [token.i for token in doc if token.head.i == token.i]
        connected = len(roots) == 1 and all(any(node == roots[0] for node in _head_path(token)) for token in doc)
        if not connected:
            missing[positions] = True
        for token, position in zip(doc, positions, strict=True):
            heads[position] = int(metadata.word_index[positions[token.head.i]])
            labels[position] = token.dep_
        sentence_reports.append({"text_id": str(text), "sentence_number": int(sentence), "tokens": len(doc), "roots": len(roots), "connected": connected})
    audit = {
        "model": nlp.meta.get("name"),
        "model_version": nlp.meta.get("version"),
        "sentences": len(sentence_reports),
        "tokens": int(sum(item["tokens"] for item in sentence_reports)),
        "coverage": float(sum(item["tokens"] for item in sentence_reports) / len(metadata.word_index)),
        "bad_root_sentences": sum(item["roots"] != 1 for item in sentence_reports),
        "disconnected_sentences": sum(not item["connected"] for item in sentence_reports),
        "sentence_reports": sentence_reports,
    }
    if audit["coverage"] != 1.0:
        raise ValueError(f"spaCy dependency audit failed: {audit}")
    return replace(metadata, head_word_index=heads, dependency_label=np.asarray(labels, dtype=str), syntax_missing=missing), audit


def _head_path(token):
    seen = set()
    while token.i not in seen:
        seen.add(token.i)
        yield token.i
        if token.head.i == token.i:
            break
        token = token.head


def _tree_distance(src: int, dst: int, head_by_word: dict[int, int]) -> int:
    def ancestors(word: int) -> dict[int, int]:
        result = {}
        while word not in result:
            result[word] = len(result)
            parent = head_by_word[word]
            if parent == word:
                break
            word = parent
        return result
    source = ancestors(src)
    target = ancestors(dst)
    return min(source[word] + target[word] for word in source.keys() & target.keys())


def build_pair_design(metadata: WordMetadata, feature_set: str = "syntax",
                      risk_set: str = "all") -> PairDesign:
    """Build all directed non-self candidate pairs, grouped by text and source."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if risk_set not in {"all", "common_forward_same_sentence", "common_forward_same_sentence_same_line"}:
        raise ValueError(f"Unknown risk set: {risk_set}")
    strict_spec = (risk_set == "common_forward_same_sentence_same_line"
                   and feature_set in STRICT_LINE_SPECIFICATION_SETS)
    selected_feature_names = (STRICT_LINE_SPECIFICATION_SETS[feature_set]
                              if strict_spec else FEATURE_SETS[feature_set])
    rows = []
    texts = []
    sources = []
    targets = []
    starts = []
    for text in sorted(set(metadata.text_id), key=_text_sort_key):
        indices = np.flatnonzero(metadata.text_id == text)
        for source_position in indices:
            src = int(metadata.word_index[source_position])
            source_start = len(rows)
            for target_position in indices:
                dst = int(metadata.word_index[target_position])
                if src == dst:
                    continue
                distance = dst - src
                log_distance = np.log1p(abs(distance))
                forward = float(distance > 0)
                same_sentence = metadata.sentence_number[source_position] == metadata.sentence_number[target_position]
                strict_line = risk_set == "common_forward_same_sentence_same_line"
                if risk_set != "all" and (distance <= 0 or not same_sentence):
                    continue
                if strict_line and (metadata.line_id is None or metadata.line_id[source_position] < 0
                                    or metadata.line_id[source_position] != metadata.line_id[target_position]):
                    continue
                frequency_ready = metadata.zipf_frequency is not None and metadata.frequency_oov is not None
                syntax_ready = metadata.head_word_index is not None
                needs_frequency = feature_set in {"lexical", "syntax", "common_core", "flexible"}
                needs_syntax = feature_set in {"syntax", "common_core", "flexible"}
                if needs_frequency and not frequency_ready:
                    raise ValueError(f"{feature_set} feature set requires frequency enrichment")
                if needs_syntax and not syntax_ready:
                    raise ValueError("syntax feature set requires syntax enrichment")
                features = (
                        forward,
                        log_distance,
                        forward * log_distance,
                        float(abs(distance) == 1),
                        metadata.log_length[source_position],
                        metadata.log_length[target_position],
                        metadata.cloze_logit[source_position],
                        metadata.cloze_logit[target_position],
                        metadata.cloze_missing[source_position],
                        metadata.cloze_missing[target_position],
                        metadata.terminal_punctuation[source_position],
                        metadata.terminal_punctuation[target_position],
                        float(same_sentence),
                        0.0 if same_sentence else float(np.sign(metadata.sentence_number[target_position] - metadata.sentence_number[source_position])),
                )
                if frequency_ready:
                    features += (metadata.zipf_frequency[source_position], metadata.zipf_frequency[target_position], metadata.frequency_oov[source_position], metadata.frequency_oov[target_position])
                if syntax_ready:
                    src_head = int(metadata.head_word_index[source_position])
                    dst_head = int(metadata.head_word_index[target_position])
                    dependency_missing = not same_sentence or bool(metadata.syntax_missing is not None and (metadata.syntax_missing[source_position] or metadata.syntax_missing[target_position]))
                    if not dependency_missing:
                        sentence_positions = indices[metadata.sentence_number[indices] == metadata.sentence_number[source_position]]
                        head_map = {int(metadata.word_index[p]): int(metadata.head_word_index[p]) for p in sentence_positions}
                        tree_distance = _tree_distance(src, dst, head_map)
                    else:
                        tree_distance = 0
                    features += (float(dst_head == src), float(src_head == dst), float(tree_distance), float(dependency_missing))
                if feature_set == "common_core" or strict_spec:
                    sentence_positions = indices[metadata.sentence_number[indices] == metadata.sentence_number[target_position]]
                    sentence_positions = sentence_positions[np.argsort(metadata.word_index[sentence_positions])]
                    target_ordinal = int(np.flatnonzero(sentence_positions == target_position)[0])
                    denominator = max(1, len(sentence_positions) - 1)
                    common_values = (
                        log_distance, float(abs(distance) == 1), metadata.log_length[target_position],
                        metadata.zipf_frequency[target_position] if frequency_ready else 0.0,
                        metadata.frequency_oov[target_position] if frequency_ready else 0.0,
                        metadata.terminal_punctuation[target_position],
                        float(dst_head == src) if syntax_ready else 0.0,
                        float(src_head == dst) if syntax_ready else 0.0,
                        float(tree_distance) if syntax_ready else 0.0, target_ordinal / denominator,
                        float(target_ordinal < 2), float(target_ordinal >= len(sentence_positions) - 2),
                    )
                    names = COMMON_CORE_FEATURE_NAMES
                    values = common_values
                    if feature_set == "flexible":
                        relative = target_ordinal / denominator
                        values += (log_distance ** 2, log_distance ** 3,
                                   log_distance * relative)
                        names = STRICT_LINE_SPECIFICATION_SETS["flexible"]
                    lookup = dict(zip(names, values, strict=True))
                    features = tuple(lookup[name] for name in selected_feature_names)
                rows.append(features[:len(selected_feature_names)])
                texts.append(text)
                sources.append(src)
                targets.append(dst)
            if len(rows) > source_start:
                starts.append(source_start)
    starts.append(len(rows))
    design = PairDesign(
        features=np.asarray(rows, dtype=np.float64),
        text_id=np.asarray(texts, dtype=str),
        src_word=np.asarray(sources, dtype=np.int64),
        dst_word=np.asarray(targets, dtype=np.int64),
        group_start=np.asarray(starts, dtype=np.int64),
        feature_names=selected_feature_names,
    )
    return design


def count_vector(design: PairDesign, edges: dict[tuple[str, int, int], float]) -> np.ndarray:
    return np.asarray(
        [edges.get((text, int(src), int(dst)), 0.0) for text, src, dst in zip(design.text_id, design.src_word, design.dst_word, strict=True)],
        dtype=np.float64,
    )


def multinomial_nll_gradient(
    coefficients: np.ndarray, design: PairDesign, counts: np.ndarray, l2: float = 0.0
) -> tuple[float, np.ndarray]:
    scores = design.features @ coefficients
    starts = design.group_start[:-1]
    lengths = np.diff(design.group_start)
    totals = np.add.reduceat(counts, starts)
    maxima = np.maximum.reduceat(scores, starts)
    exponentials = np.exp(scores - np.repeat(maxima, lengths))
    denominators = np.add.reduceat(exponentials, starts)
    log_denominators = maxima + np.log(denominators)
    probabilities = exponentials / np.repeat(denominators, lengths)
    loss = (
        0.5 * l2 * float(coefficients @ coefficients)
        + float(totals @ log_denominators)
        - float(counts @ scores)
    )
    gradient = l2 * coefficients + design.features.T @ (
        np.repeat(totals, lengths) * probabilities - counts
    )
    return float(loss), gradient


def fit_baseline(
    design: PairDesign, counts: np.ndarray, *, l2: float = 1.0, maxiter: int = 300
) -> BaselineModel:
    result = minimize(
        multinomial_nll_gradient,
        np.zeros(design.features.shape[1]),
        args=(design, np.asarray(counts, dtype=np.float64), l2),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter},
    )
    if not result.success:
        raise RuntimeError(f"Baseline optimization failed: {result.message}")
    return BaselineModel(np.asarray(result.x), l2, result)


def predict_probabilities(design: PairDesign, coefficients: np.ndarray) -> np.ndarray:
    scores = design.features @ coefficients
    starts = design.group_start[:-1]
    lengths = np.diff(design.group_start)
    maxima = np.maximum.reduceat(scores, starts)
    exponentials = np.exp(scores - np.repeat(maxima, lengths))
    return exponentials / np.repeat(np.add.reduceat(exponentials, starts), lengths)


def residual_vector(counts: np.ndarray, probabilities: np.ndarray, group_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.diff(group_start)
    exposure = np.repeat(np.add.reduceat(counts, group_start[:-1]), lengths)
    expected = exposure * probabilities
    residual = (counts - expected) / np.sqrt(expected * (1.0 - probabilities) + 1e-12)
    return residual, exposure
