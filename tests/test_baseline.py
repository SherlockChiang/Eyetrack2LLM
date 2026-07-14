import numpy as np
from scipy.optimize import check_grad

from eyetrack2llm.baseline import (
    PairDesign,
    build_pair_design,
    enrich_spacy_syntax,
    enrich_word_frequencies,
    fit_baseline,
    multinomial_nll_gradient,
    residual_vector,
    WordMetadata,
)


def synthetic_design():
    metadata = WordMetadata(
        text_id=np.array(["1"] * 4),
        word_index=np.arange(4),
        log_length=np.log([3, 5, 4, 7]),
        cloze_logit=np.array([0.0, -1.0, 0.5, 1.0]),
        cloze_missing=np.array([True, False, False, False]),
        terminal_punctuation=np.array([False, False, False, True]),
        sentence_number=np.ones(4, dtype=int),
    )
    return build_pair_design(metadata, "basic")


def enriched_metadata():
    return WordMetadata(
        text_id=np.array(["1"] * 4), word_index=np.arange(4),
        log_length=np.log([3, 5, 4, 7]), cloze_logit=np.zeros(4),
        cloze_missing=np.zeros(4, dtype=bool), terminal_punctuation=np.array([False] * 3 + [True]),
        sentence_number=np.ones(4, dtype=int), surface=np.array(["The", "quick", "fox", "jumps."]),
    )


def common_metadata():
    metadata = enrich_word_frequencies(enriched_metadata())
    return WordMetadata(**{**metadata.__dict__, "frequency_oov": np.array([False, True, False, True]),
        "head_word_index": np.array([1, 1, 1, 2]),
        "dependency_label": np.array(["det", "ROOT", "nsubj", "punct"]),
        "syntax_missing": np.zeros(4, dtype=bool)})


def test_probabilities_sum_to_one_per_source():
    design = synthetic_design()
    probabilities = fit_baseline(design, np.ones(len(design.features))).predict(design)
    for start, stop in zip(design.group_start[:-1], design.group_start[1:], strict=True):
        np.testing.assert_allclose(probabilities[start:stop].sum(), 1.0)


def test_forward_subset_rebuilds_risk_sets_and_drops_terminal_sources():
    full = synthetic_design()
    design = full.filter(full.dst_word > full.src_word)
    assert np.all(design.dst_word > design.src_word)
    assert set(design.src_word) == {0, 1, 2}
    assert 3 not in design.src_word
    assert np.all(np.diff(design.group_start) > 0)
    probabilities = fit_baseline(design, np.ones(len(design.features))).predict(design)
    for start, stop in zip(design.group_start[:-1], design.group_start[1:], strict=True):
        np.testing.assert_allclose(probabilities[start:stop].sum(), 1.0)


def test_analytic_gradient():
    design = synthetic_design()
    counts = np.arange(1, len(design.features) + 1, dtype=float)
    point = np.linspace(-0.2, 0.2, design.features.shape[1])
    error = check_grad(
        lambda value: multinomial_nll_gradient(value, design, counts, 0.3)[0],
        lambda value: multinomial_nll_gradient(value, design, counts, 0.3)[1],
        point,
    )
    assert error < 1e-4


def test_fit_beats_uniform_and_residual_mask_uses_source_exposure():
    design = synthetic_design()
    counts = np.zeros(len(design.features))
    for start, stop in zip(design.group_start[:-1], design.group_start[1:], strict=True):
        forward = np.flatnonzero(design.dst_word[start:stop] > design.src_word[start])
        counts[start + (forward[0] if len(forward) else 0)] = 20
    model = fit_baseline(design, counts, l2=0.01)
    probabilities = model.predict(design)
    fitted_nll = -counts @ np.log(probabilities)
    uniform_nll = sum(
        counts[start:stop].sum() * np.log(stop - start)
        for start, stop in zip(design.group_start[:-1], design.group_start[1:], strict=True)
    )
    assert fitted_nll < uniform_nll
    residual, exposure = residual_vector(counts, probabilities, design.group_start)
    singleton = np.repeat(np.diff(design.group_start), np.diff(design.group_start)) == 1
    assert np.isnan(residual[singleton]).all()
    assert np.isfinite(residual[~singleton]).all()
    mask = exposure >= 5
    assert mask.all()
    assert mask[counts == 0].all()


def test_singleton_conditional_group_has_undefined_residual():
    residual, exposure = residual_vector(np.array([7.]), np.array([1.]), np.array([0, 1]))
    assert exposure.tolist() == [7]
    assert np.isnan(residual[0])


def test_word_frequencies_are_finite_and_have_oov_flags():
    metadata = enrich_word_frequencies(enriched_metadata())
    assert np.isfinite(metadata.zipf_frequency).all()
    assert metadata.frequency_oov.dtype == bool
    assert metadata.zipf_frequency[0] > 0


def test_syntax_graph_and_pair_features_with_real_parser():
    import pytest

    spacy = pytest.importorskip("spacy", reason="real-parser integration requires the analysis extra")
    try:
        parser = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("real-parser integration requires: python -m spacy download en_core_web_sm")

    metadata = enrich_word_frequencies(enriched_metadata())
    metadata, audit = enrich_spacy_syntax(metadata, parser)
    assert audit["coverage"] == 1.0
    assert audit["bad_root_sentences"] == 0
    assert audit["disconnected_sentences"] == 0
    design = build_pair_design(metadata, "syntax")
    assert design.features.shape[1] == 22
    assert np.isfinite(design.features).all()
    tree_distance = design.features[:, design.feature_names.index("dependency_tree_distance")]
    assert (tree_distance >= 1).all()


def test_known_dependency_graph_direct_edges_and_tree_distance():
    metadata = enrich_word_frequencies(enriched_metadata())
    metadata = WordMetadata(
        **{name: getattr(metadata, name) for name in (
            "text_id", "word_index", "log_length", "cloze_logit", "cloze_missing",
            "terminal_punctuation", "sentence_number", "surface", "zipf_frequency", "frequency_oov"
        )},
        head_word_index=np.array([1, 1, 1, 2]),
        dependency_label=np.array(["det", "ROOT", "nsubj", "punct"]),
        syntax_missing=np.zeros(4, dtype=bool),
    )
    design = build_pair_design(metadata, "syntax")
    src_head = design.feature_names.index("src_head_of_dst")
    dst_head = design.feature_names.index("dst_head_of_src")
    distance = design.feature_names.index("dependency_tree_distance")
    edge = np.flatnonzero((design.src_word == 1) & (design.dst_word == 3))[0]
    assert design.features[edge, src_head] == 0
    assert design.features[edge, dst_head] == 0
    assert design.features[edge, distance] == 2
    direct = np.flatnonzero((design.src_word == 1) & (design.dst_word == 2))[0]
    assert design.features[direct, src_head] == 1
    assert design.features[direct, distance] == 1


def test_namespaced_text_ids_support_natural_sorting():
    metadata = enriched_metadata()
    metadata = WordMetadata(
        **{
            name: (np.array(["NR:10"] * 4) if name == "text_id" else getattr(metadata, name))
            for name in metadata.__dataclass_fields__
        }
    )
    design = build_pair_design(metadata, "basic")
    assert set(design.text_id) == {"NR:10"}


def test_common_core_schema_rank_and_no_group_constants():
    design = build_pair_design(common_metadata(), "common_core", "common_forward_same_sentence")
    assert design.feature_names == build_pair_design(common_metadata(), "common_core", "common_forward_same_sentence").feature_names
    assert design.group_constant_features() == ()
    assert 0 < design.design_rank() <= design.features.shape[1]


def test_common_risk_set_is_forward_and_same_sentence():
    metadata = common_metadata()
    metadata = WordMetadata(**{**metadata.__dict__, "sentence_number": np.array([1, 1, 2, 2]),
        "head_word_index": np.array([0, 0, 2, 2])})
    design = build_pair_design(metadata, "common_core", "common_forward_same_sentence")
    sentence = dict(zip(metadata.word_index, metadata.sentence_number, strict=True))
    assert np.all(design.dst_word > design.src_word)
    assert all(sentence[int(src)] == sentence[int(dst)] for src, dst in zip(design.src_word, design.dst_word, strict=True))


def test_common_strict_line_risk_set_excludes_other_lines():
    metadata = common_metadata()
    metadata = WordMetadata(**{**metadata.__dict__, "line_id": np.array([0, 0, 1, 1])})
    design = build_pair_design(metadata, "common_core", "common_forward_same_sentence_same_line")
    lines = dict(zip(metadata.word_index, metadata.line_id, strict=True))
    assert all(lines[int(src)] == lines[int(dst)] for src, dst in zip(design.src_word, design.dst_word, strict=True))


def test_strict_line_specifications_have_fixed_features_and_conditional_rank():
    metadata = common_metadata()
    metadata = WordMetadata(**{**metadata.__dict__, "line_id": np.zeros(4, dtype=int)})
    expected = {"position_only": 5, "lexical": 9, "syntax": 12, "flexible": 15}
    for specification, columns in expected.items():
        design = build_pair_design(metadata, specification, "common_forward_same_sentence_same_line")
        assert design.features.shape[1] == columns
        assert np.isfinite(design.features).all()
        assert design.group_constant_features() == ()
        assert 0 < design.design_rank() <= columns
