# Cross-Corpus Measurement Non-Equivalence Audit

## Scope

This is a computational audit of observable measurement non-equivalence between Provo and ZuCo, not a claim that strict psychometric measurement invariance holds or fails. Both corpora use `common_forward_same_sentence_same_line`, the same ordered 12-feature `common_core` design, conditional rank 12, shared raw feature units, and source-conditional multinomial models with `L2=1`.

The inferential unit is always text. Feature differences use one aggregate per text and text-equal standardized mean differences. Bootstrap samples independently resample Provo and ZuCo texts with replacement. Coefficient bootstrap multiplicities weight each text's counts on the unchanged candidate design, exactly representing text resampling without treating edges as independent. Domain classification uses only per-text aggregate design/exposure summaries, stratified five-fold CV, balanced accuracy/AUC, and 500 label permutations.

## Measures

- Risk-set geometry: words, candidate pairs, source groups, candidates per source, forward distance, adjacency, and line span.
- Observed exposure: events per source, zero-destination proportion, destination concentration, and entropy.
- All 12 feature distributions: text-equal SMD and 95% text-bootstrap interval.
- Nuisance fit: corpus-specific full-sample coefficients and independently bootstrapped Provo-minus-ZuCo differences.
- Calibration: coefficients fitted in one corpus transported unchanged to the other, compared with within-corpus five-fold text-crossfit and uniform predictions using text-equal mean NLL.
- Residual diagnostics: per-text Pearson-residual location, variance, tails, dispersion, and exposure.
- Omnibus domain distinguishability: corpus classification from per-text aggregate summaries only.

## Reproduction

```powershell
python scripts/analyze_cross_corpus_invariance.py
python scripts/verify_results.py
python -m pytest
```

Outputs are `data/processed/cross_corpus_measurement_invariance.json` and the compact `cross_corpus_measurement_invariance.csv`. The JSON records definitions, limitations, seed, repeats, runtime, complete summaries, and per-text geometry/exposure values.

## Interpretation Rule

Strong observable non-equivalence means the ZuCo transfer null cannot be uniquely attributed to ineffective gaze supervision: corpus measurement geometry, exposure, or nuisance calibration may contribute. Near-equivalence would narrow that explanation but would not prove psychometric invariance. With only two observational corpora, task, reader, layout, and text-composition causes are not separately identified; neither classification nor coefficient differences establish construct validity.

## Frozen Results

The run used seed `20260711`, 1,000 independent two-corpus text bootstraps, and 500 text-label permutations. Computation after corpus construction took `238.133` seconds on the audited CPU environment.

- Provo versus ZuCo text means were `49.564` versus `22.015` words, `304.891` versus `91.235` candidate pairs, `44.909` versus `19.400` source groups, and `6.774` versus `4.624` candidates per source.
- Mean forward distance was `5.237` versus `3.530`; adjacency was `.1510` versus `.2193`; maximum within-line span averaged `16.091` versus `9.245`.
- Events per source were `51.372` versus `7.700`. Zero-destination proportions were `.5034` versus `.5352`, concentration `.5782` versus `.6622`, and entropy `.7426` versus `.5250`.
- The largest absolute feature SMDs were log distance `3.465` (95% bootstrap CI `[3.130, 3.794]`) and adjacency `-2.483` (`[-2.741, -2.218]`). Target length was close (`0.022`, `[-0.259, 0.278]`), while several lexical, syntax, and position features differed.
- Seven of 12 coefficient-difference intervals excluded zero: log distance, adjacency, target length, target frequency, source-head relation, target relative position, and target-first-two; target relative position was largest (`-1.677`, `[-3.185, -0.228]`). This count is descriptive across correlated coefficients, not a multiplicity-adjusted test family.
- Text-equal NLL in Provo was `.87811` for within-corpus crossfit, `.88547` for transported ZuCo coefficients, and `1.73279` for uniform. In ZuCo it was `.84912`, `.85759`, and `1.38392`, respectively. Thus transport was slightly worse than within-corpus prediction (`.00736` and `.00847`) but much better than uniform.
- Mean per-text Pearson dispersion was `3.150` in Provo and `1.529` in ZuCo; the fraction of residuals with absolute value above two was `.1651` versus `.0783` in the complete 1,000-bootstrap artifact.
- Text-level domain classification achieved balanced accuracy `.99091` and AUC `1.000`; both add-one permutation p-values were `.001996` (`1/501`). Sample imbalance therefore does not explain the classification result.

The combined pattern is strong observable non-equivalence in geometry, exposure, feature distributions, and residual dispersion, alongside relatively small conditional-NLL transport penalties. It does not support a binary declaration that measurement invariance either holds or fails. It does mean the ZuCo transfer null cannot be uniquely attributed to ineffective supervision; measurement and corpus differences remain a quantitatively supported alternative contributor.
