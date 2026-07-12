# Reproducibility

## Environment

- Python `>=3.10`; audited runs used Python 3.12.
- Install with `python -m pip install -e ".[test,analysis,figures]"` and `python -m spacy download en_core_web_sm`.
- Audited auxiliary artifacts record CPU PyTorch `2.13.0+cpu`; Transformers `5.13.0`, spaCy `3.8.14`, `en_core_web_sm 3.8.0`, and wordfreq `3.1.1` were used.
- Peak RSS was not recorded. Except where an artifact records runtime, analyses were not benchmarked; no runtime or hardware throughput is inferred.

## Data And Hashes

- Provo URLs, license, citations, and official SHA-256 hashes are in [`data.md`](data.md#provo-corpus).
- ZuCo 1.0 is obtained from <https://osf.io/q3zws/> under CC BY 4.0; conversion reports record subject-file hashes.
- Raw corpora, generated fixation CSVs, checkpoints, embedding caches, and MAT files are local and excluded from the compact artifact archive.

## Quick Verification

This path does not rerun model fitting or resampling. It verifies frozen values, regenerates Markdown/CSV/figures, and tests provenance correspondence.

```powershell
python scripts/generate_manuscript_assets.py --root C:\Users\vince\Eyetrack2LLM --timestamp 2026-07-12T00:00:00+00:00
python scripts/verify_results.py
python -m pytest tests/test_manuscript_assets.py
```

Runtime: not benchmarked. The current revision intentionally does not run `scripts/convert_arxiv_manuscript.py`.

## Full Reproduction

Run from `C:\Users\vince\Eyetrack2LLM`. Commands below state the complete main-analysis CLI rather than referring to another section. Parameters shown are the frozen/default parameters represented by current artifacts.

1. Convert Provo and reconstruct strict lines:

```powershell
python scripts/convert_provo.py data/raw/Provo_Corpus-Eyetracking_Data.csv data/raw/Provo_Corpus-Predictability_Norms.csv data/processed/provo_fixations.csv data/processed/provo_words.json
python scripts/analyze_provo.py data/processed/provo_fixations.csv data/processed/provo_words.json data/processed/provo_analysis.json
```

Verify `provo_conversion_strictline_report.json`, 2,739/2,740 mapped words, and conservative exclusion of the ambiguous word. Runtime: not benchmarked.

2. Run simulation validation:

```powershell
python scripts/run_residual_recovery_simulation.py --output data/processed/residual_recovery_simulation.json --csv-output data/processed/residual_recovery_simulation.csv --replicates 80 --seed 20260711
```

The grid uses 30 sources, six destinations, 24 events per subject-source, subject counts `4,12,42,84`, latent effects `0,0.55`, and concentrations `120,8`. Runtime: not benchmarked.

3. Run strict-line common-core reliability:

```powershell
python scripts/analyze_provo_independent_reliability.py --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --fixations data/processed/provo_fixations_with_lines.csv --output data/processed/provo_commoncore_strictline_independent_reliability.json --repeats 100 --seed 20260711 --risk-set common_forward_same_sentence_same_line
```

Each randomized partition of the fixed sample contains two non-overlapping 42-reader halves, each with its own five-fold text-cross-fitted nuisance model. Runtime: not benchmarked.

4. Run the four-specification curve and negative control:

```powershell
python scripts/analyze_provo_specification_curve.py --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --fixations data/processed/provo_fixations_with_lines.csv --output data/processed/provo_strictline_specification_curve.json --csv-output data/processed/provo_strictline_specification_curve.csv --repeats 100 --null-repeats 500 --seed 20260711
```

Runtime: not benchmarked.

5. Run Provo fixed-step auxiliary training for all optimization seeds:

```powershell
python scripts/run_auxiliary_experiment.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --feature-set common_core --risk-set common_forward_same_sentence_same_line --steps 50 --seed 101 --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/provo_auxiliary_strictline_fixed50_seed101.json
python scripts/run_auxiliary_experiment.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --feature-set common_core --risk-set common_forward_same_sentence_same_line --steps 50 --seed 202 --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/provo_auxiliary_strictline_fixed50_seed202.json
python scripts/run_auxiliary_experiment.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --feature-set common_core --risk-set common_forward_same_sentence_same_line --steps 50 --seed 303 --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/provo_auxiliary_strictline_fixed50_seed303.json
python scripts/run_auxiliary_experiment.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --feature-set common_core --risk-set common_forward_same_sentence_same_line --steps 50 --seed 404 --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/provo_auxiliary_strictline_fixed50_seed404.json
python scripts/run_auxiliary_experiment.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --feature-set common_core --risk-set common_forward_same_sentence_same_line --steps 50 --seed 505 --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/provo_auxiliary_strictline_fixed50_seed505.json
```

Runtime: not benchmarked.

6. Run Provo fixed-split text inference:

```powershell
python scripts/analyze_provo_text_inference.py data/processed/provo_auxiliary_strictline_fixed50_seed101.json data/processed/provo_auxiliary_strictline_fixed50_seed202.json data/processed/provo_auxiliary_strictline_fixed50_seed303.json data/processed/provo_auxiliary_strictline_fixed50_seed404.json data/processed/provo_auxiliary_strictline_fixed50_seed505.json --output data/processed/provo_auxiliary_strictline_fixed50_text_inference.json --csv-output data/processed/provo_auxiliary_strictline_fixed50_text_inference.csv --bootstrap 100000 --signflips 99999 --seed 20260712
```

Runtime: not benchmarked.

7. Convert all 12 ZuCo NR files, then run strict-line transfer:

```powershell
python scripts/convert_zuco.py
python scripts/run_zuco_transfer.py --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/zuco_transfer_strictline_fixed50.json
```

The transfer uses fixed step-50 scorers, all 12 readers, NR 101-300, and the same-sentence same-line forward risk set. Runtime: not benchmarked.

8. Run ZuCo criterion and nested reader/text uncertainty:

```powershell
python scripts/audit_zuco_criterion_uncertainty.py --output data/processed/zuco_strictline_criterion_uncertainty.json --csv-output data/processed/zuco_strictline_criterion_uncertainty.csv --checkpoint-dir data/processed/strictline_fixed50 --cache data/processed/zuco_transfer_bert.pt --reader-bootstraps 200 --text-bootstraps 200 --provo-subsets 200 --seed 20260711
```

This also enumerates all 462 unique 6/6 partitions. Runtime: not benchmarked.

9. Run the cross-corpus non-equivalence audit:

```powershell
python scripts/analyze_cross_corpus_invariance.py --output data/processed/cross_corpus_measurement_invariance.json --csv-output data/processed/cross_corpus_measurement_invariance.csv --bootstrap 1000 --permutations 500 --seed 20260711
```

The artifact records `238.133` seconds wall time. The machine model and peak RSS were not recorded.

10. Run delete-one-text reaggregation and regenerate assets:

```powershell
python scripts/analyze_text_influence.py
python scripts/generate_manuscript_assets.py --root C:\Users\vince\Eyetrack2LLM
python scripts/verify_results.py
python -m pytest
```

Runtime: not benchmarked.

## Interpretation Checks

- Reliability partitions are not 100 independent population samples.
- Provo optimization seeds are perturbations, not inferential replicates.
- Transfer inference is text-equal; edges and seeds are not independent units.
- Fixed-12 ZuCo text intervals condition on observed readers; nested intervals target reader-and-text generalization under empirical resampling.
- Delete-one-text reaggregation does not refit the nuisance model.
- Historical common-core and mixed-risk outputs must not populate current primary assets.
