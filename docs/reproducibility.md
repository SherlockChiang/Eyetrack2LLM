# Reproducibility

## Environment

Use Python 3.10 or newer and run commands from the repository root:

```bash
python -m pip install ".[test,analysis]"
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
python -m pytest
```

Tests require no corpus or model download. Analyses require official Provo or ZuCo inputs described in [data.md](data.md). The versioned manuscript release must provide the compact aggregate-results attachment described below. Paths are relative and work across platforms.

The syntax analyses pin `en_core_web_sm==3.8.0` to the official wheel above; its SHA-256 is `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`. Verify the installed model with `python -c "import importlib.metadata as m; assert m.version('en-core-web-sm') == '3.8.0'"`. The BERT analyses pin `google-bert/bert-base-uncased` to revision `86b5e0934494bd15c9632b12f734a8a67f723594`. Cache schema v2 hashes the resolved model/config/tokenizer files, exact word sequences and tokenization, masking policy, and relevant software versions. A legacy or mismatched cache is rebuilt. Checkpoint schema v3 additionally stores the residual-support policy, matching provenance, and deterministic trainable-state hashes; transfer rejects a mismatch.

## Main Analysis

Convert Provo and reconstruct strict lines:

```bash
python scripts/convert_provo.py data/raw/Provo_Corpus-Eyetracking_Data.csv data/raw/Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv data/processed/provo_fixations_with_lines.csv --report data/processed/provo_conversion_strictline_report.json --line-map data/processed/provo_word_line_map.csv --line-audit data/processed/provo_word_line_audit.json
```

Run simulation, independent-half reliability, and the specification curve:

```bash
python scripts/run_residual_recovery_simulation.py --output-json data/processed/residual_recovery_simulation.json --output-csv data/processed/residual_recovery_simulation.csv --replicates 80 --seed 20260711
python scripts/analyze_provo_independent_reliability.py --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --fixations data/processed/provo_fixations_with_lines.csv --output data/processed/provo_commoncore_strictline_independent_reliability.json --repeats 100 --reader-bootstraps 200 --seed 20260711
python scripts/analyze_provo_specification_curve.py --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --fixations data/processed/provo_fixations_with_lines.csv --output data/processed/provo_strictline_specification_curve.json --csv-output data/processed/provo_strictline_specification_curve.csv --repeats 100 --null-repeats 500 --seed 20260711
```

Run the fixed-step auxiliary experiment for seeds 101, 202, 303, 404, and 505. This PowerShell loop uses the canonical filenames consumed by the generator and verifier:

```powershell
foreach ($seed in 101,202,303,404,505) { python scripts/run_auxiliary_experiment.py --feature-set common_core --risk-set common_forward_same_sentence_same_line --steps 50 --eval-every 50 --fixed-step 50 --seed $seed --split-seed 20260711 --retain-predictions --cache data/processed/provo_auxiliary_bert.pt --checkpoint-dir data/processed/strictline_fixed50 --output "data/processed/provo_auxiliary_strictline_fixed50_seed${seed}.json" }
```

Combine those five outputs for descriptive text-level reaggregation:

```bash
python scripts/analyze_provo_text_inference.py data/processed/provo_auxiliary_strictline_fixed50_seed101.json data/processed/provo_auxiliary_strictline_fixed50_seed202.json data/processed/provo_auxiliary_strictline_fixed50_seed303.json data/processed/provo_auxiliary_strictline_fixed50_seed404.json data/processed/provo_auxiliary_strictline_fixed50_seed505.json --output data/processed/provo_auxiliary_strictline_fixed50_text_inference.json --csv-output data/processed/provo_auxiliary_strictline_fixed50_text_inference.csv --bootstrap 100000 --seed 20260712
```

Convert all 12 official ZuCo NR files using the Bash or PowerShell loop in [zuco.md](zuco.md), then evaluate transfer and sensitivity:

```bash
python scripts/run_zuco_transfer.py --checkpoint-dir data/processed/strictline_fixed50 --cache data/processed/zuco_transfer_bert.pt --output data/processed/zuco_transfer_strictline_fixed50.json
python scripts/audit_zuco_criterion_uncertainty.py --output data/processed/zuco_strictline_criterion_uncertainty.json --csv-output data/processed/zuco_strictline_criterion_uncertainty.csv --checkpoint-dir data/processed/strictline_fixed50 --cache data/processed/zuco_transfer_bert.pt --reader-bootstraps 200 --text-bootstraps 200 --provo-subsets 200 --seed 20260711
python scripts/analyze_zuco_edge_threshold_sensitivity.py data/processed/zuco_transfer_strictline_fixed50.json --output data/processed/zuco_edge_threshold_sensitivity.json --csv-output data/processed/zuco_edge_threshold_sensitivity.csv --bootstrap 100000 --seed 20260713
```

Run every additional analysis consumed by the manuscript generator:

```bash
python scripts/analyze_half_specific_baseline_audit.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --output data/processed/provo_half_specific_baseline_audit.json --csv-output data/processed/provo_half_specific_baseline_audit.csv --repeats 100 --seed 20260711 --min-exposure 5
python scripts/analyze_target_selection_decomposition.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --output data/processed/provo_target_selection_decomposition.json --csv-output data/processed/provo_target_selection_decomposition.csv --repeats 100 --null-repeats 25 --seed 20260711
python scripts/analyze_residual_exposure_diagnostics.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --output data/processed/provo_residual_exposure_diagnostics.json --csv-output data/processed/provo_residual_exposure_diagnostics.csv --repeats 100 --seed 20260711
python scripts/analyze_cross_corpus_invariance.py --output data/processed/cross_corpus_measurement_invariance.json --csv-output data/processed/cross_corpus_measurement_invariance.csv --bootstrap 1000 --permutations 500 --seed 20260711
python scripts/analyze_text_influence.py --provo data/processed/provo_strictline_specification_curve.json --zuco data/processed/zuco_transfer_strictline_fixed50.json --output data/processed/text_influence_diagnostics.json --csv-output data/processed/text_influence_diagnostics.csv --resamples 10000
python scripts/audit_provo_word_reconciliation.py
python scripts/analyze_provo_reconciled_sensitivity.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --reconciliation data/processed/provo_word_position_reconciliation.json --original data/processed/provo_strictline_specification_curve.json --output data/processed/provo_reconciled_sensitivity.json --repeats 100 --seed 20260711 --min-exposure 5
```

The exploratory 50/100/200-step budget grid uses seeds 101, 202, and 303. Run `run_auxiliary_experiment.py` for each step/seed pair with matching `--steps`, `--fixed-step`, and a unique output such as `provo_auxiliary_strictline_budget_step50_seed101.json`, then summarize all nine files:

```bash
python scripts/summarize_provo_budget.py data/processed/provo_auxiliary_strictline_budget_step50_seed101.json data/processed/provo_auxiliary_strictline_budget_step50_seed202.json data/processed/provo_auxiliary_strictline_budget_step50_seed303.json data/processed/provo_auxiliary_strictline_budget_step100_seed101.json data/processed/provo_auxiliary_strictline_budget_step100_seed202.json data/processed/provo_auxiliary_strictline_budget_step100_seed303.json data/processed/provo_auxiliary_strictline_budget_step200_seed101.json data/processed/provo_auxiliary_strictline_budget_step200_seed202.json data/processed/provo_auxiliary_strictline_budget_step200_seed303.json --output data/processed/provo_auxiliary_strictline_budget_sensitivity.json --csv-output data/processed/provo_auxiliary_strictline_budget_learning_curves.csv
```

## Manuscript And Release Closure

After all aggregate results exist, generate every figure, table, source-data CSV, manifest, and provenance file, then build and verify the exact-member release attachment:

```bash
python -m pip install ".[figures]"
python scripts/generate_manuscript_assets.py --timestamp 2026-07-12T00:00:00+00:00
python scripts/verify_results.py --manuscript-assets
python scripts/build_compact_artifact_bundle.py
python scripts/verify_results.py --release-archive dist/Eyetrack2LLM-compact-artifacts.zip
python scripts/verify_results.py --full-local-results
python scripts/check_release.py --public
```

From a clean source checkout plus the downloaded release ZIP and its adjacent `compact_artifact_bundle_manifest.json`, one command verifies exact ZIP membership and hashes, extracts aggregate artifacts, regenerates manuscript assets, re-verifies provenance, and rebuilds the ZIP:

```bash
python scripts/rebuild_release_assets.py dist/Eyetrack2LLM-compact-artifacts.zip
```

On the present Windows/Python 3.12 workstation, the pinned BERT hidden-state caches were approximately 38 MiB for Provo and 10 MiB for ZuCo; the 20 trainable checkpoints were below 5 MiB total. Five fixed-50 Provo runs plus cache construction completed within tens of minutes on CPU, while the 200-draw nuisance-refit procedures can take substantially longer. Exact wall time varies by CPU, filesystem, and whether a validated cache already exists. The release excludes raw data, fixation tables, BERT caches, and checkpoints.

The outer Provo reader resampling keeps the 55 observed texts fixed and is a sensitivity analysis, not a joint reader-text confidence interval. Reader partitions and optimization seeds must not be treated as independent population samples.
