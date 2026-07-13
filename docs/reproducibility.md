# Reproducibility

## Environment

Use Python 3.10 or newer and run commands from the repository root:

```bash
python -m pip install ".[test,analysis]"
python -m pytest
```

Tests require no corpus or model download. Analyses require official Provo or ZuCo inputs described in [data.md](data.md). Paths below are relative and work across platforms.

## Main Analysis

Convert Provo and reconstruct strict lines:

```bash
python scripts/convert_provo.py data/raw/Provo_Corpus-Eyetracking_Data.csv data/raw/Provo_Corpus-Predictability_Norms.csv data/processed/provo_fixations.csv data/processed/provo_words.json
```

Run simulation, independent-half reliability, and the specification curve:

```bash
python scripts/run_residual_recovery_simulation.py --output data/processed/residual_recovery_simulation.json --csv-output data/processed/residual_recovery_simulation.csv --replicates 80 --seed 20260711
python scripts/analyze_provo_independent_reliability.py --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --fixations data/processed/provo_fixations_with_lines.csv --output data/processed/provo_commoncore_strictline_independent_reliability.json --repeats 100 --seed 20260711 --risk-set common_forward_same_sentence_same_line
python scripts/analyze_provo_specification_curve.py --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --fixations data/processed/provo_fixations_with_lines.csv --output data/processed/provo_strictline_specification_curve.json --csv-output data/processed/provo_strictline_specification_curve.csv --repeats 100 --null-repeats 500 --seed 20260711
```

Run the fixed-step auxiliary experiment for seeds 101, 202, 303, 404, and 505, changing `--seed` and `--output` for each run:

```bash
python scripts/run_auxiliary_experiment.py --fixations data/processed/provo_fixations_with_lines.csv --main-csv data/raw/Provo_Corpus-Eyetracking_Data.csv --feature-set common_core --risk-set common_forward_same_sentence_same_line --steps 50 --seed 101 --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/provo_auxiliary_strictline_fixed50_seed101.json
```

Combine those five outputs for text-level inference:

```bash
python scripts/analyze_provo_text_inference.py data/processed/provo_auxiliary_strictline_fixed50_seed101.json data/processed/provo_auxiliary_strictline_fixed50_seed202.json data/processed/provo_auxiliary_strictline_fixed50_seed303.json data/processed/provo_auxiliary_strictline_fixed50_seed404.json data/processed/provo_auxiliary_strictline_fixed50_seed505.json --output data/processed/provo_auxiliary_strictline_fixed50_text_inference.json --csv-output data/processed/provo_auxiliary_strictline_fixed50_text_inference.csv --bootstrap 100000 --seed 20260712
```

Convert the official ZuCo NR files, then evaluate transfer and uncertainty:

```bash
python scripts/convert_zuco.py
python scripts/run_zuco_transfer.py --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/zuco_transfer_strictline_fixed50.json
python scripts/audit_zuco_criterion_uncertainty.py --output data/processed/zuco_strictline_criterion_uncertainty.json --csv-output data/processed/zuco_strictline_criterion_uncertainty.csv --checkpoint-dir data/processed/strictline_fixed50 --cache data/processed/zuco_transfer_bert.pt --reader-bootstraps 200 --text-bootstraps 200 --provo-subsets 200 --seed 20260711
```

Additional public diagnostics:

```bash
python scripts/analyze_target_selection_decomposition.py --help
python scripts/analyze_residual_exposure_diagnostics.py --help
```

Runtime and memory depend on corpus size and hardware and were not benchmarked. Reliability partitions and optimization seeds must not be treated as independent population samples.
