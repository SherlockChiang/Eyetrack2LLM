# Eyetrack2LLM

Eyetrack2LLM evaluates fixed-sample agreement of cross-fitted gaze-transition residual patterns, fixed-split alignment with a constructed residual target, and cross-corpus scorer performance against a corpus-locally recalibrated criterion. The observational endpoint is destination allocation conditional on a retained transition already being forward, within-sentence, and within-line. It is not an intended saccade target, a first-pass-only measure, an unconditional next-fixation model, semantic distance, or causal effect; it can include rereading or regression-after-forward contexts.

## Install

Python 3.10 or newer is supported.

```bash
python -m pip install ".[test]"
python -m pytest
```

The test suite is self-contained and downloads no data, models, or checkpoints.

## Data

Provo and ZuCo participant data are not distributed here. Obtain them from their official sources and follow their licenses; see [data access](docs/data.md) and the [ZuCo conversion notes](docs/zuco.md). Raw, participant-level, and full processed analysis files belong under local `data/raw/` and `data/processed/` directories. Aggregate summaries are public only when explicitly included in a versioned release attachment; none are in the current `PUBLIC_FILES.txt` allowlist.

## Minimal Reproduction

Run commands from the repository root. The complete argument sets and expected local paths are in [reproducibility.md](docs/reproducibility.md).

```bash
python scripts/convert_provo.py --help
python scripts/run_residual_recovery_simulation.py --help
python scripts/analyze_provo_independent_reliability.py --help
python scripts/analyze_provo_specification_curve.py --help
python scripts/run_auxiliary_experiment.py --help
python scripts/analyze_provo_text_inference.py --help
python scripts/convert_zuco.py --help
python scripts/run_zuco_transfer.py --help
python scripts/audit_zuco_criterion_uncertainty.py --help
python scripts/audit_line_partition_identity.py --help
```

Methods and scope are summarized in [methods.md](docs/methods.md) and [limitations.md](docs/limitations.md). Release [`v0.3.2`](https://github.com/SherlockChiang/Eyetrack2LLM/releases/tag/v0.3.2) standardizes publication figure typography and layout, removes plot-area label collisions, and preserves the complete appendix simulation grid. Its immutable source archive is [10.5281/zenodo.21369962](https://doi.org/10.5281/zenodo.21369962), under concept DOI [10.5281/zenodo.21322671](https://doi.org/10.5281/zenodo.21322671). Release attachments provide the verified aggregate-result bundle, arXiv source, compiled PDF/log, and exact-member manifests without participant-level data, checkpoints, or caches.

## Citation And License

Use [CITATION.cff](CITATION.cff) to cite the software. Repository-authored code and documentation are MIT licensed; third-party datasets retain their own terms.
