# Eyetrack2LLM

Eyetrack2LLM evaluates whether cross-fitted gaze-transition residual relations show fixed-split alignment and transport as candidate supervision for language models. The observational endpoint is destination allocation conditional on a retained transition already being forward, within-sentence, and within-line. It is not an intended saccade target, a first-pass-only measure, an unconditional next-fixation model, semantic distance, or causal effect; it can include rereading or regression-after-forward contexts.

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
```

Methods and scope are summarized in [methods.md](docs/methods.md) and [limitations.md](docs/limitations.md). The source repository includes the manuscript-asset generator, scientific verifier, and exact-member compact-archive builder. Releases through `v0.1.2` are historical; this scientific revision is `v0.2.0`, and its prepared aggregate attachment has no matching permanent DOI yet.

## Citation And License

Use [CITATION.cff](CITATION.cff) to cite the software. Repository-authored code and documentation are MIT licensed; third-party datasets retain their own terms.
