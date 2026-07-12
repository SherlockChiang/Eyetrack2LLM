# Eyetrack2LLM

Eyetrack2LLM tests whether cross-fitted gaze-transition residual relations are reproducible, learnable, and transportable as cognitive supervision for language models. The estimand is destination allocation conditional on the next retained transition being forward, within sentence, and within line. It is not an unconditional next-fixation model, semantic distance, or causal effect.

## Install

Python 3.10 or newer is supported.

```bash
python -m pip install ".[test]"
python -m pytest
```

The test suite is self-contained and downloads no data, models, or checkpoints.

## Data

Provo and ZuCo participant data are not distributed here. Obtain them from their official sources and follow their licenses; see [data access](docs/data.md) and the [ZuCo conversion notes](docs/zuco.md). Local raw and processed files belong under `data/raw/` and `data/processed/`.

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

Methods and scope are summarized in [methods.md](docs/methods.md) and [limitations.md](docs/limitations.md). The immutable [Zenodo v0.1.2 archive](https://doi.org/10.5281/zenodo.21322672) preserves the historical release.

## Citation And License

Use [CITATION.cff](CITATION.cff) to cite the software. Repository-authored code and documentation are MIT licensed; third-party datasets retain their own terms.
