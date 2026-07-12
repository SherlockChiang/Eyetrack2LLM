# Eyetrack2LLM

Eyetrack2LLM tests when cross-fitted gaze-transition residual relations are reproducible, learnable, useful, and transportable as cognitive supervision for language models.

The estimand is destination allocation conditional on the next retained transition being forward, within sentence, and within line. It is not an unconditional next-fixation model or semantic distance. Simulation and corpus results show why reliability is necessary but insufficient evidence for cognitive supervision; see the [paper](manuscript/manuscript.md) and [limitations](docs/limitations.md) for the bounded claims.

## Install And Verify

Python 3.10 or newer is supported.

```bash
python -m pip install -e ".[test]"
python -m pytest
python scripts/verify_results.py
python scripts/check_release.py --public
```

These commands require no corpus download, model download, checkpoint, or generated `data/processed` directory. The default verifier checks frozen source data, tables, provenance, and the self-contained arXiv source. Full local result verification is available after obtaining the restricted inputs:

```bash
python scripts/verify_results.py --full-local-results
```

## Reproduction

- [Methods](docs/methods.md)
- [Data acquisition and licensing](docs/data.md)
- [Environment](docs/environment.md)
- [Complete commands](docs/reproducibility.md)
- [ZuCo conversion](docs/zuco.md)

Raw and processed participant-level corpora, model checkpoints, caches, and generated bundles are not distributed in Git. High-cost manuscript regeneration is documented but is not run in CI.

## Artifacts

- [Release v0.1.0](https://github.com/SherlockChiang/Eyetrack2LLM/releases/tag/v0.1.0) contains the compact processed-results archive used for archive-level hash verification.
- [`arxiv/`](arxiv/) is the complete, self-contained paper source.
- [`manuscript/`](manuscript/) contains frozen PNG figures, source data, tables, and SHA-256 provenance.

## Citation And License

Use [`CITATION.cff`](CITATION.cff) to cite the software and accompanying paper. Repository-authored code and documentation are MIT licensed; third-party datasets retain their own terms.
