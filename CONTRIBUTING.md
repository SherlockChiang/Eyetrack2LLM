# Contributing to Eyetrack2LLM

Contributions that improve correctness, reproducibility, documentation, or test coverage are welcome. For substantial methodological or scientific changes, open an issue before investing in an implementation.

## Development Setup

Use Python 3.10 or newer in an isolated environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

Install `.[analysis]` and the `en_core_web_sm` model only for analysis and real-parser integration work. Install `.[figures]` when changing generated manuscript assets. See `docs/environment.md` for verified versions.

## Changes And Tests

- Keep changes focused and preserve the existing public API unless a breaking change is justified.
- Follow the surrounding Python style, use descriptive names, and add concise comments only where intent is not evident.
- Add or update tests for behavioral changes. Run `python -m pytest` before opening a pull request.
- Run `python scripts/verify_results.py` and `python scripts/check_release.py --public` when changing public repository, release, data, or citation files.
- Do not regenerate or alter scientific outputs merely to make tests pass. Explain intentional result changes and include exact provenance.

## Issues And Pull Requests

Use the issue templates for reproducible bugs and scoped feature requests. Pull requests should state the motivation, implementation, tests run, and any effect on data, estimands, claims, or generated artifacts. Keep unrelated changes separate.

## Data And Privacy

Do not commit raw or processed corpus data, participant-level records, checkpoints, embeddings, caches, credentials, or restricted material. Contributors must obtain Provo and ZuCo independently and comply with their licenses and participant-data terms. Use synthetic fixtures in tests.

## Artifacts And Provenance

New public artifacts must identify their generating command, inputs, versions, random seeds, and hashes where practical. Generated bundles belong in local `dist/` and should be attached to a release only after review. Never include raw data in an artifact bundle.

`PUBLIC_FILES.txt` is the release boundary. Review `python scripts/stage_public.py` before using its explicit `--stage` option; do not stage the working tree wholesale.

## Scientific Integrity

Do not select results, seeds, texts, specifications, thresholds, or figures because they support a preferred conclusion. Report predefined analyses and relevant failures, distinguish exploratory from confirmatory work, preserve uncertainty, and document all exclusions or post hoc decisions.

By contributing, you agree that your contributions are licensed under the repository's MIT License and that you will follow the Code of Conduct.
