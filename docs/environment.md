# Environment

This is a verified direct-version record, not a complete transitive lock file. The environment used for the final repository audit on 2026-07-12 reported:

```text
numpy==2.5.1
scipy==1.18.0
torch==2.13.0
transformers==5.13.1
spacy==3.8.14
en-core-web-sm==3.8.0
thinc==8.3.13
confection==1.3.3
wordfreq==3.1.1
matplotlib==3.11.0
pytest==9.1.1
```

These versions were read from the active Python 3.12 environment. They do not claim bit-for-bit environment reconstruction, and `torch` availability can vary by platform and package index. The project metadata specifies supported ranges except where an analysis dependency is deliberately fixed.

## Installation

Core development and CI:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

Full analysis and figures:

```bash
python -m pip install -e ".[test,analysis,figures]"
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
python -m pytest
```

The official `en_core_web_sm` 3.8.0 wheel has SHA-256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`. The spaCy model is intentionally not a core test dependency. Its real-parser integration test skips with a stated reason when spaCy or `en_core_web_sm` is unavailable. All other core failures remain failures.
