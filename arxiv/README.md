# arXiv cognitive-theory draft

This directory is a self-contained, standard `article` source tree. It does not use a journal-specific class and does not require BibTeX because `main.tex` inputs the committed `main.bbl` directly.

## Compile

From this directory, run:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

On PowerShell the same commands work. `make` runs the same two passes when GNU Make and `pdflatex` are available. A portable Tectonic build can use `tectonic --keep-logs --outdir ../dist main.tex`; Tectonic may download its official TeX bundle cache outside this source directory.

## Remaining metadata

- The permanent Zenodo archive DOI will be inserted after the versioned release is deposited.

## Source list

- `main.tex`: complete manuscript and appendices.
- `main.bbl`: verified references, included directly for no-BibTeX compilation.
- `references.bib`: machine-readable copy of the same verified references.
- `tables/*.tex`: four main tables, full simulation grid, and supplementary Tables S1--S4.
- `figures/*.pdf`: four embedded vector figures.
- `Makefile`: two-pass pdfLaTeX build.

Generate the submission ZIP from the repository root with `python scripts/build_arxiv_bundle.py`. The generated archive has `main.tex` at its top level.

## Submission blocker

- The permanent Zenodo archive DOI remains to be inserted.
