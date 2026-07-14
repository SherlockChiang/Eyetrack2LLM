# Manuscript Assets

## Reproduction

```powershell
python -m pip install -e ".[test,figures]"
python scripts/generate_manuscript_assets.py
python scripts/verify_results.py --manuscript-assets
python -m pytest tests/test_manuscript_assets.py
```

Set `SOURCE_DATE_EPOCH` or pass `--timestamp` to make the informational provenance timestamp fixed. Plotted values, tables, source-data CSVs, and their ordering are deterministic independently of that timestamp. Matplotlib PDF metadata may vary across environments; provenance records the exact bytes produced by the current run.

## Output Design

- Figures use the Cognitive Science double-column width of 7.2 inches. Panels and labels remain readable when reduced; SVG and PDF are the submission formats and 180-dpi PNG files are previews.
- The palette uses the Okabe-Ito colorblind-safe blue (`#0072B2`), orange (`#E69F00`), green (`#009E73`), vermilion (`#D55E00`), and purple (`#CC79A7`), supplemented with neutral gray. Line type and marker shape carry distinctions in addition to color.
- Figure 1 explicitly marks external human construct validation as future and not run. It does not imply a completed human experiment.
- Figure 2 uses frozen cell means and 2.5th/97.5th replicate quantiles. Overdispersion is encoded by line style.
- Figure 3 plots all 100 observed split summaries with two-half-average NLL and uses violins/ranges for 500 one-split destination-label destruction controls balanced across the split bank. These are descriptive controls, not calibrated permutation inference.
- Figure 4 gives five separate Provo optimization-seed points, the three primary ZuCo text-level contrasts with descriptive text-resampling and reader-refit sensitivity intervals, and a compact panel of observable differences in cross-corpus measurement conditions.

## Provenance

`manuscript/artifact_manifest.csv` inventories the allowlisted aggregate inputs and generated outputs and their SHA256 values. `manuscript/artifact_provenance.json` records, for every figure and table, input paths and hashes, JSON field paths and transformations, output paths and hashes, the generation-script hash, and acquisition-document references. Raw Provo and ZuCo corpora are deliberately not hashed again; acquisition locations and source hashes are documented in `docs/data.md` and `docs/zuco.md`.

Every figure has a corresponding CSV in `manuscript/source_data/`. Tables are emitted as both Markdown and CSV. Captions are maintained in `manuscript/figure_captions.md`; all figure text is English ASCII.
