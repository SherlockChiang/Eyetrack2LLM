# Text Influence Diagnostics

## Frozen Design

This analysis asks whether the primary Provo reliability and ZuCo transfer conclusions are driven by individual texts. Every eligible text is deleted exactly once. No model is refit after deletion, no text is selected for exclusion, and no result is used to alter the frozen sample, specifications, thresholds, or claims.

The formal outputs are `data/processed/text_influence_diagnostics.json` and `.csv`. The JSON has `status: complete` and records units, estimands, seeds, resampling, thresholds, and decision rules. The CSV has 1,236 deletion rows: `4 x 3 x 55 = 660` Provo rows and `3 x 192 = 576` ZuCo rows.

## Provo Reliability

The specification-curve artifact schema was extended so every observed repeat contains compact `per_text_metrics`: 55 text keys with the three paired reliability metrics. The frozen `100` observed and `25` null analyses were rerun with seed `20260711` and unchanged data, split, folds, specifications, model, exposure rules, and negative control. The complete pre-rerun and post-rerun `summary` objects were exactly equal, not merely equal within floating-point tolerance.

The estimand is reproduced exactly: within each of 100 repeats take the median over retained texts, then take the median over repeats. Ranges below are across all 55 leave-one-text-out estimates.

| Specification | Metric | Full | LOTO range | Max abs change | Most influential |
|---|---|---:|---:|---:|---|
| position-only | edge weighted | 0.765110 | 0.762048-0.766799 | 0.003062 | 8 |
| position-only | source-equal flatten | 0.812892 | 0.811748-0.815164 | 0.002271 | 1 |
| position-only | per-source Fisher equal | 0.961924 | 0.961361-0.962541 | 0.000617 | 6 |
| lexical | edge weighted | 0.471656 | 0.468240-0.475493 | 0.003837 | 5 |
| lexical | source-equal flatten | 0.528009 | 0.525711-0.530548 | 0.002539 | 5 |
| lexical | per-source Fisher equal | 0.847732 | 0.842198-0.852229 | 0.005535 | 26 |
| syntax | edge weighted | 0.465266 | 0.462075-0.468070 | 0.003191 | 3 |
| syntax | source-equal flatten | 0.519321 | 0.514593-0.522121 | 0.004728 | 8 |
| syntax | per-source Fisher equal | 0.841873 | 0.839033-0.845985 | 0.004112 | 2 |
| flexible | edge weighted | 0.423124 | 0.419801-0.427721 | 0.004597 | 2 |
| flexible | source-equal flatten | 0.484830 | 0.481805-0.488086 | 0.003256 | 2 |
| flexible | per-source Fisher equal | 0.840557 | 0.837958-0.844100 | 0.003543 | 28 |

There is no sign reversal or crossing of the zero-reliability threshold in any of the 12 analyses. Thus the conclusion that reliability is positive under all four predefined specifications is not attributable to one Provo text. This does not make the reliability magnitude specification-invariant.

## ZuCo Transfer

The frozen unit is each comparison's 192 `per_text_seed_averaged_differences`. Delete-one means use the analytic identity `(sum(values) - deleted) / 191`. For each deletion, the artifact records a deterministic 10,000-draw descriptive text-reaggregation interval using derived seeds, plus the full-sample delete-one jackknife SE and descriptive normal interval. No sign-flip p value is reported because exchangeability is not established by the observational design.

| Contrast | Full | LOTO range | Max abs change | Most influential | Descriptive jackknife normal interval |
|---|---:|---:|---:|---|---:|
| gaze vs MLM | 0.021186 | 0.019188-0.024051 | 0.002865 | NR:112 | 0.001900-0.040473 |
| gaze vs shuffled | 0.033270 | 0.030874-0.035084 | 0.002396 | NR:212 | 0.016327-0.050214 |
| gaze vs position | 0.017632 | 0.015415-0.022357 | 0.004725 | NR:112 | -0.006762-0.042025 |

No contrast reverses sign. Across every deletion, the fixed-reader descriptive interval pattern also remains unchanged. This direct reaggregation sensitivity describes the fixed 12-reader, 192-text estimand only and does not resolve population-level reader-text uncertainty.

## Limits

LOTO diagnoses single-text influence only. It does not rule out influence from clusters of similar texts, corpus composition, readers, layout, parser errors, model choice, or omitted nuisance structure. Descriptive text-resampling intervals are Monte Carlo summaries with fixed 10,000-draw resolution. The diagnostics justify neither excluding the named texts nor re-estimating results after exclusion; the names identify maximum changes within an exhaustive fixed-result audit.
