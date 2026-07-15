# Secondary Forward Destination-Allocation Decomposition

## Status And Scope

This analysis is `complete`, `post_hoc`, `secondary`, and `theory_guided`. Its categories and rules were frozen before inspecting its own diagnostic outcomes, but it was not prespecified. It does not modify or replace the frozen primary strict-line pipeline and was not used to select a new primary specification. The candidate universe remains forward, same-sentence, same-line destinations.

The categories describe **token separation** `d = destination token index - source token index`: adjacent `d=1`, near nonadjacent `d=2-3`, and far same-line `d>=4`. The artifact key `near_skip` is retained only as a historical machine field. A near-nonadjacent event is not evidence of first-pass word skipping because intervening words may have been fixated earlier; the categories are also not visual angle, true saccade amplitude, or target intention.

## Results

Values are medians across 100 randomized 42/42 splits of the same fixed 84-reader sample; brackets are partition IQRs. The table reports text-equal edge-weighted partition agreement. Source-equal flatten estimates are retained in the artifact and lead to the same ordering.

| Specification | Adjacent | Near skip | Far same-line |
|---|---:|---:|---:|
| Position-only | `.9109 [.9082, .9138]` | `.8533 [.8495, .8570]` | `.2161 [.2024, .2286]` |
| Lexical | `.7382 [.7328, .7465]` | `.6043 [.5918, .6136]` | `.1392 [.1291, .1530]` |
| Syntax | `.7327 [.7242, .7399]` | `.5965 [.5860, .6049]` | `.1319 [.1244, .1466]` |
| Flexible | `.7325 [.7238, .7401]` | `.5959 [.5869, .6064]` | `.1032 [.0964, .1125]` |

| Category | Candidate edges | Sources | Observed nonzero edges | Observed mass | Eligible edges |
|---|---:|---:|---:|---:|---:|
| Adjacent | 2,468 | 2,468 | 2,461 | 65,789 | 2,444 |
| Near skip | 4,182 | 2,214 | 3,884 | 56,877 | 4,179 |
| Far same-line | 10,119 | 1,743 | 1,863 | 3,813 | 10,115 |

All defined edge-weighted and source-equal observed medians exceed all 25 destination-label destruction controls. This is descriptive separation only because each control uses one split rather than reproducing the observed 100-split statistic; no p value is defined. Adjacent has exactly one candidate within each source/category. Its across-source edge correlation is defined, but source-equal within-source correlation and per-source Fisher aggregation are undefined (`0` defined split replicates) and are not imputed. Near-skip per-source Fisher values are close to one partly because most contributing sources have only two category candidates, making each defined two-point correlation nearly `+1` or `-1`; these values should not be treated as the main effect size.

## Interpretation

The position-only pattern agreement is concentrated in adjacent and near-nonadjacent transitions and is weak among far same-line candidates. It is not explained solely by an adjacent-versus-nonadjacent category contrast because near-nonadjacent agreement remains high after conditioning on that category. This describes stable destination-separation structure across the fixed reader sample. It does not confirm first-pass skipping, parafoveal processing, semantic integration, target selection, or any specific cognitive mechanism.

No category-level significance comparison is made from edges; text is the descriptive/inferential aggregation layer. Full-sample residual variance/quantiles and all six specification-pair identities within category are in `data/processed/provo_target_selection_decomposition.json`. Frozen Provo auxiliary JSONs and ZuCo transfer artifacts do not retain identifiable per-edge test predictions needed for category stratification, so neither model was approximated or retrained.

## Reproduction

```powershell
python scripts/analyze_target_selection_decomposition.py
python -m pytest
python scripts/verify_results.py
```
