# ZuCo 1.0 conservative conversion

## Source and license

- Corpus: Zurich Cognitive Language Processing Corpus (ZuCo 1.0), natural reading task.
- Official project: <https://osf.io/q3zws/>.
- License reported by the OSF API: Creative Commons Attribution 4.0 International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>. Redistribution must retain attribution and identify modifications.
- Local input: `data/raw/resultsZJS_NR.mat`.
- SHA-256: `EE156114B0A8F48007A74DE02B13A72A0611AF1590B0062EEB2A95BD36EA2D1C`.

GECO was rejected for this transition prototype because its commonly distributed word-level tables provide aggregated reading measures rather than the unambiguous, complete fixation-order-to-word mapping required here. Inferring a scanpath from aggregates would manufacture order and regressions. ZuCo exposes `allFixations` and each word's MATLAB 1-based `fixPositions`, so the original sequence can be recovered without that guess.

## Conversion policy

Run:

```powershell
python scripts/convert_zuco.py data/raw/resultsZJS_NR.mat data/processed/zuco_zjs_nr_fixations.csv data/processed/zuco_zjs_nr_report.json data/processed/zuco_zjs_nr_words.json
```

The converter loads MATLAB data with `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)`. Subject and task must either be explicit or match the strict `results<SUBJECT>_<TASK>.mat` filename. Sentence IDs preserve the original array slot (`NR:<1-based slot>`). ZJS has text-only placeholders in slots 1-50: their `word`, `allFixations`, and `wordbounds` fields are empty, so they are reported missing and valid output begins at `NR:51`.

Only a fixation claimed by exactly one word's `fixPositions` is emitted. Word indices are 0-based, fixation order remains MATLAB 1-based, and duration is converted from 500 Hz samples with `duration_ms = duration_samples * 2`. Outside, invalid, out-of-range, and conflicting fixations are excluded without renumbering; downstream extraction must use `require_consecutive_order=True` so gaps are not bridged.

`wordbounds` rows are interpreted as `[left, top, right, bottom]` with closed boundaries (`left <= x <= right`, `top <= y <= bottom`). Closed shared edges can yield multiple geometric candidates. Geometry is an audit only: a small center displacement never causes automatic reassignment. The official `fixPositions` mapping remains authoritative.

## ZJS/NR audit

Conversion report:

| Measure | Count |
|---|---:|
| sentence slots | 300 |
| valid sentences | 250 |
| missing sentences | 50 |
| all fixations | 4,500 |
| uniquely mapped/written | 3,249 |
| outside (no word claim) | 1,251 |
| duplicate mapping conflicts | 0 |
| out-of-range positions | 0 |
| mapped centers outside claimed bounds | 1,146 |
| `nFixations` mismatches | 0 |
| invalid durations | 0 |
| unique geometric candidates | 3,426 |
| no geometric candidate | 1,067 |
| multiple geometric candidates | 7 |

Using `extract_events(..., require_consecutive_order=True)` yields 2,999 transitions: 2,089 forward, 457 regression, and 453 refixation events. All 250 valid sentences have at least one written fixation. Per-sentence unique-word coverage has minimum 7.69%, median 50.00%, mean 48.09%, and maximum 90.91%.

## Limitations

- This is one participant. Split-half or inter-participant reliability is not identifiable and was deliberately not run.
- The 1,146 bounds mismatches are substantial. They may reflect coordinate conventions, preprocessing, calibration drift, or AOI assignment rules. They make geometry-based remapping unsafe, but do not by themselves invalidate the explicit word-level `fixPositions` relation.
- Excluding 1,251 outside fixations breaks scanpaths into shorter contiguous segments. This is preferable to inventing transitions, but reduces coverage.
- ZuCo sentences can be multi-line. The unified CSV currently has no `line_id`, so events are categorized as forward/regression/refixation rather than line-return.
- More subjects are required before using this corpus for stable transition targets or cross-corpus replication.

## Full natural-reading cohort

All 12 official ZuCo 1.0 NR subject files were downloaded and verified against their OSF SHA-256 values. ZJS lacks slots 1-50 and ZPH lacks slots 51-100; the other ten subjects contain all 300 slots. Across all available data there are 51,826 forward, 18,657 regression, and 15,962 refixation events.

The primary equal-half analysis uses all 12 subjects and the common 200 sentences `NR:101-300`. Each sentence is split 1,000 times into 6/6 subjects, then sentence medians are summarized across texts:

| Event | Row-normalized raw reliability, median [IQR] |
|---|---:|
| Forward | 0.754 [0.706, 0.798] |
| Regression | 0.499 [0.430, 0.592] |
| Refixation | 1.000 [1.000, 1.000], structurally degenerate |

Forward reliability rises monotonically with aggregation size on the same 200 sentences: 0.613 at 4 subjects, 0.663 at 6, 0.701 at 8, 0.731 at 10, and 0.757 at 12. Regression union-observed edge correlation remains unstable (`-0.071`), so the stable cross-corpus target is restricted to source-conditioned forward transitions.

Full output is stored in `data/processed/zuco_nr_full_reliability.json`.

## Superseded transfer result

The formerly reported positive transfer table from `data/processed/zuco_zero_shot_transfer.json` is invalid for primary use. Its softmax risk set retained backward candidates and terminal source groups, so its positive comparisons are superseded and must not be quoted as evidence. The retained strict result is the later common-core fixed-step-50 analysis below; it does not support transfer.

## Subject uncertainty analyses

`python scripts/analyze_provo_independent_reliability.py` estimates Provo 42/42 reliability with separate five-fold text-crossfit common-core nuisance fits in each half and a pooled-baseline control. Raw residuals are never clipped, a source is eligible only at exposure 5 in both halves, and the JSON reports edge-weighted, source-equal weighted-flatten, and per-source Fisher-equal correlations. Output is `data/processed/provo_commoncore_independent_reliability.json`; use `--repeats 20` only as an explicitly labelled pilot.

`python scripts/analyze_zuco_subject_stability.py` keeps the fixed common-core step-50 checkpoints unchanged while rebuilding ZuCo five-fold baselines and raw residuals for all 12 leave-one-subject-out targets and balanced random subsets at k=4,6,8,10,12. The primary exposure threshold is `max(4, round(10*k/12))`; gaze-minus-MLM, shuffled, and position effects are text-equal Fisher-z differences. Output is `data/processed/zuco_commoncore_subject_stability.json`; reducing `--subsets-per-k` marks the result as a pilot.

## Four-subject pilot

Run `python scripts/analyze_zuco_pilot.py`. The script reads the ZJS, ZDN, ZDM, and ZAB processed files in memory and writes `data/processed/zuco_4subject_pilot.json`. Because ZJS lacks slots 1-50, all reliability analyses use only the 250 common sentences `NR:51` through `NR:300`. Across that range, `content`, `words`, and `bounds` are exactly identical for all four subjects.

Events require consecutive original fixation order and are grouped by subject and text. Counts over the common 250 sentences are:

| Subject | mapped fixations | events | forward | regression | refixation | official mapping coverage | bounds mismatch / mapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZJS | 3,249 | 2,999 | 2,089 | 457 | 453 | 72.20% | 35.27% |
| ZDN | 3,865 | 3,616 | 2,083 | 1,201 | 332 | 95.04% | 25.39% |
| ZDM | 4,631 | 4,384 | 2,584 | 1,373 | 427 | 94.82% | 33.74% |
| ZAB | 5,052 | 4,802 | 3,482 | 660 | 660 | 89.10% | 65.99% |

Reliability enumerates the three unique unordered 2-vs-2 partitions exactly. The table reports sentence-level medians and IQRs; features are the common flattened matrix cells per sentence.

| Type / normalization | valid sentences | raw half correlation, median [IQR] | Spearman-Brown, median [IQR] | common features, median [IQR] |
|---|---:|---:|---:|---:|
| forward / row | 245 | 0.439 [0.317, 0.527] | 0.610 [0.482, 0.690] | 171 [84, 336] |
| forward / global | 245 | 0.332 [0.248, 0.430] | 0.498 [0.398, 0.602] | 361 [196, 784] |
| regression / row | 238 | 0.296 [-0.025, 0.468] | 0.457 [-0.052, 0.637] | 38.5 [24, 75.75] |
| regression / global | 245 | 0.129 [-0.007, 0.208] | 0.229 [-0.013, 0.345] | 361 [196, 784] |
| refixation / row | 177 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 25 [16, 38] |
| refixation / global | 232 | 0.335 [0.105, 0.557] | 0.502 [0.190, 0.715] | 400 [196, 784] |

The refixation/row value is degenerate rather than evidence of perfect agreement: a refixation row has only its diagonal destination, so every nonempty normalized row equals one. Sentence-level subject-pair transition-distribution JSD is also high: median 0.814 bits for forward and 1.000 bit for both regression and refixation (valid comparisons 1,482, 1,338, and 926). The JSON additionally includes an overall descriptive correlation formed by flattening each sentence separately and concatenating those vectors; sentence edge positions remain distinct, so different sentence nodes are never merged.

Four participants are not enough to estimate a stable population target. The broad partition/sentence variability, weak regression agreement, high JSD, and mapping diagnostics support downloading the remaining eight subjects before making target-quality claims. The full 12-subject sample is worthwhile for a better-powered reliability estimate, but it should not be assumed in advance to rescue sparse sentence-level event types.

## Full 12-subject reliability

Run `python scripts/analyze_zuco_full.py`. The analysis reads all processed files for ZAB, ZDM, ZDN, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW, ZMG, and ZPH and writes `data/processed/zuco_nr_full_reliability.json`. Metadata `content`, `words`, and `bounds` are identical whenever a sentence is jointly present. ZJS lacks NR 1-50 and ZPH lacks NR 51-100; the other ten participants contain all 300 sentences. Across all available subject/sentence records there are 51,826 forward, 18,657 regression, and 15,962 refixation events.

The primary statistic is sentence-level split-half Pearson correlation under row normalization. A row enters only when both halves have source exposure; all destination cells in that row remain candidates. Each sentence uses 1,000 deterministic random splits, the split correlations are reduced to a sentence median, and the table summarizes those sentence medians. Global normalization includes every matrix cell and therefore can be inflated by sparse shared zeros. Correlation restricted to the union of observed edges is included only as a selection-biased description in the JSON. Spearman-Brown is exact only for equal halves and is explicitly marked as an approximation for the 11-person 5/6 split.

| Set | Participants / sentences | forward row raw | regression row raw | refixation row raw |
|---|---|---:|---:|---:|
| A | 11 excluding ZPH; NR 51-300 | 0.736 [0.691, 0.780] | 0.487 [0.395, 0.581] | 1.000 [1.000, 1.000] |
| B | all 12; NR 101-300 | 0.754 [0.706, 0.798] | 0.499 [0.430, 0.592] | 1.000 [1.000, 1.000] |
| C sensitivity | 10 excluding ZJS/ZPH; NR 1-300 | 0.729 [0.678, 0.774] | 0.500 [0.402, 0.576] | 1.000 [1.000, 1.000] |

The refixation row result remains structurally degenerate: each exposed source row has only its diagonal destination, so a normalized nonempty row is necessarily one. It must not be interpreted as empirical perfect agreement. In set B, union-observed raw count correlations are 0.564 for forward, -0.071 for regression, and 0.404 for refixation, reinforcing that sparse regression edge identity is not stable even though row-normalized source-conditioned distributions show moderate agreement.

The subject-count sensitivity uses only the common NR 101-300 range. Each of 100 replicates is one random subject subset and one equal split, summarized across all 200 sentences; repeats are not treated as independent observations.

| Subjects | forward row raw median [IQR] |
|---:|---:|
| 4 | 0.613 [0.573, 0.645] |
| 6 | 0.663 [0.616, 0.687] |
| 8 | 0.701 [0.683, 0.720] |
| 10 | 0.731 [0.716, 0.749] |
| 12 | 0.757 [0.749, 0.762] |

The monotonic sample-size curve shows a real aggregation benefit and supports constructing a useful **forward row-normalized auxiliary target** from the full sample. It does not establish a stable sentence-level regression-edge target, and refixation-row reliability is uninformative. Bounds mismatch rates are substantial and heterogeneous (about 25%-66% of mapped fixations), although official mapping coverage is generally high except for ZJS; the conservative official `fixPositions` policy remains necessary.

For modeling, ZuCo is suitable as an auxiliary transfer or replication domain, not as evidence that a Provo-trained target transfers unchanged. Provo-to-ZuCo auxiliary transfer is worth running for forward transitions with corpus-specific normalization and held-out ZuCo evaluation. Regression should be exploratory or down-weighted, and refixation should use a non-degenerate target/statistic rather than row-normalized reliability.

## Provo zero-shot transfer

**Current primary result (strict-line):** recovered line layouts constrain candidate construction, counts, cross-fitted baseline residuals, training, and evaluation in both corpora. `data/processed/zuco_transfer_strictline_fixed50.json` contains 18,247 pairs, 3,880 source groups, and 193 valid texts. Text-equal Fisher-z gaze-minus-MLM is `0.01275` (95% CI `[-0.00890, 0.03289]`, p=`.2441`), gaze-minus-shuffled is `0.02114` (`[0.00723, 0.03531]`, p=`.0035`), and gaze-minus-position is `0.00909` (`[-0.01075, 0.02822]`, p=`.3660`). Gaze exceeds only shuffle, so transfer is not supported.

The analyses below document the correction history and are superseded for the paper's primary result.

The previous `data/processed/zuco_zero_shot_transfer.json` result is invalidated: its softmax risk sets incorrectly retained backward candidates and terminal source groups. It is superseded by `data/processed/zuco_zero_shot_transfer_fixed50_forwardrisk.json`.

The corrected corpus-specific result is also not positive at the text unit: gaze-minus-MLM is `-0.0003` (95% CI `[-0.0213, 0.0185]`, p=`.976`) and gaze-minus-shuffled is `0.0096` (`[-0.0039, 0.0230]`, p=`.166`; 194 texts). The earlier common-core result in `zuco_transfer_commoncore_fixed50.json` also used 194 valid texts: gaze-minus-MLM `-0.0022` (`[-0.0352, 0.0278]`, p=`.895`), gaze-minus-shuffled `0.0148` (`[0.0065, 0.0235]`, p=`.0002`), and gaze-minus-position `0.0048` (`[-0.0243, 0.0300]`, p=`.741`). These values are retained as superseded history, not the paper's primary table.

`python scripts/run_zuco_transfer.py --checkpoint-dir data/processed/provo_checkpoints_fixed50 --output data/processed/zuco_zero_shot_transfer_fixed50_forwardrisk.json` evaluates the five fixed-step-50 Provo checkpoint triplets on clean BERT final hidden states for the all-12-subject common range NR 101-300. ZuCo words are forced tokens for spaCy and exact pre-tokenized words for BERT. Metadata uses log cleaned length, `wordfreq`, punctuation, sentence number 1, and explicitly missing cloze values. The complete syntax design is filtered to `dst > src` before count construction, cross-fit baseline fitting, residual construction, and target evaluation. Thus every source softmax contains only forward candidates and terminal words have no group.

The non-semantic baseline is fit by deterministic five-fold sentence cross-fitting: each fit uses 160 sentences and predicts 40, so a sentence's outcomes never predict their own baseline. Source exposure must be at least 10. Evaluation uses unclipped raw Pearson residuals. No ZuCo value trains or selects a Provo checkpoint: all three conditions use the pre-registered step 50. Overall raw correlations are descriptive; primary comparisons give each text equal weight, require at least four edges, average correlations on the Fisher-z scale, and report text bootstrap and sign-flip inference. Real gaze must exceed both MLM-only and shuffled controls to count as relation transfer; even that would not establish improved MLM efficiency. Steps 150, 250, and 300 remain sensitivity analyses rather than selection candidates.

## Frozen representation and boundary sensitivity

`python scripts/run_zuco_enhancements.py` produces two pre-specified analyses from the common-core fixed-step-50 checkpoints. `fresh_probe_representation.json` discards each Provo gaze head, freezes the adapter, and trains a fresh rank-8 directed probe on each fold's 160 ZuCo sentences. Held-out targets never enter probe fitting, target baselines are out-of-fold, scaling uses only the probe training fold, and evaluation is correlation against raw residuals. This is supervised ZuCo probing of representation quality, not zero-shot transfer. Epochs, optimizer settings, initialization, folds, and schedule are identical across conditions.

`zuco_transfer_sensitivity.json` is a historical common-core sensitivity using target-side masks. It is superseded by `zuco_transfer_strictline_fixed50.json`, in which vertical-interval overlap defines same-line identity before candidate/count construction in ZuCo and Provo is trained on its reconstructed same-line risk set. The historical file must not be described as the current primary analysis.
