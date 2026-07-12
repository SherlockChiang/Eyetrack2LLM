# ZuCo 1.0 Conversion

Obtain the ZuCo 1.0 natural-reading files from the [official OSF project](https://osf.io/q3zws/). The corpus is CC BY 4.0; retain attribution and identify modifications. Raw MATLAB files and converted participant data are local-only.

```bash
python scripts/convert_zuco.py data/raw/resultsZJS_NR.mat data/processed/zuco_zjs_nr_fixations.csv data/processed/zuco_zjs_nr_report.json data/processed/zuco_zjs_nr_words.json
```

The converter recovers fixation order from each word's MATLAB 1-based `fixPositions`. It emits a fixation only when exactly one word claims it, preserves original order, converts 500 Hz duration samples to milliseconds, and reports missing, conflicting, invalid, and out-of-range entries. Downstream extraction requires consecutive original fixation order so excluded fixations are not bridged.

Word bounds are used only for audit. Geometry never overrides the official mapping because shared boundaries, preprocessing, and calibration can make geometric assignment ambiguous. Sentence IDs preserve the source array slot, and word indices in outputs are zero-based.

For the full natural-reading cohort, convert all 12 official subject files before running transfer. The strict transfer analysis uses source-conditioned forward transitions on the common sentence range:

```bash
python scripts/run_zuco_transfer.py --checkpoint-dir data/processed/strictline_fixed50 --output data/processed/zuco_transfer_strictline_fixed50.json
python scripts/audit_zuco_criterion_uncertainty.py --output data/processed/zuco_strictline_criterion_uncertainty.json --csv-output data/processed/zuco_strictline_criterion_uncertainty.csv --checkpoint-dir data/processed/strictline_fixed50 --cache data/processed/zuco_transfer_bert.pt --reader-bootstraps 200 --text-bootstraps 200 --provo-subsets 200 --seed 20260711
```

ZuCo differs from Provo in participants, texts, task, and layout. A transfer result therefore does not isolate corpus effects, and failure to transfer is not evidence of universal non-transportability.
