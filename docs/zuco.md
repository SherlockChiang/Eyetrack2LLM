# ZuCo 1.0 Conversion

Obtain the ZuCo 1.0 `results<SUBJECT>_NR.mat` files from the [official OSF project](https://osf.io/q3zws/). This file family is Task 2, normal reading of biographical Wikipedia relation-extraction sentences, distinct from the sentiment normal-reading and task-specific relation-search paradigms. Complete sentences were displayed at the reader's own pace; the source study used multiple-choice content questions after 68 of the full task's 300 sentences. The corpus is CC BY 4.0; retain attribution and identify modifications. Raw MATLAB files and converted participant data are local-only.

The 12 analyzed `.mat` files were reacquired from the official OSF project on 2026-07-15 and verified against the SHA-256 values returned by the OSF API. `data/processed/line_partition_identity_audit.json` records every official file ID, URL, byte size, and SHA-256. Raw files remain local-only.

```bash
python scripts/convert_zuco.py data/raw/resultsZJS_NR.mat data/processed/zuco_zjs_nr_fixations.csv data/processed/zuco_zjs_nr_report.json data/processed/zuco_zjs_nr_words.json
```

The converter recovers fixation order from each word's MATLAB 1-based `fixPositions`. It emits a fixation only when exactly one word claims it, preserves original order, converts 500 Hz duration samples to milliseconds, and reports missing, conflicting, invalid, and out-of-range entries. Downstream extraction requires consecutive original fixation order so excluded fixations are not bridged.

Official `fixPositions` determine fixation-to-word assignment. Word bounds reconstruct line identity used by downstream same-line candidate construction; they never override the official word mapping because shared boundaries, preprocessing, and calibration can make geometric assignment ambiguous. The completed audit found that all 11 nonreference participant layouts exactly matched the reference bounds for the 4,423 words in `NR:101`--`NR:300`; line-partition discrepancies were zero. Analysis entry points fail closed unless all 12 participants have matching word sequences and derived word-to-line partitions for every analyzed text. Sentence IDs preserve the `NR` file family and original one-based source-array slot, and word indices in outputs are zero-based. The analyzed range is a contiguous common-complete 200-slot subset; it is not a newly assigned sentence numbering scheme. The first 100 task slots are outside this complete-case subset. No retained-versus-excluded text-property audit is available, so cross-corpus claims are scoped to `NR:101`--`NR:300`, not the full 300-sentence Task-2 collection.

For the full natural-reading cohort, convert all 12 official subject files before running transfer. The strict transfer analysis uses source-conditioned forward transitions on the common sentence range:

```bash
subjects=(ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH)
for subject in "${subjects[@]}"; do lower=$(printf '%s' "$subject" | tr '[:upper:]' '[:lower:]'); python scripts/convert_zuco.py "data/raw/results${subject}_NR.mat" "data/processed/zuco_${lower}_nr_fixations.csv" "data/processed/zuco_${lower}_nr_report.json" "data/processed/zuco_${lower}_nr_words.json" --subject "$subject" --task NR; done
python scripts/run_zuco_transfer.py --checkpoint-dir data/processed/strictline_fixed50 --cache data/processed/zuco_transfer_bert.pt --output data/processed/zuco_transfer_strictline_fixed50.json
python scripts/audit_zuco_criterion_uncertainty.py --output data/processed/zuco_strictline_criterion_uncertainty.json --csv-output data/processed/zuco_strictline_criterion_uncertainty.csv --checkpoint-dir data/processed/strictline_fixed50 --cache data/processed/zuco_transfer_bert.pt --reader-bootstraps 200 --text-bootstraps 200 --provo-subsets 200 --seed 20260711
python scripts/analyze_zuco_edge_threshold_sensitivity.py data/processed/zuco_transfer_strictline_fixed50.json --output data/processed/zuco_edge_threshold_sensitivity.json --csv-output data/processed/zuco_edge_threshold_sensitivity.csv --bootstrap 100000 --seed 20260713
```

PowerShell conversion:

```powershell
$subjects = 'ZAB','ZDM','ZDN','ZGW','ZJM','ZJN','ZJS','ZKB','ZKH','ZKW','ZMG','ZPH'; foreach ($subject in $subjects) { $lower = $subject.ToLower(); python scripts/convert_zuco.py "data/raw/results${subject}_NR.mat" "data/processed/zuco_${lower}_nr_fixations.csv" "data/processed/zuco_${lower}_nr_report.json" "data/processed/zuco_${lower}_nr_words.json" --subject $subject --task NR }
```

ZuCo differs from Provo in participants, texts, task, and layout. A transfer result therefore does not isolate corpus effects, and failure to transfer is not evidence of universal non-transportability.
