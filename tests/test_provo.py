import csv
import json

from eyetrack2llm.provo import build_provo_line_map, cluster_vertical_intervals, convert_provo_fixations


def _write(path, fieldnames, rows):
    with path.open("w", encoding="cp1252", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_provo_conversion_is_conservative(tmp_path):
    main = tmp_path / "main.csv"
    fixations = tmp_path / "fix.csv"
    output = tmp_path / "output.csv"
    report_path = tmp_path / "report.json"
    _write(
        main,
        [
            "RECORDING_SESSION_LABEL",
            "Participant_ID",
            "TRIAL_INDEX",
            "Text_ID",
            "IA_ID",
            "Word_Number",
            "IA_TOP", "IA_BOTTOM",
        ],
        [
            {"RECORDING_SESSION_LABEL": "r1", "Participant_ID": "s1", "TRIAL_INDEX": "1", "Text_ID": "8", "IA_ID": "1", "Word_Number": "NA", "IA_TOP": "70", "IA_BOTTOM": "154"},
            {"RECORDING_SESSION_LABEL": "r1", "Participant_ID": "s1", "TRIAL_INDEX": "1", "Text_ID": "8", "IA_ID": "2", "Word_Number": "2", "IA_TOP": "70.2", "IA_BOTTOM": "154.1"},
            {"RECORDING_SESSION_LABEL": "r1", "Participant_ID": "s1", "TRIAL_INDEX": "1", "Text_ID": "8", "IA_ID": "3", "Word_Number": "3", "IA_TOP": "154", "IA_BOTTOM": "246"},
        ],
    )
    fields = [
        "RECORDING_SESSION_LABEL",
        "TRIAL_INDEX",
        "page",
        "CURRENT_FIX_INTEREST_AREA_INDEX",
        "CURRENT_FIX_INDEX",
        "CURRENT_FIX_DURATION",
    ]
    _write(
        fixations,
        fields,
        [
            dict(zip(fields, ["r1", "1", "8", "1", "1", "100"])),
            dict(zip(fields, ["r1", "1", "8", ".", "2", "120"])),
            dict(zip(fields, ["r1", "1", "8", "2", "3", "130"])),
            dict(zip(fields, ["r1", "2", "8", "2", "1", "140"])),
        ],
    )
    report = convert_provo_fixations(main, fixations, output, report_path=report_path)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["word_index"] for row in rows] == ["0", "1"]
    assert [row["fixation_order"] for row in rows] == ["1", "3"]
    assert report.written_rows == 2
    assert report.excluded_outside_aoi == 1
    assert report.excluded_missing_trial == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["written_rows"] == 2


def test_vertical_intervals_cluster_overlap_not_integer_top():
    assert cluster_vertical_intervals([(70.0, 154.0), (70.4, 153.8), (154.1, 246.0)]) == [0, 0, 1]


def test_ambiguous_bounds_exclude_fixations(tmp_path):
    main, fixations, output = tmp_path / "main.csv", tmp_path / "fix.csv", tmp_path / "out.csv"
    fields = ["RECORDING_SESSION_LABEL", "Participant_ID", "TRIAL_INDEX", "Text_ID", "IA_ID", "Word_Number", "IA_TOP", "IA_BOTTOM"]
    _write(main, fields, [dict(zip(fields, ["r1", "s1", "1", "18", "2", "2", "70", "154"])),
                          dict(zip(fields, ["r2", "s2", "1", "18", "2", "2", "246", "330"]))])
    fix_fields = ["RECORDING_SESSION_LABEL", "TRIAL_INDEX", "page", "CURRENT_FIX_INTEREST_AREA_INDEX", "CURRENT_FIX_INDEX", "CURRENT_FIX_DURATION"]
    _write(fixations, fix_fields, [dict(zip(fix_fields, ["r1", "1", "18", "2", "1", "100"]))])
    report = convert_provo_fixations(main, fixations, output)
    assert report.written_rows == 0
    assert report.excluded_missing_line == 1


def test_majority_bounds_cannot_hide_a_different_line_partition(tmp_path):
    main = tmp_path / "main.csv"
    fields = ["RECORDING_SESSION_LABEL", "Participant_ID", "TRIAL_INDEX", "Text_ID", "IA_ID", "Word_Number", "IA_TOP", "IA_BOTTOM"]
    _write(main, fields, [
        dict(zip(fields, ["r1", "s1", "1", "1", "1", "1", "0", "10"])),
        dict(zip(fields, ["r2", "s2", "1", "1", "1", "1", "0", "10"])),
        dict(zip(fields, ["r3", "s3", "1", "1", "1", "1", "20", "30"])),
        dict(zip(fields, ["r1", "s1", "1", "1", "2", "2", "0", "10"])),
        dict(zip(fields, ["r2", "s2", "1", "1", "2", "2", "0", "10"])),
        dict(zip(fields, ["r3", "s3", "1", "1", "2", "2", "0", "10"])),
        dict(zip(fields, ["r1", "s1", "1", "1", "3", "3", "20", "30"])),
        dict(zip(fields, ["r2", "s2", "1", "1", "3", "3", "20", "30"])),
        dict(zip(fields, ["r3", "s3", "1", "1", "3", "3", "20", "30"])),
    ])
    try:
        build_provo_line_map(main)
    except ValueError as error:
        assert "line-partition variant mismatch" in str(error)
    else:
        raise AssertionError("minority cross-line AOI variant was not rejected")
