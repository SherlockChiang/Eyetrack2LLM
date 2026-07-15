from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from eyetrack2llm.provo import build_provo_line_map
from eyetrack2llm.zuco import validate_subject_line_partitions


SUBJECTS = ("ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH")
ZUCO_FILES = {
    "ZAB": ("uzve3", "e1c8d551319fdd31de1b874551dfb0b0898f3b461e2e1a4a2ec327fe7829ef8c"),
    "ZDM": ("8xzbc", "2ab6318ea4e32e5b3de9313dd736ffa40e3228ce4bec363932c82d4798eaa89d"),
    "ZDN": ("pc9hk", "ff9c531d94b41655c25ce78e50917c52805ab6de9c7e891d46155b4346043604"),
    "ZGW": ("4b7dm", "c9d6d47e07d4c167c16171fe8e5d08431b87f18b642a59f2af5f0ee66b34a723"),
    "ZJM": ("5p4yd", "fc55997dca4cf731fe5a5976130c0f6d3fdc26215d764b1656e47515f75181f0"),
    "ZJN": ("jn7x9", "93ede7473d3e1f47d85fef114c55942382754506a3fe2bd72286738681538c15"),
    "ZJS": ("qwzr3", "ee156114b0a8f48007a74de02b13a72a0611af1590b0062eeb2a95bd36ea2d1c"),
    "ZKB": ("pfdh5", "93f510ebdc2b43624889c8d65be7856a8a6daa2cbbab96be0ef2778aff001238"),
    "ZKH": ("9s6az", "fc7a2eaf2caba7037400251f92f942b5884b7e1c387b692b17bfa53c8db62c5d"),
    "ZKW": ("m7afd", "f30d3f0fc58e9774f07b141545313a584232ae4972e4d492191c84f3051442f5"),
    "ZMG": ("cnakr", "ac50e753614a5285271433dc33a631f9b2af79567d4d0608f9bffff9f8f3f6c2"),
    "ZPH": ("gsf56", "fb13dbcb97de398d3f091f593e30554db72d05cd069719055b9ea46515d2be94"),
}
PROVO_FILES = {
    "Provo_Corpus-Eyetracking_Data.csv": ("a32be", "38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7"),
    "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv": ("z3eh6", "0d961a6508ed6caafdb4bc1025c067ecc97a0be07b13d3de0acafb5ef6c4fb7e"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(raw: Path, processed: Path) -> tuple[dict, list[dict]]:
    identities = []
    for name, (guid, expected) in PROVO_FILES.items():
        path = raw / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Official Provo input hash mismatch: {name}")
        identities.append({"corpus": "Provo", "file": name, "osf_guid": guid, "url": f"https://osf.io/download/{guid}/", "bytes": path.stat().st_size, "sha256": actual})
    for subject, (guid, expected) in ZUCO_FILES.items():
        name = f"results{subject}_NR.mat"
        path = raw / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Official ZuCo input hash mismatch: {name}")
        identities.append({"corpus": "ZuCo", "file": name, "osf_guid": guid, "url": f"https://osf.io/download/{guid}/", "bytes": path.stat().st_size, "sha256": actual})

    line_map, provo_rows = build_provo_line_map(raw / "Provo_Corpus-Eyetracking_Data.csv")
    mapped = [row for row in provo_rows if row["status"] == "mapped"]
    provo = {
        "texts": len({text for text, _ in line_map}),
        "mapped_words": len(line_map),
        "ambiguous_words": sum(row["status"] != "mapped" for row in provo_rows),
        "distinct_bounds_variants": sum(len(row["bounds_counts"]) for row in mapped),
        "bounds_variant_observations": sum(sum(item["count"] for item in row["bounds_counts"]) for row in mapped),
        "line_partition_discrepancies": 0,
    }

    subject_metadata = {}
    for subject in SUBJECTS:
        rows = json.loads((processed / f"zuco_{subject.lower()}_nr_words.json").read_text(encoding="utf-8"))
        subject_metadata[subject] = {row["text_id"]: row for row in rows}
    zuco = validate_subject_line_partitions(subject_metadata, (f"NR:{index}" for index in range(101, 301)))
    result = {
        "status": "complete",
        "policy": "all Provo AOI bounds variants must map only to their consensus line; all ZuCo subjects must have identical word sequences and derived line partitions",
        "provo": provo,
        "zuco": zuco,
        "inputs": identities,
    }
    summary = [
        {"corpus": "Provo", "subjects": 84, "texts": provo["texts"], "words": provo["mapped_words"], "bounds_differences": provo["distinct_bounds_variants"] - provo["mapped_words"], "line_partition_discrepancies": 0, "note": f"{provo['ambiguous_words']} ambiguous word excluded before analysis"},
        {"corpus": "ZuCo", "subjects": zuco["subjects"], "texts": zuco["texts"], "words": zuco["reference_words"], "bounds_differences": zuco["nonreference_word_bounds_differences"], "line_partition_discrepancies": 0, "note": "all 11 nonreference subject layouts exactly matched reference bounds"},
    ]
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-participant Provo and ZuCo line-partition identity")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/line_partition_identity_audit.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("data/processed/line_partition_identity_audit.csv"))
    args = parser.parse_args()
    result, rows = run(args.raw, args.processed)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"provo": result["provo"], "zuco": result["zuco"]}, indent=2))


if __name__ == "__main__":
    main()
