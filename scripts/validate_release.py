#!/usr/bin/env python3
"""Validate the portable RoboJudge release bundle."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    gold_path = ROOT / "data/test/final800.jsonl"
    gold_rows = [json.loads(line) for line in gold_path.open(encoding="utf-8") if line.strip()]
    if len(gold_rows) != 800 or len({row["item_id"] for row in gold_rows}) != 800:
        fail("canonical gold must contain 800 unique items")

    expected_train_counts = {
        "data/train/physical_adherence.json": 12351,
        "data/train/instruction_alignment.json": 11520,
    }
    for relative, expected_count in expected_train_counts.items():
        rows = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        if len(rows) != expected_count:
            fail(f"{relative} must contain {expected_count:,} records")

    with (ROOT / "metadata/manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    for entry in manifest:
        path = ROOT / entry["path"]
        if not path.is_file():
            fail(f"missing release file: {entry['path']}")
        if path.stat().st_size != int(entry["bytes"]):
            fail(f"size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            fail(f"SHA256 mismatch: {entry['path']}")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/evaluate.py"),
            str(ROOT / "results/final800/predictions/robojudge_submission.jsonl"),
            str(gold_path),
        ],
        check=True,
    )
    print(f"Validated {len(manifest)} release files and canonical dataset counts.")


if __name__ == "__main__":
    main()
