#!/usr/bin/env python3
"""Regenerate deterministic SHA256 manifests for release files."""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "metadata/manifest.csv"
JSON_PATH = ROOT / "metadata/manifest.json"
EXCLUDED = {CSV_PATH.relative_to(ROOT).as_posix(), JSON_PATH.relative_to(ROOT).as_posix()}


def tracked_and_untracked_files():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    for raw in sorted(filter(None, output.split(b"\0"))):
        relative = raw.decode("utf-8")
        if relative in EXCLUDED or "/__pycache__/" in f"/{relative}/":
            continue
        path = ROOT / relative
        if path.is_file():
            yield relative, path


def describe(relative, path):
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            lines += block.count(b"\n")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "lines": lines,
    }


def main():
    entries = [describe(relative, path) for relative, path in tracked_and_untracked_files()]
    JSON_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "bytes", "sha256", "lines"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(entries)
    print(f"wrote {len(entries)} entries")


if __name__ == "__main__":
    main()
