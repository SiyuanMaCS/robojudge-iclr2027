#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRED_DIR="$ROOT_DIR/results/final800/predictions"
MANIFEST="$PRED_DIR/parts_manifest.tsv"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r file size_bytes sha256 parts; do
  [[ -n "$file" ]] || continue
  out="$PRED_DIR/$file"
  : > "$out"
  IFS=',' read -ra part_names <<< "$parts"
  for part in "${part_names[@]}"; do
    cat "$PRED_DIR/parts/$part" >> "$out"
  done

  actual_size="$(stat -c '%s' "$out")"
  if [[ "$actual_size" != "$size_bytes" ]]; then
    echo "Size mismatch for $file: got $actual_size expected $size_bytes" >&2
    exit 1
  fi

  actual_sha="$(sha256sum "$out" | awk '{print $1}')"
  if [[ "$actual_sha" != "$sha256" ]]; then
    echo "SHA256 mismatch for $file: got $actual_sha expected $sha256" >&2
    exit 1
  fi

  echo "Reconstructed $file"
done
