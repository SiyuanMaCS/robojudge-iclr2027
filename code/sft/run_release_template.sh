#!/usr/bin/env bash
set -euo pipefail

# Portable SFT launch template. It keeps the release data paths relative.
# Required env vars:
#   MODEL_NAME_OR_PATH: base Qwen3.5-9B model path or HF id
#   OUTPUT_DIR: destination checkpoint directory
# Optional env vars:
#   DATASET_DIR: release data directory; defaults to ${ROOT}/data
#   LLAMAFACTORY_DIR: external LLaMAFactory checkout; when unset, use installed CLI from PATH
#   LLAMAFACTORY_CLI: llamafactory-cli executable
#   NPROC_PER_NODE: number of GPUs for LLaMAFactory torchrun; defaults to 1
#   CONFIG_OUT: rendered config path

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data}"
export DATASET_DIR
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CONFIG_TEMPLATE="${ROOT}/code/sft/train_release_template.yaml"
CONFIG_OUT="${CONFIG_OUT:-${ROOT}/code/sft/train_release_rendered.yaml}"

if [[ -z "${MODEL_NAME_OR_PATH:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "ERROR: set MODEL_NAME_OR_PATH and OUTPUT_DIR" >&2
  exit 2
fi

python3 - "$CONFIG_TEMPLATE" "$CONFIG_OUT" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding='utf-8').read()
text = text.replace('${MODEL_NAME_OR_PATH}', os.environ['MODEL_NAME_OR_PATH'])
text = text.replace('${OUTPUT_DIR}', os.environ['OUTPUT_DIR'])
text = text.replace('${DATASET_DIR}', os.environ['DATASET_DIR'])
open(dst, 'w', encoding='utf-8').write(text)
PY

if [[ -n "${LLAMAFACTORY_DIR:-}" ]]; then
  cd "$LLAMAFACTORY_DIR"
fi
exec "$LLAMAFACTORY_CLI" train "$CONFIG_OUT"
