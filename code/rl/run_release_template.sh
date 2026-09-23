#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_STEPS="${MAX_STEPS:--1}"
SAVE_STEPS="${SAVE_STEPS:-50}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"

if [[ -z "${VERO_REPO:-}" || -z "${MODEL_CKPT:-}" || -z "${OUTPUT_DIR:-}" || -z "${DATA_PA:-}" || -z "${DATA_IA:-}" ]]; then
  echo "ERROR: set VERO_REPO, MODEL_CKPT, OUTPUT_DIR, DATA_PA, and DATA_IA" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${ROOT}/code/rl:${VERO_REPO}:${PYTHONPATH:-}"
accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16   --use_fsdp   --fsdp_version 1   --fsdp_sharding_strategy FULL_SHARD   --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP   --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer   --fsdp_backward_prefetch BACKWARD_PRE   --fsdp_state_dict_type FULL_STATE_DICT   --fsdp_use_orig_params true   --fsdp_cpu_ram_efficient_loading true   --fsdp_sync_module_states true   --fsdp_activation_checkpointing true   "${ROOT}/code/rl/train.py"   --repo "$VERO_REPO"   --data "$DATA_IA"   --data "$DATA_PA"   --ckpt "$MODEL_CKPT"   --out "$OUTPUT_DIR"   --shuffle   --max-steps "$MAX_STEPS"   --per-device-train-batch-size 2   --gradient-accumulation-steps 8   --steps-per-generation 1   --num-generations 8   --max-completion-length "$MAX_COMPLETION_LENGTH"   --learning-rate 1e-6   --optim adafactor   --save-steps "$SAVE_STEPS"   --save-only-model   --logging-steps 1   --attn-implementation sdpa
