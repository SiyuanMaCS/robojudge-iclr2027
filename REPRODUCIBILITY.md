# Reproducibility Guide

This bundle is intended for reviewer reproduction of the RoboJudge ICLR submission. It contains code, final800 labels, released prediction JSONL files, metrics, prompts, metadata, and links to the large checkpoints.

## Repositories

- GitHub release bundle: https://github.com/SiyuanMaCS/robojudge-iclr2027
- HF reviewer bundle mirror: https://huggingface.co/datasets/HuggingFriends/robojudge-iclr2027-reviewer-bundle
- HF checkpoint repo: https://huggingface.co/HuggingFriends/robojudge-iclr2027-checkpoints

The HF repos are private for submission review; grant reviewer access before sharing the final links.

## Environment

Create the release environment from the checked-out repository:

```bash
conda env create -f env/environment-release.yml
conda activate robojudge-iclr2027-release
```

If conda is not available, `requirements.txt` lists the Python package set used by the release bundle. GPU inference requires a CUDA/PyTorch stack compatible with the target machine.

## Data And Predictions

The final evaluation labels are in:

```text
data/test/final800.jsonl
```

Most prediction files are stored directly in `results/final800/predictions/`. Two large baseline prediction files are split into numbered parts to stay below GitHub's single-file size limit. Reconstruct them after cloning with:

```bash
bash scripts/join_prediction_parts.sh
```

The split manifest is:

```text
results/final800/predictions/parts_manifest.tsv
```

## Evaluate Released Predictions

The lightweight evaluator takes prediction JSONL first and gold JSONL second. For the final RoboJudge result:

```bash
python code/evaluate.py \
  results/final800/predictions/robojudge_rl_latest.jsonl \
  data/test/final800.jsonl
```

For the SFT reference checkpoint prediction:

```bash
python code/evaluate.py \
  results/final800/predictions/robojudge_sft_latest.jsonl \
  data/test/final800.jsonl
```

Table-ready metrics used by the release are also stored in:

```text
results/final800/metrics/metrics.tsv
results/final800/metrics/metrics.json
```

## Download Checkpoints

Install or update the Hugging Face CLI, then download the model repo:

```bash
pip install -U huggingface_hub
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --local-dir checkpoints_hf
```

Download only the final SFT+RL checkpoint:

```bash
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --include 'final_rl_200step/*' \
  --local-dir checkpoints_hf
```

Download only the SFT reference checkpoint:

```bash
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --include 'sft_reference_epoch2_step374/*' \
  --local-dir checkpoints_hf
```

Checkpoint metadata is in `checkpoints/CHECKPOINTS.json`.

## Run Inference

The release inference wrapper expects a local checkpoint path and local video files. The repository includes labels, metadata, prompts, and prediction outputs; it does not vendor the raw video files.

```bash
CKPT=checkpoints_hf/final_rl_200step \
DATA_ROOT=/path/to/final800/video/root \
OUT=outputs/robojudge_rl_latest.jsonl \
bash code/run_inference.sh
```

For multi-GPU or array jobs, set `SHARD` and `NUM_SHARDS`:

```bash
CKPT=checkpoints_hf/final_rl_200step \
DATA_ROOT=/path/to/final800/video/root \
OUT=outputs/part-000.jsonl \
SHARD=0 NUM_SHARDS=8 \
bash code/run_inference.sh
```

Concatenate shard outputs before running `code/evaluate.py`.

## Training Reproduction

SFT templates are under `code/sft/`; RL templates are under `code/rl/`. These files preserve the RoboJudge-specific configuration and launch shape. Large raw training workspaces, intermediate checkpoints, and cluster-specific paths are intentionally excluded from this reviewer bundle.
