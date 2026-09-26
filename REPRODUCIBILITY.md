# Reproducibility Guide

This bundle is the complete open reproducibility release for RoboJudge. It
contains training and inference code, final benchmark labels, released
prediction JSONL files, metrics, prompts, metadata, and checkpoint links.

## Repositories

- GitHub release bundle: https://github.com/SiyuanMaCS/robojudge-iclr2027
- HF release bundle mirror: https://huggingface.co/datasets/HuggingFriends/robojudge-iclr2027-reviewer-bundle
- HF checkpoint repo: https://huggingface.co/HuggingFriends/robojudge-iclr2027-checkpoints
- HF video assets: https://huggingface.co/datasets/HuggingFriends/mllm-as-embodied-world-judge

All release repositories listed above are public.

## Environment

Create the release environment from the checked-out repository:

```bash
conda env create -f env/environment-release.yml
conda activate robojudge-iclr2027-release
```

If conda is not available, `requirements.txt` lists the Python package set used by the release bundle. GPU inference requires a CUDA/PyTorch stack compatible with the target machine.

## Data And Predictions

The reviewed final evaluation labels used by the paper are in:

```text
data/test/final800.jsonl
```

The training JSON files use paths rooted at `data_root/`. Point `DATA_ROOT` in
the inference and training launchers to a local checkout of the HF video-assets
repository. All 12,351 training-video references and all 800 test video/frame
pairs were verified against that repository before archival.
The released files contain 12,351 PA records and 11,520 IA records; the 831
verified real demonstrations are included in the PA file.

Most prediction files are stored directly in `results/final800/predictions/`. Two large baseline prediction files are split into numbered parts to stay below GitHub's single-file size limit. Reconstruct them after cloning with:

```bash
bash scripts/join_prediction_parts.sh
```

The split manifest is:

```text
results/final800/predictions/parts_manifest.tsv
```

## Evaluate Released Predictions

The lightweight evaluator takes prediction JSONL first and gold JSONL second. For the final RoboJudge result reported in the paper:

```bash
python code/evaluate.py \
  results/final800/predictions/robojudge_submission.jsonl \
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

Regenerate the repository-wide SHA256 manifest after changing release files:

```bash
python scripts/generate_release_manifest.py
```

Reproduce the paired source-clip bootstrap confidence intervals with:

```bash
python code/bootstrap_ci.py \
  --output results/final800/metrics/bootstrap_ci.json
```

The script uses 10,000 paired replicates by default, treats each
`dataset/task/episode` as a source-clip cluster, stratifies resampling by source
corpus, and preserves the paper's pooled PA+IA definition of Overall Pearson
correlation.

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

Download the exact one-epoch SFT reference checkpoint with:

```bash
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --include 'sft_reference_1epoch/*' \
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

SFT templates are under `code/sft/`; RL templates are under `code/rl/`. The
final SFT run uses one epoch, and the release RL launcher defaults to 200 steps.
These files preserve the RoboJudge-specific configuration and launch shape. Large raw
training workspaces, intermediate checkpoints, and cluster-specific paths are
intentionally excluded from the release bundle.

The exact LLaMAFactory `0.9.6.dev0` source snapshot used by SFT is included at
`third_party/llamafactory-0.9.6.dev0/` and mirrored as
`archives/llamafactory-0.9.6.dev0-robojudge-snapshot.tar.gz` in the HF release
bundle. Both copies include the Apache-2.0 license. Install the local snapshot with:

```bash
pip install ./third_party/llamafactory-0.9.6.dev0
```

## Archived Paper Source

The HF release bundle also stores
`archives/Siyuan-RoboJudge-paper-09b9837.tar.gz`, a source snapshot of the paper
at commit `09b9837b964206d599b284ab3d4084a1bd9bd2ee`.

## Validate The Archive

Run the local integrity and canonical-metric checks with:

```bash
python scripts/validate_release.py
```
