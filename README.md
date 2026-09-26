# RoboJudge

RoboJudge provides human-aligned evaluation for embodied video generation along
two axes: **Physical Adherence (PA)** and **Instruction Alignment (IA)**. This
repository is the complete open release of benchmark metadata, training data,
training and inference code, prompts, model outputs, metrics, and reproducibility
artifacts.

- Project page: <https://siyuanmacs.github.io/robojudge-iclr2027/>
- Video dataset: <https://huggingface.co/datasets/HuggingFriends/mllm-as-embodied-world-judge>
- Full release mirror: <https://huggingface.co/datasets/HuggingFriends/robojudge-iclr2027-reviewer-bundle>
- RoboJudge-9B: <https://huggingface.co/HuggingFriends/RoboJudge-9B>
- BF16 checkpoints: <https://huggingface.co/HuggingFriends/robojudge-iclr2027-checkpoints>
- Supplementary examples: [download ZIP](assets/robojudge_supplementary_material.zip)

## Release contents

```text
code/                       training, inference, evaluation, and bootstrap code
data/train/                 released PA and IA training records
data/test/final800.jsonl    final expert-adjudicated benchmark labels
results/final800/           released predictions, metrics, and manifests
checkpoints/                model file manifests and download instructions
third_party/                exact LLaMAFactory source snapshot used for SFT
assets/                     project-page image and supplementary sample ZIP
```

## Data

The released training files contain:

- `data/train/physical_adherence.json`: 12,351 records, comprising 11,520
  generated videos and 831 verified successful real demonstrations.
- `data/train/instruction_alignment.json`: 11,520 generated-video records.
- `data/test/final800.jsonl`: 800 benchmark videos with final PA/IA labels and
  six diagnostic sub-scores.

The referenced videos are hosted in the public
[Hugging Face dataset](https://huggingface.co/datasets/HuggingFriends/mllm-as-embodied-world-judge).
The 22 MB supplementary ZIP contains 10 labeled test examples covering all 8
source corpora and 10 video-generator families.

## Quick start

Create the release environment:

```bash
conda env create -f env/environment-release.yml
conda activate robojudge-iclr2027-release
```

Download the final BF16 checkpoint:

```bash
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --include 'final_rl_200step/*' \
  --local-dir checkpoints_hf
```

Run inference:

```bash
CKPT=checkpoints_hf/final_rl_200step \
DATA_ROOT=/path/to/mllm-as-embodied-world-judge \
OUT=outputs/robojudge_predictions.jsonl \
bash code/run_inference.sh
```

Evaluate the released submission predictions:

```bash
python code/evaluate.py \
  results/final800/predictions/robojudge_submission.jsonl \
  data/test/final800.jsonl
```

This reproduces PA $r=0.644617$, IA $r=0.775028$, and pooled Overall
$r=0.719211$.

## Training

The final SFT configuration is in `code/sft/train.yaml`; a portable launcher is
provided in `code/sft/run_release_template.sh`. The final run used one epoch,
full-parameter SFT, a global batch size of 64, and a learning rate of $10^{-5}$.

RL code is under `code/rl/`. The release launcher defaults to 200 GRPO steps,
matching the final checkpoint. The exact LLaMAFactory `0.9.6.dev0` source used
for SFT is vendored under `third_party/llamafactory-0.9.6.dev0/`.

## Results and integrity

- `results/final800/metrics/metrics.tsv`: table-ready metrics.
- `results/final800/metrics/bootstrap_ci.json`: paired cluster-bootstrap CIs.
- `metadata/manifest.csv` and `metadata/manifest.json`: repository-wide file
  sizes and SHA256 hashes.
- `checkpoints/CHECKPOINTS.json`: checkpoint-level sizes and SHA256 hashes.

Reconstruct the two prediction files split for GitHub's single-file size limit:

```bash
bash scripts/join_prediction_parts.sh
```

Regenerate and validate the release:

```bash
python scripts/generate_release_manifest.py
python scripts/validate_release.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete workflow and
[ARCHIVE_STATUS.md](ARCHIVE_STATUS.md) for the verified off-machine asset map.

## License

The RoboJudge code in this repository is released under the Apache License 2.0.
Model and dataset use are additionally subject to the licenses and terms listed
on their respective Hugging Face cards and the licenses of upstream source
datasets. The vendored LLaMAFactory snapshot retains its original Apache-2.0
license.
