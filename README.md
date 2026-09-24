# RoboJudge ICLR 2027 Release Bundle

Clean GitHub release bundle for the RoboJudge ICLR submission. The root keeps final data, judge outputs, evaluation metrics, and RoboJudge reproduction code.

Reviewer mirror: <https://huggingface.co/datasets/HuggingFriends/robojudge-iclr2027-reviewer-bundle>.

Large checkpoints: <https://huggingface.co/HuggingFriends/robojudge-iclr2027-checkpoints>.

Paper repository reference: <https://github.com/yqi19/Siyuan-RoboJudge.git>, branch `main`, confirmed paper commit `bcc4078`.

`prompt.txt` records the exact PA/IA inference prompts used by `code/inference.py`.

## Data

- `data/train/physical_adherence.json`
- `data/train/instruction_alignment.json`
- `data/test/final800.jsonl`
- `data/dataset_info.json`

## Final800 Results

- `results/final800/predictions/` - judge result JSONL files used by the paper table.
- `results/final800/predictions/parts/` - numbered chunks for the two prediction files that exceed GitHub's 100 MB single-file limit.
- `results/final800/predictions/parts_manifest.tsv` - expected size, SHA256, and part list for the split files.
- `results/final800/manifest.csv` - model name, rows, sha256, and release-relative prediction path.
- `results/final800/metrics/metrics.tsv` - recomputed table metrics.

Reconstruct the split prediction files after cloning with:

```bash
bash scripts/join_prediction_parts.sh
```

## RoboJudge Code

- `code/inference.py` and `code/run_inference.sh`
- `code/sft/train.yaml`, `code/sft/run_release_template.sh`
- `code/rl/train.py`, `code/rl/run_release_template.sh`
- `code/evaluate.py`
- `scripts/join_prediction_parts.sh`

SFT reproduction expects LLaMAFactory to be installed externally; this bundle keeps only the RoboJudge-specific config and launch template.

See `REPRODUCIBILITY.md` for reviewer-oriented setup, checkpoint download, inference, and evaluation commands.

## Checkpoints

- `checkpoints/README.md`
- `checkpoints/CHECKPOINTS.json`

Final RL 200-step and SFT reference checkpoints are referenced there. Model weights are not copied into this GitHub-sized bundle; they are stored in the HF checkpoint repo above.

Large model checkpoints and raw run directories are intentionally excluded from git.
