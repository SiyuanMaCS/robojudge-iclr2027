# Checkpoints

Large model weights are stored in the private Hugging Face checkpoint repository instead of this GitHub-sized bundle:

- HF model repo: https://huggingface.co/HuggingFriends/robojudge-iclr2027-checkpoints

## Released Checkpoints

Final RoboJudge SFT+RL checkpoint used for the main submission result:

`HuggingFriends/robojudge-iclr2027-checkpoints/final_rl_200step`

SFT reference checkpoint used for the SFT ablation/reproduction:

`HuggingFriends/robojudge-iclr2027-checkpoints/sft_reference_epoch2_step374`

Download both checkpoints with:

```bash
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --local-dir checkpoints_hf
```

Download only the final RoboJudge checkpoint with:

```bash
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --include 'final_rl_200step/*' \
  --local-dir checkpoints_hf
```

Machine-readable details are in `checkpoints/CHECKPOINTS.json`.
