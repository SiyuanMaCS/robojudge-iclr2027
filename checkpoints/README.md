# Checkpoints

Large model weights are stored in the public Hugging Face checkpoint repository
instead of this GitHub-sized bundle:

- HF model repo: https://huggingface.co/HuggingFriends/robojudge-iclr2027-checkpoints

## Released Checkpoints

Final RoboJudge SFT+RL checkpoint used for the main submission result. The
released weights are a standard sharded BF16 inference copy of the verified
FP32 training checkpoint:

`HuggingFriends/robojudge-iclr2027-checkpoints/final_rl_200step`

The final checkpoint was verified at HF commit
`22df40134103728ffa125220c2e8b10abc494a72`. Its original FP32 weights are
also preserved at `HuggingFriends/RoboJudge-9B/model.safetensors`; the remote
SHA256 matches the source hash recorded in `CHECKPOINTS.json`.

The exact one-epoch SFT reference checkpoint is also public:

`HuggingFriends/robojudge-iclr2027-checkpoints/sft_reference_1epoch`

Both public directories contain standard sharded BF16 safetensors and were
verified against the file-level hashes in `CHECKPOINTS.json`.

Download the uploaded final checkpoint with:

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

Download only the one-epoch SFT reference with:

```bash
hf download HuggingFriends/robojudge-iclr2027-checkpoints \
  --repo-type model \
  --include 'sft_reference_1epoch/*' \
  --local-dir checkpoints_hf
```

Machine-readable details are in `checkpoints/CHECKPOINTS.json`.
