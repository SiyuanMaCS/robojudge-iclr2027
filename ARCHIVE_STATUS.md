# Archive Status

This file records the off-machine copies needed after the original training
workspace is retired.

## Source repositories

- Paper source: https://github.com/yqi19/Siyuan-RoboJudge
- Release bundle: https://github.com/SiyuanMaCS/robojudge-iclr2027
- HF release bundle: https://huggingface.co/datasets/HuggingFriends/robojudge-iclr2027-reviewer-bundle
- HF video assets: https://huggingface.co/datasets/HuggingFriends/mllm-as-embodied-world-judge
- HF checkpoints: https://huggingface.co/HuggingFriends/robojudge-iclr2027-checkpoints
- Original FP32 final model: https://huggingface.co/HuggingFriends/RoboJudge-9B

The GitHub release, HF release bundle, video dataset, and model repositories are
public.

## Verified assets

- All 12,351 training video paths and all 800 benchmark video/frame pairs
  referenced by the release metadata exist in the HF video-assets repository.
- Canonical public metadata is stored under `robojudge_release/` at HF dataset
  commit `2a1a8d9f2e7e2bd2f2a6b924128d21cbe5329ef6`.
  - PA training SHA256: `def35bf16596a175a370268743a726d538b40a561b14ee0390cdbd969831d738`
  - IA training SHA256: `f43491bbcd9abee88135715bc040af9ebc9fcbec38ea317fa7bc3a167b93d582`
  - Final800 SHA256: `65315b722a3dff92d88e370e351d4518f0ca6ab42d74e4fd69bee50047e66bef`
  - Supplementary ZIP SHA256: `b138316d79a4361ed96f2c446fffbc5c872fde42d8da8f2839d4442c405b22f4`
- `data/test/final800.jsonl` is the reviewed label set used by the paper.
- `results/final800/predictions/robojudge_submission.jsonl` is the final
  RoboJudge prediction set used by the paper.
- `code/evaluate.py` reproduces PA 0.644617, IA 0.775028, and pooled Overall
  0.719211 for that prediction set.
- `code/bootstrap_ci.py` deterministically reproduces the reported paired
  cluster-bootstrap confidence intervals.
- The exact LLaMAFactory 0.9.6.dev0 source snapshot and a paper-source archive
  are stored in the HF release bundle under `archives/`.
  - HF archive commit: `4d1b64db55b18b1440a0b1161951c53a6b61c8c9`
  - Paper commit: `09b9837b964206d599b284ab3d4084a1bd9bd2ee`
  - Paper archive SHA256: `e7088219428beb967b361e28efbfb4921ac9370a8ee69ef5ac1102c8883ef67d`
  - LLaMAFactory archive SHA256: `29e5cd8f3f2a2debe7522af7ee0b11fa5107e301646a84e5f3d263c6e4b3145b`

## Checkpoints

See `checkpoints/CHECKPOINTS.json` for file-level SHA256 checksums and upload
status. The final RL release is stored as standard sharded BF16 safetensors;
the original FP32 checkpoint is independently preserved in `RoboJudge-9B`, and
its remote SHA256 matches the source hash retained in the manifest. The BF16
copy and the one-epoch SFT checkpoint were verified at HF commit
`22df40134103728ffa125220c2e8b10abc494a72`. Both checkpoint directories are
public and match the file-level SHA256 manifest.
