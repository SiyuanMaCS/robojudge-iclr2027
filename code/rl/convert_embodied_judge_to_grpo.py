#!/usr/bin/env python3
"""Convert embodied_world_judge_sft_mixed.json (SFT chat format) into veRL RL
JSONL format so it can be used to smoke-test GRPO training.

The source data is a video-judging SFT dataset: given an instruction + video
frames, the assistant outputs JSON like
    {"reasoning": "...", "instruction_alignment": 5}
or
    {"reasoning": "...", "physical_adherence": 1}

For RL we drop the assistant turn (the policy will generate it) and instead:
  - Extract the single numeric score (1-5) from the assistant JSON as the
    ground-truth "answer".
  - Keep the user turn's original "Output JSON only: {...}" instruction
    UNCHANGED (the policy must still output raw JSON, e.g.
    {"reasoning": "...", "physical_adherence": <1-5>}). A dedicated reward
    function (vero_reward/embodied_judge_reward.py) checks this exact format.
  - Emit reward_model.ground_truth / extra_info.metric fields consumed by
    that reward function.

Usage:
    python scripts/convert_embodied_judge_to_grpo.py \
        --input /mnt/pengruotian.prt/datasets/embodied_world_judge_sft_mixed.json \
        --output-dir /mnt/pengruotian.prt/vero/vero-rl/data \
        --val-ratio 0.02
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

_SCORE_KEY_PATTERN = re.compile(r'"(?P<key>[a-z_]+)"\s*:\s*(?P<value>\d+)\s*[},]')

# qwen_vl_utils defaults are fps=2.0, min_frames=4, max_frames=768; these clips
# are only a few seconds long so 32 is plenty and keeps the mm-token budget in check.
# Override via --video-fps / --video-min-frames / --video-max-frames.
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_VIDEO_MIN_FRAMES = 4
DEFAULT_VIDEO_MAX_FRAMES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to embodied_world_judge_sft_mixed.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val JSONL files")
    parser.add_argument("--train-filename", default="embodied_judge_grpo_train.jsonl")
    parser.add_argument("--val-filename", default="embodied_judge_grpo_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Fraction of samples randomly picked for val (ignored if --val-count is set); train always keeps all converted samples")
    parser.add_argument("--val-count", type=int, default=None, help="Fixed number of samples randomly picked for val; train always keeps all converted samples (val may overlap train)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples (debugging)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS, help="Frames per second to sample from each video")
    parser.add_argument("--video-min-frames", type=int, default=DEFAULT_VIDEO_MIN_FRAMES)
    parser.add_argument("--video-max-frames", type=int, default=DEFAULT_VIDEO_MAX_FRAMES)
    return parser.parse_args()


def extract_score(assistant_content: str) -> tuple[str, int] | None:
    """Return (metric_name, score) parsed from the assistant JSON, or None if unparsable."""
    try:
        obj = json.loads(assistant_content)
    except (json.JSONDecodeError, TypeError):
        return None
    for key, value in obj.items():
        if key == "reasoning":
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return key, int(value)
    return None


def convert_sample(
    raw: dict,
    record_id: str,
    *,
    video_fps: float,
    video_min_frames: int,
    video_max_frames: int,
) -> dict | None:
    messages = raw.get("messages") or []
    if len(messages) < 3:
        return None
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    if user_msg is None or assistant_msg is None:
        return None

    score_info = extract_score(assistant_msg.get("content", ""))
    if score_info is None:
        return None
    metric_name, score = score_info
    if not (1 <= score <= 5):
        return None

    prompt_messages = []
    if system_msg is not None:
        prompt_messages.append({"role": "system", "content": system_msg.get("content", "")})
    prompt_messages.append({"role": "user", "content": user_msg.get("content", "")})

    images = list(raw.get("images") or [])
    # verl's process_video() requires each entry to be a dict with a "video"
    # key (plain path strings raise NotImplementedError) -- unlike images,
    # which accept bare strings. Embed explicit qwen_vl_utils sampling rules
    # here too, since rl_dataset.py calls process_video(video) with no config
    # overrides; per-sample dict keys are the only way to control fps.
    videos = [
        {"video": video_path, "fps": video_fps, "min_frames": video_min_frames, "max_frames": video_max_frames}
        for video_path in (raw.get("videos") or [])
    ]

    return {
        "id": record_id,
        "data_source": "embodied_world_judge_sft_mixed",
        "prompt": prompt_messages,
        "images": images,
        "videos": videos,
        "ability": metric_name,
        "reward_model": {"style": "rule", "ground_truth": str(score)},
        "extra_info": {
            "id": record_id,
            "reward_type": "embodied_judge",
            "answer": str(score),
            "metric": metric_name,
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    with input_path.open(encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if args.limit is not None:
        raw_data = raw_data[: args.limit]

    converted = []
    skipped = 0
    for idx, raw in enumerate(raw_data):
        record_id = f"embodied_judge_{idx:06d}"
        sample = convert_sample(
            raw,
            record_id,
            video_fps=args.video_fps,
            video_min_frames=args.video_min_frames,
            video_max_frames=args.video_max_frames,
        )
        if sample is None:
            skipped += 1
            continue
        converted.append(sample)

    random.shuffle(converted)
    if args.val_count is not None:
        val_count = min(args.val_count, len(converted))
    else:
        val_count = max(1, int(len(converted) * args.val_ratio)) if converted else 0
    # val is sampled independently and may overlap with train; train always
    # keeps the full converted set (nothing is held out).
    val_samples = converted[:val_count]
    train_samples = converted

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / args.train_filename
    val_path = output_dir / args.val_filename

    with train_path.open("w", encoding="utf-8") as handle:
        for sample in train_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as handle:
        for sample in val_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Total input samples: {len(raw_data)}")
    print(f"Video sampling: fps={args.video_fps} min_frames={args.video_min_frames} max_frames={args.video_max_frames}")
    print(f"Converted: {len(converted)}  Skipped (unparsable): {skipped}")
    print(f"Train: {len(train_samples)} -> {train_path}")
    print(f"Val:   {len(val_samples)} -> {val_path}")


if __name__ == "__main__":
    main()
