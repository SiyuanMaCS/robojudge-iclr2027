#!/usr/bin/env python3
"""Plot signed PA scoring bias relative to human ratings.

This script reads final800 human labels and selected model prediction JSONL files,
then writes a compact diverging bar chart. It uses Pillow to avoid requiring a
full plotting stack in the release repository.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data/label_review_20260923/final800_label_review_20260923.jsonl"
PRED_DIR = ROOT / "results/final800/predictions"
OUT_PATH = ROOT / "figures/signed_bias_pa.png"

MODELS = [
    ("RoboJudge", "robojudge_submission.jsonl"),
    ("GPT-5.5", "gpt_5_5_4fps.jsonl"),
    ("Gemini-3.7", "gemini_3_7_flash.jsonl"),
    ("Qwen3.5-9B", "qwen3_5_9b_4fps.jsonl"),
    ("Cosmos-Reason2-2B", "cosmos_reason2_2b.jsonl"),
]

COLORS = {
    "lower": "#3b82c4",
    "exact": "#d6d9df",
    "higher": "#d95f5f",
    "text": "#20242a",
    "muted": "#666f7a",
    "grid": "#e7e9ee",
    "bg": "#ffffff",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parse_score(row: dict, axis: str = "PA") -> int | None:
    keys = ("physical_adherence", "score", "pa_score")
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                score = int(value)
            except (TypeError, ValueError):
                continue
            return score if 1 <= score <= 5 else None

    text = row.get("predict") or row.get("response") or row.get("output") or ""
    if not isinstance(text, str):
        return None

    starts = list(re.finditer(r"\{", text))[-5:]
    for match in reversed(starts):
        candidate = text[match.start():]
        end = candidate.rfind("}")
        if end < 0:
            continue
        try:
            obj = json.loads(candidate[: end + 1])
        except json.JSONDecodeError:
            continue
        for key in keys:
            if key in obj:
                try:
                    score = int(obj[key])
                except (TypeError, ValueError):
                    continue
                return score if 1 <= score <= 5 else None

    match = re.search(r'"physical_adherence"\s*:\s*([1-5])', text)
    return int(match.group(1)) if match else None


def load_gold() -> dict[str, int]:
    gold: dict[str, int] = {}
    with GOLD_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            gold[row["item_id"]] = int(row["physical_adherence"])
    return gold


def load_predictions(filename: str, item_ids: set[str]) -> dict[str, int]:
    scores: dict[str, int] = {}
    with (PRED_DIR / filename).open() as f:
        for line in f:
            row = json.loads(line)
            item_id = row.get("item_id")
            if item_id not in item_ids:
                continue
            # Some release prediction files omit an explicit axis field and store
            # only the relevant score key; keep those rows if a PA score parses.
            if row.get("axis") not in (None, "PA"):
                continue
            score = parse_score(row)
            if score is not None:
                scores[item_id] = score
    return scores


def compute_stats() -> list[dict]:
    gold = load_gold()
    item_ids = set(gold)
    stats = []
    for model_name, filename in MODELS:
        preds = load_predictions(filename, item_ids)
        lower = exact = higher = 0
        for item_id, pred_score in preds.items():
            delta = pred_score - gold[item_id]
            if delta < 0:
                lower += 1
            elif delta > 0:
                higher += 1
            else:
                exact += 1
        n = lower + exact + higher
        stats.append({
            "model": model_name,
            "n": n,
            "lower": lower / n * 100,
            "exact": exact / n * 100,
            "higher": higher / n * 100,
        })
    return stats


def draw_chart(stats: list[dict]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1120, 470
    margin_l, margin_r = 230, 70
    plot_l, plot_r = margin_l, width - margin_r
    center_x = (plot_l + plot_r) // 2
    scale = (plot_r - plot_l) / 100.0
    row_h, bar_h = 60, 30
    top = 120

    img = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    font_title = load_font(26, bold=True)
    font_label = load_font(18)
    font_label_bold = load_font(18, bold=True)
    font_small = load_font(15)
    font_axis = load_font(14)

    draw.text((margin_l, 30), "Signed PA Scoring Bias Relative to Human Ratings", fill=COLORS["text"], font=font_title)
    draw.text((margin_l, 66), "Bars show whether model scores are lower than, equal to, or higher than human PA labels.", fill=COLORS["muted"], font=font_small)

    # Axis ticks for percent points around the centered exact segment.
    for pct in [-50, -25, 0, 25, 50]:
        x = center_x + pct * scale
        draw.line((x, top - 28, x, top + row_h * len(stats) - 8), fill=COLORS["grid"], width=1)
        label = f"{abs(pct)}%" if pct else "0"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, top - 50), label, fill=COLORS["muted"], font=font_axis)

    draw.text((plot_l, top - 78), "Lower than human", fill=COLORS["lower"], font=font_small)
    exact_label = "Exact match"
    bbox = draw.textbbox((0, 0), exact_label, font=font_small)
    draw.text((center_x - (bbox[2] - bbox[0]) / 2, top - 78), exact_label, fill=COLORS["muted"], font=font_small)
    higher_label = "Higher than human"
    bbox = draw.textbbox((0, 0), higher_label, font=font_small)
    draw.text((plot_r - (bbox[2] - bbox[0]), top - 78), higher_label, fill=COLORS["higher"], font=font_small)

    for idx, stat in enumerate(stats):
        y = top + idx * row_h
        model_font = font_label_bold if stat["model"] == "RoboJudge" else font_label
        draw.text((34, y + 5), stat["model"], fill=COLORS["text"], font=model_font)

        lower_w = stat["lower"] * scale
        exact_w = stat["exact"] * scale
        higher_w = stat["higher"] * scale
        exact_l = center_x - exact_w / 2
        exact_r = center_x + exact_w / 2
        lower_l = exact_l - lower_w
        higher_r = exact_r + higher_w

        draw.rounded_rectangle((lower_l, y, exact_l, y + bar_h), radius=4, fill=COLORS["lower"])
        draw.rectangle((exact_l, y, exact_r, y + bar_h), fill=COLORS["exact"])
        draw.rounded_rectangle((exact_r, y, higher_r, y + bar_h), radius=4, fill=COLORS["higher"])

        values = f'{stat["lower"]:.1f} / {stat["exact"]:.1f} / {stat["higher"]:.1f}'
        draw.text((plot_r - 155, y + 34), values, fill=COLORS["muted"], font=font_small)

    footer = "Numbers are lower / exact / higher percentages on final800 PA labels."
    draw.text((margin_l, height - 46), footer, fill=COLORS["muted"], font=font_small)
    img.save(OUT_PATH)


def main() -> None:
    draw_chart(compute_stats())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
