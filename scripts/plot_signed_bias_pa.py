#!/usr/bin/env python3
"""Plot signed PA/IA scoring bias relative to human ratings."""

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
    ("Cosmos-R2-2B", "cosmos_reason2_2b.jsonl"),
    ("VideoScore2", "videoscore2_8b.jsonl"),
]

AXES = [
    ("PA", "physical_adherence", ("physical_adherence", "pa_score", "score")),
    ("IA", "instruction_alignment", ("instruction_alignment", "ia_score", "score")),
]

COLORS = {
    "lower": "#2f6fb0",
    "exact": "#d8dce3",
    "higher": "#c84f4f",
    "text": "#1f252d",
    "muted": "#69717d",
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


def parse_score(row: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
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
            if key not in obj:
                continue
            try:
                score = int(obj[key])
            except (TypeError, ValueError):
                continue
            return score if 1 <= score <= 5 else None

    for key in keys:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*([1-5])', text)
        if match:
            return int(match.group(1))
    return None


def load_gold(gold_key: str) -> dict[str, int]:
    gold: dict[str, int] = {}
    with GOLD_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            gold[row["item_id"]] = int(row[gold_key])
    return gold


def load_predictions(filename: str, axis_name: str, keys: tuple[str, ...], item_ids: set[str]) -> dict[str, int]:
    scores: dict[str, int] = {}
    with (PRED_DIR / filename).open() as f:
        for line in f:
            row = json.loads(line)
            item_id = row.get("item_id")
            if item_id not in item_ids:
                continue
            if row.get("axis") not in (None, axis_name):
                continue
            score = parse_score(row, keys)
            if score is not None:
                scores[item_id] = score
    return scores


def compute_stats() -> dict[str, list[dict]]:
    all_stats: dict[str, list[dict]] = {}
    for axis_name, gold_key, keys in AXES:
        gold = load_gold(gold_key)
        item_ids = set(gold)
        stats = []
        for model_name, filename in MODELS:
            preds = load_predictions(filename, axis_name, keys, item_ids)
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
            stats.append(
                {
                    "model": model_name,
                    "lower": lower / n * 100,
                    "exact": exact / n * 100,
                    "higher": higher / n * 100,
                }
            )
        all_stats[axis_name] = stats
    return all_stats


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2 - 1), text, fill=fill, font=font)


def draw_bar(
    draw: ImageDraw.ImageDraw,
    stat: dict,
    x: int,
    y: int,
    bar_w: int,
    bar_h: int,
    font_pct: ImageFont.ImageFont,
) -> None:
    segments = [
        ("lower", stat["lower"], "white"),
        ("exact", stat["exact"], COLORS["text"]),
        ("higher", stat["higher"], "white"),
    ]
    for seg_idx, (key, value, text_color) in enumerate(segments):
        w = bar_w * value / 100.0
        x0, x1 = x, x + w
        if seg_idx == 0:
            draw.rounded_rectangle((x0, y, x1, y + bar_h), radius=4, fill=COLORS[key])
            draw.rectangle((x1 - 4, y, x1, y + bar_h), fill=COLORS[key])
        elif seg_idx == len(segments) - 1:
            draw.rounded_rectangle((x0, y, x1, y + bar_h), radius=4, fill=COLORS[key])
            draw.rectangle((x0, y, x0 + 4, y + bar_h), fill=COLORS[key])
        else:
            draw.rectangle((x0, y, x1, y + bar_h), fill=COLORS[key])

        if value >= 15.0:
            draw_centered(draw, (x0, y, x1, y + bar_h), f"{value:.0f}%", font_pct, text_color)
        x = x1


def draw_chart(all_stats: dict[str, list[dict]]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1180, 360
    img = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    font_title = load_font(22, bold=True)
    font_label = load_font(15)
    font_label_bold = load_font(15, bold=True)
    font_panel = load_font(15, bold=True)
    font_small = load_font(13)
    font_pct = load_font(12, bold=True)

    label_x = 155
    left_pa = 195
    left_ia = 690
    bar_w = 430
    bar_h = 20
    row_gap = 39
    top = 105

    draw.text((left_pa, 24), "Signed scoring bias vs. human ratings", fill=COLORS["text"], font=font_title)

    legend = [("Lower", "lower"), ("Exact", "exact"), ("Higher", "higher")]
    lx = left_pa
    for label, key in legend:
        draw.rounded_rectangle((lx, 60, lx + 16, 72), radius=2, fill=COLORS[key])
        draw.text((lx + 22, 57), label, fill=COLORS["muted"], font=font_small)
        lx += 92

    draw.text((left_pa, 83), "PA", fill=COLORS["text"], font=font_panel)
    draw.text((left_ia, 83), "IA", fill=COLORS["text"], font=font_panel)

    for idx, model in enumerate([name for name, _ in MODELS]):
        y = top + idx * row_gap
        model_font = font_label_bold if model == "RoboJudge" else font_label
        label_w = text_width(draw, model, model_font)
        draw.text((label_x - label_w, y + 1), model, fill=COLORS["text"], font=model_font)

        draw_bar(draw, all_stats["PA"][idx], left_pa, y, bar_w, bar_h, font_pct)
        draw_bar(draw, all_stats["IA"][idx], left_ia, y, bar_w, bar_h, font_pct)

    img.save(OUT_PATH)


def main() -> None:
    draw_chart(compute_stats())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
