#!/usr/bin/env python3
"""Create a close-up ASCII portrait and an optional tonal opacity map.

The text file remains readable and editable by hand. The JSON map stores only
ASCII characters and opacity values; it does not contain the source photo.
Best results come from a crop where the face and hair fill most of the frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Light to dark. The varied glyph shapes resemble the source README's portrait.
CHARACTERS = " .,'`-_:;!i1tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


def parse_crop(value: str) -> tuple[int, int, int, int]:
    try:
        numbers = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Crop must contain four integers.") from exc
    if len(numbers) != 4:
        raise argparse.ArgumentTypeError("Crop format is left,top,right,bottom.")
    left, top, right, bottom = numbers
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("Crop right/bottom must exceed left/top.")
    return left, top, right, bottom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input photograph.")
    parser.add_argument(
        "--crop",
        type=parse_crop,
        help="Optional crop as left,top,right,bottom pixels.",
    )
    parser.add_argument("--cols", type=int, default=60, help="ASCII grid width.")
    parser.add_argument("--rows", type=int, default=34, help="ASCII grid height.")
    parser.add_argument(
        "--contrast", type=float, default=1.15, help="Final contrast multiplier."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.55,
        help="Darkness gamma. Raise this to make skin tones lighter.",
    )
    parser.add_argument(
        "--edge-strength",
        type=float,
        default=0.28,
        help="Amount of glasses, hair, and facial-edge enhancement.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("portrait.txt"), help="ASCII text output."
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=Path("portrait_map.json"),
        help="Per-character opacity JSON output.",
    )
    return parser.parse_args()


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate the main foreground subject and soften its boundary."""
    height, width = rgb.shape[:2]
    margin_x = max(2, width // 50)
    margin_y = max(2, height // 50)
    rectangle = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y,
    )
    grabcut_mask = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        rgb,
        grabcut_mask,
        rectangle,
        background,
        foreground,
        7,
        cv2.GC_INIT_WITH_RECT,
    )
    binary = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)

    # Keep the largest connected component so background furniture is discarded.
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count > 1:
        foreground_index = 1 + int(
            np.argmax(statistics[1:, cv2.CC_STAT_AREA])
        )
        binary = (labels == foreground_index).astype(np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2,
    )
    return cv2.GaussianBlur(binary.astype(np.float32), (9, 9), 0)


def direction_character(gx: float, gy: float) -> str:
    """Choose a text stroke parallel to a detected image edge."""
    angle = (math.atan2(gy, gx) + math.pi / 2) % math.pi
    if angle < math.pi / 8 or angle >= 7 * math.pi / 8:
        return "_"
    if angle < 3 * math.pi / 8:
        return "\\"
    if angle < 5 * math.pi / 8:
        return "|"
    return "/"


def generate_ascii(
    image_path: Path,
    crop: tuple[int, int, int, int] | None,
    cols: int,
    rows: int,
    contrast: float,
    gamma: float,
    edge_strength: float,
) -> tuple[list[str], list[list[list[str | float]]]]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to open image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if crop:
        left, top, right, bottom = crop
        rgb = rgb[top:bottom, left:right]
    if rgb.size == 0:
        raise ValueError("The selected crop is outside the image.")

    mask = foreground_mask(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 35, 35)
    local = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.addWeighted(gray, 0.72, local, 0.28, 0)

    pil_image = Image.fromarray(gray)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    pil_image = pil_image.filter(
        ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3)
    )
    gray = np.asarray(pil_image, dtype=np.float32)
    gray = gray * mask + 255 * (1 - mask)

    darkness = np.clip(1 - gray / 255, 0, 1) ** gamma
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx * gx + gy * gy) * mask
    foreground_edges = magnitude[mask > 0.25]
    normalizer = (
        float(np.percentile(foreground_edges, 97))
        if foreground_edges.size
        else float(np.percentile(magnitude, 97))
    )
    edges = np.clip(magnitude / max(normalizer, 1e-6), 0, 1)

    resize = lambda array: cv2.resize(
        array, (cols, rows), interpolation=cv2.INTER_AREA
    )
    small_darkness = resize(darkness)
    small_mask = resize(mask)
    small_edges = resize(edges)
    small_gx = resize(gx)
    small_gy = resize(gy)

    portrait: list[list[list[str | float]]] = []
    for row_index in range(rows):
        output_row: list[list[str | float]] = []
        for column_index in range(cols):
            subject = float(small_mask[row_index, column_index])
            base = float(small_darkness[row_index, column_index])
            edge = float(small_edges[row_index, column_index])
            tone = float(
                np.clip(base + edge_strength * edge * (1 - 0.45 * base), 0, 1)
            )

            if subject < 0.10 or tone < 0.025:
                character = " "
                opacity = 0.0
            elif edge > 0.42 and base < 0.38 and edge > base:
                character = direction_character(
                    float(small_gx[row_index, column_index]),
                    float(small_gy[row_index, column_index]),
                )
                opacity = min(1.0, 0.58 + 0.42 * edge)
            else:
                character_index = int(round(tone * (len(CHARACTERS) - 1)))
                character = CHARACTERS[character_index]
                if character == " " and tone > 0.04:
                    character = "."
                opacity = min(1.0, 0.32 + 0.68 * tone**0.55)

            opacity *= min(1.0, subject / 0.40)
            output_row.append([character, round(float(opacity), 2)])
        portrait.append(output_row)

    # Remove empty borders while keeping text and map dimensions identical.
    while portrait and not any(cell[0] != " " for cell in portrait[0]):
        portrait.pop(0)
    while portrait and not any(cell[0] != " " for cell in portrait[-1]):
        portrait.pop()
    if not portrait:
        raise ValueError("No foreground portrait was produced. Try a tighter crop.")

    visible_columns = [
        column_index
        for row in portrait
        for column_index, cell in enumerate(row)
        if cell[0] != " "
    ]
    left = min(visible_columns)
    right = max(visible_columns) + 1
    portrait = [row[left:right] for row in portrait]

    lines = ["".join(str(cell[0]) for cell in row).rstrip() for row in portrait]
    return lines, portrait


def main() -> int:
    args = parse_args()
    if args.cols < 20 or args.rows < 15:
        raise ValueError("Use at least 20 columns and 15 rows.")
    if args.gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")

    lines, portrait = generate_ascii(
        args.image,
        args.crop,
        args.cols,
        args.rows,
        args.contrast,
        args.gamma,
        args.edge_strength,
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    map_data = {
        "version": 1,
        "source": "Generated ASCII character and opacity map; source image not embedded.",
        "rows": portrait,
    }
    args.map_output.write_text(
        json.dumps(map_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(lines)} lines to {args.output}")
    print(f"Wrote tonal map to {args.map_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
