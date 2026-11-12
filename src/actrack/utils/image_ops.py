from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from actrack.types import BBox


def crop_image(image: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = image.shape[:2]
    b = bbox.clamp(w, h)
    x1, y1, x2, y2 = b.as_xyxy()
    return image[int(y1) : int(y2), int(x1) : int(x2)]


def draw_labeled_boxes(
    image: np.ndarray,
    boxes: list[tuple[str, BBox, tuple[int, int, int]]],
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        return arr
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    arr = arr[:, :, :3]
    h, w = arr.shape[:2]
    canvas = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(canvas)
    for label, bbox, color in boxes:
        b = bbox.clamp(w, h)
        x1, y1, x2, y2 = b.as_xyxy()
        xy = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        draw.rectangle(xy, outline=color, width=3)
        text = str(label).strip() or "BOX"
        tx = xy[0] + 2
        ty = max(0, xy[1] - 14)
        tw = max(28, 7 * len(text) + 6)
        tx = min(max(0, tx), max(0, w - tw - 1))
        ty = min(max(0, ty), max(0, h - 13))
        text_right = min(w - 1, tx + tw)
        text_bottom = min(h - 1, ty + 12)
        draw.rectangle([tx, ty, text_right, text_bottom], fill=color)
        draw.text((min(w - 1, tx + 2), min(h - 1, ty + 1)), text, fill=(255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


