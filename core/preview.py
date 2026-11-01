"""目視確認用のプレビュー生成（輪郭描画・和集合クロップ）。numpy のみ。

`preview` 出力は**表示専用**であり、`HOTSCREEN` / `MASK` の計算結果には影響しない。
"""

from __future__ import annotations

import numpy as np

from .geometry import bbox_to_pixels
from .raster import rasterize_polygons
from .types import Detection

__all__ = ["CLASS_COLORS", "class_color", "crop_union", "draw_outline"]

CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 64, 64),
    (64, 128, 255),
    (64, 220, 96),
    (255, 200, 48),
    (200, 80, 255),
    (48, 224, 224),
    (255, 140, 0),
    (200, 200, 200),
)
"""クラスIDごとの描画色（決定的）。16クラスを巡回して使う。"""


def class_color(class_id: int) -> tuple[int, int, int]:
    """クラスIDに対応する RGB 色を返す（同じIDは常に同じ色）。"""
    return CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]


def _boundary(bitmap: np.ndarray) -> np.ndarray:
    """マスクの外周1画素を返す（4近傍の内側判定）。

    画像端は「外側＝偽」として扱うため、端に接するマスクでも輪郭線が欠けない。
    """
    if bitmap.size == 0:
        return bitmap.copy()
    inner = bitmap.copy()
    inner[1:, :] &= bitmap[:-1, :]
    inner[:-1, :] &= bitmap[1:, :]
    inner[:, 1:] &= bitmap[:, :-1]
    inner[:, :-1] &= bitmap[:, 1:]
    inner[0, :] = False
    inner[-1, :] = False
    inner[:, 0] = False
    inner[:, -1] = False
    return bitmap & ~inner


def _rect_mask(bbox: tuple[float, float, float, float], height: int, width: int) -> np.ndarray:
    bitmap = np.zeros((height, width), dtype=bool)
    region = bbox_to_pixels(bbox, width, height)
    if region is not None:
        left, top, right, bottom = region
        bitmap[top:bottom, left:right] = True
    return bitmap


def draw_outline(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """元画像に検出マスクの輪郭をクラス別の色で描く。

    `masks` が無い検出は **bbox 矩形**の輪郭を描く（SPEC §4-2-2）。

    Args:
        image: `float32[H, W, C]`（0.0–1.0）。
        detections: 描画対象の検出。

    Returns:
        入力と同形の `float32` 画像（入力は不変）。
    """
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    canvas = array.copy()
    for detection in detections:
        color = np.asarray(class_color(detection.class_id), dtype=np.float32) / 255.0
        if detection.masks:
            for polygon in detection.masks:
                edges = _boundary(rasterize_polygons([polygon], height, width))
                canvas[edges] = color
            continue
        edges = _boundary(_rect_mask(detection.bbox, height, width))
        canvas[edges] = color
    return canvas


def crop_union(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """全検出の和集合 bbox でクロップした画像を返す（検出0件なら元画像）。

    Args:
        image: `float32[H, W, C]`。
        detections: クロップ範囲の算出に使う検出。

    Returns:
        `float32[h, w, C]`。`bbox` は `floor`/`ceil` で外側へ丸め、画像端でクリップする。
    """
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    if not detections:
        return array.copy()

    left = min(int(np.floor(det.bbox[0])) for det in detections)
    top = min(int(np.floor(det.bbox[1])) for det in detections)
    right = max(int(np.ceil(det.bbox[2])) for det in detections)
    bottom = max(int(np.ceil(det.bbox[3])) for det in detections)
    region = bbox_to_pixels((left, top, right, bottom), width, height)
    if region is None:
        return array.copy()
    x1, y1, x2, y2 = region
    return array[y1:y2, x1:x2].copy()
