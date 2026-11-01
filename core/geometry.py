"""座標変換（letterbox・xywh↔xyxy・原画像座標への復元）。

`cv2` を使わないため、リサイズもここで numpy 実装する。
"""

from __future__ import annotations

import numpy as np

from .constants import LETTERBOX_PAD_VALUE
from .types import BBox, LetterboxMeta

__all__ = [
    "bbox_to_pixels",
    "compute_letterbox",
    "clip_bbox",
    "letterbox",
    "map_bbox_to_source",
    "map_points_to_source",
    "resize_bilinear",
    "xywh_to_xyxy",
]


def compute_letterbox(src_width: int, src_height: int, dst_size: int) -> LetterboxMeta:
    """アスペクト比を保って `dst_size` に収める倍率とパディング量を返す。"""
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"invalid source size: {src_width}x{src_height}")
    if dst_size <= 0:
        raise ValueError(f"invalid destination size: {dst_size}")

    scale = min(dst_size / src_width, dst_size / src_height)
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))
    pad_x = (dst_size - new_width) // 2
    pad_y = (dst_size - new_height) // 2
    return LetterboxMeta(
        src_width=src_width,
        src_height=src_height,
        dst_size=dst_size,
        scale_x=new_width / src_width,
        scale_y=new_height / src_height,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
    )


def resize_bilinear(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """バイリニア縮小/拡大（HWC / HW 両対応）。"""
    if image.ndim not in (2, 3):
        raise ValueError(f"unsupported image shape: {image.shape}")

    height, width = image.shape[:2]
    if (height, width) == (new_height, new_width):
        return image

    ys = (np.arange(new_height, dtype=np.float32) + 0.5) * (height / new_height) - 0.5
    xs = (np.arange(new_width, dtype=np.float32) + 0.5) * (width / new_width) - 0.5
    ys = np.clip(ys, 0.0, height - 1)
    xs = np.clip(xs, 0.0, width - 1)

    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)

    if image.ndim == 3:
        wy = wy[:, None, None]
        wx = wx[None, :, None]
    else:
        wy = wy[:, None]
        wx = wx[None, :]

    rows_top = image[y0]
    rows_bottom = image[y1]
    top = rows_top[:, x0] * (1.0 - wx) + rows_top[:, x1] * wx
    bottom = rows_bottom[:, x0] * (1.0 - wx) + rows_bottom[:, x1] * wx
    out = top * (1.0 - wy) + bottom * wy
    return out.astype(image.dtype, copy=False)


def letterbox(
    image: np.ndarray,
    dst_size: int,
    *,
    pad_value: int = LETTERBOX_PAD_VALUE,
) -> tuple[np.ndarray, LetterboxMeta]:
    """画像を letterbox する（HWC uint8 RGB を想定）。"""
    if image.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {image.shape}")
    height, width = image.shape[:2]
    meta = compute_letterbox(width, height, dst_size)

    new_width, new_height = meta.scaled_size
    resized = resize_bilinear(image, new_height, new_width)

    pad_x = int(meta.pad_x)
    pad_y = int(meta.pad_y)
    canvas = np.full((dst_size, dst_size, image.shape[2]), pad_value, dtype=image.dtype)
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, meta


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """中心xywh → 左上右下（同一次元の配列を想定）。"""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    center_x, center_y, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ),
        axis=1,
    )


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """原画像の範囲へクリップする。"""
    x1, y1, x2, y2 = bbox
    return (
        float(min(max(x1, 0.0), width)),
        float(min(max(y1, 0.0), height)),
        float(min(max(x2, 0.0), width)),
        float(min(max(y2, 0.0), height)),
    )


def map_bbox_to_source(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """letterbox 入力座標の xyxy を原画像座標へ戻す。"""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale_x
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale_y
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, meta.src_width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, meta.src_height)
    return out


def map_points_to_source(points: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """letterbox 入力座標の点列（x, y）を原画像座標へ戻す。"""
    if points.size == 0:
        return points.reshape(0, 2)
    out = points.astype(np.float32, copy=True)
    out[:, 0] = (out[:, 0] - meta.pad_x) / meta.scale_x
    out[:, 1] = (out[:, 1] - meta.pad_y) / meta.scale_y
    out[:, 0] = np.clip(out[:, 0], 0.0, meta.src_width)
    out[:, 1] = np.clip(out[:, 1], 0.0, meta.src_height)
    return out


def bbox_to_pixels(
    bbox: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
    *,
    expand: int = 0,
) -> tuple[int, int, int, int] | None:
    """連続値の bbox を整数の矩形へ丸める（`x1,y1` は floor、`x2,y2` は ceil）。

    画像端でクリップする。`expand` を指定すると外側へ広げてからクリップする
    （`mask_source="none"` の検出を矩形マスク化するときの `mask_expand` 用）。

    Args:
        bbox: `(x1, y1, x2, y2)`（原画像px）。
        width: 画像幅。
        height: 画像高さ。
        expand: 外側へ広げる量（px）。負値は不可。

    Returns:
        `(x1, y1, x2, y2)`。面積が0になる（画像外）場合は `None`。

    Raises:
        ValueError: `expand` が負の場合。
    """
    expand = int(expand)
    if expand < 0:
        raise ValueError(f"expand must be >= 0: {expand}")

    left = max(int(np.floor(bbox[0])) - expand, 0)
    top = max(int(np.floor(bbox[1])) - expand, 0)
    right = min(int(np.ceil(bbox[2])) + expand, width)
    bottom = min(int(np.ceil(bbox[3])) + expand, height)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
