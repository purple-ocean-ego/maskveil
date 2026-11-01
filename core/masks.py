"""マスク生成（プロトタイプ合成 → 2値化 → 輪郭ポリゴン化）。

`cv2` を使わないため、輪郭抽出はピクセル境界の連結で行う。
座標系は呼び出し側が指定する（モデル入力座標 → 原画像座標は `geometry.map_points_to_source`）。
"""

from __future__ import annotations

import math

import numpy as np

from .types import Point, Polygon

__all__ = [
    "Polygon",
    "compress_collinear",
    "crop_mask_to_bbox",
    "ellipse_polygon",
    "mask_logits",
    "mask_to_polygons",
    "polygon_area",
    "scale_polygon",
    "sigmoid",
    "translate_polygon",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数値安定なシグモイド。"""
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def mask_logits(coeffs: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """検出ごとの係数とプロトタイプからマスクのロジットを作る。

    Args:
        coeffs: `(32,)` の係数。
        prototypes: `(32, H, W)` のプロトタイプ。

    Returns:
        `(H, W)` のロジット（しきい値0で sigmoid>0.5 と等価）。
    """
    channels, height, width = prototypes.shape
    if coeffs.shape[-1] != channels:
        raise ValueError(f"coeffs has {coeffs.shape[-1]} channels, expected {channels}")
    flat = prototypes.reshape(channels, height * width)
    return (coeffs.reshape(1, channels) @ flat).reshape(height, width)


def crop_mask_to_bbox(
    logits: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """bbox（入力座標）の外側を **-1.0（ロジット）で潰す**。

    マスク格子はプロトタイプ解像度（例 320）なので、`floor`/`ceil` により
    最大 1 格子（入力座標で 4px 程度）外側へ張り出すことがある（安全側の誤差）。
    """
    height, width = logits.shape
    x1 = max(0, min(width, int(math.floor(bbox[0] * scale_x))))
    y1 = max(0, min(height, int(math.floor(bbox[1] * scale_y))))
    x2 = max(0, min(width, int(math.ceil(bbox[2] * scale_x))))
    y2 = max(0, min(height, int(math.ceil(bbox[3] * scale_y))))
    if x2 <= x1 or y2 <= y1:
        return np.full_like(logits, -1.0)
    cropped = np.full_like(logits, -1.0)
    cropped[y1:y2, x1:x2] = logits[y1:y2, x1:x2]
    return cropped


def mask_to_polygons(mask: np.ndarray) -> list[Polygon]:
    """2値マスクから輪郭ポリゴン（連結成分ごと）を抽出する。

    ピクセルの外周に沿って辺を張り、始点→終点の連結で閉路を作る。
    戻り値は面積の大きい順。
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    ys, xs = np.nonzero(padded[1:-1, 1:-1])
    if ys.size == 0:
        return []

    # (始点, 終点) の差分。外周を時計回りにたどる
    directions = (
        ((0, 0), (1, 0), (ys, xs + 1)),
        ((1, 0), (1, 1), (ys + 1, xs + 2)),
        ((1, 1), (0, 1), (ys + 2, xs + 1)),
        ((0, 1), (0, 0), (ys + 1, xs)),
    )

    edges: dict[Point, list[Point]] = {}
    for start_offset, end_offset, (block_y, block_x) in directions:
        selected = ~padded[block_y, block_x]
        for y, x in zip(ys[selected].tolist(), xs[selected].tolist(), strict=True):
            start = (float(x + start_offset[0]), float(y + start_offset[1]))
            end = (float(x + end_offset[0]), float(y + end_offset[1]))
            edges.setdefault(start, []).append(end)

    polygons: list[Polygon] = []
    while edges:
        start = next(iter(edges))
        ring: list[Point] = [start]
        current = start
        while True:
            targets = edges.get(current)
            if not targets:
                break
            nxt = targets.pop(0)
            if not targets:
                del edges[current]
            ring.append(nxt)
            current = nxt
            if current == start:
                break
        if len(ring) >= 4:
            compressed = compress_collinear(ring)
            if len(compressed) >= 3 and abs(polygon_area(compressed)) > 0:
                polygons.append(compressed)

    polygons.sort(key=lambda poly: abs(polygon_area(poly)), reverse=True)
    return polygons


def compress_collinear(points: Polygon) -> Polygon:
    """直線上に並ぶ中間点を除去する（始点・終点が同一の閉路を想定）。"""
    if len(points) <= 3:
        return points
    closed = points[0] == points[-1]
    body = list(points[:-1] if closed else points)
    if len(body) < 3:
        return points

    result: list[Point] = []
    count = len(body)
    for index in range(count):
        prev_x, prev_y = body[index - 1]
        cur_x, cur_y = body[index]
        next_x, next_y = body[(index + 1) % count]
        cross = (cur_x - prev_x) * (next_y - cur_y) - (cur_y - prev_y) * (next_x - cur_x)
        if cross != 0:
            result.append((cur_x, cur_y))
    if len(result) < 3:
        result = body
    if closed:
        result.append(result[0])
    return tuple(result)


def polygon_area(points: Polygon) -> float:
    """靴ひも公式による符号付き面積。"""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def scale_polygon(points: Polygon, scale_x: float, scale_y: float) -> Polygon:
    """ポリゴンの座標を拡大/縮小する。"""
    return tuple((x * scale_x, y * scale_y) for x, y in points)


def translate_polygon(points: Polygon, dx: float, dy: float) -> Polygon:
    """ポリゴンを平行移動する。"""
    return tuple((x + dx, y + dy) for x, y in points)


def ellipse_polygon(
    bbox: tuple[float, float, float, float],
    *,
    num_points: int = 32,
) -> Polygon:
    """bbox の内接楕円を返す（**デバッグ用**。既定では使わない）。"""
    if num_points < 8:
        raise ValueError("num_points must be >= 8")
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    radius_x, radius_y = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    points: list[Point] = []
    for index in range(num_points):
        angle = 2.0 * math.pi * index / num_points
        points.append(
            (center_x + radius_x * math.cos(angle), center_y + radius_y * math.sin(angle))
        )
    points.append(points[0])
    return tuple(points)
