"""ポリゴンのラスタ化・マスクの膨張・検出マスクの合成（numpy のみ）。

`core/masks.py` が「ラスタ → ポリゴン」を担うのに対し、本モジュールは逆方向
（ポリゴン → ラスタ）と膨張・合成を担う。両者は**画素中心規則**で一致するため、
`mask_to_polygons()` → `rasterize_polygons()` の往復で画素が失われない。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .geometry import bbox_to_pixels
from .masks import mask_to_polygons
from .types import Detection, Polygon

__all__ = [
    "compose_masks",
    "dilate_mask",
    "expand_detections",
    "rasterize_polygons",
]


def rasterize_polygons(
    polygons: list[Polygon] | tuple[Polygon, ...],
    height: int,
    width: int,
) -> np.ndarray:
    """閉路ポリゴン列を2値マスクへ塗りつぶす（even-odd 規則）。

    画素 `(row, col)` の中心 `(col + 0.5, row + 0.5)` がポリゴンの内側なら 1。
    この規則は `mask_to_polygons()` が返す「画素境界座標」と往復して一致する。

    Args:
        polygons: 閉路ポリゴンの列（座標は画像px）。
        height: 出力マスクの高さ。
        width: 出力マスクの幅。

    Returns:
        `bool[height, width]` のマスク。

    Raises:
        ValueError: `height` / `width` が正でない場合。
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid mask size: {width}x{height}")

    mask = np.zeros((height, width), dtype=bool)
    row_centers = np.arange(height, dtype=np.float64) + 0.5

    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        start_x, start_y = points[:-1, 0], points[:-1, 1]
        end_x, end_y = points[1:, 0], points[1:, 1]
        rows: list[np.ndarray] = []
        crossings: list[np.ndarray] = []
        for x1, y1, x2, y2 in zip(start_x, start_y, end_x, end_y, strict=True):
            if y1 == y2:
                continue
            # 半開区間 [min, max) で数えることで頂点の二重計上を避ける
            lower, upper = (y1, y2) if y1 < y2 else (y2, y1)
            selected = (row_centers >= lower) & (row_centers < upper)
            if not selected.any():
                continue
            ratio = (row_centers[selected] - y1) / (y2 - y1)
            rows.append(np.flatnonzero(selected))
            crossings.append(x1 + ratio * (x2 - x1))
        if not rows:
            continue

        row_index = np.concatenate(rows)
        x_cross = np.concatenate(crossings)
        order = np.lexsort((x_cross, row_index))
        row_index = row_index[order]
        x_cross = x_cross[order]
        unique_rows, first_positions = np.unique(row_index, return_index=True)
        for row, group in zip(unique_rows, np.split(x_cross, first_positions[1:]), strict=True):
            if group.size < 2:
                continue
            for index in range(0, group.size - 1, 2):
                left, right = float(group[index]), float(group[index + 1])
                # 画素中心 col + 0.5 が [left, right) に入る col の範囲
                first = max(int(np.ceil(left - 0.5)), 0)
                last = min(int(np.ceil(right - 0.5)), width)
                if last > first:
                    mask[row, first:last] = True

    return mask


def _l1_distance(mask: np.ndarray, cap: int) -> np.ndarray:
    """真の画素までの4近傍距離（マンハッタン距離）を `cap` で頭打ちにして返す。"""
    height, width = mask.shape
    limit = cap + 1
    distance = np.where(mask, 0, limit).astype(np.int32)

    # 1次元の距離変換: min_k (dist[k] + |index - k|) = accumulate(dist - index) + index
    columns = np.arange(width, dtype=np.int32)
    distance = np.minimum.accumulate(distance - columns, axis=1) + columns
    np.clip(distance, 0, limit, out=distance)
    reversed_distance = distance[:, ::-1]
    reversed_distance = np.minimum.accumulate(reversed_distance - columns, axis=1) + columns
    np.clip(reversed_distance, 0, limit, out=reversed_distance)
    distance = reversed_distance[:, ::-1]

    rows = np.arange(height, dtype=np.int32).reshape(-1, 1)
    distance = np.minimum.accumulate(distance - rows, axis=0) + rows
    np.clip(distance, 0, limit, out=distance)
    reversed_distance = distance[::-1, :]
    reversed_distance = np.minimum.accumulate(reversed_distance - rows, axis=0) + rows
    np.clip(reversed_distance, 0, limit, out=reversed_distance)
    return reversed_distance[::-1, :]


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """4近傍 dilation を `px` 回適用する（`mask_expand` の実体）。

    Args:
        mask: `bool[H, W]`（真偽値以外は `!= 0` で解釈）。
        px: 膨張量（px）。0 は無変化。

    Returns:
        `bool[H, W]`。`px` 回の4近傍膨張と同じ結果（マンハッタン距離 `<= px`）。

    Raises:
        ValueError: `px` が負の場合。
    """
    px = int(px)
    if px < 0:
        raise ValueError(f"px must be >= 0: {px}")
    bitmap = np.asarray(mask) != 0
    if px == 0 or not bitmap.any():
        return bitmap.copy()
    return _l1_distance(bitmap, px) <= px


def expand_detections(detections: list[Detection], px: int) -> list[Detection]:
    """検出のマスク形状を外側へ `px` 膨張させ、ポリゴンを再抽出する。

    `bbox` と `mask_source` は**変更しない**。`masks` が `None` の検出（bbox 矩形で
    扱う検出）は、膨張後の矩形を作れないため**そのまま返す**（矩形の外挿は
    `compose_masks(expand_px=...)` が担う）。

    Args:
        detections: 対象の検出。
        px: 膨張量（px）。0 は無変化。

    Returns:
        膨張後の検出（入力の `Detection` は不変）。

    Raises:
        ValueError: `px` が負の場合。
    """
    px = int(px)
    if px < 0:
        raise ValueError(f"px must be >= 0: {px}")
    if px == 0:
        return list(detections)

    expanded: list[Detection] = []
    for detection in detections:
        if detection.masks is None:
            expanded.append(detection)
            continue

        # 窓は「bbox ∪ 全ポリゴン点」から作る。bbox だけで作ると、輪郭モデルの
        # マスクが検出 bbox より外へはみ出したときに切り落とされ「膨張」が
        # 「収縮」になってしまう（Refiner 経路で実際に起こる）。
        xs = [detection.bbox[0], detection.bbox[2]]
        ys = [detection.bbox[1], detection.bbox[3]]
        for polygon in detection.masks:
            for x, y in polygon:
                xs.append(x)
                ys.append(y)

        left = max(int(np.floor(min(xs))) - px, 0)
        top = max(int(np.floor(min(ys))) - px, 0)
        right = int(np.ceil(max(xs))) + px
        bottom = int(np.ceil(max(ys))) + px
        window_width = right - left
        window_height = bottom - top
        if window_width <= 0 or window_height <= 0:
            expanded.append(detection)
            continue

        local = [tuple((x - left, y - top) for x, y in polygon) for polygon in detection.masks]
        grown = dilate_mask(rasterize_polygons(local, window_height, window_width), px)
        rings = mask_to_polygons(grown)
        if not rings:
            expanded.append(detection)
            continue
        expanded.append(
            replace(
                detection,
                masks=tuple(tuple((x + left, y + top) for x, y in ring) for ring in rings),
            )
        )
    return expanded


def compose_masks(
    detections: list[Detection],
    height: int,
    width: int,
    *,
    expand_px: int = 0,
) -> np.ndarray:
    """検出列から合成マスクを作る。

    `masks` がある検出はポリゴンを、無い検出は `bbox` 矩形（`expand_px` だけ外挿し、
    画像端でクリップ）を塗る。

    Args:
        detections: 対象の検出。
        height: 出力マスクの高さ。
        width: 出力マスクの幅。
        expand_px: `masks` を持たない検出の矩形を外挿する量（px）。

    Returns:
        `float32[height, width]`（1.0 = 対象）。
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid mask size: {width}x{height}")
    expand_px = int(expand_px)
    if expand_px < 0:
        raise ValueError(f"expand_px must be >= 0: {expand_px}")

    bitmap = np.zeros((height, width), dtype=bool)
    for detection in detections:
        if detection.masks:
            bitmap |= rasterize_polygons(detection.masks, height, width)
            continue
        region = bbox_to_pixels(detection.bbox, width, height, expand=expand_px)
        if region is not None:
            left, top, right, bottom = region
            bitmap[top:bottom, left:right] = True

    return bitmap.astype(np.float32)
