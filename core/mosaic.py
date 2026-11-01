"""マスクによる隠蔽加工（モザイク / ぼかし / 黒塗り）。numpy のみ。

すべて `float32 0–1` のまま処理し、8bit 量子化は保存時に行う（`nodes` の責務）。
`mask > 0.0` の比較だけを適用条件とし、閾値の調整や判断は行わない（SPEC §6-6）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["CENSOR_MODES", "apply_censor"]

CENSOR_MODES: tuple[str, ...] = ("mosaic", "blur", "black")
"""`MosaicByMask.mode` が取りうる値（加工の種類。判断パラメータではない）。"""


def apply_censor(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str,
    block_size: int,
) -> np.ndarray:
    """マスク領域を隠蔽した画像を返す。

    Args:
        image: `float32[H, W, C]`（`C=3` を想定。0.0–1.0）。
        mask: `[H, W]`。`> 0.0` の画素が対象。
        mode: `mosaic` / `blur` / `black`。
        block_size: `mosaic` のブロック辺、`blur` の半径（px）。

    Returns:
        入力と同形の `float32` 画像（マスクが全ゼロなら入力と画素一致）。

    Raises:
        ValueError: `mode` が未知、`block_size` が1未満、形状が不一致の場合。
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"image must be (H, W, C): got shape {array.shape}")
    if mode not in CENSOR_MODES:
        raise ValueError(f"unknown mode: {mode!r} (expected one of {CENSOR_MODES})")
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1: {block_size}")

    bitmap = np.asarray(mask) > 0.0
    if bitmap.shape != array.shape[:2]:
        raise ValueError(f"mask shape {bitmap.shape} does not match image {array.shape[:2]}")
    if not bitmap.any():
        return array.copy()

    if mode == "black":
        censored = array.copy()
        censored[bitmap] = 0.0
        return censored
    if mode == "mosaic":
        return _mosaic(array, bitmap, block_size)
    return _blur_composite(array, bitmap, block_size)


def _mosaic(image: np.ndarray, bitmap: np.ndarray, block_size: int) -> np.ndarray:
    """マスクが1画素でも掛かるブロックを、そのブロックの平均色で置換する。

    触れたブロックだけをスライス単位で置換する（全画素分の中間配列を作らない）。
    ブロック平均は**マスク外の画素も含めた**ブロック全体の平均である（SPEC §6-3）。
    """
    height, width = bitmap.shape
    rows, cols = np.nonzero(bitmap)
    grid_width = (width + block_size - 1) // block_size
    touched = np.unique((rows // block_size) * grid_width + (cols // block_size))

    output = image.copy()
    for key in touched.tolist():
        block_row, block_col = divmod(key, grid_width)
        top = block_row * block_size
        left = block_col * block_size
        bottom = min(top + block_size, height)
        right = min(left + block_size, width)
        output[top:bottom, left:right] = image[top:bottom, left:right].mean(axis=(0, 1))
    return output


def _blur_composite(image: np.ndarray, bitmap: np.ndarray, radius: int) -> np.ndarray:
    """画像全体をぼかしてから、マスク領域だけ合成する（境界の混色を防ぐ）。"""
    blurred = _box_blur(image, radius)
    return np.where(bitmap[:, :, None], blurred, image).astype(np.float32)


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """box blur を2回（縦横それぞれ）掛けてガウスぼかしに近似する。"""
    blurred = image
    for _ in range(2):
        for axis in (0, 1):
            blurred = _blur_axis(blurred, radius, axis)
    return blurred


def _blur_axis(image: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """1軸方向の box blur（端は reflect パディング）。

    画像がカーネルより小さい場合は、実際に取れる半径へ縮めて正規化する
    （`reflect` は pad 幅が次元以上だと成立しないため）。
    """
    length = image.shape[axis]
    effective = min(radius, max(length - 1, 0))
    kernel = 2 * effective + 1

    if effective > 0:
        widths = [(0, 0)] * image.ndim
        widths[axis] = (effective, effective)
        padded = np.pad(image, widths, mode="reflect")
    else:
        padded = image

    zero_shape = list(padded.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate(
        [np.zeros(zero_shape, dtype=np.float32), np.cumsum(padded, axis=axis, dtype=np.float32)],
        axis=axis,
    )
    lower = np.take(cumulative, np.arange(0, length), axis=axis)
    upper = np.take(cumulative, np.arange(kernel, kernel + length), axis=axis)
    return ((upper - lower) / float(kernel)).astype(np.float32)
