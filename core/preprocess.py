"""前処理（letterbox → RGB → 1/255 正規化 → CHW）。"""

from __future__ import annotations

import numpy as np

from .geometry import letterbox
from .types import LetterboxMeta


def to_model_input(image: np.ndarray, dst_size: int) -> tuple[np.ndarray, LetterboxMeta]:
    """HWC uint8 RGB 画像を ONNX 入力（1, 3, S, S）へ変換する。

    Args:
        image: `(H, W, 3)` の uint8 RGB 配列。
        dst_size: モデルの入力一辺（例 640）。

    Returns:
        `(tensor, meta)`。`tensor` は float32・0〜1・NCHW。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {image.shape}")

    boxed, meta = letterbox(image, dst_size)
    tensor = boxed.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor[None, ...]), meta
