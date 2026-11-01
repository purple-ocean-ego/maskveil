"""ONNX 出力テンソルの軸正規化（検出・輪郭で共通）。"""

from __future__ import annotations

import numpy as np


def normalize_channels(output: np.ndarray, expected: int, *, label: str) -> np.ndarray:
    """`(channels, anchors)` / `(anchors, channels)` のどちらでも `(anchors, channels)` に揃える。

    Args:
        output: モデル出力（バッチ次元は任意）。
        expected: 期待するチャネル数（例: 4 + クラス数）。
        label: エラーメッセージ用の名称（`"detector"` / `"segmentation"`）。

    Raises:
        ValueError: どちらの軸も `expected` と一致しない場合。
    """
    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError(f"unexpected {label} output shape: {raw.shape}")
    if raw.shape[0] == expected:
        return raw.T
    if raw.shape[1] == expected:
        return raw
    raise ValueError(f"{label} output has shape {raw.shape}; expected one axis to be {expected}")
