"""`preview` 出力（outline / crop / none）の組み立て（SPEC §4-2-2）。

`preview` は**表示専用**であり、`HOTSCREEN` / `MASK` の計算結果には影響しない。
バッチ運用では `none`（＝入力をそのまま返す）を推奨する。
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.preview import crop_union, draw_outline
from ..core.types import Detection
from ._convert import numpy_to_image
from ._envelope import dict_to_detection

__all__ = ["PREVIEW_MODES", "build_preview"]

PREVIEW_MODES: tuple[str, ...] = ("outline", "crop", "none")
"""`preview_mode` の選択肢（`outline` が既定）。"""


def build_preview(
    image: torch.Tensor,
    rgb_u8: np.ndarray,
    detections: list[dict[str, object]],
    mode: str,
) -> torch.Tensor:
    """`preview_mode` に応じた確認用画像を返す。

    Args:
        image: 元の `IMAGE` テンソル（`none` の素通しと device/dtype の復元に使う）。
        rgb_u8: `uint8[B, H, W, C]` の元画像。
        detections: ノード用検出 dict の一覧。
        mode: `outline` / `crop` / `none`。

    Returns:
        `IMAGE` テンソル。`crop` は先頭バッチのみ（`B=1`）、それ以外は `B` を保つ。

    Raises:
        ValueError: `mode` が未知の場合。
    """
    if mode not in PREVIEW_MODES:
        raise ValueError(f"unknown preview_mode: {mode!r} (expected one of {PREVIEW_MODES})")
    if mode == "none":
        return image

    grouped: dict[int, list[Detection]] = {}
    for item in detections:
        grouped.setdefault(int(item["batch_index"]), []).append(dict_to_detection(item))

    # core の描画・クロップ API は float32 0–1 を要求する（SPEC §2-1）
    normalized = rgb_u8.astype(np.float32) / 255.0

    if mode == "crop":
        return numpy_to_image(crop_union(normalized[0], grouped.get(0, [])), like=image)

    drawn = [
        draw_outline(normalized[index], grouped.get(index, []))
        for index in range(normalized.shape[0])
    ]
    return numpy_to_image(np.stack(drawn), like=image)
