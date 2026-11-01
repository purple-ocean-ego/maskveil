"""`SEGS`（Impact Pack 互換）の組み立て。

Impact Pack が無い環境でも出力できるよう、互換構造を自前で作る。
`SEGS` は**互換ビュー**であり、正は `HOTSCREEN` である。
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch

from ..core.geometry import bbox_to_pixels
from ..core.raster import compose_masks
from ._envelope import dict_to_detection

__all__ = ["SEG", "build_segs"]

SEG = namedtuple(  # noqa: PYI024 - Impact Pack 互換のため namedtuple を使う
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=(None, None, 0.0, None, None, "", None),
)
"""Impact Pack の `SEG` と同形の namedtuple（既定値はすべて None 相当）。"""


def build_segs(
    image: torch.Tensor,
    detections: list[dict[str, object]],
    *,
    width: int,
    height: int,
    batch_index: int = 0,
) -> tuple[tuple[int, int], list[SEG]]:
    """`SEGS` を組み立てる（`batch_index` の画像のみを対象にする）。

    複数バッチでも `SEGS` は先頭（既定 `0`）の画像だけを表す。全バッチの検出は
    `HOTSCREEN.detections` と `MASK` が保持する。

    `crop_region` は `mask_expand` を含むマスク全体を囲む（膨張分が欠落すると
    Impact Pack 系の下流で未被覆が生まれるため）。画像端では画像内へクリップする。

    Args:
        image: `IMAGE` テンソル `[B, H, W, C]`。
        detections: ノード用検出 dict の一覧（`batch_index` で絞り込む）。
        width: 元画像の幅。
        height: 元画像の高さ。
        batch_index: 対象のバッチ添字。

    Returns:
        `((H, W), [SEG, ...])`。
    """
    segs: list[SEG] = []
    for item in detections:
        if int(item["batch_index"]) != batch_index:
            continue
        detection = dict_to_detection(item)
        full_mask = compose_masks(
            [detection],
            height,
            width,
            expand_px=int(item["expanded_px"]),
        )

        region = bbox_to_pixels(item["bbox"], width, height)  # type: ignore[arg-type]
        if region is None:
            continue

        # `crop_region` はマスク全体を含む（膨張分が SEGS から欠落すると下流で未被覆になる）
        rows, cols = np.nonzero(full_mask)
        if rows.size:
            x1, y1, x2, y2 = region
            region = (
                min(x1, int(cols.min())),
                min(y1, int(rows.min())),
                max(x2, int(cols.max()) + 1),
                max(y2, int(rows.max()) + 1),
            )
        x1, y1, x2, y2 = region
        cropped_mask = torch.from_numpy(np.ascontiguousarray(full_mask[y1:y2, x1:x2]))
        cropped_image = image[batch_index : batch_index + 1, y1:y2, x1:x2, :].clone()
        segs.append(
            SEG(
                cropped_image=cropped_image,
                cropped_mask=cropped_mask,
                confidence=float(item["score"]),
                crop_region=region,
                bbox=region,
                label=str(item["class_name"]),
                control_net_wrapper=None,
            )
        )
    return (height, width), segs
