"""`HOTSCREEN` エンベロープの組み立てと検証。

`HOTSCREEN` は dict であり、`image` を不変に保つことが唯一の不変条件である。
本モジュールは「dict を作る／読む」処理だけを持ち、推論や加工は行わない。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..core.types import Detection

__all__ = [
    "DETECTION_KEYS",
    "ENVELOPE_KEYS",
    "SCHEMA_VERSION",
    "base_meta",
    "detection_json",
    "detection_to_dict",
    "dict_to_detection",
    "make_envelope",
    "validate_envelope",
]

SCHEMA_VERSION = 1
"""現在のエンベロープ版数。`1` 以外は実行時エラーにする。"""

ENVELOPE_KEYS: tuple[str, ...] = (
    "schema_version",
    "image",
    "mosaic_image",
    "mask",
    "detections",
    "meta",
)

DETECTION_KEYS: tuple[str, ...] = (
    "class_id",
    "class_name",
    "score",
    "bbox",
    "masks",
    "mask_source",
    "batch_index",
    "expanded_px",
)
"""検出 dict のキー（`core` の6キー + ノードが付与する2キー）。"""


def validate_envelope(hotscreen: Any) -> dict[str, Any]:
    """`HOTSCREEN` の必須キーと版数を検証して返す。

    Args:
        hotscreen: 検証対象（`Detector` / `Refiner` / `MosaicByMask` の入力）。

    Returns:
        検証済みの dict（新しい dict ではなく入力そのもの。書き換えは呼び出し側で行う）。

    Raises:
        ValueError: dict でない、必須キー不足、`schema_version` が1以外。
    """
    if not isinstance(hotscreen, dict):
        raise ValueError(f"HOTSCREEN must be a dict: got {type(hotscreen).__name__}")

    missing = [key for key in ENVELOPE_KEYS if key not in hotscreen]
    if missing:
        raise ValueError(f"HOTSCREEN に必須キーがありません: {missing}")

    version = hotscreen["schema_version"]
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version!r} (expected {SCHEMA_VERSION})")
    return hotscreen


def make_envelope(
    *,
    image: Any,
    mask: Any,
    detections: list[dict[str, Any]],
    meta: dict[str, Any],
    mosaic_image: Any = None,
) -> dict[str, Any]:
    """`HOTSCREEN` を `ENVELOPE_KEYS` の順で組み立てる（生成箇所の重複を防ぐ）。

    Args:
        image: 元画像テンソル（以後どのノードも書き換えない）。
        mask: 直近の検出系が出したマスク。
        detections: ノード用検出 dict の一覧。
        meta: 実行の記録。
        mosaic_image: 最後に加工した画像（未加工は `None`）。

    Returns:
        `HOTSCREEN` dict。
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "image": image,
        "mosaic_image": mosaic_image,
        "mask": mask,
        "detections": detections,
        "meta": meta,
    }


def base_meta(*, size: Iterable[int]) -> dict[str, Any]:
    """`meta` の初期値を作る。

    Args:
        size: `[B, H, W]`。

    Returns:
        `det` / `refine` / `mask_expand` / `size` / `processing` を持つ dict。
    """
    return {
        "det": None,
        "refine": None,
        "mask_expand": 0,
        "size": [int(value) for value in size],
        "processing": [],
    }


def detection_to_dict(
    detection: Detection,
    *,
    batch_index: int,
    expanded_px: int,
) -> dict[str, Any]:
    """`Detection` をノード用 dict（8キー）へ変換する。

    `masks` は形状が無い場合も `None` として**キーを残す**（下流が欠損で落ちないため）。

    Args:
        detection: 変換元の検出。
        batch_index: 属する画像のバッチ添字。
        expanded_px: 直近に適用した膨張量（px）。

    Returns:
        `DETECTION_KEYS` の順序を持つ dict。
    """
    # 丸めは JSON 化（detection_json）のときだけ行う。メモリ上の dict は生値で持ち、
    # Refiner が戻して再ラスタ化しても座標がずれないようにする。
    masks = None
    if detection.masks is not None:
        masks = [[[float(x), float(y)] for x, y in polygon] for polygon in detection.masks]
    return {
        "class_id": int(detection.class_id),
        "class_name": str(detection.class_name),
        "score": float(detection.score),
        "bbox": [float(value) for value in detection.bbox],
        "masks": masks,
        "mask_source": detection.mask_source,
        "batch_index": int(batch_index),
        "expanded_px": int(expanded_px),
    }


def dict_to_detection(item: dict[str, Any]) -> Detection:
    """ノード用 dict を `Detection` へ戻す（`Refiner` の入力に使う）。

    Args:
        item: `detection_to_dict()` が作った dict（`batch_index` / `expanded_px` は無視）。

    Returns:
        `core.types.Detection`。
    """
    masks = item.get("masks")
    polygons = (
        tuple(tuple((float(x), float(y)) for x, y in polygon) for polygon in masks)
        if masks
        else None
    )
    bbox = tuple(float(value) for value in item["bbox"])
    return Detection(
        class_id=int(item["class_id"]),
        class_name=str(item["class_name"]),
        score=float(item["score"]),
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        masks=polygons,
        mask_source=item.get("mask_source", "none"),
    )


def _rounded_for_json(item: dict[str, Any]) -> dict[str, Any]:
    """JSON 出力用に丸めた検出 dict を返す（score 4桁 / 座標 2桁）。"""
    rounded = dict(item)
    rounded["score"] = round(float(item["score"]), 4)
    rounded["bbox"] = [round(float(value), 2) for value in item["bbox"]]
    if item.get("masks"):
        rounded["masks"] = [
            [[round(float(x), 2), round(float(y), 2)] for x, y in polygon]
            for polygon in item["masks"]
        ]
    return rounded


def detection_json(
    *,
    model: str,
    input_size: int,
    params: dict[str, Any],
    detections: list[dict[str, Any]],
) -> str:
    """`detection` 出力（STRING）を組み立てる。

    CLI 版との差分は「`image` を持たない」「`source` を持つ」「`model.path` が相対パス」
    「`params` に `top_k` / `mask_expand` を持つ」の4点。

    Args:
        model: `folder_paths` の相対パス。
        input_size: 実際に使った入力一辺。
        params: `classes` / `conf` / `iou` / `top_k` / `mask_expand`（`meta.det.params` と共有）。
        detections: 検出 dict の一覧（生値。ここで丸める）。

    Returns:
        JSON 文字列（UTF-8・非 ASCII をそのまま出す）。
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "comfyui",
        "model": {"path": model, "input_size": int(input_size)},
        "params": params,
        "detections": [_rounded_for_json(item) for item in detections],
        "count": len(detections),
    }
    return json.dumps(payload, ensure_ascii=False)
