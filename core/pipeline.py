"""輪郭付与のパイプライン（CLI と ComfyUI ノードで共有する純ロジック）。

`tools/refine.py` は引数解釈と JSON 入出力だけを担当し、対象選別・推論・割当はここに置く。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .class_mapping import SEG_CLASS_TO_ID, seg_target
from .constants import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    MASK_SOURCE_BBOX_ELLIPSE,
    MASK_SOURCE_MODEL_SEG,
    MASK_SOURCE_NONE,
)
from .masks import ellipse_polygon
from .refiner import Refiner, assign_masks, bbox_iou
from .types import Detection

__all__ = [
    "RefineOptions",
    "RefinePlan",
    "apply_ellipse_masks",
    "plan_targets",
    "refine_with_refiner",
]


@dataclass(frozen=True, slots=True)
class RefineOptions:
    """輪郭付与の実行オプション。"""

    mask_source: str = MASK_SOURCE_MODEL_SEG
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    seg_crop: bool = False
    allowed_classes: frozenset[int] | None = None


@dataclass(frozen=True, slots=True)
class RefinePlan:
    """どの検出をどの輪郭クラスで処理するか。"""

    target_indices: tuple[int, ...]
    seg_class_ids: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.target_indices)

    @property
    def is_empty(self) -> bool:
        return not self.target_indices


def plan_targets(
    detections: list[Detection],
    allowed_classes: frozenset[int] | None = None,
) -> RefinePlan:
    """輪郭を付与できる検出を選び、必要な seg クラス集合を導出する。"""
    indices: list[int] = []
    seg_classes: set[int] = set()
    for index, det in enumerate(detections):
        target = seg_target(det.class_name)
        if target is None:
            continue
        if allowed_classes is not None and det.class_id not in allowed_classes:
            continue
        indices.append(index)
        seg_classes.add(SEG_CLASS_TO_ID[target])
    return RefinePlan(tuple(indices), tuple(sorted(seg_classes)))


def apply_ellipse_masks(
    detections: list[Detection],
    plan: RefinePlan,
) -> list[Detection]:
    """bbox 内接楕円を付与する（**デバッグ用**）。対象外の検出は変更しない。"""
    targets = set(plan.target_indices)
    return [
        replace(
            det,
            masks=(ellipse_polygon(det.bbox),),
            mask_source=MASK_SOURCE_BBOX_ELLIPSE,
        )
        if index in targets
        else det
        for index, det in enumerate(detections)
    ]


def refine_with_refiner(
    image: np.ndarray,
    detections: list[Detection],
    plan: RefinePlan,
    refiner: Refiner,
    *,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    seg_crop: bool = False,
) -> list[Detection]:
    """輪郭モデルを実行してマスクを割り当てる。対象外の検出は変更しない。"""
    if plan.is_empty:
        return list(detections)

    seg_classes = list(plan.seg_class_ids)
    if seg_crop:
        return _refine_by_crop(image, detections, plan, refiner, conf=conf, iou=iou)

    candidates = refiner.detect(image, class_ids=seg_classes, conf=conf, iou=iou)
    return assign_masks(detections, candidates, indices=set(plan.target_indices))


def _refine_by_crop(
    image: np.ndarray,
    detections: list[Detection],
    plan: RefinePlan,
    refiner: Refiner,
    *,
    conf: float,
    iou: float,
) -> list[Detection]:
    """検出ごとに bbox 周辺をクロップして輪郭を取る（計測用）。

    割当は全体推論と同じ規則（同クラス・bbox IoU 最大の1対1）。
    """
    refined = list(detections)
    for index in plan.target_indices:
        det = detections[index]
        candidates = refiner.detect_region(
            image, det.bbox, class_ids=list(plan.seg_class_ids), conf=conf, iou=iou
        )
        target = seg_target(det.class_name)
        best_position: int | None = None
        best_iou = 0.0
        for position, candidate in enumerate(candidates):
            if candidate.masks is None:
                continue
            if candidate.class_name != target:
                continue
            overlap = bbox_iou(det.bbox, candidate.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_position = position
        if best_position is None:
            refined[index] = replace(det, masks=None, mask_source=MASK_SOURCE_NONE)
            continue
        refined[index] = replace(
            det,
            masks=candidates[best_position].masks,
            mask_source=MASK_SOURCE_MODEL_SEG,
        )
    return refined
