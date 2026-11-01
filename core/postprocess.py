"""後処理（デコード・class-aware NMS・原画像座標への復元）。"""

from __future__ import annotations

import numpy as np

from .constants import DEFAULT_CONF, DEFAULT_IOU
from .geometry import map_bbox_to_source, xywh_to_xyxy
from .labels import LABELS, class_name
from .tensors import normalize_channels
from .types import Detection, LetterboxMeta

__all__ = ["decode_detections", "iou_matrix", "nms"]


def iou_matrix(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """1つの box と複数 box の IoU を返す。"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0.0, None
    )
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """貪欲法の NMS。残すインデックスをスコア降順で返す。"""
    if boxes.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    return keep


def decode_detections(
    output: np.ndarray,
    *,
    meta: LetterboxMeta,
    class_ids: tuple[int, ...] | list[int],
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
) -> list[Detection]:
    """YOLO11 の検出出力 `(1, 4+nc, N)` を原画像座標の検出リストへ変換する。"""
    raw = normalize_channels(output, 4 + len(LABELS), label="detector")

    boxes_xywh = raw[:, :4]
    class_scores = raw[:, 4:]

    detections: list[Detection] = []
    for cid in class_ids:
        scores = class_scores[:, cid]
        candidates = np.flatnonzero(scores >= conf)
        if candidates.size == 0:
            continue
        boxes = xywh_to_xyxy(boxes_xywh[candidates])
        keep = nms(boxes, scores[candidates], iou)
        if not keep:
            continue
        kept_boxes = boxes[keep]
        kept_scores = scores[candidates][keep]
        mapped = map_bbox_to_source(kept_boxes, meta)
        for bbox, score in zip(mapped, kept_scores, strict=True):
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=class_name(cid),
                    score=float(score),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
            )

    detections.sort(key=lambda det: det.score, reverse=True)
    return detections
