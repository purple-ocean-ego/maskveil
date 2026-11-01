"""輪郭モデル（YOLO11-Seg）の推論ラッパー。

1枚につき1回だけ推論し、対応する全クラスのマスクを取得する。
パスの探索・検証は行わない。検証済みの絶対パスを受け取る。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .class_mapping import SEG_CLASSES, seg_target
from .constants import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    MASK_CHANNELS,
    MASK_SOURCE_MODEL_SEG,
    MASK_SOURCE_NONE,
)
from .geometry import map_bbox_to_source, map_points_to_source, xywh_to_xyxy
from .masks import crop_mask_to_bbox, mask_logits, mask_to_polygons, scale_polygon
from .onnx_utils import resolve_input_size
from .postprocess import iou_matrix, nms
from .preprocess import to_model_input
from .tensors import normalize_channels
from .types import BBox, Detection, LetterboxMeta, Polygon

__all__ = ["Refiner", "assign_masks", "bbox_iou", "decode_segmentation"]


def bbox_iou(first: BBox, second: BBox) -> float:
    ious = iou_matrix(
        np.asarray(first, dtype=np.float32),
        np.asarray([second], dtype=np.float32),
    )
    return float(ious[0]) if ious.size else 0.0


def assign_masks(
    detections: list[Detection],
    candidates: list[Detection],
    *,
    indices: set[int] | None = None,
) -> list[Detection]:
    """検出に対応する輪郭を割り当てる（1つの輪郭は1回だけ使う）。

    対応する輪郭クラスが無い検出、`indices` に含まれない検出、または bbox の重なりが
    無い検出は `mask_source="none"`。
    """
    used: set[int] = set()
    assigned: list[Detection] = []
    for index, det in enumerate(detections):
        if indices is not None and index not in indices:
            assigned.append(det)
            continue
        target = seg_target(det.class_name)
        best_index: int | None = None
        best_iou = 0.0
        if target is not None:
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in used:
                    continue
                if candidate.class_name != target or candidate.masks is None:
                    continue
                overlap = bbox_iou(det.bbox, candidate.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = candidate_index
        if best_index is None:
            assigned.append(replace(det, masks=None, mask_source=MASK_SOURCE_NONE))
            continue
        used.add(best_index)
        assigned.append(
            replace(
                det,
                masks=candidates[best_index].masks,
                mask_source=MASK_SOURCE_MODEL_SEG,
            )
        )
    return assigned


def decode_segmentation(
    outputs: list[np.ndarray],
    *,
    meta: LetterboxMeta,
    input_size: int,
    class_ids: tuple[int, ...] | list[int],
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
) -> list[Detection]:
    """YOLO11-Seg の出力（予測＋プロトタイプ）を原画像座標のマスク付き検出へ変換する。"""
    if len(outputs) < 2:
        raise ValueError(f"expected 2 outputs (predictions, prototypes), got {len(outputs)}")

    predictions = normalize_channels(
        outputs[0],
        4 + len(SEG_CLASSES) + MASK_CHANNELS,
        label="segmentation",
    )
    prototypes = np.asarray(outputs[1], dtype=np.float32)
    if prototypes.ndim == 4:
        prototypes = prototypes[0]
    if prototypes.ndim != 3:
        raise ValueError(f"unexpected prototype shape: {prototypes.shape}")

    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4 : 4 + len(SEG_CLASSES)]
    coefficients = predictions[:, 4 + len(SEG_CLASSES) :]

    proto_height, proto_width = prototypes.shape[1], prototypes.shape[2]
    to_proto_x = proto_width / input_size
    to_proto_y = proto_height / input_size
    to_input_x = input_size / proto_width
    to_input_y = input_size / proto_height

    detections: list[Detection] = []
    for class_index in class_ids:
        class_scores = scores[:, class_index]
        candidates = np.flatnonzero(class_scores >= conf)
        if candidates.size == 0:
            continue
        boxes = xywh_to_xyxy(boxes_xywh[candidates])
        keep = nms(boxes, class_scores[candidates], iou)
        if not keep:
            continue
        mapped_boxes = map_bbox_to_source(boxes[keep], meta)
        for position, mapped in zip(keep, mapped_boxes, strict=True):
            anchor = candidates[position]
            logits = mask_logits(coefficients[anchor], prototypes)
            cropped = crop_mask_to_bbox(
                logits,
                tuple(float(v) for v in boxes[position]),
                scale_x=to_proto_x,
                scale_y=to_proto_y,
            )
            polygons = mask_to_polygons(cropped > 0)
            masks: tuple[Polygon, ...] | None = None
            mask_source = MASK_SOURCE_NONE
            if polygons:
                rings: list[Polygon] = []
                for polygon in polygons:
                    scaled = scale_polygon(polygon, to_input_x, to_input_y)
                    points = map_points_to_source(np.asarray(scaled, dtype=np.float32), meta)
                    rings.append(tuple((float(x), float(y)) for x, y in points))
                masks = tuple(rings)
                mask_source = MASK_SOURCE_MODEL_SEG
            detections.append(
                Detection(
                    class_id=int(class_index),
                    class_name=SEG_CLASSES[class_index],
                    score=float(class_scores[anchor]),
                    bbox=(
                        float(mapped[0]),
                        float(mapped[1]),
                        float(mapped[2]),
                        float(mapped[3]),
                    ),
                    masks=masks,
                    mask_source=mask_source,
                )
            )

    detections.sort(key=lambda det: det.score, reverse=True)
    return detections


class Refiner:
    """輪郭モデルのセッション（静的入力）。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_size = resolve_input_size(self.session, input_size, label="refiner")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(
        self,
        image: np.ndarray,
        *,
        class_ids: tuple[int, ...] | list[int],
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> list[Detection]:
        """`(H, W, 3)` uint8 RGB 画像から、マスク付き検出を返す。"""
        if not class_ids:
            raise ValueError("class_ids must not be empty")

        tensor, meta = to_model_input(image, self.input_size)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return decode_segmentation(
            outputs,
            meta=meta,
            input_size=self.input_size,
            class_ids=class_ids,
            conf=conf,
            iou=iou,
        )

    def detect_region(
        self,
        image: np.ndarray,
        bbox: BBox,
        *,
        class_ids: tuple[int, ...] | list[int],
        padding: float = 0.1,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> list[Detection]:
        """bbox 周辺をクロップして推論する（計測用）。座標は元画像系で返す。"""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        left = max(0, int(round(x1 - pad_x)))
        top = max(0, int(round(y1 - pad_y)))
        right = min(width, int(round(x2 + pad_x)))
        bottom = min(height, int(round(y2 + pad_y)))
        if right <= left or bottom <= top:
            return []

        crop = image[top:bottom, left:right]
        detections = self.detect(crop, class_ids=class_ids, conf=conf, iou=iou)
        shifted: list[Detection] = []
        for det in detections:
            masks = None
            if det.masks is not None:
                masks = tuple(
                    tuple((x + left, y + top) for x, y in polygon) for polygon in det.masks
                )
            shifted.append(
                replace(
                    det,
                    bbox=(
                        det.bbox[0] + left,
                        det.bbox[1] + top,
                        det.bbox[2] + left,
                        det.bbox[3] + top,
                    ),
                    masks=masks,
                )
            )
        return shifted
