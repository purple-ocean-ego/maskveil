"""輪郭出力のデコード（`decode_segmentation`）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from maskveil.core.class_mapping import SEG_CLASSES
from maskveil.core.constants import MASK_CHANNELS, MASK_SOURCE_MODEL_SEG, MASK_SOURCE_NONE
from maskveil.core.geometry import compute_letterbox
from maskveil.core.masks import polygon_area
from maskveil.core.refiner import decode_segmentation

PROTO_SIZE = 320
INPUT_SIZE = 1280
PENIS = SEG_CLASSES.index("penis")


def _prototypes(blobs: list[tuple[int, int, int, int]]) -> np.ndarray:
    """指定矩形だけ正のロジットを持つプロトタイプを作る。"""
    prototype = np.full((1, 32, PROTO_SIZE, PROTO_SIZE), -1.0, dtype=np.float32)
    for row_start, row_end, col_start, col_end in blobs:
        prototype[0, 0, row_start:row_end, col_start:col_end] = 1.0
    return prototype


def _predictions(
    box_xywh: tuple[float, float, float, float],
    *,
    class_index: int = PENIS,
    score: float = 0.9,
) -> np.ndarray:
    predictions = np.zeros((1, 4 + len(SEG_CLASSES) + MASK_CHANNELS, 1), dtype=np.float32)
    predictions[0, 0:4, 0] = box_xywh
    predictions[0, 4 + class_index, 0] = score
    predictions[0, 4 + len(SEG_CLASSES), 0] = 1.0  # 係数は先頭プロトタイプのみ参照
    return predictions


def test_decode_single_component_restores_original_coordinates() -> None:
    meta = compute_letterbox(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    outputs = [
        _predictions((600.0, 500.0, 400.0, 300.0)),
        _prototypes([(100, 150, 120, 180)]),
    ]
    detections = decode_segmentation(
        outputs, meta=meta, input_size=INPUT_SIZE, class_ids=(PENIS,), conf=0.2, iou=0.45
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "penis"
    assert detection.mask_source == MASK_SOURCE_MODEL_SEG
    assert detection.masks is not None
    assert len(detection.masks) == 1
    polygon = detection.masks[0]
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    # プロトタイプ 1px = 入力 4px（1280 / 320）
    assert min(xs) == pytest.approx(480.0, abs=4.0)
    assert max(xs) == pytest.approx(720.0, abs=4.0)
    assert min(ys) == pytest.approx(400.0, abs=4.0)
    assert max(ys) == pytest.approx(600.0, abs=4.0)
    assert abs(polygon_area(polygon)) == pytest.approx(240 * 200, rel=0.1)


def test_decode_keeps_all_components() -> None:
    meta = compute_letterbox(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    outputs = [
        _predictions((400.0, 400.0, 800.0, 800.0)),
        _prototypes([(10, 20, 10, 20), (100, 150, 120, 180)]),
    ]
    detections = decode_segmentation(
        outputs, meta=meta, input_size=INPUT_SIZE, class_ids=(PENIS,), conf=0.2, iou=0.45
    )

    assert len(detections[0].masks or ()) == 2
    largest, smallest = detections[0].masks  # 面積の大きい順
    assert abs(polygon_area(largest)) > abs(polygon_area(smallest))


def test_decode_without_positive_mask_returns_none() -> None:
    meta = compute_letterbox(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    outputs = [_predictions((600.0, 500.0, 400.0, 300.0)), _prototypes([])]
    detections = decode_segmentation(
        outputs, meta=meta, input_size=INPUT_SIZE, class_ids=(PENIS,), conf=0.2, iou=0.45
    )

    assert detections[0].masks is None
    assert detections[0].mask_source == MASK_SOURCE_NONE


def test_decode_rejects_unexpected_shapes() -> None:
    meta = compute_letterbox(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    with pytest.raises(ValueError):
        decode_segmentation(
            [np.zeros((1, 42, 5), dtype=np.float32), np.zeros((1, 32, 320, 320))],
            meta=meta,
            input_size=INPUT_SIZE,
            class_ids=(PENIS,),
            conf=0.2,
            iou=0.45,
        )
    with pytest.raises(ValueError):
        decode_segmentation(
            [np.zeros((1, 43, 5), dtype=np.float32)],
            meta=meta,
            input_size=INPUT_SIZE,
            class_ids=(PENIS,),
            conf=0.2,
            iou=0.45,
        )
