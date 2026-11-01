"""後処理（デコード・NMS・座標復元）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from maskveil.core.geometry import compute_letterbox
from maskveil.core.labels import LABELS
from maskveil.core.postprocess import decode_detections, iou_matrix, nms

NUM_CLASSES = len(LABELS)


def _empty_output(anchors: int) -> np.ndarray:
    return np.zeros((1, 4 + NUM_CLASSES, anchors), dtype=np.float32)


def _set(
    output: np.ndarray,
    anchor: int,
    *,
    box: tuple[float, float, float, float],
    scores: dict[int, float],
) -> None:
    output[0, 0:4, anchor] = box
    for class_index, score in scores.items():
        output[0, 4 + class_index, anchor] = score


def _identity_meta(size: int = 640):
    return compute_letterbox(size, size, size)


def test_decode_basic_and_score_order() -> None:
    output = _empty_output(3)
    _set(output, 0, box=(100, 100, 40, 40), scores={11: 0.9})
    _set(output, 1, box=(100, 100, 40, 40), scores={11: 0.8})  # NMS で抑制される
    _set(output, 2, box=(300, 300, 20, 20), scores={11: 0.5, 3: 0.6})

    detections = decode_detections(
        output, meta=_identity_meta(), class_ids=(3, 11), conf=0.2, iou=0.45
    )

    assert [det.class_id for det in detections] == [11, 3, 11]
    assert detections[0].score == pytest.approx(0.9, abs=1e-6)
    assert detections[0].bbox == pytest.approx((80.0, 80.0, 120.0, 120.0))
    assert detections[1].class_name == "FEMALE_GENITALIA_EXPOSED"
    assert detections[2].bbox == pytest.approx((290.0, 290.0, 310.0, 310.0))


def test_decode_accepts_anchor_major_layout() -> None:
    output = _empty_output(1)
    _set(output, 0, box=(100, 100, 40, 40), scores={11: 0.9})
    anchor_major = output[0].T  # (N, C)

    detections = decode_detections(
        anchor_major, meta=_identity_meta(), class_ids=(11,), conf=0.2, iou=0.45
    )
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9, abs=1e-6)


def test_decode_filters_by_class() -> None:
    output = _empty_output(2)
    _set(output, 0, box=(100, 100, 40, 40), scores={11: 0.9, 3: 0.9})
    _set(output, 1, box=(300, 300, 20, 20), scores={0: 0.8})

    detections = decode_detections(
        output, meta=_identity_meta(), class_ids=(11,), conf=0.2, iou=0.45
    )
    assert {det.class_id for det in detections} == {11}
    assert len(detections) == 1


def test_decode_conf_boundary_is_inclusive() -> None:
    output = _empty_output(1)
    _set(output, 0, box=(100, 100, 40, 40), scores={11: 0.2})
    detections = decode_detections(
        output, meta=_identity_meta(), class_ids=(11,), conf=0.2, iou=0.45
    )
    assert len(detections) == 1


def test_decode_is_class_aware_nms() -> None:
    """同一 bbox でもクラスが異なれば両方残す（モザイク漏れ防止）。"""
    output = _empty_output(1)
    _set(output, 0, box=(100, 100, 40, 40), scores={11: 0.9, 3: 0.8})
    detections = decode_detections(
        output, meta=_identity_meta(), class_ids=(3, 11), conf=0.2, iou=0.45
    )
    assert len(detections) == 2


def test_decode_restores_letterboxed_coordinates() -> None:
    # 2560x3840 の原画像 → 640 入力（pad_x=106, scale_y=1/6）
    meta = compute_letterbox(2560, 3840, 640)
    output = _empty_output(1)
    # 原画像の中心 (600, 1920)・サイズ 100x100 を入力座標へ写してから与える
    input_cx = 600.0 * meta.scale_x + meta.pad_x
    input_cy = 1920.0 * meta.scale_y + meta.pad_y
    _set(
        output,
        0,
        box=(input_cx, input_cy, 100.0 * meta.scale_x, 100.0 * meta.scale_y),
        scores={11: 0.9},
    )

    detections = decode_detections(output, meta=meta, class_ids=(11,), conf=0.2, iou=0.45)
    x1, y1, x2, y2 = detections[0].bbox
    assert (x1 + x2) / 2 == pytest.approx(600.0, abs=1.0)
    assert (y1 + y2) / 2 == pytest.approx(1920.0, abs=1.0)
    assert (x2 - x1) == pytest.approx(100.0, abs=2.0)
    assert (y2 - y1) == pytest.approx(100.0, abs=2.0)


def test_decode_is_deterministic() -> None:
    output = _empty_output(50)
    rng = np.random.default_rng(42)
    output[0, 0:4] = rng.uniform(0, 640, size=(4, 50))
    output[0, 4:] = rng.uniform(0, 1, size=(NUM_CLASSES, 50))

    first = decode_detections(output, meta=_identity_meta(), class_ids=(11, 3), conf=0.9, iou=0.45)
    second = decode_detections(output, meta=_identity_meta(), class_ids=(11, 3), conf=0.9, iou=0.45)
    assert [det.to_dict() for det in first] == [det.to_dict() for det in second]


def test_decode_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError):
        decode_detections(
            np.zeros((1, 19, 10), dtype=np.float32),
            meta=_identity_meta(),
            class_ids=(11,),
            conf=0.2,
            iou=0.45,
        )


def test_iou_matrix() -> None:
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    ious = iou_matrix(box, boxes)
    assert ious[0] == pytest.approx(1.0)
    assert ious[1] == pytest.approx(0.0)


def test_nms_iou_threshold_boundary() -> None:
    # IoU = 60 / 100 = 0.6
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 6.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0]  # 0.6 > 0.5 → 抑制
    assert nms(boxes, scores, 0.7) == [0, 1]  # 0.6 <= 0.7 → 両方残す


def test_nms_empty_and_single() -> None:
    assert nms(np.empty((0, 4)), np.empty((0,)), 0.45) == []
    assert nms(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.5]), 0.45) == [0]
