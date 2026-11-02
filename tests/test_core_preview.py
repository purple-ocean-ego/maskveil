"""`core/preview.py` の検証（プレビュー経路の回帰）。"""

from __future__ import annotations

import numpy as np

from maskveil.core.preview import CLASS_COLORS, class_color, crop_union, draw_outline
from maskveil.core.types import Detection


def _image(height: int = 12, width: int = 12) -> np.ndarray:
    return np.full((height, width, 3), 0.5, dtype=np.float32)


def _detection(class_id: int = 3, *, masks=None, bbox=(2.0, 2.0, 6.0, 6.0)):
    return Detection(
        class_id=class_id,
        class_name="FEMALE_GENITALIA_EXPOSED",
        score=0.9,
        bbox=bbox,
        masks=masks,
        mask_source="model_seg" if masks else "none",
    )


def test_class_color_is_deterministic_and_distinct() -> None:
    assert class_color(3) == class_color(3) == CLASS_COLORS[3 % len(CLASS_COLORS)]
    assert class_color(3) != class_color(4)
    assert len(CLASS_COLORS) >= 8


def test_outline_draws_polygon_edges_only() -> None:
    ring = ((2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0), (2.0, 2.0))
    image = _image()
    output = draw_outline(image, [_detection(masks=(ring,))])

    color = np.asarray(class_color(3), dtype=np.float32) / 255.0
    assert np.allclose(output[2, 2], color)  # 左上の角
    assert np.allclose(output[4, 4], image[4, 4])  # 内側は塗らない
    assert np.array_equal(output[8:, :, :], image[8:, :, :])  # 外側は不変


def test_outline_uses_bbox_rect_when_no_mask() -> None:
    image = _image()
    output = draw_outline(image, [_detection(masks=None, bbox=(2.0, 2.0, 6.0, 6.0))])

    color = np.asarray(class_color(3), dtype=np.float32) / 255.0
    assert np.allclose(output[2, 4], color)
    assert np.allclose(output[5, 2], color)
    assert np.allclose(output[4, 4], image[4, 4])


def test_outline_does_not_mutate_input() -> None:
    image = _image()
    before = image.copy()
    draw_outline(image, [_detection()])
    assert np.array_equal(image, before)


def test_outline_without_detections_returns_copy() -> None:
    image = _image()
    output = draw_outline(image, [])
    assert np.array_equal(output, image)
    assert output is not image


def test_crop_union_uses_union_of_bboxes() -> None:
    image = _image(20, 20)
    first = _detection(bbox=(2.0, 3.0, 6.0, 8.0))
    second = _detection(bbox=(5.0, 6.0, 12.0, 14.0))

    crop = crop_union(image, [first, second])
    assert crop.shape == (11, 10, 3)  # y: 3..14, x: 2..12
    assert np.array_equal(crop, image[3:14, 2:12])


def test_crop_union_rounds_outward_and_clips() -> None:
    image = _image(10, 10)
    crop = crop_union(image, [_detection(bbox=(-2.4, -1.6, 4.2, 3.9))])
    assert crop.shape == (4, 5, 3)  # y: floor(-1.6)=0 .. ceil(3.9)=4, x: 0 .. ceil(4.2)=5


def test_crop_union_without_detections_returns_copy() -> None:
    image = _image(8, 8)
    crop = crop_union(image, [])
    assert np.array_equal(crop, image)
    assert crop is not image


def test_crop_union_preserves_channels() -> None:
    crop = crop_union(_image(6, 6), [_detection(bbox=(1.0, 1.0, 3.0, 3.0))])
    assert crop.dtype == np.float32
    assert crop.shape[2] == 3


def test_draw_outline_handles_empty_image() -> None:
    output = draw_outline(np.zeros((0, 0, 3), dtype=np.float32), [_detection()])
    assert output.shape == (0, 0, 3)


def test_outline_is_drawn_at_image_edge() -> None:
    """画像端に接するマスクでも輪郭線が欠けないこと（端は外側扱い）。"""
    ring = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0))
    image = _image()
    output = draw_outline(image, [_detection(masks=(ring,))])
    color = np.asarray(class_color(3), dtype=np.float32) / 255.0

    assert np.allclose(output[0, 0], color)
    assert np.allclose(output[0, 3], color)
    assert np.allclose(output[3, 0], color)
