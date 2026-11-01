"""letterbox・座標復元・リサイズのテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from maskveil.core.geometry import (
    clip_bbox,
    compute_letterbox,
    letterbox,
    map_bbox_to_source,
    map_points_to_source,
    resize_bilinear,
    xywh_to_xyxy,
)


def test_compute_letterbox_portrait() -> None:
    meta = compute_letterbox(2560, 3840, 640)
    assert meta.pad_y == 0.0
    assert meta.pad_x == 106.0
    assert meta.scaled_size == (427, 640)
    assert meta.scale_x == pytest.approx(427 / 2560)
    assert meta.scale_y == pytest.approx(640 / 3840)


def test_compute_letterbox_landscape_and_square() -> None:
    landscape = compute_letterbox(1280, 640, 640)
    assert landscape.pad_x == 0.0
    assert landscape.pad_y == 160.0
    assert landscape.scaled_size == (640, 320)

    square = compute_letterbox(640, 640, 640)
    assert (square.pad_x, square.pad_y) == (0.0, 0.0)
    assert (square.scale_x, square.scale_y) == (1.0, 1.0)


def test_compute_letterbox_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        compute_letterbox(0, 100, 640)
    with pytest.raises(ValueError):
        compute_letterbox(100, 100, 0)


@pytest.mark.parametrize(
    ("width", "height"),
    [(2560, 3840), (3840, 2560), (640, 640), (100, 50), (333, 1001)],
)
def test_bbox_roundtrip_within_one_pixel(width: int, height: int) -> None:
    meta = compute_letterbox(width, height, 640)
    bbox = np.array([[width * 0.1, height * 0.2, width * 0.5, height * 0.6]], dtype=np.float32)

    # 原画像 → letterbox 入力座標
    forward = bbox.copy()
    forward[:, [0, 2]] = forward[:, [0, 2]] * meta.scale_x + meta.pad_x
    forward[:, [1, 3]] = forward[:, [1, 3]] * meta.scale_y + meta.pad_y

    restored = map_bbox_to_source(forward, meta)
    assert np.allclose(restored, bbox, atol=1.0)


def test_map_points_roundtrip() -> None:
    meta = compute_letterbox(2560, 3840, 640)
    points = np.array([[10.0, 20.0], [1000.0, 2000.0]], dtype=np.float32)
    forward = points.copy()
    forward[:, 0] = forward[:, 0] * meta.scale_x + meta.pad_x
    forward[:, 1] = forward[:, 1] * meta.scale_y + meta.pad_y
    assert np.allclose(map_points_to_source(forward, meta), points, atol=1.0)


def test_map_bbox_to_source_clips_to_image() -> None:
    meta = compute_letterbox(100, 100, 640)
    boxes = np.array([[-50.0, -50.0, 1000.0, 1000.0]], dtype=np.float32)
    restored = map_bbox_to_source(boxes, meta)
    assert restored[0, 0] == pytest.approx(0.0)
    assert restored[0, 1] == pytest.approx(0.0)
    assert restored[0, 2] == pytest.approx(100.0)
    assert restored[0, 3] == pytest.approx(100.0)


def test_map_bbox_empty() -> None:
    meta = compute_letterbox(100, 100, 640)
    assert map_bbox_to_source(np.empty((0, 4)), meta).shape == (0, 4)
    assert map_points_to_source(np.empty((0, 2)), meta).shape == (0, 2)


def test_xywh_to_xyxy() -> None:
    boxes = np.array([[100.0, 100.0, 40.0, 20.0]], dtype=np.float32)
    assert xywh_to_xyxy(boxes).tolist() == [[80.0, 90.0, 120.0, 110.0]]
    assert xywh_to_xyxy(np.empty((0, 4))).shape == (0, 4)


def test_clip_bbox() -> None:
    assert clip_bbox((-5.0, -5.0, 200.0, 50.0), 100, 100) == (0.0, 0.0, 100.0, 50.0)


def test_resize_bilinear_keeps_constant_image() -> None:
    image = np.full((10, 20, 3), 7, dtype=np.uint8)
    resized = resize_bilinear(image, 25, 33)
    assert resized.shape == (25, 33, 3)
    assert np.all(resized == 7)


def test_resize_bilinear_identity() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(resize_bilinear(image, 2, 2), image)


def test_resize_bilinear_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        resize_bilinear(np.zeros((2, 2, 3, 1), dtype=np.uint8), 4, 4)


def test_letterbox_shape_and_padding() -> None:
    image = np.full((50, 100, 3), 255, dtype=np.uint8)
    boxed, meta = letterbox(image, 640)
    assert boxed.shape == (640, 640, 3)
    assert boxed.dtype == np.uint8
    # 上下のパディングは 114 で埋まる（100:50 → 640x320, pad_y=160）
    assert np.all(boxed[0, :, :] == 114)
    assert np.all(boxed[160:480, 0, :] == 255)
    assert meta.pad_y == 160.0


def test_letterbox_rejects_non_hwc() -> None:
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), dtype=np.uint8), 64)
