"""マスク生成と輪郭抽出のテスト。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maskveil.core.masks import (
    compress_collinear,
    crop_mask_to_bbox,
    ellipse_polygon,
    mask_logits,
    mask_to_polygons,
    polygon_area,
    scale_polygon,
    sigmoid,
    translate_polygon,
)


def test_sigmoid_matches_threshold_semantics() -> None:
    values = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    output = sigmoid(values)
    assert output[0] < 0.5
    assert output[1] == pytest.approx(0.5)
    assert output[2] > 0.5
    assert np.all((output >= 0.0) & (output <= 1.0))


def test_mask_logits_uses_coefficients_and_prototypes() -> None:
    prototypes = np.zeros((2, 4, 4), dtype=np.float32)
    prototypes[0] = 1.0
    coeffs = np.array([2.0, -2.0], dtype=np.float32)
    logits = mask_logits(coeffs, prototypes)
    assert logits.shape == (4, 4)
    assert np.allclose(logits, 2.0)


def test_mask_logits_rejects_channel_mismatch() -> None:
    with pytest.raises(ValueError):
        mask_logits(np.zeros(3, dtype=np.float32), np.zeros((2, 4, 4), dtype=np.float32))


def test_crop_mask_to_bbox_keeps_only_inside() -> None:
    logits = np.ones((10, 10), dtype=np.float32)
    cropped = crop_mask_to_bbox(logits, (2.0, 2.0, 6.0, 6.0), scale_x=1.0, scale_y=1.0)
    assert cropped[3, 3] == 1.0
    assert cropped[0, 0] == -1.0
    assert cropped[8, 8] == -1.0


def test_crop_mask_to_bbox_handles_empty_region() -> None:
    logits = np.ones((10, 10), dtype=np.float32)
    cropped = crop_mask_to_bbox(logits, (5.0, 5.0, 5.0, 5.0), scale_x=1.0, scale_y=1.0)
    assert np.all(cropped == -1.0)


def test_mask_to_polygons_rectangle() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 3:7] = True
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    polygon = polygons[0]
    # 閉路（始点=終点）で、面積は 4x4=16
    assert polygon[0] == polygon[-1]
    assert abs(polygon_area(polygon)) == pytest.approx(16.0)
    xs = [x for x, _ in polygon[:-1]]
    ys = [y for _, y in polygon[:-1]]
    assert (min(xs), min(ys), max(xs), max(ys)) == (3.0, 2.0, 7.0, 6.0)
    # 矩形は4頂点に圧縮される
    assert len(polygon) == 5


def test_mask_to_polygons_returns_largest_first() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:4, 1:4] = True
    mask[10:18, 10:18] = True
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 2
    assert abs(polygon_area(polygons[0])) > abs(polygon_area(polygons[1]))


def test_mask_to_polygons_empty() -> None:
    assert mask_to_polygons(np.zeros((5, 5), dtype=bool)) == []


def test_mask_to_polygons_accepts_non_bool() -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    assert len(mask_to_polygons(mask)) == 1


def test_compress_collinear_removes_straight_points() -> None:
    square = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]
    compressed = compress_collinear(square)
    assert (1.0, 0.0) not in compressed
    assert len(compressed) == 5


def test_compress_collinear_keeps_short_input() -> None:
    points = [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    assert compress_collinear(points) == points


def test_polygon_area_sign_and_degenerate() -> None:
    clockwise = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    assert polygon_area(clockwise) == pytest.approx(-1.0)
    assert polygon_area([(0.0, 0.0), (1.0, 1.0)]) == 0.0


def test_scale_and_translate_polygon() -> None:
    points = [(1.0, 2.0), (3.0, 4.0)]
    assert scale_polygon(points, 2.0, 0.5) == ((2.0, 1.0), (6.0, 2.0))
    assert translate_polygon(points, 10.0, -1.0) == ((11.0, 1.0), (13.0, 3.0))


def test_ellipse_polygon_is_inside_bbox() -> None:
    bbox = (10.0, 20.0, 110.0, 220.0)
    polygon = ellipse_polygon(bbox, num_points=32)
    assert len(polygon) == 33
    assert polygon[0] == polygon[-1]
    for x, y in polygon:
        assert 10.0 - 1e-6 <= x <= 110.0 + 1e-6
        assert 20.0 - 1e-6 <= y <= 220.0 + 1e-6
    # 楕円の面積 ≈ πab
    assert abs(polygon_area(polygon)) == pytest.approx(math.pi * 50 * 100, rel=0.02)


def test_ellipse_polygon_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        ellipse_polygon((0.0, 0.0, 10.0, 10.0), num_points=4)
