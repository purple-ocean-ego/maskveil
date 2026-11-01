"""ラスタ操作（`core/raster.py`）の検証（AC-2 / AC-3 の回帰）。

`dilate_mask` は当初 `min(dist[k] + |i-k|)` を誤って `min(dist[k] - (i-k))` と実装し、
**窓全体を塗りつぶす**不具合があった（実モデル検証で発覚）。ここで菱形（L1 距離）を固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from maskveil.core.masks import mask_to_polygons
from maskveil.core.raster import (
    compose_masks,
    dilate_mask,
    expand_detections,
    rasterize_polygons,
)
from maskveil.core.types import Detection


def test_rasterize_single_pixel_round_trip() -> None:
    bitmap = np.zeros((5, 6), dtype=bool)
    bitmap[2, 3] = True
    rings = mask_to_polygons(bitmap)
    assert rasterize_polygons(rings, 5, 6).tolist() == bitmap.tolist()


def test_rasterize_rect_and_concave_shape() -> None:
    bitmap = np.zeros((6, 7), dtype=bool)
    bitmap[1:3, 1:4] = True
    bitmap[3:5, 3:6] = True  # L 字（凹）
    assert rasterize_polygons(mask_to_polygons(bitmap), 6, 7).tolist() == bitmap.tolist()


def test_rasterize_polygon_uses_pixel_centers() -> None:
    # (0,0)-(2,0)-(2,2)-(0,2) は画素 (0,0),(0,1),(1,0),(1,1) を覆う
    ring = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0))
    assert rasterize_polygons([ring], 3, 3).tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, False],
    ]


def test_rasterize_rejects_empty_size() -> None:
    with pytest.raises(ValueError):
        rasterize_polygons([], 0, 4)


def test_dilate_zero_is_identity() -> None:
    bitmap = np.zeros((4, 4), dtype=bool)
    bitmap[1, 1] = True
    assert dilate_mask(bitmap, 0).tolist() == bitmap.tolist()


def test_dilate_is_diamond_not_full_window() -> None:
    bitmap = np.zeros((9, 9), dtype=bool)
    bitmap[4, 4] = True
    grown = dilate_mask(bitmap, 1)
    assert grown.sum() == 5  # 中心 + 上下左右
    assert grown[3, 4] and grown[5, 4] and grown[4, 3] and grown[4, 5]
    assert not grown[3, 3]


def test_dilate_two_steps_is_manhattan_distance() -> None:
    bitmap = np.zeros((9, 9), dtype=bool)
    bitmap[4, 4] = True
    grown = dilate_mask(bitmap, 2)
    assert grown.sum() == 13
    assert grown[2, 4] and grown[4, 6]
    assert not grown[7, 7]


def test_dilate_accumulates_over_iterations() -> None:
    bitmap = np.zeros((9, 9), dtype=bool)
    bitmap[4, 4] = True
    once = dilate_mask(bitmap, 1)
    twice = dilate_mask(once, 1)
    assert twice.sum() == dilate_mask(bitmap, 2).sum()
    assert twice.sum() > once.sum()


def test_dilate_rejects_negative() -> None:
    with pytest.raises(ValueError):
        dilate_mask(np.zeros((2, 2), dtype=bool), -1)


def test_compose_masks_prefers_polygons_over_bbox() -> None:
    ring = ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0), (2.0, 2.0))
    detection = Detection(
        class_id=3,
        class_name="FEMALE_GENITALIA_EXPOSED",
        score=0.9,
        bbox=(0.0, 0.0, 10.0, 10.0),
        masks=(ring,),
        mask_source="model_seg",
    )
    mask = compose_masks([detection], 10, 10)
    assert mask.dtype == np.float32
    assert float(mask.sum()) == 4.0  # bbox 全体(100)ではなくポリゴン(2x2)


def test_compose_masks_uses_expanded_bbox_when_no_polygon() -> None:
    detection = Detection(
        class_id=3,
        class_name="FEMALE_GENITALIA_EXPOSED",
        score=0.9,
        bbox=(2.0, 2.0, 5.0, 5.0),
        masks=None,
        mask_source="none",
    )
    plain = compose_masks([detection], 10, 10)
    grown = compose_masks([detection], 10, 10, expand_px=1)
    assert float(plain.sum()) == 9.0  # 3x3
    assert float(grown.sum()) == 25.0  # 5x5（画像端でクリップ）


def test_expand_detections_keeps_bbox_and_source() -> None:
    ring = ((2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0), (2.0, 2.0))
    detection = Detection(
        class_id=3,
        class_name="FEMALE_GENITALIA_EXPOSED",
        score=0.9,
        bbox=(2.0, 2.0, 6.0, 6.0),
        masks=(ring,),
        mask_source="model_seg",
    )
    grown = expand_detections([detection], 2)[0]

    assert grown.bbox == detection.bbox
    assert grown.mask_source == detection.mask_source
    before = compose_masks([detection], 12, 12).sum()
    after = compose_masks([grown], 12, 12).sum()
    # 4x4 → 8x8 から角の三角形（1辺2px の L1 距離で 3px × 4）を除いた 52px
    assert before == 16.0
    assert after == 52.0
    assert after < 12 * 12  # 窓全体を塗りつぶしていない


def test_expand_detections_keeps_bbox_only_detection() -> None:
    detection = Detection(
        class_id=11,
        class_name="MALE_GENITALIA_EXPOSED",
        score=0.5,
        bbox=(1.0, 1.0, 4.0, 4.0),
        masks=None,
        mask_source="none",
    )
    assert expand_detections([detection], 4) == [detection]


def test_expand_detections_never_shrinks_mask_outside_bbox() -> None:
    """輪郭マスクが検出 bbox より外へはみ出しても、膨張で縮まないこと。

    以前は作業窓を bbox からしか作っていなかったため、はみ出し分が切り落とされ
    `mask_expand` が「収縮」になっていた（Refiner 経路で未被覆が発生する）。
    """
    ring = ((90.0, 90.0), (130.0, 90.0), (130.0, 130.0), (90.0, 130.0), (90.0, 90.0))
    detection = Detection(
        class_id=3,
        class_name="FEMALE_GENITALIA_EXPOSED",
        score=0.9,
        bbox=(100.0, 100.0, 120.0, 120.0),  # ポリゴンより内側の bbox
        masks=(ring,),
        mask_source="model_seg",
    )
    baseline = float(compose_masks([detection], 300, 300).sum())

    areas = []
    for px in (1, 2, 4, 8):
        grown = expand_detections([detection], px)[0]
        area = float(compose_masks([grown], 300, 300).sum())
        assert area > baseline, f"px={px} で面積が増えていない（{area} <= {baseline}）"
        areas.append(area)

    assert areas == sorted(areas), "px を増やすほど面積が減っている"
    # bbox は膨張の影響を受けない
    assert expand_detections([detection], 4)[0].bbox == detection.bbox
