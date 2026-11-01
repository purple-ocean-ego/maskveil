"""`bbox_to_pixels`（bbox → 整数矩形）の検証。

`MASK` 合成・`SEGS.crop_region`・プレビューの矩形描画が同じ丸め規則を使うことを
1箇所で固定する（`x1,y1` は floor、`x2,y2` は ceil、画像端でクリップ）。
"""

from __future__ import annotations

import pytest

from maskveil.core.geometry import bbox_to_pixels


def test_rounds_outward() -> None:
    assert bbox_to_pixels((2.4, 3.6, 10.2, 12.9), 100, 100) == (2, 3, 11, 13)


def test_clips_to_image() -> None:
    assert bbox_to_pixels((-5.0, -3.0, 4.0, 6.0), 10, 10) == (0, 0, 4, 6)
    assert bbox_to_pixels((8.0, 8.0, 99.0, 99.0), 10, 10) == (8, 8, 10, 10)


def test_expand_grows_and_clips() -> None:
    assert bbox_to_pixels((4.0, 4.0, 8.0, 8.0), 20, 20, expand=4) == (0, 0, 12, 12)
    assert bbox_to_pixels((4.0, 4.0, 8.0, 8.0), 10, 10, expand=4) == (0, 0, 10, 10)


def test_expand_rejects_negative() -> None:
    with pytest.raises(ValueError, match="expand"):
        bbox_to_pixels((1.0, 1.0, 2.0, 2.0), 10, 10, expand=-1)


def test_returns_none_when_outside_image() -> None:
    assert bbox_to_pixels((20.0, 20.0, 30.0, 30.0), 10, 10) is None
    assert bbox_to_pixels((2.0, 2.0, 2.0, 8.0), 10, 10) is None  # 幅0
