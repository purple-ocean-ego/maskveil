"""`core/mosaic.py` の検証（AC-4）。

**画素不変は完全一致（`np.array_equal`）で検証する**。マスクが掛からないブロックは
1画素も変わらないこと、空マスクは入力と同一であることを固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from maskveil.core.mosaic import CENSOR_MODES, apply_censor


def _ramp(height: int = 8, width: int = 8) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
    return values / float(values.max())


def test_modes_are_fixed() -> None:
    assert CENSOR_MODES == ("mosaic", "blur", "black")


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        apply_censor(_ramp(), np.ones((8, 8), dtype=bool), "pixelate", 8)


def test_invalid_block_size_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        apply_censor(_ramp(), np.ones((8, 8), dtype=bool), "mosaic", 0)


def test_mask_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        apply_censor(_ramp(8, 8), np.ones((4, 4), dtype=bool), "black", 8)


def test_image_must_be_hwc() -> None:
    with pytest.raises(ValueError, match="image must be"):
        apply_censor(np.zeros((8, 8), dtype=np.float32), np.ones((8, 8), dtype=bool), "black", 8)


@pytest.mark.parametrize("mode", CENSOR_MODES)
def test_empty_mask_returns_identical_pixels(mode: str) -> None:
    image = _ramp()
    empty = np.zeros((8, 8), dtype=bool)
    output = apply_censor(image, empty, mode, 2)

    assert np.array_equal(output, image)
    assert output is not image  # 呼び出し側の配列を共有しない


def test_mosaic_replaces_whole_touched_block() -> None:
    image = _ramp(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    output = apply_censor(image, mask, "mosaic", 2)

    expected_block = image[0:2, 0:2].mean(axis=(0, 1))
    assert np.allclose(output[0:2, 0:2], expected_block)
    # ブロック内のマスク外画素も置換される（はみ出し側に倒す）
    assert not np.allclose(output[0, 1], image[0, 1])


def test_mosaic_leaves_untouched_blocks_exactly_unchanged() -> None:
    image = _ramp(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    output = apply_censor(image, mask, "mosaic", 2)

    assert np.array_equal(output[2:4, :, :], image[2:4, :, :])
    assert np.array_equal(output[:, 2:4, :], image[:, 2:4, :])


def test_mosaic_handles_partial_last_block() -> None:
    image = _ramp(5, 7)
    mask = np.zeros((5, 7), dtype=bool)
    mask[4, 6] = True  # 端数のある最終ブロック
    output = apply_censor(image, mask, "mosaic", 2)

    expected_block = image[4:5, 6:7].mean(axis=(0, 1))
    assert np.allclose(output[4, 6], expected_block)
    assert np.array_equal(output[0:2, 0:2, :], image[0:2, 0:2, :])


def test_mosaic_grid_origin_is_top_left() -> None:
    image = _ramp(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True  # (0,0) 原点のブロック [0:2, 0:2] に属する
    output = apply_censor(image, mask, "mosaic", 2)

    assert np.allclose(output[0:2, 0:2], image[0:2, 0:2].mean(axis=(0, 1)))
    assert np.array_equal(output[2:4, 2:4, :], image[2:4, 2:4, :])


def test_black_fills_only_mask() -> None:
    image = _ramp()
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 3:5] = True
    output = apply_censor(image, mask, "black", 8)

    assert np.array_equal(output[mask], np.zeros((4, 3), dtype=np.float32))
    assert np.array_equal(output[~mask], image[~mask])


def _checkerboard(height: int = 16, width: int = 16) -> np.ndarray:
    """ぼかしの効果が見えるよう、高周波（市松）の画像を作る。"""
    rows, cols = np.indices((height, width))
    pattern = ((rows + cols) % 2).astype(np.float32)
    return np.repeat(pattern[:, :, None], 3, axis=2)


def test_blur_changes_only_masked_pixels() -> None:
    image = _checkerboard(16, 16)
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:10, 6:10] = True
    output = apply_censor(image, mask, "blur", 2)

    assert np.array_equal(output[~mask], image[~mask])
    assert not np.allclose(output[mask], image[mask])


def test_blur_preserves_constant_image_exactly() -> None:
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    output = apply_censor(image, mask, "blur", 4)

    assert np.allclose(output, image)  # 端の reflect パディングでも定数は保たれる


def test_blur_radius_changes_result() -> None:
    image = _checkerboard(32, 32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    small = apply_censor(image, mask, "blur", 1)
    large = apply_censor(image, mask, "blur", 6)

    assert not np.allclose(small[mask], large[mask])


def test_mask_threshold_is_positive_not_half() -> None:
    image = _ramp(4, 4)
    soft_mask = np.zeros((4, 4), dtype=np.float32)
    soft_mask[0, 0] = 0.3  # 0.5 未満でも「マスクあり」として扱う
    output = apply_censor(image, soft_mask, "black", 2)

    assert float(output[0, 0].sum()) == 0.0


def test_output_dtype_is_float32() -> None:
    image = _ramp().astype(np.float64)
    output = apply_censor(image, np.ones((8, 8), dtype=bool), "mosaic", 2)
    assert output.dtype == np.float32
