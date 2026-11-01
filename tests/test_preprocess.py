"""前処理のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from maskveil.core.preprocess import to_model_input


def test_to_model_input_shape_dtype_and_range() -> None:
    image = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    tensor, meta = to_model_input(image, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert meta.src_width == 640
    assert meta.src_height == 480


def test_to_model_input_normalizes_white_to_one() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    tensor, _ = to_model_input(image, 64)
    assert np.allclose(tensor, 1.0)


def test_to_model_input_pads_with_gray_114() -> None:
    image = np.zeros((64, 32, 3), dtype=np.uint8)
    tensor, meta = to_model_input(image, 64)
    # 32x64 → 32x64 (scale=1) / pad_x = 16
    assert meta.pad_x == 16.0
    assert np.allclose(tensor[0, :, 0, 0], 114 / 255)
    assert np.allclose(tensor[0, :, 0, 16], 0.0)


def test_to_model_input_rejects_non_rgb() -> None:
    with pytest.raises(ValueError):
        to_model_input(np.zeros((10, 10), dtype=np.uint8), 64)
    with pytest.raises(ValueError):
        to_model_input(np.zeros((10, 10, 4), dtype=np.uint8), 64)
