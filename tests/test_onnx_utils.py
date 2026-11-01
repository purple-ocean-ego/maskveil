"""ONNX 入力サイズ解決のテスト。"""

from __future__ import annotations

import pytest

from maskveil.core.onnx_utils import resolve_input_size


class _Input:
    def __init__(self, shape: list[object]) -> None:
        self.shape = shape


class FakeSession:
    def __init__(self, shape: list[object]) -> None:
        self._shape = shape

    def get_inputs(self) -> list[_Input]:
        return [_Input(self._shape)]


def test_derives_static_square_size() -> None:
    session = FakeSession([1, 3, 1280, 1280])
    assert resolve_input_size(session, None, label="refiner") == 1280


def test_accepts_matching_request() -> None:
    session = FakeSession([1, 3, 640, 640])
    assert resolve_input_size(session, 640, label="detector") == 640


def test_rejects_mismatched_request() -> None:
    session = FakeSession([1, 3, 1280, 1280])
    with pytest.raises(ValueError, match="expects input size 1280"):
        resolve_input_size(session, 640, label="refiner")


def test_dynamic_input_requires_explicit_size() -> None:
    session = FakeSession([1, 3, "height", "width"])
    with pytest.raises(ValueError, match="dynamic input size"):
        resolve_input_size(session, None, label="refiner")
    assert resolve_input_size(session, 960, label="refiner") == 960


def test_rejects_non_square_input() -> None:
    session = FakeSession([1, 3, 640, 320])
    with pytest.raises(ValueError, match="non-square"):
        resolve_input_size(session, None, label="detector")
