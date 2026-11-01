"""公開 API の契約テスト（既定クラスを持たない・パッケージとしてロード可能）。"""

from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest

from maskveil.core.detector import Detector
from maskveil.core.refiner import Refiner

PACKAGE = pathlib.Path(__file__).resolve().parents[1]


def test_package_has_root_init() -> None:
    """ComfyUI は `<pkg>/__init__.py` を起点にロードするため必須。"""
    assert (PACKAGE / "__init__.py").is_file()


@pytest.mark.parametrize("method", [Detector.detect, Refiner.detect])
def test_class_ids_has_no_default(method) -> None:
    """AC3（ロジックに既定クラスを持たない）を API でも保証する。"""
    parameter = inspect.signature(method).parameters["class_ids"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_empty_class_ids_is_rejected() -> None:
    with pytest.raises(ValueError):
        Detector.detect(object(), np.zeros((4, 4, 3), dtype=np.uint8), class_ids=())
    with pytest.raises(ValueError):
        Refiner.detect(object(), np.zeros((4, 4, 3), dtype=np.uint8), class_ids=())
