"""ComfyUI へ公開するノード定義（`NODE_CLASS_MAPPINGS`）。

`CATEGORY` は全ノード共通で `hotscreen`。`WEB_DIRECTORY` を定義すると、
ComfyUI が `web/` 配下の JS を自動配信する（クラス選択 UI 用）。
"""

from __future__ import annotations

from .detector import DETECTOR_CLASSES_UI_DEFAULT, HotscreenDetector
from .mosaic import MosaicByMask
from .refiner import HotscreenRefiner

__all__ = [
    "DETECTOR_CLASSES_UI_DEFAULT",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "HotscreenDetector": HotscreenDetector,
    "HotscreenRefiner": HotscreenRefiner,
    "MosaicByMask": MosaicByMask,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "HotscreenDetector": "Hotscreen Detector",
    "HotscreenRefiner": "Hotscreen Refiner",
    "MosaicByMask": "Mosaic by Mask",
}

WEB_DIRECTORY = "./web"
"""`web/` 配下の JS を ComfyUI に配信させる（クラス選択のチェックボックス）。"""
