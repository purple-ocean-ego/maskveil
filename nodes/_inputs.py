"""ノード間で共通の入力ウィジェット定義。

`conf` / `iou` / `mask_expand` / `preview_mode` の既定値・範囲はノード仕様として固定されており、
複数ノードで同じ値を使う。定義を1箇所に集約し、片方だけ更新される事故を防ぐ。

各関数は**呼ぶたびに新しい dict / tuple を返す**（ComfyUI が入力定義を書き換えても
他のノードへ影響しないようにするため）。
"""

from __future__ import annotations

from typing import Any

from ..core.constants import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MASK_EXPAND
from ._preview import PREVIEW_MODES

__all__ = ["mask_expand_input", "preview_mode_input", "threshold_inputs"]


def threshold_inputs() -> dict[str, tuple[Any, ...]]:
    """`conf` / `iou`（0.0–1.0 / step 0.01）の入力定義を返す。"""
    return {
        "conf": ("FLOAT", {"default": DEFAULT_CONF, "min": 0.0, "max": 1.0, "step": 0.01}),
        "iou": ("FLOAT", {"default": DEFAULT_IOU, "min": 0.0, "max": 1.0, "step": 0.01}),
    }


def mask_expand_input() -> tuple[Any, ...]:
    """`mask_expand`（0–256 px）の入力定義を返す。"""
    return ("INT", {"default": DEFAULT_MASK_EXPAND, "min": 0, "max": 256, "step": 1})


def preview_mode_input() -> tuple[Any, ...]:
    """`preview_mode`（`outline` / `crop` / `none`）の入力定義を返す。"""
    return (list(PREVIEW_MODES), {"default": "outline"})
