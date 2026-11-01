"""ONNX セッションの入力仕様解決。"""

from __future__ import annotations

from typing import Any


def resolve_input_size(session: Any, requested: int | None, *, label: str) -> int:
    """モデルの静的入力サイズを解決し、指定値と矛盾しないことを検証する。

    Args:
        session: `onnxruntime.InferenceSession`。
        requested: 利用者が指定した入力一辺（`None` ならモデルから取得）。
        label: エラーメッセージ用の名称（`"detector"` / `"refiner"`）。

    Raises:
        ValueError: 動的入力なのに未指定、または指定値がモデルと不一致の場合。
    """
    shape = session.get_inputs()[0].shape
    model_size: int | None = None
    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        height, width = shape[2], shape[3]
        if height != width:
            raise ValueError(f"{label} model has a non-square input: {height}x{width}")
        model_size = int(height)

    if requested is None:
        if model_size is None:
            raise ValueError(f"{label} model has a dynamic input size; specify it explicitly")
        return model_size

    if model_size is not None and int(requested) != model_size:
        raise ValueError(f"{label} model expects input size {model_size}, got {requested}")
    return int(requested)
