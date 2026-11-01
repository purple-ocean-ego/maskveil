"""maskveil — ComfyUI カスタムノードパッケージ。

`core/`（純ロジック）と `tools/`（CLI）に加え、`nodes/`（ComfyUI アダプタ）を公開する。
ComfyUI 以外（CLI・単体テスト）では `torch` / `folder_paths` が無いため、
その場合だけノードの公開を見送る（`core` / `tools` は従来どおり動く）。
"""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)

_NODE_IMPORT_ERROR: ModuleNotFoundError | None = None

try:  # pragma: no cover - ComfyUI 実行時のみ通る
    from .nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        WEB_DIRECTORY,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - ComfyUI 外
    if exc.name not in {"torch", "folder_paths"}:
        raise
    _NODE_IMPORT_ERROR = exc
    _LOGGER.warning(
        "maskveil: ComfyUI ノードを公開しません（%s が見つかりません）。"
        "CLI（tools/）と core/ は利用できます。",
        exc.name,
    )
    NODE_CLASS_MAPPINGS: dict[str, type] = {}
    NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
    WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
