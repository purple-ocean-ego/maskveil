"""maskveil — 領域を検出し、マスクで隠すエンジン。

`core/`（純ロジック）と `tools/`（CLI）に加え、ComfyUI 用の `nodes/` を提供する。

ComfyUI はパッケージ属性 `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` /
`WEB_DIRECTORY` を参照してノードを登録する。これらは**参照されたときに初めて**
`nodes/` を読み込む（PEP 562 のモジュール `__getattr__`）。

この遅延読み込みにより:

- CLI（`maskveil` / `maskveil-detect` / `maskveil-refine`）の起動時に
  `torch` を探しに行かない（ComfyUI の無い環境で警告も出ない・起動が速い）
- `torch` が壊れている環境でも CLI と `core` は動く（障害の影響範囲をノードに閉じる）
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

_LAZY_ATTRIBUTES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """ComfyUI が参照したときだけ `nodes` を読み込んで値を返す。"""
    if name not in _LAZY_ATTRIBUTES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    nodes = importlib.import_module(f"{__name__}.nodes")
    value = getattr(nodes, name)
    globals()[name] = value  # 2回目以降は通常の属性として解決する
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_ATTRIBUTES)
