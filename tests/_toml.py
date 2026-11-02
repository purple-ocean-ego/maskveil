"""TOML 読み込みの互換レイヤ。

Python 3.11 以降は標準ライブラリの `tomllib`、3.10 では dev 依存の `tomli` を使う。
CI の Python 3.10 / 3.12 レグで同じテストを動かすために挟んでいる。
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):  # pragma: no cover - 3.10 では通らない
    import tomllib
else:  # pragma: no cover - 3.11 以降では通らない
    import tomli as tomllib

__all__ = ["tomllib"]
