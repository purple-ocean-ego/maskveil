"""pytest 共通設定。

リポジトリのソースを、**clone 先のディレクトリ名に依存せず** `maskveil` として解決する。

1. site-packages にインストール済みの `maskveil` がこのリポジトリを指しているなら、それを使う
   （`pip install -e .` 済みの環境）。
2. そうでなければ、パッケージの親ディレクトリを `sys.path` へ追加する（clone 名が `maskveil` の場合に効く）。
3. それでも解決できない場合（`maskveil-main` のように別名で展開した場合）は、
   `importlib` でこのリポジトリを `maskveil` として登録する。

いずれの経路でも、**解決結果がこのリポジトリを指していること**を検証する
（別バージョンの `maskveil` をテストしてしまう事故を防ぐ）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolved_module_path() -> Path | None:
    try:
        import maskveil  # noqa: PLC0415 - 解決結果を見るため遅延 import する
    except ModuleNotFoundError:
        return None
    file = getattr(maskveil, "__file__", None)
    return Path(file).resolve().parent if file else None


def _ensure_package_is_this_tree() -> None:
    resolved = _resolved_module_path()
    if resolved == ROOT:
        return

    parent = str(ROOT.parent)
    if ROOT.name == "maskveil" and parent not in sys.path:
        sys.path.insert(0, parent)
        if _resolved_module_path() == ROOT:
            return

    # 最後の手段: 別名ディレクトリでも、このリポジトリを `maskveil` として登録する
    spec = importlib.util.spec_from_file_location(
        "maskveil",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - 防御的
        raise RuntimeError(f"maskveil を解決できませんでした: {ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["maskveil"] = module
    spec.loader.exec_module(module)

    resolved = _resolved_module_path()
    if resolved != ROOT:  # pragma: no cover - 防御的
        raise RuntimeError(f"テスト対象がこのリポジトリではありません: {resolved} != {ROOT}")


_ensure_package_is_this_tree()
