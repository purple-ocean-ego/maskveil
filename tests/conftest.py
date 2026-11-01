"""pytest 共通設定。

リポジトリ直下から実行してもパッケージを import できるように、
パッケージの親ディレクトリを `sys.path` へ追加する。

**clone 先のディレクトリ名は `maskveil` である必要がある。**
`import maskveil` は親ディレクトリ直下の `maskveil/` を解決するため、
別名で clone すると import できない（その場合は `pip install -e .` を使う）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = ROOT.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))
