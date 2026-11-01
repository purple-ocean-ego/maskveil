"""ComfyUI の実際のローダ手順でパッケージを読み込めることの検証（AC-1 / C-1）。

ComfyUI はディレクトリ型のカスタムノードを

    sys_module_name = module_path.replace(".", "_x_")   # 例: /workspace/.../maskveil
    spec = importlib.util.spec_from_file_location(sys_module_name, "<dir>/__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    spec.loader.exec_module(module)

として読み込み、**`custom_nodes` を `sys.path` に追加しない**（`ComfyUI/nodes.py`）。
そのため `nodes/` 内の絶対 import（`import maskveil...`）は解決できず、
パック全体が IMPORT FAILED になる。

ここでは「パッケージの親を `sys.path` に入れない」状態で同じ手順を再現し、
相対 import でロードできることを固定する。
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

_LOADER_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys

    path = sys.argv[1]
    sys_module_name = path.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(
        sys_module_name, path + "/__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    spec.loader.exec_module(module)

    print(json.dumps({
        "nodes": sorted(module.NODE_CLASS_MAPPINGS),
        "web": module.WEB_DIRECTORY,
        "package_importable": "maskveil" in sys.modules,
    }))
    """
)


def test_package_loads_with_comfyui_loader(tmp_path) -> None:
    """マングル名ロード（`custom_nodes` 非 `sys.path`）で3ノードが公開される。"""
    target = tmp_path / "maskveil"
    shutil.copytree(
        PACKAGE,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )

    completed = subprocess.run(
        [sys.executable, "-c", _LOADER_SCRIPT, str(target)],
        # パッケージの親を sys.path に入れない（cwd も親にしない）
        cwd="/",
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["nodes"] == ["HotscreenDetector", "HotscreenRefiner", "MosaicByMask"]
    assert payload["web"] == "./web"
    # 絶対 import に戻すとここが True になり、ComfyUI では解決できなくなる
    assert payload["package_importable"] is False


@pytest.mark.parametrize(
    "path",
    sorted(PACKAGE.glob("nodes/*.py")),
    ids=lambda p: p.name,
)
def test_nodes_use_relative_imports(path) -> None:
    """`nodes/` は自パッケージを絶対名で import しない（ComfyUI ローダ対策）。"""
    text = path.read_text("utf-8")
    assert "from maskveil" not in text
    assert "import maskveil" not in text
