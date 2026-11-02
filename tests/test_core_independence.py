"""`core/` が ComfyUI（comfy / torch / PIL / cv2）に依存しないことの検証。

`tests/test_imports.py` は AST で import 文を禁止しているが、ここでは**実際に
import できない環境**（ブロッカー付きサブプロセス）で `core/` が動くことを確かめる。
依存方向が逆流すると「CLI では動くが ComfyUI では動かない」事故になるため。
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
CORE = PACKAGE / "core"
NODES = PACKAGE / "nodes"
TOOLS = PACKAGE / "tools"

BLOCKED = ("comfy", "folder_paths", "torch", "PIL", "cv2", "ultralytics")

_BLOCKER_SCRIPT = """
import importlib
import sys

BLOCKED = {blocked!r}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("blocked: " + name, name=name.split(".")[0])
        return None


sys.meta_path.insert(0, Blocker())

for module in ("raster", "mosaic", "preview", "masks", "labels", "constants", "types", "pipeline"):
    importlib.import_module("maskveil.core." + module)

import maskveil.core.raster as raster
import maskveil.core.mosaic as mosaic
import maskveil.core.preview as preview

assert hasattr(raster, "rasterize_polygons")
assert hasattr(mosaic, "apply_censor")
assert hasattr(preview, "draw_outline")
print("ok")
"""


def _python_files(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.rglob("*.py"))


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_core_imports_without_comfy_stack() -> None:
    """`core/` は ComfyUI のモジュールが無い環境でも import できる。"""
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCKER_SCRIPT.format(blocked=BLOCKED)],
        cwd=PACKAGE.parent,
        env={"PYTHONPATH": str(PACKAGE.parent), "PATH": "/usr/local/bin:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_core_has_no_forbidden_imports() -> None:
    offending = {
        str(path.relative_to(PACKAGE)): sorted(_imports(path) & set(BLOCKED))
        for path in _python_files(CORE)
        if _imports(path) & set(BLOCKED)
    }
    assert offending == {}


def test_tools_do_not_depend_on_comfy_or_torch() -> None:
    forbidden = {"comfy", "folder_paths", "torch", "cv2", "ultralytics"}
    offending = {
        str(path.relative_to(PACKAGE)): sorted(_imports(path) & forbidden)
        for path in _python_files(TOOLS)
        if _imports(path) & forbidden
    }
    assert offending == {}


def test_torch_is_confined_to_nodes() -> None:
    """`torch` を使うのは ComfyUI アダプタ（`nodes/`）だけである。"""
    production = [path for directory in (CORE, TOOLS) for path in _python_files(directory)]
    users = {str(path.relative_to(PACKAGE)) for path in production if "torch" in _imports(path)}
    assert users == set()
    assert any("torch" in _imports(path) for path in _python_files(NODES))


@pytest.mark.parametrize("name", ["raster", "mosaic", "preview"])
def test_new_core_modules_exist_and_are_pure(name: str) -> None:
    path = CORE / f"{name}.py"
    assert path.is_file()
    assert _imports(path) & set(BLOCKED) == set()


def test_core_package_exports_only_pure_modules() -> None:
    """`core/__init__.py` が ComfyUI 依存のモジュールを import していない。"""
    assert _imports(CORE / "__init__.py") & set(BLOCKED) == set()
