"""依存方向のテスト（`core` が ComfyUI に依存しないこと）。"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
CORE = PACKAGE / "core"
TOOLS = PACKAGE / "tools"

FORBIDDEN_IN_CORE = {"comfy", "folder_paths", "torch", "PIL", "cv2", "ultralytics"}
FORBIDDEN_IN_TOOLS = {"comfy", "folder_paths", "torch", "cv2", "ultralytics"}


def _imported_top_level_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _python_files(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.rglob("*.py"))


def test_core_has_python_files() -> None:
    assert len(_python_files(CORE)) >= 7


def test_core_does_not_import_comfy_or_heavy_deps() -> None:
    violations: dict[str, set[str]] = {}
    for path in _python_files(CORE):
        found = _imported_top_level_modules(path) & FORBIDDEN_IN_CORE
        if found:
            violations[path.name] = found
    assert violations == {}


def test_tools_do_not_import_comfy() -> None:
    violations: dict[str, set[str]] = {}
    for path in _python_files(TOOLS):
        found = _imported_top_level_modules(path) & FORBIDDEN_IN_TOOLS
        if found:
            violations[path.name] = found
    assert violations == {}


@pytest.mark.parametrize("module", ["maskveil.core.labels", "maskveil.core.geometry"])
def test_core_modules_are_importable(module: str) -> None:
    __import__(module)
