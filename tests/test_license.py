"""ライセンス衛生: AGPL コードを同梱していないことを機械検証する。

本パッケージは **書き出し済み ONNX の推論のみ** を行い、`ultralytics`（AGPL-3.0）を
import しない。`nodes/` は既存の `tests/test_imports.py` の対象外なので、ここで補う。
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _toml import tomllib

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
PY_DIRS = ("core", "nodes", "tools")
FORBIDDEN_THIRD_PARTY = {"ultralytics"}


def _python_files() -> list[pathlib.Path]:
    return sorted(path for directory in PY_DIRS for path in (PACKAGE / directory).rglob("*.py"))


def test_python_files_are_discovered() -> None:
    files = _python_files()
    assert any("nodes" in str(path) for path in files)
    assert any("core" in str(path) for path in files)


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_agpl_imports(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & FORBIDDEN_THIRD_PARTY), f"{path} が AGPL ライブラリを import している"


def test_no_agpl_references_in_source() -> None:
    offending = []
    for path in _python_files():
        text = path.read_text("utf-8")
        if "ultralytics" in text:
            offending.append(str(path.relative_to(PACKAGE)))
    for path in (PACKAGE / "web").rglob("*.js"):
        if "ultralytics" in path.read_text("utf-8"):
            offending.append(str(path.relative_to(PACKAGE)))
    assert offending == []


def test_dependencies_exclude_agpl_packages() -> None:
    data = tomllib.loads((PACKAGE / "pyproject.toml").read_text("utf-8"))
    dependencies = data["project"]["dependencies"]
    names = {
        dependency.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        for dependency in dependencies
    }
    assert names.isdisjoint(FORBIDDEN_THIRD_PARTY)
    assert {"numpy", "onnxruntime"} <= names


def test_no_torch_in_core_dependencies() -> None:
    """`torch` は ComfyUI が提供する。パッケージの依存に足すと CLI 利用者を壊す。"""
    data = tomllib.loads((PACKAGE / "pyproject.toml").read_text("utf-8"))
    dependencies = {dependency.lower() for dependency in data["project"]["dependencies"]}
    assert not any(dependency.startswith("torch") for dependency in dependencies)


def test_readme_states_onnx_only_policy() -> None:
    readme = (PACKAGE / "README.md").read_text("utf-8")
    assert "onnxruntime" in readme
    assert "AGPL" in readme
