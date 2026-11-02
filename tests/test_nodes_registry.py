"""ノード登録の検証。

ComfyUI のノード一覧に載るために必要な属性（`NODE_CLASS_MAPPINGS` / `CATEGORY` /
`FUNCTION` / `INPUT_TYPES` / `RETURN_TYPES` / `WEB_DIRECTORY`）を、
`tests/fixtures/nodes_contract.json`と突き合わせる。

`folder_paths` は ComfyUI 本体が提供するためスタブを差し込み、
「ノード側が ComfyUI の実 API をどう使うか」だけを検証する。
"""

from __future__ import annotations

import json
import pathlib
import sys
import types
from typing import Any

import pytest

pytest.importorskip("torch", reason="torch は ComfyUI が提供する。torch 無しの CI レグではスキップ")

from maskveil import nodes as nodes_module

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = json.loads((PACKAGE / "tests" / "fixtures" / "nodes_contract.json").read_text("utf-8"))
NODES: dict[str, Any] = CONTRACT["nodes"]

HOTSCREEN_MODELS = [
    "det/hs-real-anime-y11n-640-fp32.onnx",
    "det/labels.txt",
    "seg/nsfw-anime-xl-x1280.onnx",
]
"""`get_filename_list("hotscreen")` の実測値（`.onnx` 以外も混ざる）。"""


@pytest.fixture(autouse=True)
def folder_paths_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """`folder_paths` のスタブ（ComfyUI 本体の代わり）。

    `nodes/_models.py` は `folder_paths` を**遅延 import** するため、
    `INPUT_TYPES()` を呼ぶ間だけ差し込めば足りる。
    """
    stub = types.ModuleType("folder_paths")
    stub.get_filename_list = lambda folder: (  # type: ignore[attr-defined]
        list(HOTSCREEN_MODELS) if folder == "hotscreen" else []
    )
    stub.get_full_path = lambda folder, filename: (  # type: ignore[attr-defined]
        f"/workspace/models/{folder}/{filename}"
    )
    monkeypatch.setitem(sys.modules, "folder_paths", stub)
    return stub


def _inputs(node_module: types.ModuleType, key: str) -> dict[str, Any]:
    class_type = node_module.NODE_CLASS_MAPPINGS[key]
    return class_type.INPUT_TYPES()


def _widget_specs(input_types: dict[str, Any]) -> list[tuple[str, Any]]:
    """`INPUT_TYPES()` を (name, spec) の並びへ平坦化する（required → optional）。"""
    flattened: list[tuple[str, Any]] = []
    for section in ("required", "optional"):
        for name, spec in input_types.get(section, {}).items():
            flattened.append((name, spec))
    return flattened


def test_node_class_mappings_has_three_nodes() -> None:
    assert set(nodes_module.NODE_CLASS_MAPPINGS) == set(NODES)


def test_display_names_match_contract() -> None:
    for key, spec in NODES.items():
        assert nodes_module.NODE_DISPLAY_NAME_MAPPINGS[key] == spec["title"]


@pytest.mark.parametrize("key", sorted(NODES))
def test_category_and_function(key: str) -> None:
    class_type = nodes_module.NODE_CLASS_MAPPINGS[key]
    assert class_type.CATEGORY == "hotscreen"
    assert class_type.OUTPUT_NODE is False
    assert class_type.FUNCTION in {"detect", "refine", "apply"}
    assert callable(getattr(class_type, class_type.FUNCTION))


@pytest.mark.parametrize("key", sorted(NODES))
def test_input_types_match_contract(key: str) -> None:
    input_types = _inputs(nodes_module, key)
    assert set(input_types) <= {"required", "optional"}

    flattened = _widget_specs(input_types)
    assert [name for name, _ in flattened] == [item["name"] for item in NODES[key]["inputs"]]

    for (name, spec), expected in zip(flattened, NODES[key]["inputs"], strict=True):
        required = name in input_types["required"]
        assert required is expected["required"], f"{key}.{name}: required 不一致"

        if expected["type"] == "COMBO":
            # 写像規則: COMBO は選択肢そのものが型文字列の位置に入る
            options = list(spec[0])
            if "options" in expected:
                assert options == expected["options"], f"{key}.{name}: 選択肢不一致"
                assert (spec[1] if len(spec) > 1 else {}).get("default") == expected["default"]
            continue

        assert spec[0] == expected["type"], f"{key}.{name}: 型文字列不一致"

        options = spec[1] if len(spec) > 1 else {}
        if expected["widget"] in {"number", "text", "toggle"}:
            assert options.get("default") == expected["default"]
        for bound in ("min", "max", "step"):
            if bound in expected:
                assert options.get(bound) == expected[bound], f"{key}.{name}: {bound} 不一致"


def test_detector_model_input_is_onnx_only() -> None:
    input_types = _inputs(nodes_module, "HotscreenDetector")
    model_spec = input_types["required"]["model"]
    assert list(model_spec[0]) == [
        "det/hs-real-anime-y11n-640-fp32.onnx",
        "seg/nsfw-anime-xl-x1280.onnx",
    ]
    # COMBO の既定値は先頭要素なので、default キーは持たない
    assert len(model_spec) == 1


@pytest.mark.parametrize("key", sorted(NODES))
def test_return_types_match_contract(key: str) -> None:
    class_type = nodes_module.NODE_CLASS_MAPPINGS[key]
    expected = NODES[key]["outputs"]
    assert list(class_type.RETURN_TYPES) == [item["type"] for item in expected]
    assert list(class_type.RETURN_NAMES) == [item["name"] for item in expected]


def test_web_directory_and_js_exist() -> None:
    assert nodes_module.WEB_DIRECTORY == "./web"
    assert (PACKAGE / "web" / "hotscreen_classes.js").is_file()


def test_detector_classes_ui_default() -> None:
    from maskveil.core.labels import SEXUAL_EXPOSED_LABELS

    assert ",".join(SEXUAL_EXPOSED_LABELS) == nodes_module.DETECTOR_CLASSES_UI_DEFAULT
    input_types = _inputs(nodes_module, "HotscreenDetector")
    assert (
        input_types["required"]["classes"][1]["default"] == nodes_module.DETECTOR_CLASSES_UI_DEFAULT
    )
