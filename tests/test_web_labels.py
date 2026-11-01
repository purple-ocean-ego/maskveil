"""クラス選択 UI（`web/hotscreen_classes.js`）の検証（AC-5）。

JS から Python の定数は読めないため、ラベル配列は**二重管理**になる。
本テストは JS を解析し、`core` の `LABELS` / `DET_TO_SEG` と順序込みで一致することを
機械検証する（論点6の決定事項）。ドリフトしたら pytest が落ちる。
"""

from __future__ import annotations

import pathlib
import re

import pytest

from maskveil.core.class_mapping import DET_TO_SEG
from maskveil.core.labels import LABELS, SEXUAL_EXPOSED_LABELS

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
JS = PACKAGE / "web" / "hotscreen_classes.js"
SOURCE = JS.read_text("utf-8")


def _array(name: str) -> list[str]:
    """`const NAME = [ ... ];` を解析して文字列の並びを返す。"""
    match = re.search(rf"const\s+{name}\s*=\s*\[(.*?)\];", SOURCE, re.DOTALL)
    assert match, f"{name} が JS に見つかりません"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_js_file_exists_and_registers_extension() -> None:
    assert JS.is_file()
    assert "app.registerExtension(" in SOURCE
    assert "beforeRegisterNodeDef" in SOURCE


def test_detection_labels_match_core() -> None:
    assert _array("DETECTION_LABELS") == list(LABELS)


def test_refiner_labels_match_mapped_detection_classes() -> None:
    expected = [name for name in LABELS if name in DET_TO_SEG]
    assert len(expected) == 6
    assert _array("REFINER_LABELS") == expected


def test_refiner_labels_are_in_definition_order() -> None:
    labels = _array("REFINER_LABELS")
    indices = [LABELS.index(name) for name in labels]
    assert indices == sorted(indices)


def test_node_settings_cover_both_nodes() -> None:
    assert re.search(r"HotscreenDetector:\s*\{\s*widget:\s*\"classes\"", SOURCE)
    assert re.search(r"HotscreenRefiner:\s*\{\s*widget:\s*\"refiner_classes\"", SOURCE)
    # MosaicByMask はクラス選択 UI の対象外
    assert "MosaicByMask:" not in SOURCE


def test_widget_names_match_input_types() -> None:
    from maskveil.nodes.detector import HotscreenDetector
    from maskveil.nodes.refiner import HotscreenRefiner

    # `INPUT_TYPES()` は folder_paths を要するため、名前だけを静的に確認する
    for widget in ("classes", "refiner_classes"):
        assert f'"{widget}"' in SOURCE
    assert "HotscreenDetector" in SOURCE and "HotscreenRefiner" in SOURCE
    assert HotscreenDetector.FUNCTION == "detect"
    assert HotscreenRefiner.FUNCTION == "refine"


def test_ui_default_matches_sexual_exposed_labels() -> None:
    from maskveil.nodes.detector import DETECTOR_CLASSES_UI_DEFAULT

    assert ",".join(SEXUAL_EXPOSED_LABELS) == DETECTOR_CLASSES_UI_DEFAULT
    # 初期チェックは STRING の既定値から復元される（JS 側にハードコードしない）
    assert "default_checked" not in SOURCE
    assert "widget.value" in SOURCE


def test_js_handles_all_and_empty() -> None:
    assert 'text.toLowerCase() === "all"' in SOURCE
    assert "return selected;" in SOURCE
    assert 'names.join(",")' in SOURCE


def test_reentrancy_guard_exists() -> None:
    # STRING → チェックボックスの書き戻し中は同期を止める（無限ループ防止）
    assert "syncing" in SOURCE
    assert "state.syncing = true" in SOURCE
    assert "applyValue" in SOURCE


@pytest.mark.parametrize("token", ["FEMALE_FACE", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED"])
def test_every_mapped_class_is_represented(token: str) -> None:
    assert token in _array("REFINER_LABELS")
