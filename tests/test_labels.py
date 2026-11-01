"""クラス定義と `--classes` 解決のテスト。"""

from __future__ import annotations

import pytest

from maskveil.core.labels import (
    LABEL_TO_ID,
    LABELS,
    SEXUAL_EXPOSED_LABELS,
    UnknownClassError,
    class_id,
    class_name,
    resolve_classes,
)


def test_labels_have_16_entries_in_model_order() -> None:
    assert len(LABELS) == 16
    assert LABELS[0] == "FEMALE_FACE"
    assert LABELS[3] == "FEMALE_GENITALIA_EXPOSED"
    assert LABELS[11] == "MALE_GENITALIA_EXPOSED"
    assert LABELS[-1] == "EYE"
    assert LABEL_TO_ID["MALE_GENITALIA_EXPOSED"] == 11


def test_class_name_and_id_roundtrip() -> None:
    for index, name in enumerate(LABELS):
        assert class_name(index) == name
        assert class_id(name) == index
    assert class_id("female_face") == 0
    assert class_id("  Male_Face  ") == 1


def test_class_name_and_id_reject_unknown() -> None:
    with pytest.raises(UnknownClassError):
        class_name(16)
    with pytest.raises(UnknownClassError):
        class_name(-1)
    with pytest.raises(UnknownClassError):
        class_id("NOT_A_CLASS")


def test_resolve_classes_supports_all_names_ids_and_mix() -> None:
    assert resolve_classes("all") == tuple(range(16))
    assert resolve_classes("3,11") == (3, 11)
    assert resolve_classes("MALE_GENITALIA_EXPOSED") == (11,)
    assert resolve_classes("FEMALE_FACE,11") == (0, 11)
    # 重複は除去され、昇順に正規化される
    assert resolve_classes("11,3,11") == (3, 11)


@pytest.mark.parametrize("spec", ["", "   ", "FEMALE_FACE,", ",FEMALE_FACE", "1,,2"])
def test_resolve_classes_rejects_empty_tokens(spec: str) -> None:
    with pytest.raises(UnknownClassError):
        resolve_classes(spec)


@pytest.mark.parametrize("spec", ["99", "-1", "NOT_A_CLASS", "all,1"])
def test_resolve_classes_rejects_invalid(spec: str) -> None:
    with pytest.raises(UnknownClassError):
        resolve_classes(spec)


def test_no_default_class_set_is_exposed_for_logic() -> None:
    """ロジック側に既定クラス集合が存在しないこと（UI 初期値のみ別定義）。"""
    import maskveil.core.labels as labels_module

    exported_sets = [
        value
        for name, value in vars(labels_module).items()
        if name.isupper() and isinstance(value, (set, frozenset))
    ]
    assert exported_sets == []
    # UI 初期値としての2クラスは名前で保持されているだけ（解決には使われない）
    assert SEXUAL_EXPOSED_LABELS == ("FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED")
    assert set(SEXUAL_EXPOSED_LABELS) != set(LABELS)
