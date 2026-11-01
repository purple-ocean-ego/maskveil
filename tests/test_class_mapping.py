"""検出16クラス → 輪郭7クラス対応表のテスト。"""

from __future__ import annotations

import pytest

from maskveil.core.class_mapping import (
    DET_TO_SEG,
    SEG_CLASS_TO_ID,
    SEG_CLASSES,
    has_shape_source,
    seg_target,
    seg_target_id,
)
from maskveil.core.labels import LABELS


def test_seg_classes_are_the_expected_seven() -> None:
    assert SEG_CLASSES == (
        "anus",
        "nipple",
        "penis",
        "vagina",
        "female face",
        "male face",
        "pubic hair",
    )
    assert SEG_CLASS_TO_ID["penis"] == 2


def test_mapping_keys_are_valid_detection_classes() -> None:
    assert set(DET_TO_SEG) <= set(LABELS)


def test_mapping_targets_are_valid_seg_classes() -> None:
    assert set(DET_TO_SEG.values()) <= set(SEG_CLASSES)


@pytest.mark.parametrize(
    ("det", "expected"),
    [
        ("FEMALE_GENITALIA_EXPOSED", "vagina"),
        ("MALE_GENITALIA_EXPOSED", "penis"),
        ("ANUS_EXPOSED", "anus"),
        ("FEMALE_BREAST_EXPOSED", "nipple"),
        ("FEMALE_FACE", "female face"),
        ("MALE_FACE", "male face"),
    ],
)
def test_known_mappings(det: str, expected: str) -> None:
    assert seg_target(det) == expected
    assert seg_target_id(det) == SEG_CLASS_TO_ID[expected]
    assert has_shape_source(det) is True


@pytest.mark.parametrize(
    "det",
    [
        "FEMALE_GENITALIA_COVERED",
        "BUTTOCKS_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ARMPITS_EXPOSED",
        "BELLY_EXPOSED",
        "FEET_EXPOSED",
        "EYE",
    ],
)
def test_unmapped_classes_have_no_shape_source(det: str) -> None:
    assert seg_target(det) is None
    assert seg_target_id(det) is None
    assert has_shape_source(det) is False


def test_pubic_hair_has_no_detection_source() -> None:
    assert "pubic hair" in SEG_CLASSES
    assert "pubic hair" not in set(DET_TO_SEG.values())
