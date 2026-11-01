"""検出クラス（16）と輪郭クラス（7）の対応表。

輪郭モデル（YOLO11-Seg）が持つクラスは限られている。対応先が無い検出クラスは
`mask_source: "none"` として形状を付けない（bbox のみ）。対応関係はここにデータとして持つ。
"""

from __future__ import annotations

SEG_CLASSES: tuple[str, ...] = (
    "anus",
    "nipple",
    "penis",
    "vagina",
    "female face",
    "male face",
    "pubic hair",
)

SEG_CLASS_TO_ID: dict[str, int] = {name: index for index, name in enumerate(SEG_CLASSES)}

DET_TO_SEG: dict[str, str] = {
    "FEMALE_GENITALIA_EXPOSED": "vagina",
    "MALE_GENITALIA_EXPOSED": "penis",
    "ANUS_EXPOSED": "anus",
    "FEMALE_BREAST_EXPOSED": "nipple",
    "FEMALE_FACE": "female face",
    "MALE_FACE": "male face",
}
"""検出クラス名 → 輪郭クラス名。ここに無い検出クラスは形状を付けない。"""


def seg_target(det_class_name: str) -> str | None:
    """検出クラスに対応する輪郭クラス名を返す（無ければ None）。"""
    return DET_TO_SEG.get(det_class_name)


def seg_target_id(det_class_name: str) -> int | None:
    """検出クラスに対応する輪郭クラスIDを返す（無ければ None）。"""
    target = seg_target(det_class_name)
    return None if target is None else SEG_CLASS_TO_ID[target]


def has_shape_source(det_class_name: str) -> bool:
    """その検出クラスに形状の供給元があるか。"""
    return det_class_name in DET_TO_SEG
