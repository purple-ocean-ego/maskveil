"""hotscreen 検出モデルのクラス定義（16クラス）とクラス指定の解決。

**既定クラス集合は持たない。** `--classes` 未指定は呼び出し側（CLI / ノード）のエラーとする。
"""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "FEMALE_FACE",
    "MALE_FACE",
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_COVERED",
    "FEET_EXPOSED",
    "EYE",
)

LABEL_TO_ID: dict[str, int] = {name: index for index, name in enumerate(LABELS)}

SEXUAL_EXPOSED_LABELS: tuple[str, ...] = (
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
)
"""性器露出の2クラス。**UI の初期値専用**（ロジックの既定値ではない）。"""


class UnknownClassError(ValueError):
    """クラス指定の解決に失敗した。"""


def class_name(class_id: int) -> str:
    """クラスIDからクラス名を返す。"""
    if not 0 <= class_id < len(LABELS):
        raise UnknownClassError(f"class id out of range: {class_id} (0-{len(LABELS) - 1})")
    return LABELS[class_id]


def class_id(name: str) -> int:
    """クラス名からクラスIDを返す（大文字小文字は区別しない）。"""
    key = name.strip().upper()
    try:
        return LABEL_TO_ID[key]
    except KeyError:
        raise UnknownClassError(f"unknown class name: {name!r}") from None


def resolve_classes(spec: str) -> tuple[int, ...]:
    """`--classes` の文字列をクラスIDのタプルへ解決する。

    受け付ける形式:
    - `all`（全16クラス）
    - 名前のカンマ区切り（例: `MALE_GENITALIA_EXPOSED`）
    - IDのカンマ区切り（例: `3,11`）
    - 上記の混在（例: `FEMALE_FACE,11`）

    Raises:
        UnknownClassError: 空文字・未知の名前・範囲外のID。
    """
    if spec is None or not spec.strip():
        raise UnknownClassError("class spec is empty")

    tokens = [token.strip() for token in spec.split(",")]
    if any(token == "" for token in tokens):
        raise UnknownClassError(f"class spec contains an empty token: {spec!r}")

    if len(tokens) == 1 and tokens[0].lower() == "all":
        return tuple(range(len(LABELS)))

    resolved: set[int] = set()
    for token in tokens:
        if token.lstrip("-").isdigit():
            value = int(token)
            if not 0 <= value < len(LABELS):
                raise UnknownClassError(f"class id out of range: {value} (0-{len(LABELS) - 1})")
            resolved.add(value)
        else:
            resolved.add(class_id(token))
    return tuple(sorted(resolved))
