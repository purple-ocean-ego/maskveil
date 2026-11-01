"""検出結果のデータ構造。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Polygon = tuple[Point, ...]
"""閉路ポリゴン（先頭と末尾の点は同一）。"""

MaskSource = Literal["none", "model_seg", "bbox_ellipse"]


@dataclass(frozen=True, slots=True)
class Detection:
    """1件の検出結果（座標は原画像px）。

    `masks` は連結成分ごとのポリゴン。マスクが分断された場合もすべて保持する
    （1本だけ採用すると未被覆の領域が生まれるため）。
    """

    class_id: int
    class_name: str
    score: float
    bbox: BBox
    masks: tuple[Polygon, ...] | None = None
    mask_source: MaskSource = "none"

    def to_dict(self, *, include_mask: bool = False) -> dict[str, object]:
        """JSON 化用の dict を返す。"""
        data: dict[str, object] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": round(float(self.score), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
        }
        if include_mask:
            data["mask_source"] = self.mask_source
            if self.masks is not None:
                data["masks"] = [
                    [[round(float(x), 2), round(float(y), 2)] for x, y in polygon]
                    for polygon in self.masks
                ]
        return data


@dataclass(frozen=True, slots=True)
class LetterboxMeta:
    """letterbox 前処理の逆変換に必要な情報。

    倍率は軸ごとに保持する（整数へ丸めた実サイズから算出するため、往復誤差が出ない）。
    """

    src_width: int
    src_height: int
    dst_size: int
    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0

    @property
    def src_size(self) -> tuple[int, int]:
        return self.src_width, self.src_height

    @property
    def scaled_size(self) -> tuple[int, int]:
        """パディング前の実サイズ（幅, 高さ）。"""
        return int(round(self.src_width * self.scale_x)), int(round(self.src_height * self.scale_y))


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """使用したモデルの識別情報（証跡用）。"""

    path: str
    sha256: str = field(default="")
    input_size: int = 0

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"path": self.path}
        if self.sha256:
            data["sha256"] = self.sha256
        if self.input_size:
            data["input_size"] = self.input_size
        return data
