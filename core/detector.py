"""検出モデル（hotscreen YOLO11）の推論ラッパー。

パスの探索・検証は行わない。検証済みの絶対パスを受け取る。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .constants import DEFAULT_CONF, DEFAULT_IOU
from .onnx_utils import resolve_input_size
from .postprocess import decode_detections
from .preprocess import to_model_input
from .types import Detection


class Detector:
    """hotscreen 検出モデルのセッション。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_size = resolve_input_size(self.session, input_size, label="detector")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(
        self,
        image: np.ndarray,
        *,
        class_ids: tuple[int, ...] | list[int],
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> list[Detection]:
        """`(H, W, 3)` uint8 RGB 画像から検出結果を返す（原画像座標）。"""
        if not class_ids:
            raise ValueError("class_ids must not be empty")

        tensor, meta = to_model_input(image, self.input_size)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return decode_detections(
            outputs[0],
            meta=meta,
            class_ids=class_ids,
            conf=conf,
            iou=iou,
        )
