"""`HotscreenDetector` ノード（SPEC §4-2）。

`core` の推論 API を呼び、`HOTSCREEN` / `MASK` / `SEGS` / `detection` / `preview` を
組み立てるだけの薄いアダプタ。判断ロジックは `core` に置く。
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.labels import SEXUAL_EXPOSED_LABELS, resolve_classes
from ..core.raster import compose_masks, expand_detections
from ..core.types import Detection
from ._convert import image_to_numpy, numpy_to_mask
from ._envelope import base_meta, detection_json, detection_to_dict, make_envelope
from ._inputs import mask_expand_input, preview_mode_input, threshold_inputs
from ._models import detector_for, model_list, model_metadata, resolve_model_path
from ._preview import build_preview
from ._segs import build_segs

__all__ = ["DETECTOR_CLASSES_UI_DEFAULT", "HotscreenDetector"]

DETECTOR_CLASSES_UI_DEFAULT = ",".join(SEXUAL_EXPOSED_LABELS)
"""`classes` の UI 初期値（性器露出の2クラス）。**UI 専用で、ロジックの既定値ではない。**"""


class HotscreenDetector:
    """NSFW 部位を検出し、マスクと互換ビューを出力する。"""

    CATEGORY = "hotscreen"
    FUNCTION = "detect"
    RETURN_TYPES = ("HOTSCREEN", "MASK", "SEGS", "STRING", "IMAGE")
    RETURN_NAMES = ("hotscreen", "mask", "segs", "detection", "preview")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        """ComfyUI の入力定義（SPEC §4-2 の `spec-io` を写経）。"""
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list(),),
                "classes": (
                    "STRING",
                    {"default": DETECTOR_CLASSES_UI_DEFAULT, "multiline": False},
                ),
                **threshold_inputs(),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 300, "step": 1}),
                "mask_expand": mask_expand_input(),
                "preview_mode": preview_mode_input(),
            }
        }

    def detect(
        self,
        image: torch.Tensor,
        model: str,
        classes: str,
        conf: float,
        iou: float,
        top_k: int,
        mask_expand: int,
        preview_mode: str,
    ) -> tuple[dict, torch.Tensor, tuple, str, torch.Tensor]:
        """画像バッチを検出し、`HOTSCREEN` ほかを返す。

        Args:
            image: `IMAGE [B, H, W, C]`。
            model: `folder_paths` の相対パス。
            classes: クラス指定（名前 / ID / `all` のカンマ区切り）。
            conf: スコア閾値。
            iou: NMS の IoU 閾値。
            top_k: 画像ごとのスコア上位N件（0 は無効）。
            mask_expand: マスク形状の膨張量（px）。
            preview_mode: `outline` / `crop` / `none`。

        Returns:
            `(hotscreen, mask, segs, detection, preview)`。

        Raises:
            UnknownClassError: `classes` が空・未知。
            FileNotFoundError: モデルの実体が見つからない。
            ValueError: モデルの入力サイズ不一致など。
        """
        class_ids = tuple(resolve_classes(classes))
        path = resolve_model_path(model)
        detector = detector_for(path)
        mask_expand = int(mask_expand)
        top_k = int(top_k)

        frames = image_to_numpy(image)
        batch_size, height, width = frames.shape[0], frames.shape[1], frames.shape[2]

        grouped: dict[int, list[Detection]] = {}
        for batch_index in range(batch_size):
            found = detector.detect(
                frames[batch_index],
                class_ids=class_ids,
                conf=float(conf),
                iou=float(iou),
            )
            if top_k > 0:
                # `core` はスコア降順で返すため、先頭N件がスコア上位N件になる
                found = found[:top_k]
            grouped[batch_index] = expand_detections(found, mask_expand)

        masks = np.zeros((batch_size, height, width), dtype=np.float32)
        detections: list[dict] = []
        for batch_index, found in grouped.items():
            masks[batch_index] = compose_masks(found, height, width, expand_px=mask_expand)
            detections.extend(
                detection_to_dict(item, batch_index=batch_index, expanded_px=mask_expand)
                for item in found
            )

        mask_tensor = numpy_to_mask(masks, like=image)
        params = {
            "classes": list(class_ids),
            "conf": float(conf),
            "iou": float(iou),
            "top_k": top_k,
            "mask_expand": mask_expand,
        }
        meta = base_meta(size=(batch_size, height, width))
        meta["det"] = {
            "model": model_metadata(path, input_size=detector.input_size),
            "params": params,
        }
        meta["mask_expand"] = mask_expand

        envelope = make_envelope(
            image=image,
            mask=mask_tensor,
            detections=detections,
            meta=meta,
        )
        segs = build_segs(image, detections, width=width, height=height)
        payload = detection_json(
            model=model,
            input_size=detector.input_size,
            params=params,
            detections=detections,
        )
        preview = build_preview(image, frames, detections, preview_mode)
        return envelope, mask_tensor, segs, payload, preview
