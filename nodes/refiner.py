"""`HotscreenRefiner` ノード（SPEC §4-3）。

`HOTSCREEN` を受け取り、`DET_TO_SEG` に対応のある検出だけを輪郭形状へ置換する。
**`IMAGE` 入力を持たない**（`HOTSCREEN.image` を常に基準にするため）。
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ..core.labels import resolve_classes
from ..core.pipeline import plan_targets, refine_with_refiner
from ..core.raster import compose_masks, expand_detections
from ..core.types import Detection
from ._convert import image_to_numpy, numpy_to_mask
from ._envelope import (
    detection_to_dict,
    dict_to_detection,
    make_envelope,
    validate_envelope,
)
from ._inputs import mask_expand_input, preview_mode_input, threshold_inputs
from ._models import model_list, model_metadata, refiner_for, resolve_model_path
from ._preview import build_preview

__all__ = ["HotscreenRefiner"]


class HotscreenRefiner:
    """検出のマスク形状を輪郭モデルで置換する。"""

    CATEGORY = "hotscreen"
    FUNCTION = "refine"
    RETURN_TYPES = ("HOTSCREEN", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("hotscreen", "mask", "refine", "preview")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        """ComfyUI の入力定義（SPEC §4-3 の `spec-io` を写経）。"""
        return {
            "required": {
                "hotscreen": ("HOTSCREEN",),
                "refiner_model": (model_list(),),
                "refiner_classes": ("STRING", {"default": "all", "multiline": False}),
                **threshold_inputs(),
                "seg_crop": ("BOOLEAN", {"default": False}),
                "mask_expand": mask_expand_input(),
                "preview_mode": preview_mode_input(),
            }
        }

    def refine(
        self,
        hotscreen: dict,
        refiner_model: str,
        refiner_classes: str,
        conf: float,
        iou: float,
        seg_crop: bool,
        mask_expand: int,
        preview_mode: str,
    ) -> tuple[dict, torch.Tensor, str, torch.Tensor]:
        """検出の形状を輪郭へ置換し、更新後の `HOTSCREEN` ほかを返す。

        Args:
            hotscreen: `HotscreenDetector` の出力。
            refiner_model: `folder_paths` の相対パス（輪郭モデル）。
            refiner_classes: 輪郭化する**検出クラス**（`all` は16クラス）。
            conf: 輪郭のスコア閾値。
            iou: 輪郭の NMS IoU 閾値。
            seg_crop: 検出ごとにクロップ推論する（計測用・低速）。
            mask_expand: 置換後の輪郭に適用する膨張量（px）。
            preview_mode: `outline` / `crop` / `none`。

        Returns:
            `(hotscreen, mask, refine, preview)`。

        Raises:
            ValueError: `HOTSCREEN` の版数・必須キーが不正。
            UnknownClassError: `refiner_classes` が空・未知。
            FileNotFoundError: モデルの実体が見つからない。
        """
        envelope = validate_envelope(hotscreen)
        image = envelope["image"]
        frames = image_to_numpy(image)
        batch_size, height, width = frames.shape[0], frames.shape[1], frames.shape[2]

        allowed = frozenset(resolve_classes(refiner_classes))
        path = resolve_model_path(refiner_model)
        refiner = refiner_for(path)
        mask_expand = int(mask_expand)

        entries: list[tuple[int, Detection, int]] = [
            (
                int(item["batch_index"]),
                dict_to_detection(item),  # type: ignore[arg-type]
                int(item["expanded_px"]),  # type: ignore[arg-type]
            )
            for item in envelope["detections"]
        ]

        # 範囲外の batch_index を黙って捨てると「モザイクされない検出」が生まれる
        # （SPEC §3-3 は不一致を明示エラーとする）
        out_of_range = sorted(
            {batch_index for batch_index, _, _ in entries if not 0 <= batch_index < batch_size}
        )
        if out_of_range:
            raise ValueError(
                f"batch_index out of range: {out_of_range} (image batch size {batch_size})"
            )

        updated: list[tuple[int, Detection, int] | None] = [None] * len(entries)
        targets = 0
        refined_count = 0
        for batch_index in range(batch_size):
            positions = [index for index, entry in enumerate(entries) if entry[0] == batch_index]
            if not positions:
                continue
            subset = [entries[index][1] for index in positions]
            plan = plan_targets(subset, allowed_classes=allowed)
            targets += plan.count
            result = refine_with_refiner(
                frames[batch_index],
                subset,
                plan,
                refiner,
                conf=float(conf),
                iou=float(iou),
                seg_crop=bool(seg_crop),
            )
            replaced = set(plan.target_indices)
            for offset, detection in enumerate(result):
                index = positions[offset]
                if offset in replaced and detection.mask_source == "model_seg":
                    # 置換した輪郭にだけ膨張を再適用する（保持した検出は元の膨張を維持）
                    updated[index] = (
                        batch_index,
                        expand_detections([detection], mask_expand)[0],
                        mask_expand,
                    )
                    refined_count += 1
                else:
                    updated[index] = (batch_index, detection, entries[index][2])

        refined = [entry for entry in updated if entry is not None]
        masks = np.zeros((batch_size, height, width), dtype=np.float32)
        for batch_index, detection, expanded_px in refined:
            masks[batch_index] = np.maximum(
                masks[batch_index],
                compose_masks([detection], height, width, expand_px=expanded_px),
            )
        detections = [
            detection_to_dict(detection, batch_index=batch_index, expanded_px=expanded_px)
            for batch_index, detection, expanded_px in refined
        ]

        mask_tensor = numpy_to_mask(masks, like=image)
        meta = dict(envelope["meta"])
        meta["refine"] = {
            "model": model_metadata(path, input_size=refiner.input_size),
            "params": {
                "classes": sorted(allowed),
                "conf": float(conf),
                "iou": float(iou),
                "seg_crop": bool(seg_crop),
                "mask_expand": mask_expand,
            },
        }
        meta["mask_expand"] = mask_expand

        updated_envelope = make_envelope(
            image=envelope["image"],
            mosaic_image=envelope["mosaic_image"],
            mask=mask_tensor,
            detections=detections,
            meta=meta,
        )

        summary = json.dumps(
            {
                "targets": int(targets),
                "refined": int(refined_count),
                "preserved": int(targets - refined_count),
                "mask_source": "model_seg",
            },
            ensure_ascii=False,
        )
        preview = build_preview(image, frames, detections, preview_mode)
        return updated_envelope, mask_tensor, summary, preview
