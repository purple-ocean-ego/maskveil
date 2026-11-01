"""`MosaicByMask` ノード（SPEC §4-4 / §6）。

**判断処理を持たない**（`classes` / `conf` / `threshold` / `mask_margin` を持たず、
`detections` も参照しない）。受け取ったマスクをそのまま適用するだけのノード。
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.constants import DEFAULT_BLOCK_SIZE
from ..core.mosaic import CENSOR_MODES, apply_censor
from ._convert import image_to_numpy, mask_to_float, numpy_to_image
from ._envelope import base_meta, make_envelope, validate_envelope

__all__ = ["MosaicByMask"]


class MosaicByMask:
    """マスク領域をモザイク / ぼかし / 黒塗りで隠す。"""

    CATEGORY = "hotscreen"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "HOTSCREEN")
    RETURN_NAMES = ("image", "hotscreen")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        """ComfyUI の入力定義（SPEC §4-4 の `spec-io` を写経）。"""
        return {
            "required": {
                "mode": (list(CENSOR_MODES), {"default": "mosaic"}),
                "block_size": (
                    "INT",
                    {"default": DEFAULT_BLOCK_SIZE, "min": 2, "max": 128, "step": 1},
                ),
            },
            "optional": {
                "hotscreen": ("HOTSCREEN",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def apply(
        self,
        mode: str,
        block_size: int,
        hotscreen: dict | None = None,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """マスク領域を加工し、加工後画像と更新後の `HOTSCREEN` を返す。

        ベース画像は「明示 `image` > `hotscreen.mosaic_image` > `hotscreen.image`」の順で
        解決する（明示入力が最優先）。マスクは「明示 `mask` > `hotscreen.mask`」。

        Args:
            mode: `mosaic` / `blur` / `black`。
            block_size: `mosaic` のブロック辺、`blur` の半径（px）。
            hotscreen: 任意。接続するだけで画像とマスクを内包できる。
            image: 任意。明示指定（多段モザイクでは繋がない）。
            mask: 任意。明示指定（別 Detector のマスクを差し込む入口）。

        Returns:
            `(image, hotscreen)`。マスクが全ゼロなら入力をそのまま返し、`mosaic_image` と
            `meta.processing` は更新しない。

        Raises:
            ValueError: 画像／マスクの供給が無い、バッチ数・解像度の不一致、
                `schema_version` が1以外。
        """
        envelope = validate_envelope(hotscreen) if hotscreen is not None else None

        base, source = self._resolve_image(envelope, image)
        if envelope is not None and envelope["mosaic_image"] is not None:
            mosaic_shape = tuple(envelope["mosaic_image"].shape[:3])
            image_shape = tuple(envelope["image"].shape[:3])
            if mosaic_shape != image_shape:
                raise ValueError(
                    f"mosaic_image size mismatch: image={image_shape} mosaic_image={mosaic_shape}"
                )

        mask_tensor = mask if mask is not None else (envelope["mask"] if envelope else None)
        if mask_tensor is None:
            raise ValueError(
                "no mask source: `mask` か `hotscreen`（mask を内包）を接続してください"
            )

        frames = image_to_numpy(base)
        # 加工は「受け取ったマスクをそのまま適用」なので二値化しない（SPEC §6-6）
        bitmap = mask_to_float(mask_tensor)
        batch_size, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
        if bitmap.shape[0] not in (1, batch_size):
            raise ValueError(f"batch size mismatch: image={batch_size} mask={bitmap.shape[0]}")
        if bitmap.shape[1:] != (height, width):
            raise ValueError(f"mask size mismatch: image={(height, width)} mask={bitmap.shape[1:]}")
        if bitmap.shape[0] == 1 and batch_size > 1:
            bitmap = np.broadcast_to(bitmap, (batch_size, height, width))

        if not (bitmap > 0.0).any():
            return self._passthrough(base, envelope, image, mask_tensor)

        censored = np.empty(frames.shape, dtype=np.float32)
        for batch_index in range(batch_size):
            censored[batch_index] = apply_censor(
                frames[batch_index].astype(np.float32) / 255.0,
                bitmap[batch_index],
                mode,
                int(block_size),
            )
        output = numpy_to_image(censored, like=base)

        processing = {"mode": mode, "block_size": int(block_size), "source": source}
        if envelope is None:
            meta = base_meta(size=(batch_size, height, width))
            meta["processing"] = [processing]
            new_envelope = make_envelope(
                image=image,
                mosaic_image=output,
                mask=mask_tensor,
                detections=[],
                meta=meta,
            )
            return output, new_envelope

        meta = dict(envelope["meta"])
        meta["processing"] = [*list(meta.get("processing") or []), processing]
        new_envelope = dict(envelope)
        new_envelope["mosaic_image"] = output
        new_envelope["mask"] = mask_tensor
        new_envelope["meta"] = meta
        return output, new_envelope

    @staticmethod
    def _resolve_image(
        envelope: dict | None,
        image: torch.Tensor | None,
    ) -> tuple[torch.Tensor, str]:
        """ベース画像と `processing.source` を返す（明示入力が最優先）。"""
        if image is not None:
            return image, "explicit"
        if envelope is not None:
            if envelope["mosaic_image"] is not None:
                return envelope["mosaic_image"], "mosaic_image"
            return envelope["image"], "image"
        raise ValueError("no image source: `image` か `hotscreen` を接続してください")

    @staticmethod
    def _passthrough(
        base: torch.Tensor,
        envelope: dict | None,
        image: torch.Tensor | None,
        mask_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """全ゼロマスク時の素通し（`mosaic_image` と `processing` は据え置き）。

        `image` は `_resolve_image()` が非 `None` を保証済み（`envelope` が無い場合のみここへ来る）。
        """
        if envelope is not None:
            new_envelope = dict(envelope)
            new_envelope["mask"] = mask_tensor
            return base, new_envelope

        batch_size, height, width = image_to_numpy(base).shape[:3]  # type: ignore[union-attr]
        meta = base_meta(size=(batch_size, height, width))
        new_envelope = make_envelope(
            image=image,
            mask=mask_tensor,
            detections=[],
            meta=meta,
        )
        return base, new_envelope
