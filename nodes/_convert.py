"""ComfyUI のテンソルと numpy 配列の相互変換（SPEC §3-8）。

`core/` は device 概念を持たないため、device / dtype の復元はすべてここで行う。
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "image_to_numpy",
    "mask_to_float",
    "mask_to_numpy",
    "numpy_to_image",
    "numpy_to_mask",
]


def image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """`IMAGE` テンソルを uint8 RGB の numpy 配列へ変換する。

    `(t.detach().cpu().clamp(0,1) * 255).round()` に固定する（切り捨てにしない）。

    Args:
        image: `[B, H, W, C]`（`[H, W, C]` も許容）。

    Returns:
        `uint8[B, H, W, C]`（色順は RGB 固定）。

    Raises:
        ValueError: 次元数が3でも4でもない場合。
    """
    tensor = image.detach().to("cpu")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"IMAGE must be [B,H,W,C]: got shape {tuple(tensor.shape)}")
    return (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()


def numpy_to_image(array: np.ndarray, *, like: torch.Tensor | None = None) -> torch.Tensor:
    """numpy 画像を `IMAGE` テンソルへ変換する。

    Args:
        array: `[B, H, W, C]` または `[H, W, C]`（0.0–1.0）。
        like: 変換先の device / dtype を持つテンソル（省略時は CPU / float32）。

    Returns:
        `float32[B, H, W, C]` のテンソル。
    """
    data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    if data.ndim == 3:
        data = data[None, ...]
    tensor = torch.from_numpy(data)
    if like is not None:
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    return tensor


def mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """`MASK` テンソルを bool マスクへ変換する（`> 0.5`、`[H,W]` は `[1,H,W]` に揃える）。

    検出系（`Detector` / `Refiner` の出力を再度ラスタ化する経路）向け。
    **加工系は `mask_to_float()` を使う**（0.5 で切ると SPEC §6-6 の `mask > 0.0` と矛盾する）。

    Args:
        mask: `[B, H, W]` または `[H, W]`。

    Returns:
        `bool[B, H, W]`。

    Raises:
        ValueError: 次元数が2でも3でもない場合。
    """
    tensor = mask.detach().to("cpu")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"MASK must be [B,H,W]: got shape {tuple(tensor.shape)}")
    return tensor.numpy() > 0.5


def mask_to_float(mask: torch.Tensor) -> np.ndarray:
    """`MASK` テンソルを float32 マスクへ変換する（**二値化しない**）。

    `MosaicByMask` は受け取ったマスクをそのまま適用し、適用条件は `core` 側の
    `> 0.0` だけである（SPEC §6-6）。ここで 0.5 の閾値をかけると、
    0 < v <= 0.5 のソフトマスクが無言で無視されてしまう。

    Args:
        mask: `[B, H, W]` または `[H, W]`。

    Returns:
        `float32[B, H, W]`（値域はそのまま）。

    Raises:
        ValueError: 次元数が2でも3でもない場合。
    """
    tensor = mask.detach().to("cpu")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"MASK must be [B,H,W]: got shape {tuple(tensor.shape)}")
    return tensor.numpy().astype(np.float32, copy=False)


def numpy_to_mask(bitmap: np.ndarray, *, like: torch.Tensor | None = None) -> torch.Tensor:
    """bool / float マスクを `MASK` テンソル（float32 0–1）へ変換する。

    Args:
        bitmap: `[B, H, W]` または `[H, W]`。
        like: 変換先の device / dtype を持つテンソル（省略時は CPU / float32）。

    Returns:
        `float32[B, H, W]` のテンソル。
    """
    data = np.ascontiguousarray(np.asarray(bitmap, dtype=np.float32))
    if data.ndim == 2:
        data = data[None, ...]
    tensor = torch.from_numpy(data)
    if like is not None:
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    return tensor
