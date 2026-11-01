"""CLI 共通処理（画像読込・引数検証・JSON 出力）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from maskveil.core.constants import DEFAULT_CONF, DEFAULT_IOU


def load_rgb(path: str | Path) -> np.ndarray:
    """画像を RGB の `(H, W, 3)` uint8 配列として読む。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def quantize_rgb(image: np.ndarray) -> np.ndarray:
    """`float32 0–1` の画像を `uint8` へ量子化する。

    ComfyUI の `SaveImage`（`np.clip(i, 0, 255).astype(np.uint8)`）と同じ**切り捨て**に揃える。
    これにより、同じ float32 結果から作った PNG が ComfyUI の保存経路と**画素完全一致**する
    （CLI と ComfyUI の等価性を厳密一致で検証できる）。

    なお検出0件の無加工コピーは `uint8` のまま `save_png` へ渡すため、この関数を通らない
    （入力と画素完全一致を保つ）。
    """
    array = np.asarray(image, dtype=np.float32)
    return np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)


def save_png(
    image: np.ndarray,
    path: str | Path,
    *,
    compress_level: int = 1,
) -> None:
    """RGB 画像を PNG として原子的に保存する。

    一時ファイルへ書いてから `os.replace` するため、途中で失敗しても
    既存ファイルは壊れない（`write_json` と同じ方針）。

    Args:
        image: `(H, W, 3)` の `float32 0–1`（`uint8` はそのまま保存）。
        path: 保存先。
        compress_level: PNG の圧縮レベル（0–9）。

    Raises:
        OSError: 書き込みに失敗した場合。
        ValueError: 画像の形状・型が想定外の場合。
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3): got shape {array.shape}")

    quantized = array if array.dtype == np.uint8 else quantize_rgb(array)
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    try:
        Image.fromarray(quantized, mode="RGB").save(
            temporary, format="PNG", compress_level=int(compress_level)
        )
        os.replace(temporary, target)
    finally:
        # 失敗時に `.tmp` を残さない（成功時は `os.replace` 済みで存在しない）
        temporary.unlink(missing_ok=True)


def add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """`--conf` / `--iou` を共通の既定値で追加する。"""
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF, help=f"スコア閾値（既定 {DEFAULT_CONF}）"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU, help=f"NMS の IoU 閾値（既定 {DEFAULT_IOU}）"
    )


def validate_thresholds(parser: argparse.ArgumentParser, *, conf: float, iou: float) -> None:
    """`--conf` / `--iou` の範囲を検証する（不正なら exit 2）。"""
    if not 0.0 <= conf <= 1.0:
        parser.error("--conf must be within 0.0-1.0")
    if not 0.0 < iou <= 1.0:
        parser.error("--iou must be within (0.0, 1.0]")


def write_json(payload: dict[str, Any], *, output: str | None, pretty: bool) -> None:
    """JSON を標準出力またはファイルへ書く（ファイルは原子的に置換）。

    Raises:
        OSError: 書き込みに失敗した場合。
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
    if not output:
        print(text)
        return

    path = Path(output)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def report_write_error(output: str, exc: OSError) -> int:
    """書き込み失敗を stderr に出して終了コード 1 を返す。"""
    print(f"error: failed to write {output!r}: {exc}", file=sys.stderr)
    return 1
