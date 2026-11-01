"""CLI 用のパス検証とハッシュ計算。

CLI は**フルパス必須**。環境変数・既定値・ディレクトリ探索は行わない。
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class InvalidPathError(ValueError):
    """CLI に渡されたパスが要件を満たさない。"""


def validate_model_path(value: str, *, option: str) -> Path:
    """モデルパスを検証して `Path` を返す。

    要件: 絶対パス / 存在する / 通常ファイル / 拡張子 `.onnx`。

    Raises:
        InvalidPathError: いずれかを満たさない場合。
    """
    if not value:
        raise InvalidPathError(f"{option}: path is empty")

    path = Path(value)
    if not path.is_absolute():
        raise InvalidPathError(f"{option}: an absolute path is required, got {value!r}")
    if not path.exists():
        raise InvalidPathError(f"{option}: file not found: {value}")
    if not path.is_file():
        raise InvalidPathError(f"{option}: not a regular file: {value}")
    if path.suffix.lower() != ".onnx":
        raise InvalidPathError(f"{option}: expected a .onnx file, got {value!r}")
    return path


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """ファイルの SHA-256 を返す。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
