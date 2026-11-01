"""`folder_paths` によるモデル解決と、推論セッションの寿命管理。

`folder_paths` は ComfyUI 本体が提供するため、本モジュール内で**遅延 import** する
（CLI や単体テストからパッケージを import しても失敗しないようにするため）。
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..core.constants import DET_INPUT_SIZE, SEG_INPUT_SIZE

if TYPE_CHECKING:  # pragma: no cover - 型注釈のみ
    from ..core.detector import Detector
    from ..core.refiner import Refiner

__all__ = [
    "MODEL_CATEGORY",
    "detector_for",
    "model_list",
    "model_metadata",
    "refiner_for",
    "resolve_model_path",
]

MODEL_CATEGORY = "hotscreen"
"""`extra_model_paths.yaml` で定義したモデルカテゴリ名。"""

_ONNX_SUFFIX = ".onnx"

_sessions: dict[tuple[str, int, int, int, str], object] = {}
_sha256_cache: dict[tuple[str, int, int], str] = {}
_lock = threading.Lock()


def _folder_paths():
    """`folder_paths` を import して返す（ComfyUI 外では明示エラー）。"""
    try:
        import folder_paths  # noqa: PLC0415 - ComfyUI 実行時のみ存在する
    except ModuleNotFoundError as exc:  # pragma: no cover - ComfyUI 外
        raise RuntimeError(
            "ComfyUI の folder_paths が見つかりません。ComfyUI の custom_nodes 配下で実行してください。"
        ) from exc
    return folder_paths


def model_list() -> list[str]:
    """`hotscreen` カテゴリの `.onnx` を返す（コンボボックスの選択肢）。

    `folder_paths.get_filename_list()` は `labels.txt` なども返すため、
    **`.onnx` で終わるものだけ**に絞る。

    Returns:
        相対パスのリスト（`get_filename_list` の順序を保つ）。

    Raises:
        RuntimeError: カテゴリ未登録（`extra_model_paths.yaml` 未読込）、または `.onnx` が0件。
    """
    folder_paths = _folder_paths()
    try:
        names = folder_paths.get_filename_list(MODEL_CATEGORY)
    except KeyError as exc:
        raise RuntimeError(
            f"folder_paths にカテゴリ {MODEL_CATEGORY!r} がありません。"
            " extra_model_paths.yaml の hotscreen 設定を確認してください。"
        ) from exc

    models = [name for name in names if name.lower().endswith(_ONNX_SUFFIX)]
    if not models:
        raise RuntimeError(
            f"{MODEL_CATEGORY!r} カテゴリに .onnx モデルがありません。"
            " モデルの配置先を確認してください。"
        )
    return models


def resolve_model_path(name: str) -> Path:
    """モデルの相対パスを絶対パスへ解決する。

    Args:
        name: `model_list()` が返した相対パス。

    Returns:
        解決済みの絶対パス。

    Raises:
        RuntimeError: カテゴリ未登録。
        FileNotFoundError: 実体が見つからない場合（候補の絶対パスを含める）。
    """
    folder_paths = _folder_paths()
    try:
        resolved = folder_paths.get_full_path(MODEL_CATEGORY, name)
    except KeyError as exc:
        raise RuntimeError(
            f"folder_paths にカテゴリ {MODEL_CATEGORY!r} がありません。"
            " extra_model_paths.yaml の hotscreen 設定を確認してください。"
        ) from exc

    if not resolved:
        try:
            candidates = folder_paths.get_folder_paths(MODEL_CATEGORY)
        except (AttributeError, KeyError):  # pragma: no cover - 古い ComfyUI
            candidates = []
        searched = ", ".join(str(Path(base) / name) for base in candidates) or "(不明)"
        raise FileNotFoundError(
            f"モデルが見つかりません: {MODEL_CATEGORY}/{name}（探索先: {searched}）"
        )
    return Path(resolved)


def _session_key(path: Path, input_size: int, kind: str) -> tuple[str, int, int, int, str]:
    """セッションキャッシュのキー（パス + mtime + size + 入力サイズ + 種別）。"""
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size, input_size, kind)


def _cached_session(
    path: Path,
    input_size: int,
    kind: str,
    factory: Callable[[Path, int], object],
) -> object:
    """モデルセッションをキャッシュから取得する（無ければ構築する）。

    構築はロックの外で行う（初回ロード中の数秒間、他のノードの実行を妨げないため）。
    競合した場合は先に入ったセッションを採用する。

    Args:
        path: 解決済みの絶対パス。
        input_size: 期待する入力一辺。
        kind: `"detector"` / `"refiner"`（キャッシュキーの一部）。
        factory: `(path, input_size)` を受けてセッションを作る関数。

    Returns:
        構築済みのセッション。
    """
    key = _session_key(path, input_size, kind)
    with _lock:
        cached = _sessions.get(key)
    if cached is not None:
        return cached

    session = factory(path, input_size)
    with _lock:
        return _sessions.setdefault(key, session)


def detector_for(path: Path, *, input_size: int = DET_INPUT_SIZE) -> Detector:
    """検出モデルのセッションを取得する（プロセス内でキャッシュ）。

    Args:
        path: 解決済みの絶対パス。
        input_size: 期待する入力一辺（既定 `DET_INPUT_SIZE`）。

    Returns:
        `core.detector.Detector` のインスタンス。
    """
    from ..core.detector import Detector  # noqa: PLC0415 - 依存を遅延させる

    return cast(
        "Detector",
        _cached_session(path, input_size, "detector", lambda p, s: Detector(p, input_size=s)),
    )


def refiner_for(path: Path, *, input_size: int = SEG_INPUT_SIZE) -> Refiner:
    """輪郭モデルのセッションを取得する（プロセス内でキャッシュ）。

    Args:
        path: 解決済みの絶対パス。
        input_size: 期待する入力一辺（既定 `SEG_INPUT_SIZE`）。

    Returns:
        `core.refiner.Refiner` のインスタンス。
    """
    from ..core.refiner import Refiner  # noqa: PLC0415 - 依存を遅延させる

    return cast(
        "Refiner",
        _cached_session(path, input_size, "refiner", lambda p, s: Refiner(p, input_size=s)),
    )


def model_metadata(path: Path, *, input_size: int) -> dict[str, object]:
    """モデルの識別情報（絶対パス・sha256・入力サイズ）を返す。

    `sha256` は 250MB 級のモデルでは毎回計算すると重いため、
    `(パス, mtime, size)` をキーにプロセス内でキャッシュする。

    Args:
        path: 解決済みの絶対パス。
        input_size: 実際に使った入力一辺。

    Returns:
        `{"path", "sha256", "input_size"}`。
    """
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _lock:
        digest = _sha256_cache.get(key)
    if digest is None:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        with _lock:
            _sha256_cache[key] = digest
    return {"path": str(path), "sha256": digest, "input_size": int(input_size)}
