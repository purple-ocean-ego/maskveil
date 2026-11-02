"""モデル解決と推論セッションの寿命（AC-6）。

実モデルは使わず、`folder_paths` と `Detector` / `Refiner` を差し替えて
「ComfyUI の実 API をどう使うか」「何をキャッシュするか」を検証する。
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import types
from typing import Any

import pytest

pytest.importorskip("torch", reason="torch は ComfyUI が提供する。torch 無しの CI レグではスキップ")

from maskveil.core.constants import DET_INPUT_SIZE, SEG_INPUT_SIZE
from maskveil.nodes import _models

MODELS = [
    "det/hs-real-anime-y11n-640-fp32.onnx",
    "det/labels.txt",
    "seg/nsfw-anime-xl-x1280.onnx",
]


def _folder_paths(
    *,
    names: list[str] | None = None,
    full_path: str | None = "/workspace/models/hotscreen/det/a.onnx",
    folders: list[str] | None = None,
    raise_key_error: bool = False,
    has_get_folder_paths: bool = True,
) -> types.ModuleType:
    stub = types.ModuleType("folder_paths")

    def get_filename_list(folder: str) -> list[str]:
        if raise_key_error:
            raise KeyError(folder)
        return list(MODELS if names is None else names)

    def get_full_path(folder: str, filename: str) -> str | None:
        if raise_key_error:
            raise KeyError(folder)
        return full_path

    stub.get_filename_list = get_filename_list  # type: ignore[attr-defined]
    stub.get_full_path = get_full_path  # type: ignore[attr-defined]
    if has_get_folder_paths:
        stub.get_folder_paths = lambda folder: (  # type: ignore[attr-defined]
            list(folders or ["/workspace/models/hotscreen"])
        )
    return stub


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """プロセス内キャッシュをテストごとに初期化する。"""
    _models._sessions.clear()  # noqa: SLF001 - キャッシュの検証が目的
    _models._sha256_cache.clear()  # noqa: SLF001


def _install(monkeypatch: pytest.MonkeyPatch, stub: types.ModuleType | None) -> None:
    if stub is None:
        monkeypatch.delitem(sys.modules, "folder_paths", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "folder_paths", stub)


def test_model_list_filters_to_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _folder_paths())
    assert _models.model_list() == [
        "det/hs-real-anime-y11n-640-fp32.onnx",
        "seg/nsfw-anime-xl-x1280.onnx",
    ]


def test_model_list_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _folder_paths(names=["a.ONNX", "b.txt"]))
    assert _models.model_list() == ["a.ONNX"]


def test_model_list_raises_when_category_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _folder_paths(raise_key_error=True))
    with pytest.raises(RuntimeError, match="hotscreen"):
        _models.model_list()


def test_model_list_raises_when_no_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _folder_paths(names=["det/labels.txt"]))
    with pytest.raises(RuntimeError, match="onnx"):
        _models.model_list()


def test_model_list_raises_outside_comfyui(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, None)
    with pytest.raises(RuntimeError, match="folder_paths"):
        _models.model_list()


def test_resolve_model_path_returns_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _folder_paths(full_path="/workspace/models/hotscreen/det/a.onnx"))
    assert _models.resolve_model_path("det/a.onnx") == pathlib.Path(
        "/workspace/models/hotscreen/det/a.onnx"
    )


def test_resolve_model_path_reports_searched_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _folder_paths(full_path=None, folders=["/models/hotscreen", "/alt/hotscreen"]),
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        _models.resolve_model_path("det/missing.onnx")

    message = str(excinfo.value)
    assert "/models/hotscreen/det/missing.onnx" in message
    assert "/alt/hotscreen/det/missing.onnx" in message


def test_resolve_model_path_tolerates_missing_get_folder_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _folder_paths(full_path=None, has_get_folder_paths=False))
    with pytest.raises(FileNotFoundError, match="不明"):
        _models.resolve_model_path("det/missing.onnx")


def test_resolve_model_path_raises_when_category_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _folder_paths(raise_key_error=True))
    with pytest.raises(RuntimeError, match="extra_model_paths.yaml"):
        _models.resolve_model_path("det/a.onnx")


class _FakeDetector:
    created: list[tuple[pathlib.Path, int]] = []

    def __init__(self, path: pathlib.Path, *, input_size: int) -> None:
        self.path = path
        self.input_size = input_size
        type(self).created.append((path, input_size))


class _FakeRefiner(_FakeDetector):
    created: list[tuple[pathlib.Path, int]] = []


@pytest.fixture()
def model_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"onnx-bytes")
    return path


def test_detector_session_is_cached(
    monkeypatch: pytest.MonkeyPatch, model_file: pathlib.Path
) -> None:
    from maskveil.core import detector as detector_module

    monkeypatch.setattr(detector_module, "Detector", _FakeDetector)
    _FakeDetector.created.clear()

    first = _models.detector_for(model_file)
    second = _models.detector_for(model_file)

    assert first is second
    assert len(_FakeDetector.created) == 1
    assert _FakeDetector.created[0][1] == DET_INPUT_SIZE


def test_detector_cache_invalidates_on_mtime_change(
    monkeypatch: pytest.MonkeyPatch, model_file: pathlib.Path
) -> None:
    from maskveil.core import detector as detector_module

    monkeypatch.setattr(detector_module, "Detector", _FakeDetector)
    first = _models.detector_for(model_file)
    model_file.write_bytes(b"onnx-bytes-v2")
    second = _models.detector_for(model_file)

    assert first is not second


def test_detector_cache_key_includes_input_size(
    monkeypatch: pytest.MonkeyPatch, model_file: pathlib.Path
) -> None:
    from maskveil.core import detector as detector_module

    monkeypatch.setattr(detector_module, "Detector", _FakeDetector)
    default = _models.detector_for(model_file)
    other = _models.detector_for(model_file, input_size=SEG_INPUT_SIZE)

    assert default is not other


def test_refiner_session_is_cached(
    monkeypatch: pytest.MonkeyPatch, model_file: pathlib.Path
) -> None:
    from maskveil.core import refiner as refiner_module

    monkeypatch.setattr(refiner_module, "Refiner", _FakeRefiner)
    _FakeRefiner.created.clear()

    first = _models.refiner_for(model_file)
    second = _models.refiner_for(model_file)

    assert first is second
    assert _FakeRefiner.created[0][1] == SEG_INPUT_SIZE


def test_detector_and_refiner_have_separate_cache_entries(
    monkeypatch: pytest.MonkeyPatch, model_file: pathlib.Path
) -> None:
    from maskveil.core import detector as detector_module
    from maskveil.core import refiner as refiner_module

    monkeypatch.setattr(detector_module, "Detector", _FakeDetector)
    monkeypatch.setattr(refiner_module, "Refiner", _FakeRefiner)

    detector = _models.detector_for(model_file)
    refiner = _models.refiner_for(model_file)
    assert detector is not refiner


def test_model_metadata_hashes_once(
    model_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original_open = pathlib.Path.open

    def counting_open(self: pathlib.Path, *args: Any, **kwargs: Any):
        calls.append(str(self))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", counting_open)

    first = _models.model_metadata(model_file, input_size=DET_INPUT_SIZE)
    second = _models.model_metadata(model_file, input_size=DET_INPUT_SIZE)

    assert first == second
    assert first["sha256"] == hashlib.sha256(b"onnx-bytes").hexdigest()
    assert first["input_size"] == DET_INPUT_SIZE
    assert first["path"] == str(model_file)
    assert calls == [str(model_file)]  # 2回目はキャッシュから返る


def test_model_metadata_rehashes_after_change(model_file: pathlib.Path) -> None:
    first = _models.model_metadata(model_file, input_size=DET_INPUT_SIZE)
    model_file.write_bytes(b"changed")
    second = _models.model_metadata(model_file, input_size=DET_INPUT_SIZE)

    assert first["sha256"] != second["sha256"]
    assert second["sha256"] == hashlib.sha256(b"changed").hexdigest()


def test_node_uses_resolved_path_and_default_input_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    from maskveil.nodes import detector as detector_node
    from maskveil.nodes import refiner as refiner_node

    assert inspect.signature(detector_node.detector_for).parameters["input_size"].default == (
        DET_INPUT_SIZE
    )
    assert inspect.signature(refiner_node.refiner_for).parameters["input_size"].default == (
        SEG_INPUT_SIZE
    )

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        detector_node, "resolve_model_path", lambda name: pathlib.Path("/abs") / name
    )

    def fake_detector_for(path: pathlib.Path, **kwargs: Any) -> Any:
        captured["path"] = path
        raise RuntimeError("stop")

    monkeypatch.setattr(detector_node, "detector_for", fake_detector_for)
    with pytest.raises(RuntimeError, match="stop"):
        detector_node.HotscreenDetector().detect(
            image=None,
            model="det/a.onnx",
            classes="all",
            conf=0.2,
            iou=0.45,
            top_k=0,
            mask_expand=4,
            preview_mode="none",
        )
    assert captured["path"] == pathlib.Path("/abs/det/a.onnx")
