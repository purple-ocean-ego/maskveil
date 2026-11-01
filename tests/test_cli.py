"""CLI（引数解釈・パス検証）のテスト。推論は実行しない。"""

from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image

from maskveil.tools.detect import main
from maskveil.tools.paths import InvalidPathError, sha256_file, validate_model_path


def test_list_classes_prints_sixteen_lines(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-classes"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 16
    assert out[0].endswith("FEMALE_FACE")
    assert out[11].endswith("MALE_GENITALIA_EXPOSED")


@pytest.mark.parametrize(
    "argv",
    [
        [],  # 何も無い
        ["image.png", "--model", "/abs/does-not-exist.onnx"],  # --classes なし
        ["image.png", "--classes", "all"],  # --model なし
        ["--classes", "all", "--model", "/abs/does-not-exist.onnx"],  # IMAGE なし
    ],
)
def test_argument_errors_exit_with_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_unknown_class_exits_with_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["image.png", "--classes", "NOT_A_CLASS", "--model", "/abs/m.onnx"])
    assert excinfo.value.code == 2


def test_relative_model_path_exits_with_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["image.png", "--classes", "all", "--model", "models/m.onnx"])
    assert excinfo.value.code == 2


def test_validate_model_path_accepts_existing_onnx(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"dummy")
    assert validate_model_path(str(model), option="--model") == model


@pytest.mark.parametrize(
    ("name", "content"),
    [("model.pt", b"x"), ("model.onnx.txt", b"x")],
)
def test_validate_model_path_rejects_non_onnx(tmp_path, name: str, content: bytes) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(InvalidPathError):
        validate_model_path(str(path), option="--model")


def test_validate_model_path_rejects_relative_and_missing(tmp_path) -> None:
    with pytest.raises(InvalidPathError):
        validate_model_path("model.onnx", option="--model")
    with pytest.raises(InvalidPathError):
        validate_model_path(str(tmp_path / "nope.onnx"), option="--model")
    with pytest.raises(InvalidPathError):
        validate_model_path(str(tmp_path), option="--model")


def test_sha256_file(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"hotscreen")
    assert sha256_file(path) == hashlib.sha256(b"hotscreen").hexdigest()


def test_relative_image_path_is_resolved(tmp_path, monkeypatch, capsys) -> None:
    from maskveil.core.types import Detection as _Detection

    class _FakeDetector:
        def __init__(self, model_path, *, input_size=None) -> None:
            self.input_size = input_size or 640

        def detect(self, image, *, class_ids, conf, iou):
            return [_Detection(0, "FEMALE_FACE", 0.9, (1.0, 2.0, 3.0, 4.0))]

    monkeypatch.setattr("maskveil.tools.detect.Detector", _FakeDetector)
    monkeypatch.chdir(tmp_path)

    image = tmp_path / "img.png"
    image.write_bytes(b"x")
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    # 画像は読めないので、パス解決の検証には load_rgb の差し替えが必要
    import maskveil.tools.detect as detect_module

    monkeypatch.setattr(
        detect_module,
        "load_rgb",
        lambda path: __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
    )

    assert main(["img.png", "--classes", "all", "--model", str(model)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["image"]["path"] == str(image)


def test_output_write_failure_returns_1(tmp_path, monkeypatch, capsys) -> None:
    from maskveil.core.types import Detection as _Detection

    class _FakeDetector:
        def __init__(self, model_path, *, input_size=None) -> None:
            self.input_size = input_size or 640

        def detect(self, image, *, class_ids, conf, iou):
            return [_Detection(0, "FEMALE_FACE", 0.9, (1.0, 2.0, 3.0, 4.0))]

    monkeypatch.setattr("maskveil.tools.detect.Detector", _FakeDetector)
    import maskveil.tools.detect as detect_module

    monkeypatch.setattr(
        detect_module,
        "load_rgb",
        lambda path: __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
    )
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    image = tmp_path / "img.png"
    image.write_bytes(b"x")

    assert (
        main(
            [
                str(image),
                "--classes",
                "all",
                "--model",
                str(model),
                "--output",
                "/proc/nope/out.json",
            ]
        )
        == 1
    )
    assert "failed to write" in capsys.readouterr().err


def test_model_input_size_mismatch_exits_with_2(tmp_path, monkeypatch) -> None:
    class _RejectingDetector:
        def __init__(self, model_path, *, input_size=None) -> None:
            raise ValueError("detector model expects input size 640, got 320")

    monkeypatch.setattr("maskveil.tools.detect.Detector", _RejectingDetector)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(image)

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                str(image),
                "--classes",
                "all",
                "--model",
                str(model),
                "--input-size",
                "320",
            ]
        )
    assert excinfo.value.code == 2
