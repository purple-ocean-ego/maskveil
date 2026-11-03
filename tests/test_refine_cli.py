"""輪郭 CLI（refine）のテスト。推論は実行しない。"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from maskveil.tools.refine import main


def _write_image(path) -> None:
    Image.new("RGB", (64, 64), (10, 20, 30)).save(path)


def _payload(image_path) -> dict:
    return {
        "schema_version": 1,
        "image": {"path": str(image_path), "width": 64, "height": 64},
        "model": {"path": "/abs/det.onnx", "sha256": "x", "input_size": 640},
        "params": {"conf": 0.2, "iou": 0.45, "classes": [11]},
        "detections": [
            {
                "class_id": 11,
                "class_name": "MALE_GENITALIA_EXPOSED",
                "score": 0.7,
                "bbox": [10.0, 10.0, 40.0, 40.0],
            },
            {
                "class_id": 13,
                "class_name": "FEET_COVERED",
                "score": 0.5,
                "bbox": [5.0, 5.0, 20.0, 20.0],
            },
        ],
        "count": 2,
    }


def _run(monkeypatch, argv: list[str], payload) -> int:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    return main(argv)


def test_missing_refiner_model_exits_with_2(monkeypatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, [], "{}")
    assert excinfo.value.code == 2


def test_relative_refiner_model_exits_with_2(monkeypatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, ["--refiner-model", "seg.onnx"], "{}")
    assert excinfo.value.code == 2


def test_broken_json_returns_1(monkeypatch, capsys) -> None:
    assert _run(monkeypatch, ["--mask-source", "bbox_ellipse"], "{not json") == 1
    assert "invalid input JSON" in capsys.readouterr().err


def test_unsupported_schema_returns_1(monkeypatch) -> None:
    payload = {"schema_version": 99, "image": {"path": "x"}, "detections": []}
    assert _run(monkeypatch, ["--mask-source", "bbox_ellipse"], payload) == 1


def test_missing_image_returns_1(tmp_path, monkeypatch) -> None:
    payload = _payload(tmp_path / "nope.png")
    assert _run(monkeypatch, ["--mask-source", "bbox_ellipse"], payload) == 1


def test_bbox_ellipse_assigns_debug_masks(tmp_path, monkeypatch, capsys) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    assert _run(monkeypatch, ["--mask-source", "bbox_ellipse"], _payload(image)) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["schema_version"] == 2
    assert result["model"]["seg"] is None
    assert result["refine"] == {
        "targets": 1,
        "refined": 1,
        "preserved": 0,
        "mask_source": "bbox_ellipse",
    }
    sources = [det["mask_source"] for det in result["detections"]]
    assert sources == ["bbox_ellipse", "none"]
    polygons = result["detections"][0]["masks"]
    assert len(polygons) == 1
    mask = polygons[0]
    assert len(mask) == 33
    xs = [point[0] for point in mask]
    assert min(xs) >= 10.0 and max(xs) <= 40.0


def test_classes_filter_restricts_refine_targets(tmp_path, monkeypatch, capsys) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    assert (
        _run(
            monkeypatch,
            ["--mask-source", "bbox_ellipse", "--classes", "FEET_COVERED"],
            _payload(image),
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["refine"]["targets"] == 0
    assert result["params"]["refine"]["classes"] == [13]


def test_no_targets_does_not_load_model(tmp_path, monkeypatch, capsys) -> None:
    """対応表に載らないクラスだけなら、壊れたモデルパスでも推論しない。"""
    image = tmp_path / "img.png"
    _write_image(image)
    model = tmp_path / "dummy.onnx"
    model.write_bytes(b"not a real model")

    payload = _payload(image)
    payload["detections"] = [payload["detections"][1]]  # FEET_COVERED のみ
    payload["count"] = 1

    assert _run(monkeypatch, ["--refiner-model", str(model)], payload) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["refine"]["targets"] == 0
    assert result["detections"][0]["mask_source"] == "none"


def test_output_file_is_written(tmp_path, monkeypatch) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    out = tmp_path / "out.json"
    assert (
        _run(
            monkeypatch,
            ["--mask-source", "bbox_ellipse", "--output", str(out)],
            _payload(image),
        )
        == 0
    )
    saved = json.loads(out.read_text())
    assert saved["schema_version"] == 2
    assert saved["count"] == 2


def _fake_refiner_class(candidate_masks):
    from maskveil.core.types import Detection as _Detection

    class _FakeRefiner:
        input_size = 1280

        def __init__(self, model_path, *, input_size=None) -> None:
            self.model_path = model_path
            self.input_size = input_size or 1280

        def detect(self, image, *, class_ids, conf, iou):
            return [
                _Detection(
                    class_id=2,
                    class_name="penis",
                    score=0.8,
                    bbox=(10.0, 10.0, 40.0, 40.0),
                    masks=candidate_masks,
                    mask_source="model_seg",
                )
            ]

    return _FakeRefiner


def test_v2_rerun_preserves_existing_masks(tmp_path, monkeypatch, capsys) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    assert _run(monkeypatch, ["--mask-source", "bbox_ellipse"], _payload(image)) == 0
    first = json.loads(capsys.readouterr().out)

    assert (
        _run(
            monkeypatch,
            ["--mask-source", "bbox_ellipse", "--classes", "FEET_COVERED"],
            first,
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)

    assert second["refine"] == {
        "targets": 0,
        "refined": 1,
        "preserved": 1,
        "mask_source": "bbox_ellipse",
    }
    assert second["detections"][0]["masks"] == first["detections"][0]["masks"]
    assert second["model"] == first["model"]  # model.det.det に二重ネストしない
    assert second["params"]["det"] == first["params"]["det"]


def test_image_size_mismatch_returns_1(tmp_path, monkeypatch, capsys) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    payload = _payload(image)
    payload["image"]["width"] = 128
    assert _run(monkeypatch, ["--mask-source", "bbox_ellipse"], payload) == 1
    assert "image size mismatch" in capsys.readouterr().err


def test_model_seg_path_assigns_masks(tmp_path, monkeypatch, capsys) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    model = tmp_path / "seg.onnx"
    model.write_bytes(b"dummy")
    masks = (((20.0, 20.0), (30.0, 20.0), (30.0, 30.0)),)
    monkeypatch.setattr("maskveil.tools.refine.Refiner", _fake_refiner_class(masks))

    assert _run(monkeypatch, ["--refiner-model", str(model)], _payload(image)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["refine"]["targets"] == 1
    assert result["refine"]["refined"] == 1
    assert result["detections"][0]["masks"] == [[list(p) for p in masks[0]]]
    assert result["model"]["seg"]["input_size"] == 1280


def test_unknown_mask_source_in_input_returns_1(tmp_path, monkeypatch) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    payload = _payload(image)
    payload["schema_version"] = 2
    payload["detections"][0]["mask_source"] = "bogus"
    assert _run(monkeypatch, ["--mask-source", "bbox_ellipse"], payload) == 1


def test_output_write_failure_returns_1(tmp_path, monkeypatch, capsys) -> None:
    image = tmp_path / "img.png"
    _write_image(image)
    assert (
        _run(
            monkeypatch,
            ["--mask-source", "bbox_ellipse", "--output", "/proc/nope/out.json"],
            _payload(image),
        )
        == 1
    )
    assert "failed to write" in capsys.readouterr().err


def test_refiner_input_size_mismatch_exits_with_2(tmp_path, monkeypatch) -> None:
    class _RejectingRefiner:
        def __init__(self, model_path, *, input_size=None) -> None:
            raise ValueError("refiner model expects input size 1280, got 640")

    monkeypatch.setattr("maskveil.tools.refine.Refiner", _RejectingRefiner)
    image = tmp_path / "img.png"
    _write_image(image)
    model = tmp_path / "seg.onnx"
    model.write_bytes(b"x")

    with pytest.raises(SystemExit) as excinfo:
        _run(
            monkeypatch,
            ["--refiner-model", str(model), "--refiner-input-size", "640"],
            _payload(image),
        )
    assert excinfo.value.code == 2


@pytest.mark.parametrize("size", ["0", "-1"])
def test_refiner_input_size_must_be_positive(monkeypatch, capsys, size: str) -> None:
    """0 や負の入力サイズは引数エラー（exit 2）として拒否する。"""
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, ["--refiner-input-size", size], "{}")
    assert excinfo.value.code == 2
    assert "--refiner-input-size must be positive" in capsys.readouterr().err
