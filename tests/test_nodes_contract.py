"""`HotscreenDetector` の出力契約（AC-2）。

推論はモデル依存で遅いため、`detector_for()` をスタブに差し替えて
**ノードが組み立てる構造**（`HOTSCREEN` / `MASK` / `SEGS` / `detection` / `preview`）を検証する。
座標・マスク・膨張の意味は `core` 側のテストで別途担保する。
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="torch は ComfyUI が提供する。torch 無しの CI レグではスキップ"
)

from maskveil.core.types import Detection
from maskveil.nodes import detector as detector_node
from maskveil.nodes.detector import HotscreenDetector

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = json.loads((PACKAGE / "tests" / "fixtures" / "ac1_contract.json").read_text("utf-8"))

MODEL = "det/fake.onnx"

SQUARE_MASKS = (((2.0, 3.0), (10.0, 3.0), (10.0, 12.0), (2.0, 12.0), (2.0, 3.0)),)


def _detection(score: float = 0.9, *, masks=SQUARE_MASKS, mask_source: str = "model_seg"):
    return Detection(
        class_id=3,
        class_name="FEMALE_GENITALIA_EXPOSED",
        score=score,
        bbox=(2.0, 3.0, 10.0, 12.0),
        masks=masks,
        mask_source=mask_source,
    )


class _StubDetector:
    """`core.detector.Detector` と同じ `detect()` シグネチャを持つスタブ。"""

    input_size = 640

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.calls: list[dict[str, Any]] = []

    def detect(self, image: np.ndarray, *, class_ids, conf, iou):  # noqa: ANN001
        self.calls.append(
            {"shape": image.shape, "class_ids": tuple(class_ids), "conf": conf, "iou": iou}
        )
        return list(self.detections)


@pytest.fixture()
def stub() -> _StubDetector:
    return _StubDetector([_detection()])


@pytest.fixture(autouse=True)
def patched_models(stub: _StubDetector):
    """モデル解決とメタデータを固定する（実ファイルを要求しない）。"""
    with (
        patch.object(
            detector_node, "resolve_model_path", return_value=pathlib.Path("/tmp/fake.onnx")
        ),
        patch.object(detector_node, "detector_for", return_value=stub),
        patch.object(
            detector_node,
            "model_metadata",
            return_value={"path": "/tmp/fake.onnx", "sha256": "0" * 64, "input_size": 640},
        ),
    ):
        yield


def _image(batch: int = 1, height: int = 32, width: int = 24) -> torch.Tensor:
    values = torch.arange(batch * height * width * 3, dtype=torch.float32)
    image = (values % 256.0) / 255.0
    return image.reshape(batch, height, width, 3)


def _run(**overrides):
    kwargs = {
        "image": _image(),
        "model": MODEL,
        "classes": "FEMALE_GENITALIA_EXPOSED",
        "conf": 0.2,
        "iou": 0.45,
        "top_k": 0,
        "mask_expand": 4,
        "preview_mode": "none",
    }
    kwargs.update(overrides)
    return HotscreenDetector().detect(**kwargs)


def test_hotscreen_envelope_keys_and_types() -> None:
    envelope, mask, segs, payload, preview = _run()

    assert list(envelope) == CONTRACT["hotcreen_keys"]
    assert envelope["schema_version"] == 1
    assert isinstance(envelope["image"], torch.Tensor)
    assert envelope["mosaic_image"] is None
    assert isinstance(envelope["mask"], torch.Tensor)
    assert envelope["mask"] is mask
    assert isinstance(envelope["detections"], list)
    assert isinstance(envelope["meta"], dict)
    assert isinstance(segs, tuple) and len(segs) == 2
    assert isinstance(payload, str)
    assert isinstance(preview, torch.Tensor)


def test_mask_shape_and_values() -> None:
    batch = 2
    _, mask, _, _, _ = _run(image=_image(batch=batch), preview_mode="none")

    assert tuple(mask.shape) == (batch, 32, 24)
    assert mask.dtype == torch.float32
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    for index in range(batch):
        assert mask[index].sum() > 0


def test_detection_dict_has_node_keys() -> None:
    envelope, _, _, _, _ = _run(image=_image(batch=2))
    detections = envelope["detections"]

    assert len(detections) == 2
    for batch_index, item in enumerate(detections):
        assert list(item) == [
            "class_id",
            "class_name",
            "score",
            "bbox",
            "masks",
            "mask_source",
            "batch_index",
            "expanded_px",
        ]
        assert item["batch_index"] == batch_index
        assert item["expanded_px"] == 4
        assert item["class_name"] == "FEMALE_GENITALIA_EXPOSED"


def test_mask_expand_grows_mask_but_not_bbox() -> None:
    from maskveil.core.raster import dilate_mask, rasterize_polygons

    _, raw_mask, _, _, _ = _run(mask_expand=0)
    envelope, grown_mask, _, _, _ = _run(mask_expand=4)

    expected = dilate_mask(rasterize_polygons(SQUARE_MASKS, 32, 24), 4).sum()
    assert int(raw_mask.sum()) == 8 * 9  # ポリゴン (2,3)-(10,12)
    assert int(grown_mask.sum()) == int(expected)  # core の膨張と一致（形状も含めて一致）
    # bbox は膨張の影響を受けない
    assert envelope["detections"][0]["bbox"] == [2.0, 3.0, 10.0, 12.0]


def test_detection_dict_keeps_raw_coordinates() -> None:
    """メモリ上の dict は生値、JSON は丸め。"""
    fractional = Detection(
        class_id=3,
        class_name="FEMALE_GENITALIA_EXPOSED",
        score=0.123456,
        bbox=(2.3456789, 3.4567, 10.111, 12.999),
        masks=(((2.3456789, 3.1), (10.5, 3.1), (10.5, 12.5), (2.3456789, 3.1)),),
        mask_source="model_seg",
    )
    stub = _StubDetector([fractional])
    with patch.object(detector_node, "detector_for", return_value=stub):
        envelope, _, _, payload, _ = _run(mask_expand=0)

    item = envelope["detections"][0]
    assert item["bbox"][0] == pytest.approx(2.3456789)
    assert item["score"] == pytest.approx(0.123456)
    assert item["masks"][0][1][0] == pytest.approx(10.5)

    data = json.loads(payload)["detections"][0]
    assert data["bbox"] == [2.35, 3.46, 10.11, 13.0]
    assert data["score"] == 0.1235


def test_top_k_is_applied_per_batch() -> None:
    stub = _StubDetector([_detection(0.9), _detection(0.8), _detection(0.7)])
    with patch.object(detector_node, "detector_for", return_value=stub):
        envelope, _, _, payload, _ = _run(image=_image(batch=2), top_k=2)

    per_batch = [item for item in envelope["detections"] if item["batch_index"] == 0]
    assert len(per_batch) == 2
    assert [item["score"] for item in per_batch] == [0.9, 0.8]
    assert len(envelope["detections"]) == 4
    assert json.loads(payload)["params"]["top_k"] == 2


def test_segs_structure_is_impact_pack_compatible() -> None:
    _, mask, segs, _, _ = _run(image=_image(batch=2), mask_expand=4)
    size, items = segs

    assert size == (32, 24)
    # SEGS は先頭バッチのみ（全バッチは HOTSCREEN / MASK が保持する）
    assert len(items) == 1
    seg = items[0]
    assert list(seg._fields) == CONTRACT["segs_fields"]
    assert seg.label == "FEMALE_GENITALIA_EXPOSED"
    assert seg.confidence == pytest.approx(0.9)
    assert seg.control_net_wrapper is None
    assert seg.crop_region == seg.bbox
    # crop_region は検出 bbox を含み、膨張したマスクも取りこぼさない
    assert seg.bbox[0] <= 2 and seg.bbox[1] <= 3
    assert seg.bbox[2] >= 10 and seg.bbox[3] >= 12
    height, width = seg.bbox[3] - seg.bbox[1], seg.bbox[2] - seg.bbox[0]
    assert tuple(seg.cropped_image.shape) == (1, height, width, 3)
    assert tuple(seg.cropped_mask.shape) == (height, width)
    assert seg.cropped_mask.dtype == torch.float32
    # マスク全体がクロップに収まっている（膨張分の欠落なし）
    assert int((seg.cropped_mask > 0).sum()) == int(mask[0].sum())


def test_segs_crop_region_expands_with_mask_expand() -> None:
    _, _, raw, _, _ = _run(mask_expand=0)
    _, _, grown, _, _ = _run(mask_expand=6)

    assert raw[1][0].bbox == (2, 3, 10, 12)
    assert grown[1][0].bbox[0] < raw[1][0].bbox[0]
    assert grown[1][0].bbox[1] < raw[1][0].bbox[1]
    assert grown[1][0].bbox[2] > raw[1][0].bbox[2]
    assert grown[1][0].bbox[3] > raw[1][0].bbox[3]


def test_seg_bbox_rounds_outward() -> None:
    fractional = Detection(
        class_id=11,
        class_name="MALE_GENITALIA_EXPOSED",
        score=0.5,
        bbox=(2.4, 3.6, 10.2, 12.9),
        masks=None,
        mask_source="none",
    )
    stub = _StubDetector([fractional])
    with patch.object(detector_node, "detector_for", return_value=stub):
        # mask_expand=0 なら crop_region は丸めた bbox そのもの
        _, _, segs, _, _ = _run(mask_expand=0)

    assert segs[1][0].bbox == (2, 3, 11, 13)  # x1,y1 は floor / x2,y2 は ceil


def test_detection_json_matches_spec() -> None:
    _, _, _, payload, _ = _run(classes="FEMALE_GENITALIA_EXPOSED,MALE_GENITALIA_EXPOSED")
    data = json.loads(payload)

    assert data["schema_version"] == 1
    assert data["source"] == "comfyui"
    assert data["model"] == {"path": MODEL, "input_size": 640}
    assert data["params"] == {
        "classes": [3, 11],
        "conf": 0.2,
        "iou": 0.45,
        "top_k": 0,
        "mask_expand": 4,
    }
    assert data["count"] == len(data["detections"]) == 1
    assert "image" not in data  # ComfyUI に入力パスは無い


def test_zero_detections_pass_through() -> None:
    stub = _StubDetector([])
    image = _image()
    with patch.object(detector_node, "detector_for", return_value=stub):
        envelope, mask, segs, payload, preview = _run(image=image, preview_mode="outline")

    assert envelope["detections"] == []
    assert float(mask.sum()) == 0.0
    assert segs == ((32, 24), [])
    assert json.loads(payload)["count"] == 0
    original = (image.clamp(0, 1) * 255.0).round().to(torch.uint8)
    assert torch.equal((preview.clamp(0, 1) * 255.0).round().to(torch.uint8), original)


def test_preview_modes() -> None:
    image = _image(batch=2)
    _, _, _, _, none_preview = _run(image=image, preview_mode="none")
    assert none_preview is image

    _, _, _, _, outline = _run(image=image, preview_mode="outline")
    assert tuple(outline.shape) == (2, 32, 24, 3)

    _, _, _, _, crop = _run(image=image, preview_mode="crop")
    assert tuple(crop.shape) == (1, 9, 8, 3)


def test_preview_does_not_change_results() -> None:
    _, none_mask, none_segs, none_payload, _ = _run(preview_mode="none")
    _, outline_mask, outline_segs, outline_payload, _ = _run(preview_mode="outline")

    assert torch.equal(none_mask, outline_mask)
    assert none_segs[0] == outline_segs[0]
    assert none_payload == outline_payload


def test_invalid_preview_mode_raises() -> None:
    with pytest.raises(ValueError, match="preview_mode"):
        _run(preview_mode="zoom")


@pytest.mark.parametrize("classes", ["", "NOT_A_CLASS"])
def test_invalid_classes_raise(classes: str) -> None:
    from maskveil.core.labels import UnknownClassError

    with pytest.raises(UnknownClassError):
        _run(classes=classes)


def test_all_classes_are_accepted() -> None:
    _, _, _, payload, _ = _run(classes="all")
    assert len(json.loads(payload)["params"]["classes"]) == 16


def test_input_image_is_not_modified() -> None:
    image = _image()
    before = image.clone()
    _run(image=image)
    assert torch.equal(image, before)


def test_detector_forwards_conf_iou_and_top_k_to_model() -> None:
    """UI の conf / iou がモデルへ渡っていること（ハードコード検出用）。"""
    stub = _StubDetector([_detection()])
    with patch.object(detector_node, "detector_for", return_value=stub):
        _run(conf=0.31, iou=0.27, top_k=0)

    assert stub.calls[0]["conf"] == pytest.approx(0.31)
    assert stub.calls[0]["iou"] == pytest.approx(0.27)


def test_mask_expand_grows_bbox_rectangle_at_node_level() -> None:
    """実 Detector の出力（mask_source="none"）でも矩形が膨張すること。"""
    bbox_only = Detection(
        class_id=11,
        class_name="MALE_GENITALIA_EXPOSED",
        score=0.7,
        bbox=(6.0, 6.0, 10.0, 10.0),
        masks=None,
        mask_source="none",
    )
    stub = _StubDetector([bbox_only])
    with patch.object(detector_node, "detector_for", return_value=stub):
        plain_env, plain_mask, _, _, _ = _run(image=_image(height=32, width=24), mask_expand=0)
        grown_env, grown_mask, _, _, _ = _run(image=_image(height=32, width=24), mask_expand=4)

    assert float(plain_mask.sum()) == 16.0  # 4x4
    assert float(grown_mask.sum()) == 144.0  # 12x12（上下左右へ4px ずつ）
    assert grown_env["detections"][0]["bbox"] == plain_env["detections"][0]["bbox"]
