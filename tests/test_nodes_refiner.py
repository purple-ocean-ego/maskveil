"""`HotscreenRefiner` の契約。

輪郭モデルの推論は約3秒/枚かかるため、`refiner_for()` をスタブに差し替え、
**置換の対象選別・保持規則・JSON 件数・meta の扱い**を検証する。
"""

from __future__ import annotations

import json
import pathlib
import sys
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="torch は ComfyUI が提供する。torch 無しの CI レグではスキップ"
)

from maskveil.core.types import Detection
from maskveil.nodes import refiner as refiner_node
from maskveil.nodes.refiner import HotscreenRefiner

MODEL = "seg/fake.onnx"

VAGINA_RING = (((100.0, 100.0), (140.0, 100.0), (140.0, 150.0), (100.0, 150.0), (100.0, 100.0)),)


def _item(
    class_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
    *,
    masks: Any = None,
    mask_source: str = "none",
    batch_index: int = 0,
    expanded_px: int = 4,
) -> dict[str, Any]:
    polygons = (
        [[[float(x), float(y)] for x, y in ring] for ring in masks] if masks is not None else None
    )
    return {
        "class_id": class_id,
        "class_name": class_name,
        "score": 0.7,
        "bbox": [float(value) for value in bbox],
        "masks": polygons,
        "mask_source": mask_source,
        "batch_index": batch_index,
        "expanded_px": expanded_px,
    }


def _envelope(items: list[dict[str, Any]], *, batch: int = 1, size: tuple[int, int] = (256, 256)):
    height, width = size
    image = torch.zeros((batch, height, width, 3), dtype=torch.float32)
    return {
        "schema_version": 1,
        "image": image,
        "mosaic_image": None,
        "mask": torch.zeros((batch, height, width), dtype=torch.float32),
        "detections": items,
        "meta": {
            "det": {"model": {"path": "/abs/det.onnx", "sha256": "a" * 64, "input_size": 640}},
            "refine": None,
            "mask_expand": 4,
            "size": [batch, height, width],
            "processing": [],
        },
    }


class _StubRefiner:
    """`core.refiner.Refiner` と同じ `detect()` / `detect_region()` を持つスタブ。"""

    input_size = 1280

    def __init__(self, candidates: list[Detection]) -> None:
        self.candidates = candidates
        self.calls: list[list[int]] = []
        self.region_calls: list[list[int]] = []
        self.params: list[dict[str, Any]] = []

    def detect(self, image: np.ndarray, *, class_ids, conf, iou):  # noqa: ANN001
        self.calls.append(list(class_ids))
        self.params.append({"conf": conf, "iou": iou, "kind": "detect"})
        return list(self.candidates)

    def detect_region(self, image: np.ndarray, bbox, *, class_ids, conf, iou, padding=0.1):  # noqa: ANN001, ARG002
        self.region_calls.append(list(class_ids))
        self.params.append({"conf": conf, "iou": iou, "kind": "detect_region"})
        return list(self.candidates)


def _vagina_candidate() -> Detection:
    return Detection(
        class_id=3,
        class_name="vagina",
        score=0.8,
        bbox=(100.0, 100.0, 140.0, 150.0),
        masks=VAGINA_RING,
        mask_source="model_seg",
    )


@pytest.fixture(autouse=True)
def stubbed_folder_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """`INPUT_TYPES()` が使う `folder_paths` のスタブ。"""
    stub = types.ModuleType("folder_paths")
    stub.get_filename_list = lambda folder: (  # type: ignore[attr-defined]
        ["det/a.onnx", "seg/b.onnx"] if folder == "hotscreen" else []
    )
    stub.get_full_path = lambda folder, filename: (  # type: ignore[attr-defined]
        f"/workspace/models/{folder}/{filename}"
    )
    monkeypatch.setitem(sys.modules, "folder_paths", stub)


@pytest.fixture(autouse=True)
def patched_models():
    with (
        patch.object(
            refiner_node, "resolve_model_path", return_value=pathlib.Path("/tmp/seg.onnx")
        ),
        patch.object(
            refiner_node,
            "model_metadata",
            return_value={"path": "/tmp/seg.onnx", "sha256": "b" * 64, "input_size": 1280},
        ),
    ):
        yield


def _run(
    envelope: dict[str, Any],
    *,
    candidates: list[Detection],
    refiner_classes: str = "all",
    mask_expand: int = 4,
    preview_mode: str = "none",
    conf: float = 0.2,
    iou: float = 0.45,
    seg_crop: bool = False,
):
    stub = _StubRefiner(candidates)
    with patch.object(refiner_node, "refiner_for", return_value=stub):
        result = HotscreenRefiner().refine(
            hotscreen=envelope,
            refiner_model=MODEL,
            refiner_classes=refiner_classes,
            conf=conf,
            iou=iou,
            seg_crop=seg_crop,
            mask_expand=mask_expand,
            preview_mode=preview_mode,
        )
    return (*result, stub)


def test_refiner_has_no_image_input() -> None:
    input_types = HotscreenRefiner.INPUT_TYPES()
    names = set(input_types["required"]) | set(input_types.get("optional", {}))
    assert "image" not in names
    assert list(HotscreenRefiner.RETURN_TYPES) == ["HOTSCREEN", "MASK", "STRING", "IMAGE"]
    assert list(HotscreenRefiner.RETURN_NAMES) == ["hotscreen", "mask", "refine", "preview"]


def test_refiner_classes_use_detection_vocabulary() -> None:
    from maskveil.core.labels import LABELS, UnknownClassError

    input_types = HotscreenRefiner.INPUT_TYPES()
    assert input_types["required"]["refiner_classes"][1]["default"] == "all"

    # `all` は検出16クラス（輪郭クラスではない）
    envelope = _envelope([])
    with patch.object(refiner_node, "refiner_for", return_value=_StubRefiner([])):
        HotscreenRefiner().refine(envelope, MODEL, "all", 0.2, 0.45, False, 4, "none")
    with pytest.raises(UnknownClassError):
        HotscreenRefiner().refine(envelope, MODEL, "vagina", 0.2, 0.45, False, 4, "none")
    assert len(LABELS) == 16


def test_replaces_only_mapped_classes() -> None:
    items = [
        _item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0)),
        _item(2, "FEMALE_GENITALIA_COVERED", (10.0, 10.0, 30.0, 30.0)),
        _item(15, "EYE", (40.0, 40.0, 50.0, 50.0)),
    ]
    envelope, mask, summary, _preview, stub = _run(
        _envelope(items), candidates=[_vagina_candidate()]
    )

    by_class = {item["class_name"]: item for item in envelope["detections"]}
    replaced = by_class["FEMALE_GENITALIA_EXPOSED"]
    assert replaced["mask_source"] == "model_seg"
    assert replaced["masks"] is not None
    assert replaced["bbox"] == [100.0, 100.0, 140.0, 150.0]

    for name in ("FEMALE_GENITALIA_COVERED", "EYE"):
        assert by_class[name]["mask_source"] == "none"
        assert by_class[name]["masks"] is None

    data = json.loads(summary)
    assert data == {"targets": 1, "refined": 1, "preserved": 0, "mask_source": "model_seg"}
    assert stub.calls == [[3]]  # 輪郭モデルには vagina(3) だけを要求する
    assert float(mask.sum()) > 0


def test_preserved_when_no_candidate() -> None:
    items = [_item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0))]
    envelope, _, summary, _, _ = _run(_envelope(items), candidates=[])

    item = envelope["detections"][0]
    assert item["mask_source"] == "none"
    assert item["masks"] is None
    assert json.loads(summary)["preserved"] == 1
    assert json.loads(summary)["refined"] == 0


def test_refiner_classes_limits_targets() -> None:
    items = [
        _item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0)),
        _item(11, "MALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0)),
    ]
    _, _, summary, _, stub = _run(
        _envelope(items),
        candidates=[_vagina_candidate()],
        refiner_classes="FEMALE_GENITALIA_EXPOSED",
    )

    assert json.loads(summary)["targets"] == 1
    assert stub.calls == [[3]]


def test_mask_expand_applies_only_to_replaced() -> None:
    items = [
        _item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0), expanded_px=0),
        _item(2, "FEMALE_GENITALIA_COVERED", (10.0, 10.0, 30.0, 30.0), expanded_px=0),
    ]
    envelope, _, _, _, _ = _run(_envelope(items), candidates=[_vagina_candidate()], mask_expand=6)

    by_class = {item["class_name"]: item for item in envelope["detections"]}
    assert by_class["FEMALE_GENITALIA_EXPOSED"]["expanded_px"] == 6
    assert by_class["FEMALE_GENITALIA_COVERED"]["expanded_px"] == 0


def test_meta_refine_does_not_break_meta_det() -> None:
    items = [_item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0))]
    envelope, _, _, _, _ = _run(_envelope(items), candidates=[_vagina_candidate()])

    meta = envelope["meta"]
    assert meta["det"] == {
        "model": {"path": "/abs/det.onnx", "sha256": "a" * 64, "input_size": 640}
    }
    assert meta["refine"]["model"]["input_size"] == 1280
    assert meta["refine"]["params"] == {
        "classes": list(range(16)),
        "conf": 0.2,
        "iou": 0.45,
        "seg_crop": False,
        "mask_expand": 4,
    }
    assert meta["mask_expand"] == 4
    assert meta["processing"] == []


def test_input_envelope_is_not_mutated() -> None:
    items = [_item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0))]
    envelope = _envelope(items)
    before_detections = [dict(item) for item in envelope["detections"]]
    before_meta = json.loads(json.dumps(envelope["meta"]))
    image = envelope["image"]

    result, _, _, _, _ = _run(envelope, candidates=[_vagina_candidate()])

    assert result is not envelope
    assert envelope["detections"] == before_detections
    assert envelope["meta"] == before_meta
    assert result["image"] is image
    assert result["meta"] is not envelope["meta"]


def test_empty_detections_returns_zero_mask() -> None:
    envelope, mask, summary, preview, stub = _run(_envelope([]), candidates=[_vagina_candidate()])

    assert envelope["detections"] == []
    assert float(mask.sum()) == 0.0
    assert json.loads(summary)["targets"] == 0
    assert stub.calls == []


def test_batches_are_refined_per_image() -> None:
    items = [
        _item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0), batch_index=0),
        _item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0), batch_index=1),
    ]
    envelope, mask, summary, _, stub = _run(
        _envelope(items, batch=2), candidates=[_vagina_candidate(), _vagina_candidate()]
    )

    assert len(stub.calls) == 2
    assert json.loads(summary)["targets"] == 2
    assert [item["batch_index"] for item in envelope["detections"]] == [0, 1]
    assert float(mask[0].sum()) > 0 and float(mask[1].sum()) > 0


def test_invalid_schema_version_raises() -> None:
    envelope = _envelope([])
    envelope["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        HotscreenRefiner().refine(envelope, MODEL, "all", 0.2, 0.45, False, 4, "none")


def test_missing_envelope_key_raises() -> None:
    envelope = _envelope([])
    del envelope["mask"]
    with pytest.raises(ValueError, match="必須キー"):
        HotscreenRefiner().refine(envelope, MODEL, "all", 0.2, 0.45, False, 4, "none")


def test_forwards_conf_iou_to_refiner() -> None:
    items = [_item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0))]
    _, _, _, _, stub = _run(_envelope(items), candidates=[_vagina_candidate()], conf=0.33, iou=0.22)

    assert stub.params[0]["conf"] == pytest.approx(0.33)
    assert stub.params[0]["iou"] == pytest.approx(0.22)
    assert stub.params[0]["kind"] == "detect"


def test_seg_crop_uses_detect_region_instead_of_detect() -> None:
    items = [_item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0))]
    envelope, _, summary, _, stub = _run(
        _envelope(items), candidates=[_vagina_candidate()], seg_crop=True
    )

    assert stub.calls == []
    assert stub.region_calls == [[3]]
    assert envelope["detections"][0]["mask_source"] == "model_seg"
    assert json.loads(summary)["refined"] == 1


def test_seg_crop_preserves_when_region_has_no_candidate() -> None:
    items = [_item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0))]
    _, _, summary, _, _ = _run(_envelope(items), candidates=[], seg_crop=True)

    assert json.loads(summary)["preserved"] == 1
    assert json.loads(summary)["refined"] == 0


def test_out_of_range_batch_index_raises() -> None:
    items = [_item(3, "FEMALE_GENITALIA_EXPOSED", (100.0, 100.0, 140.0, 150.0), batch_index=5)]
    with pytest.raises(ValueError, match="batch_index out of range"):
        _run(_envelope(items, batch=1), candidates=[_vagina_candidate()])
