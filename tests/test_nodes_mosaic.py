"""`MosaicByMask` の契約（AC-4）。

**判断処理を持たない**こと（`classes` / `conf` / `threshold` / `mask_margin` を持たず、
`detections` を参照しない）と、ベース画像・マスクの解決順、画素不変を検証する。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from maskveil.nodes.mosaic import MosaicByMask


def _image(height: int = 16, width: int = 16, batch: int = 1) -> torch.Tensor:
    values = np.arange(batch * height * width * 3, dtype=np.float32) % 256.0
    return torch.from_numpy(values.reshape(batch, height, width, 3) / 255.0)


def _mask(height: int = 16, width: int = 16, box: tuple[int, int, int, int] = (4, 4, 8, 8)):
    bitmap = np.zeros((1, height, width), dtype=np.float32)
    x1, y1, x2, y2 = box
    bitmap[0, y1:y2, x1:x2] = 1.0
    return torch.from_numpy(bitmap)


def _envelope(
    *,
    batch: int = 1,
    size: tuple[int, int] = (16, 16),
    detections: list[dict[str, Any]] | None = None,
    mosaic_image: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    height, width = size
    return {
        "schema_version": 1,
        "image": _image(height, width, batch),
        "mosaic_image": mosaic_image,
        "mask": _mask(height, width) if mask is None else mask,
        "detections": detections if detections is not None else [],
        "meta": {
            "det": None,
            "refine": None,
            "mask_expand": 0,
            "size": [batch, height, width],
            "processing": [],
        },
    }


def _quantized(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()


def test_input_types_have_no_judgment_parameters() -> None:
    input_types = MosaicByMask.INPUT_TYPES()
    names = set(input_types["required"]) | set(input_types["optional"])

    assert names == {"mode", "block_size", "hotscreen", "image", "mask"}
    assert not names & {"classes", "conf", "iou", "top_k", "threshold", "mask_margin"}
    assert list(MosaicByMask.RETURN_TYPES) == ["IMAGE", "HOTSCREEN"]
    assert list(MosaicByMask.RETURN_NAMES) == ["image", "hotscreen"]
    assert MosaicByMask.INPUT_TYPES()["required"]["block_size"][1] == {
        "default": 48,
        "min": 2,
        "max": 128,
        "step": 1,
    }


def test_explicit_image_wins_over_hotscreen() -> None:
    envelope = _envelope()
    explicit = _image() * 0.0 + 0.2
    output, updated = MosaicByMask().apply("black", 8, hotscreen=envelope, image=explicit)

    assert np.array_equal(_quantized(output)[0, 4:8, 4:8], np.zeros((4, 4, 3), dtype=np.uint8))
    assert updated["meta"]["processing"][-1]["source"] == "explicit"


def test_mosaic_image_is_used_for_second_pass() -> None:
    first = _image() * 0.0 + 0.4
    envelope = _envelope(mosaic_image=first)
    output, updated = MosaicByMask().apply("black", 8, hotscreen=envelope)

    assert np.array_equal(_quantized(output)[0, 0:4, 0:4], np.full((4, 4, 3), 102, dtype=np.uint8))
    assert updated["meta"]["processing"][-1]["source"] == "mosaic_image"


def test_hotscreen_image_is_used_when_no_mosaic_image() -> None:
    envelope = _envelope()
    output, updated = MosaicByMask().apply("black", 8, hotscreen=envelope)

    assert updated["meta"]["processing"][-1]["source"] == "image"
    outside = ~_mask().numpy()[0].astype(bool)
    assert np.array_equal(_quantized(output)[0][outside], _quantized(envelope["image"])[0][outside])


def test_no_image_source_raises() -> None:
    with pytest.raises(ValueError, match="no image source"):
        MosaicByMask().apply("black", 8)


def test_no_mask_source_raises() -> None:
    with pytest.raises(ValueError, match="no mask source"):
        MosaicByMask().apply("black", 8, image=_image())


def test_schema_version_other_than_one_raises() -> None:
    envelope = _envelope()
    envelope["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        MosaicByMask().apply("black", 8, hotscreen=envelope)


def test_zero_mask_returns_input_and_keeps_processing() -> None:
    envelope = _envelope(mask=torch.zeros((1, 16, 16)), mosaic_image=None)
    output, updated = MosaicByMask().apply("mosaic", 8, hotscreen=envelope)

    assert np.array_equal(_quantized(output), _quantized(envelope["image"]))
    assert updated["mosaic_image"] is None
    assert updated["meta"]["processing"] == []


def test_zero_mask_without_hotscreen_returns_input() -> None:
    image = _image()
    output, updated = MosaicByMask().apply("black", 8, image=image, mask=torch.zeros((1, 16, 16)))

    assert output is image
    assert updated["mosaic_image"] is None
    assert updated["detections"] == []
    assert updated["meta"]["processing"] == []
    assert updated["meta"]["size"] == [1, 16, 16]


def test_pixels_outside_mask_are_identical() -> None:
    image = _image()
    mask = _mask()
    output, _ = MosaicByMask().apply("mosaic", 4, image=image, mask=mask)

    inside = mask.numpy()[0].astype(bool)
    outside = ~inside
    assert np.array_equal(_quantized(output)[0][outside], _quantized(image)[0][outside])


@pytest.mark.parametrize("mode", ["mosaic", "blur", "black"])
def test_all_modes_change_masked_pixels(mode: str) -> None:
    image = _image()
    mask = _mask()
    output, _ = MosaicByMask().apply(mode, 4, image=image, mask=mask)

    inside = mask.numpy()[0].astype(bool)
    assert not np.array_equal(_quantized(output)[0][inside], _quantized(image)[0][inside])


def test_processing_history_is_appended() -> None:
    envelope = _envelope()
    _, first = MosaicByMask().apply("black", 8, hotscreen=envelope)
    _, second = MosaicByMask().apply("blur", 4, hotscreen=first)

    assert first["meta"]["processing"] == [{"mode": "black", "block_size": 8, "source": "image"}]
    assert second["meta"]["processing"] == [
        {"mode": "black", "block_size": 8, "source": "image"},
        {"mode": "blur", "block_size": 4, "source": "mosaic_image"},
    ]


def test_envelope_image_is_preserved() -> None:
    envelope = _envelope()
    _, updated = MosaicByMask().apply("black", 8, hotscreen=envelope)

    assert updated["image"] is envelope["image"]
    assert updated["mosaic_image"] is not None
    assert updated["detections"] is envelope["detections"]
    assert updated is not envelope
    assert updated["meta"] is not envelope["meta"]


def test_detections_are_not_used() -> None:
    item = {
        "class_id": 3,
        "class_name": "FEMALE_GENITALIA_EXPOSED",
        "score": 0.9,
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "masks": None,
        "mask_source": "none",
        "batch_index": 0,
        "expanded_px": 0,
    }
    without = _envelope()
    with_detections = _envelope(detections=[item])

    output_a, _ = MosaicByMask().apply("black", 8, hotscreen=without)
    output_b, updated = MosaicByMask().apply("black", 8, hotscreen=with_detections)

    assert np.array_equal(_quantized(output_a), _quantized(output_b))
    assert updated["detections"] == [item]


def test_mask_hw_is_broadcast_to_batch() -> None:
    image = _image(batch=2)
    mask_2d = _mask()[0]
    output, _ = MosaicByMask().apply("black", 8, image=image, mask=mask_2d)

    assert tuple(output.shape) == (2, 16, 16, 3)
    assert np.array_equal(_quantized(output)[0], _quantized(output)[1])


def test_batch_mismatch_raises() -> None:
    mismatched = torch.zeros((3, 16, 16), dtype=torch.float32)
    mismatched[:, 4:8, 4:8] = 1.0
    with pytest.raises(ValueError, match="batch size mismatch"):
        MosaicByMask().apply("black", 8, image=_image(batch=2), mask=mismatched)


def test_mask_size_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mask size mismatch"):
        MosaicByMask().apply("black", 8, image=_image(16, 16), mask=_mask(8, 8, (1, 1, 4, 4)))


def test_mosaic_image_size_mismatch_raises() -> None:
    envelope = _envelope()
    envelope["mosaic_image"] = _image(8, 8)
    with pytest.raises(ValueError, match="mosaic_image size mismatch"):
        MosaicByMask().apply("black", 8, hotscreen=envelope)


def _image_value(height: int, width: int, value: float, batch: int = 1) -> torch.Tensor:
    return torch.full((batch, height, width, 3), value, dtype=torch.float32)


def test_mask_applies_per_batch() -> None:
    """バッチごとに「自分の画像 × 自分のマスク」で加工されること。

    `frames[0]` / `bitmap[0]` を全バッチへ使い回す実装（よくある取り違え）を検出する。
    """
    images = torch.cat(
        [
            _image_value(16, 16, 0.1),
            _image_value(16, 16, 0.5),
            _image_value(16, 16, 0.9),
        ],
        dim=0,
    )
    masks = torch.zeros((3, 16, 16), dtype=torch.float32)
    masks[0, 0:4, 0:4] = 1.0  # batch0: 左上
    # batch1 は全ゼロ（加工されない）
    masks[2, 12:16, 12:16] = 1.0  # batch2: 右下

    output, _ = MosaicByMask().apply("black", 4, image=images, mask=masks)

    assert np.array_equal(_quantized(output)[0, 0:4, 0:4], np.zeros((4, 4, 3), dtype=np.uint8))
    assert np.array_equal(_quantized(output)[0, 12:16, 12:16], _quantized(images)[0, 12:16, 12:16])
    # マスク全ゼロのバッチは完全に不変
    assert np.array_equal(_quantized(output)[1], _quantized(images)[1])
    assert np.array_equal(_quantized(output)[2, 12:16, 12:16], np.zeros((4, 4, 3), dtype=np.uint8))
    assert np.array_equal(_quantized(output)[2, 0:4, 0:4], _quantized(images)[2, 0:4, 0:4])


def test_zero_mask_keeps_previous_mosaic_image() -> None:
    """2段目に検出0件の画像が来ても、1段目のモザイクが消えないこと。"""
    previous = _image_value(16, 16, 0.25)
    envelope = _envelope(mosaic_image=previous)
    before_processing = list(envelope["meta"]["processing"])

    output, updated = MosaicByMask().apply(
        "black", 8, hotscreen=envelope, mask=torch.zeros((1, 16, 16))
    )

    assert updated["mosaic_image"] is previous
    assert np.array_equal(_quantized(output), _quantized(previous))
    assert updated["meta"]["processing"] == before_processing


def test_explicit_mask_wins_over_hotscreen_mask() -> None:
    """明示 `mask` が `hotscreen.mask` より優先されること（解決順）。"""
    envelope = _envelope()  # hotscreen.mask は 4:8, 4:8 を覆う
    explicit = torch.zeros((1, 16, 16), dtype=torch.float32)
    explicit[0, 10:14, 10:14] = 1.0

    output, updated = MosaicByMask().apply("black", 8, hotscreen=envelope, mask=explicit)

    image = _quantized(envelope["image"])[0]
    # 明示マスクの領域だけが加工される
    assert np.array_equal(_quantized(output)[0, 10:14, 10:14], np.zeros((4, 4, 3), dtype=np.uint8))
    assert np.array_equal(_quantized(output)[0, 4:8, 4:8], image[4:8, 4:8])
    assert updated["mask"] is explicit


def test_soft_mask_below_half_is_applied() -> None:
    """0 < v <= 0.5 のソフトマスクも適用対象。"""
    image = _image()
    soft = torch.zeros((1, 16, 16), dtype=torch.float32)
    soft[0, 2:6, 2:6] = 0.3

    output, updated = MosaicByMask().apply("black", 8, image=image, mask=soft)

    assert np.array_equal(_quantized(output)[0, 2:6, 2:6], np.zeros((4, 4, 3), dtype=np.uint8))
    assert updated["meta"]["processing"] == [
        {"mode": "black", "block_size": 8, "source": "explicit"}
    ]


def test_soft_mask_zero_is_ignored() -> None:
    image = _image()
    zero = torch.zeros((1, 16, 16), dtype=torch.float32)
    output, updated = MosaicByMask().apply("black", 8, image=image, mask=zero)

    assert np.array_equal(_quantized(output), _quantized(image))
    assert updated["meta"]["processing"] == []
