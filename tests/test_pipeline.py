"""輪郭付与パイプライン（対象選別・割当・crop モード）のテスト。"""

from __future__ import annotations

import numpy as np

from maskveil.core.constants import (
    MASK_SOURCE_BBOX_ELLIPSE,
    MASK_SOURCE_MODEL_SEG,
    MASK_SOURCE_NONE,
)
from maskveil.core.pipeline import (
    RefineOptions,
    apply_ellipse_masks,
    plan_targets,
    refine_with_refiner,
)
from maskveil.core.types import Detection

MASKS_A = (((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 1.0)),)
MASKS_B = (((5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 5.0)),)


def _det(clazz: str, bbox, *, class_id: int = 0, score: float = 0.9, masks=None, source="none"):
    return Detection(
        class_id=class_id,
        class_name=clazz,
        score=score,
        bbox=bbox,
        masks=masks,
        mask_source=source,
    )


class FakeRefiner:
    """`Refiner` の代役（推論せず固定の候補を返す）。"""

    input_size = 1280

    def __init__(self, candidates=None, region_candidates=None) -> None:
        self.candidates = candidates or []
        self.region_candidates = region_candidates or []
        self.region_calls = 0

    def detect(self, image, *, class_ids, conf, iou):  # noqa: ANN001, ARG002
        return list(self.candidates)

    def detect_region(self, image, bbox, *, class_ids, conf, iou):  # noqa: ANN001, ARG002
        self.region_calls += 1
        return list(self.region_candidates)


def test_plan_targets_selects_mapped_and_respects_filter() -> None:
    detections = [
        _det("MALE_GENITALIA_EXPOSED", (0.0, 0.0, 10.0, 10.0), class_id=11),
        _det("FEMALE_FACE", (10.0, 10.0, 20.0, 20.0), class_id=0),
        _det("FEET_EXPOSED", (20.0, 20.0, 30.0, 30.0), class_id=14),
    ]

    plan = plan_targets(detections)
    assert plan.target_indices == (0, 1)
    assert plan.count == 2
    assert plan.seg_class_ids  # penis / female face

    filtered = plan_targets(detections, frozenset({11}))
    assert filtered.target_indices == (0,)
    assert filtered.count == 1


def test_plan_targets_empty_when_no_mapping() -> None:
    plan = plan_targets([_det("EYE", (0.0, 0.0, 1.0, 1.0), class_id=15)])
    assert plan.is_empty is True


def test_apply_ellipse_masks_only_targets() -> None:
    detections = [
        _det("MALE_GENITALIA_EXPOSED", (0.0, 0.0, 10.0, 10.0), class_id=11),
        _det("FEET_EXPOSED", (0.0, 0.0, 10.0, 10.0), class_id=14),
    ]
    refined = apply_ellipse_masks(detections, plan_targets(detections))

    assert refined[0].mask_source == MASK_SOURCE_BBOX_ELLIPSE
    assert refined[0].masks is not None
    assert refined[1].mask_source == MASK_SOURCE_NONE
    assert refined[1].masks is None


def test_refine_with_refiner_assigns_by_overlap() -> None:
    detections = [_det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11)]
    candidate = _det("penis", (12.0, 12.0, 48.0, 48.0), masks=MASKS_A, source="model_seg")
    refined = refine_with_refiner(
        np.zeros((10, 10, 3), dtype=np.uint8),
        detections,
        plan_targets(detections),
        FakeRefiner([candidate]),
    )

    assert refined[0].masks == MASKS_A
    assert refined[0].mask_source == MASK_SOURCE_MODEL_SEG


def test_refine_with_empty_plan_keeps_existing_masks() -> None:
    existing = _det(
        "FEET_EXPOSED", (0.0, 0.0, 10.0, 10.0), class_id=14, masks=MASKS_B, source="model_seg"
    )
    refined = refine_with_refiner(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [existing],
        plan_targets([existing]),
        FakeRefiner(),
    )
    assert refined[0].masks == MASKS_B


def test_crop_mode_chooses_highest_iou_not_score() -> None:
    detections = [_det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11)]
    far = _det("penis", (0.0, 0.0, 100.0, 100.0), masks=MASKS_B, score=0.99, source="model_seg")
    near = _det("penis", (12.0, 12.0, 48.0, 48.0), masks=MASKS_A, score=0.5, source="model_seg")
    refined = refine_with_refiner(
        np.zeros((10, 10, 3), dtype=np.uint8),
        detections,
        plan_targets(detections),
        FakeRefiner(region_candidates=[far, near]),
        seg_crop=True,
    )
    assert refined[0].masks == MASKS_A


def test_refine_options_defaults() -> None:
    options = RefineOptions()
    assert options.mask_source == MASK_SOURCE_MODEL_SEG
    assert options.seg_crop is False
    assert options.allowed_classes is None
