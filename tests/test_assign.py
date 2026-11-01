"""輪郭の割り当て（`assign_masks`）のテスト。"""

from __future__ import annotations

from maskveil.core.refiner import assign_masks
from maskveil.core.types import Detection

POLYGON = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0))
MASKS = (POLYGON,)


def _det(
    class_name: str,
    bbox: tuple[float, float, float, float],
    *,
    class_id: int = 0,
    score: float = 0.9,
    masks=None,
    mask_source: str = "none",
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        score=score,
        bbox=bbox,
        masks=masks,
        mask_source=mask_source,
    )


def test_assigns_matching_class_by_bbox_overlap() -> None:
    det = _det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11)
    candidate = _det("penis", (12.0, 12.0, 48.0, 48.0), masks=MASKS, mask_source="model_seg")
    [assigned] = assign_masks([det], [candidate])
    assert assigned.masks == MASKS
    assert assigned.mask_source == "model_seg"


def test_unmapped_class_stays_none() -> None:
    det = _det("FEET_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=14)
    candidate = _det("penis", (10.0, 10.0, 50.0, 50.0), masks=MASKS, mask_source="model_seg")
    [assigned] = assign_masks([det], [candidate])
    assert assigned.masks is None
    assert assigned.mask_source == "none"


def test_no_overlap_stays_none() -> None:
    det = _det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11)
    candidate = _det("penis", (500.0, 500.0, 600.0, 600.0), masks=MASKS, mask_source="model_seg")
    [assigned] = assign_masks([det], [candidate])
    assert assigned.mask_source == "none"


def test_candidate_is_used_only_once() -> None:
    first = _det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11, score=0.9)
    second = _det("MALE_GENITALIA_EXPOSED", (11.0, 11.0, 49.0, 49.0), class_id=11, score=0.8)
    candidate = _det("penis", (10.0, 10.0, 50.0, 50.0), masks=MASKS, mask_source="model_seg")
    assigned = assign_masks([first, second], [candidate])
    assert [det.mask_source for det in assigned] == ["model_seg", "none"]


def test_highest_overlap_wins() -> None:
    det = _det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11)
    weak = _det(
        "penis", (0.0, 0.0, 100.0, 100.0), masks=(((9.0, 9.0), (9.0, 9.0), (9.0, 9.0)),), score=0.95
    )
    strong = _det("penis", (12.0, 12.0, 48.0, 48.0), masks=MASKS, mask_source="model_seg")
    [assigned] = assign_masks([det], [weak, strong])
    assert assigned.masks == MASKS


def test_indices_filter_restricts_targets() -> None:
    det = _det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11)
    candidate = _det("penis", (10.0, 10.0, 50.0, 50.0), masks=MASKS, mask_source="model_seg")
    [assigned] = assign_masks([det], [candidate], indices=set())
    assert assigned.mask_source == "none"


def test_candidate_without_mask_is_ignored() -> None:
    det = _det("MALE_GENITALIA_EXPOSED", (10.0, 10.0, 50.0, 50.0), class_id=11)
    candidate = _det("penis", (10.0, 10.0, 50.0, 50.0), masks=None)
    [assigned] = assign_masks([det], [candidate])
    assert assigned.mask_source == "none"
