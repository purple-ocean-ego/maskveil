"""輪郭 CLI（`refine`）: 検出JSON + 輪郭モデル → マスク付きJSON。

stdin の検出JSON（schema_version 1/2）を読み、対応表に該当する検出へ
形状ポリゴン（原画像座標）を付与して schema_version 2 で返す。

v2 入力（refine 済み）を再投入しても、対象外の検出のマスクは保持される（冪等）。

例::

    python -m maskveil.tools.detect img.png --classes all \
      --model /abs/det.onnx | python -m maskveil.tools.refine \
      --refiner-model /abs/seg.onnx > refined.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from maskveil.core.constants import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    MASK_SOURCE_BBOX_ELLIPSE,
    MASK_SOURCE_MODEL_SEG,
    MASK_SOURCE_NONE,
    MASK_SOURCES,
)
from maskveil.core.labels import UnknownClassError, resolve_classes
from maskveil.core.pipeline import (
    RefineOptions,
    apply_ellipse_masks,
    plan_targets,
    refine_with_refiner,
)
from maskveil.core.refiner import Refiner
from maskveil.core.types import Detection, MaskSource, ModelInfo, Polygon
from maskveil.tools._common import (
    add_refiner_input_size_arg,
    load_rgb,
    report_write_error,
    validate_input_size,
    validate_thresholds,
    write_json,
)
from maskveil.tools.paths import InvalidPathError, sha256_file, validate_model_path

SCHEMA_VERSION = 2
SUPPORTED_INPUT_SCHEMAS = (1, 2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maskveil-refine",
        description="検出JSON（stdin）の検出へ形状マスクを付与し、JSON（stdout）で返す。",
    )
    parser.add_argument(
        "--refiner-model",
        help="輪郭モデル(.onnx)の絶対パス（--mask-source model_seg のとき必須）",
    )
    add_refiner_input_size_arg(parser)
    parser.add_argument(
        "--classes",
        help="絞り込む検出クラス（名前 / ID / all）。省略時は入力JSONの全検出を対象",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="輪郭モデルのスコア閾値")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="輪郭モデルの NMS IoU 閾値")
    parser.add_argument(
        "--mask-source",
        choices=MASK_SOURCES,
        default=MASK_SOURCE_MODEL_SEG,
        help="形状の生成方法（bbox_ellipse はデバッグ用）",
    )
    parser.add_argument(
        "--seg-crop",
        action="store_true",
        help="検出 bbox 周辺をクロップして輪郭推論する（計測用。遅い）",
    )
    parser.add_argument("--output", help="出力JSONのパス（省略時は標準出力）")
    parser.add_argument("--pretty", action="store_true", help="JSON を整形して出力")
    return parser


def _parse_polygons(raw: Any) -> tuple[Polygon, ...] | None:
    """JSON の `masks` をポリゴンのタプルへ変換する。"""
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("'masks' must be a non-empty list of polygons")
    polygons: list[Polygon] = []
    for ring in raw:
        if not isinstance(ring, list) or len(ring) < 3:
            raise ValueError("each mask polygon must have at least 3 points")
        polygons.append(tuple((float(point[0]), float(point[1])) for point in ring))
    return tuple(polygons)


def _parse_detections(payload: dict[str, Any]) -> list[Detection]:
    detections: list[Detection] = []
    for raw in payload["detections"]:
        mask_source = str(raw.get("mask_source", MASK_SOURCE_NONE))
        if mask_source not in (MASK_SOURCE_NONE, *MASK_SOURCES):
            raise ValueError(f"unknown mask_source: {mask_source!r}")
        detections.append(
            Detection(
                class_id=int(raw["class_id"]),
                class_name=str(raw["class_name"]),
                score=float(raw["score"]),
                bbox=(
                    float(raw["bbox"][0]),
                    float(raw["bbox"][1]),
                    float(raw["bbox"][2]),
                    float(raw["bbox"][3]),
                ),
                masks=_parse_polygons(raw.get("masks")),
                mask_source=cast(MaskSource, mask_source),
            )
        )
    return detections


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    if payload.get("schema_version") not in SUPPORTED_INPUT_SCHEMAS:
        raise ValueError(f"unsupported schema_version: {payload.get('schema_version')!r}")
    for key in ("image", "detections"):
        if key not in payload:
            raise ValueError(f"input JSON is missing '{key}'")
    image = payload["image"]
    if not isinstance(image, dict) or "path" not in image:
        raise ValueError("input JSON is missing 'image.path'")
    if not isinstance(payload["detections"], list):
        raise ValueError("'detections' must be a list")
    return payload


def _det_model_info(payload: dict[str, Any]) -> Any:
    """v1 はフラット、v2 は `model.det` に検出モデル情報を持つ。"""
    model = payload.get("model")
    if isinstance(model, dict) and "det" in model:
        return model.get("det")
    return model


def _det_params(payload: dict[str, Any]) -> Any:
    """v1 はフラット、v2 は `params.det` に検出パラメータを持つ。"""
    params = payload.get("params")
    if isinstance(params, dict) and "det" in params:
        return params.get("det")
    return params


def _build_result(
    *,
    payload: dict[str, Any],
    image_info: dict[str, Any],
    detections: list[Detection],
    plan_count: int,
    preserved: int,
    seg_info: ModelInfo | None,
    options: RefineOptions,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "image": image_info,
        "model": {"det": _det_model_info(payload), "seg": seg_info.to_dict() if seg_info else None},
        "params": {
            "det": _det_params(payload),
            "refine": {
                "mask_source": options.mask_source,
                "classes": sorted(options.allowed_classes) if options.allowed_classes else "all",
                "conf": options.conf,
                "iou": options.iou,
                "seg_crop": options.seg_crop,
            },
        },
        "detections": [det.to_dict(include_mask=True) for det in detections],
        "count": len(detections),
        "refine": {
            "targets": plan_count,
            "refined": sum(1 for det in detections if det.masks is not None),
            "preserved": preserved,
            "mask_source": options.mask_source,
        },
    }


def _resolve_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[frozenset[int] | None, Path | None, RefineOptions]:
    """引数を検証し、実行オプションとモデルパスを返す（不正なら exit 2）。"""
    validate_thresholds(parser, conf=args.conf, iou=args.iou)
    validate_input_size(parser, "--refiner-input-size", args.refiner_input_size)

    allowed: frozenset[int] | None = None
    if args.classes:
        try:
            allowed = frozenset(resolve_classes(args.classes))
        except UnknownClassError as exc:
            parser.error(str(exc))

    model_path: Path | None = None
    if args.mask_source == MASK_SOURCE_MODEL_SEG:
        if not args.refiner_model:
            parser.error(f"--refiner-model is required when --mask-source {MASK_SOURCE_MODEL_SEG}")
        try:
            model_path = validate_model_path(args.refiner_model, option="--refiner-model")
        except InvalidPathError as exc:
            parser.error(str(exc))

    options = RefineOptions(
        mask_source=args.mask_source,
        conf=args.conf,
        iou=args.iou,
        seg_crop=args.seg_crop,
        allowed_classes=allowed,
    )
    return allowed, model_path, options


def _read_input() -> tuple[dict[str, Any], list[Detection]]:
    """stdin の JSON を読み、検証済みの検出リストへ変換する。"""
    payload = _validate_payload(json.loads(sys.stdin.read()))
    return payload, _parse_detections(payload)


def _load_image(payload: dict[str, Any]) -> tuple[Any, dict[str, Any]] | None:
    """画像を読み、JSON の寸法と一致することを検証する（不一致・失敗は None）。"""
    image_path = Path(str(payload["image"]["path"])).resolve()
    try:
        image = load_rgb(image_path)
    except OSError as exc:
        print(f"error: failed to read image {str(image_path)!r}: {exc}", file=sys.stderr)
        return None

    height, width = image.shape[:2]
    declared_width = payload["image"].get("width")
    declared_height = payload["image"].get("height")
    if (
        declared_width is not None
        and declared_height is not None
        and (int(declared_width) != width or int(declared_height) != height)
    ):
        print(
            f"error: image size mismatch: JSON says {declared_width}x{declared_height}, "
            f"file is {width}x{height}",
            file=sys.stderr,
        )
        return None
    return image, {"path": str(image_path), "width": int(width), "height": int(height)}


def _build_refiner(
    parser: argparse.ArgumentParser,
    model_path: Path,
    input_size: int | None,
) -> Refiner:
    """輪郭モデルをロードする（入力サイズ不一致は引数エラー扱い）。"""
    try:
        return Refiner(model_path, input_size=input_size)
    except ValueError as exc:
        parser.error(str(exc))


def _refine(
    parser: argparse.ArgumentParser,
    *,
    image: Any,
    detections: list[Detection],
    plan: Any,
    options: RefineOptions,
    model_path: Path | None,
    input_size: int | None,
) -> tuple[list[Detection] | None, ModelInfo | None]:
    """マスクを付与する（実行時エラーは None を返す）。"""
    if options.mask_source == MASK_SOURCE_BBOX_ELLIPSE:
        return apply_ellipse_masks(detections, plan), None
    if plan.is_empty:
        return list(detections), None

    try:
        assert model_path is not None  # noqa: S101 - model_seg では検証済み
        refiner = _build_refiner(parser, model_path, input_size)
        refined = refine_with_refiner(
            image,
            detections,
            plan,
            refiner,
            conf=options.conf,
            iou=options.iou,
            seg_crop=options.seg_crop,
        )
        info = ModelInfo(
            path=str(model_path),
            sha256=sha256_file(model_path),
            input_size=refiner.input_size,
        )
        return refined, info
    except Exception as exc:  # noqa: BLE001 - CLI は実行時エラーを exit 1 に写像する
        print(f"error: refine failed: {exc}", file=sys.stderr)
        return None, None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    allowed, model_path, options = _resolve_options(parser, args)

    try:
        payload, detections = _read_input()
    except (json.JSONDecodeError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"error: invalid input JSON: {exc}", file=sys.stderr)
        return 1

    loaded = _load_image(payload)
    if loaded is None:
        return 1
    image, image_info = loaded

    plan = plan_targets(detections, allowed)
    targets = set(plan.target_indices)
    preserved = sum(
        1 for index, det in enumerate(detections) if index not in targets and det.masks is not None
    )

    refined, seg_info = _refine(
        parser,
        image=image,
        detections=detections,
        plan=plan,
        options=options,
        model_path=model_path,
        input_size=args.refiner_input_size,
    )
    if refined is None:
        return 1

    result = _build_result(
        payload=payload,
        image_info=image_info,
        detections=refined,
        plan_count=plan.count,
        preserved=preserved,
        seg_info=seg_info,
        options=options,
    )

    try:
        write_json(result, output=args.output, pretty=args.pretty)
    except OSError as exc:
        return report_write_error(str(args.output), exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
