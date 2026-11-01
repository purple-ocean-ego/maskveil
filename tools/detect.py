"""検出 CLI（`detect`）: IMAGE + 検出モデル → 検出JSON。

例::

    python -m maskveil.tools.detect input.png \
      --classes MALE_GENITALIA_EXPOSED \
      --model /abs/path/det.onnx > det.json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from maskveil.core.detector import Detector
from maskveil.core.labels import LABELS, UnknownClassError, resolve_classes
from maskveil.core.types import ModelInfo
from maskveil.tools._common import (
    add_threshold_args,
    load_rgb,
    report_write_error,
    validate_thresholds,
    write_json,
)
from maskveil.tools.paths import InvalidPathError, sha256_file, validate_model_path

SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maskveil-detect",
        description="hotscreen 検出モデルで1枚推論し、原画像座標のbboxをJSONで出力する。",
    )
    parser.add_argument("image", nargs="?", help="入力画像のパス")
    parser.add_argument("--model", help="検出モデル(.onnx)の絶対パス（必須）")
    parser.add_argument(
        "--classes",
        help="対象クラス（名前 / ID / all のカンマ区切り）。未指定はエラー（既定クラスなし）",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="モデル入力の一辺（省略時はモデルから取得）",
    )
    add_threshold_args(parser)
    parser.add_argument("--output", help="出力JSONのパス（省略時は標準出力）")
    parser.add_argument("--pretty", action="store_true", help="JSON を整形して出力")
    parser.add_argument("--list-classes", action="store_true", help="クラス一覧を表示して終了")
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="スコア上位N件に絞る（0 は無効）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_classes:
        for index, name in enumerate(LABELS):
            print(f"{index}\t{name}")
        return 0

    if not args.classes:
        parser.error("--classes is required (no default classes; use 'all' for every class)")
    if not args.model:
        parser.error("--model is required (absolute path to a .onnx file)")
    if not args.image:
        parser.error("IMAGE is required")
    validate_thresholds(parser, conf=args.conf, iou=args.iou)
    if args.input_size is not None and args.input_size <= 0:
        parser.error("--input-size must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be zero or positive")

    try:
        class_ids = resolve_classes(args.classes)
        model_path = validate_model_path(args.model, option="--model")
    except (UnknownClassError, InvalidPathError) as exc:
        parser.error(str(exc))

    image_path = Path(args.image).resolve()
    try:
        image = load_rgb(image_path)
    except OSError as exc:
        print(f"error: failed to read image {str(image_path)!r}: {exc}", file=sys.stderr)
        return 1

    try:
        detector = Detector(model_path, input_size=args.input_size)
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - モデル破損などは実行時エラー
        print(f"error: failed to load model: {exc}", file=sys.stderr)
        return 1

    try:
        detections = detector.detect(image, class_ids=class_ids, conf=args.conf, iou=args.iou)
        input_size = detector.input_size
        model_sha256 = sha256_file(model_path)
    except Exception as exc:  # noqa: BLE001 - CLI は実行時エラーを exit 1 に写像する
        print(f"error: inference failed: {exc}", file=sys.stderr)
        return 1

    if args.top_k > 0:
        detections = detections[: args.top_k]

    height, width = image.shape[:2]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "image": {"path": str(image_path), "width": int(width), "height": int(height)},
        "model": ModelInfo(
            path=str(model_path),
            sha256=model_sha256,
            input_size=input_size,
        ).to_dict(),
        "params": {
            "conf": args.conf,
            "iou": args.iou,
            "classes": list(class_ids),
        },
        "detections": [det.to_dict() for det in detections],
        "count": len(detections),
    }

    try:
        write_json(payload, output=args.output, pretty=args.pretty)
    except OSError as exc:
        return report_write_error(str(args.output), exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
