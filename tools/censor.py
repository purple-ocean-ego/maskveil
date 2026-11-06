"""モザイク CLI（`censor`）: 画像 → 検出 →（任意で輪郭）→ モザイク → PNG。

ComfyUI・WebUI を起動せず、1コマンドでモザイク済み PNG を作る。処理は
`detect →（--refine 指定時のみ）refine → compose_masks → apply_censor → 保存` で、
各段は `core` の関数をそのまま使う（ノードと同一の画素結果になる）。

例::

    python -m maskveil.tools.censor /abs/input \\
      --det-model /abs/det.onnx \\
      --output-dir /abs/input_censored \\
      --report /abs/input_censored/report.json

出力は常に PNG。ファイル名は `NNNN_<元の名前>.png`（**フォルダ毎に 0001 起点**）で、
入力のディレクトリ構造を `--output-dir` 配下へ再現する。

終了コードは `0`（全件成功。検出0件の `clean` を含む）／`1`（1件以上の失敗。処理は
最後まで継続する）／`2`（引数エラー。処理を開始しない）。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from maskveil.core.constants import DEFAULT_BLOCK_SIZE, DEFAULT_MASK_EXPAND
from maskveil.core.detector import Detector
from maskveil.core.labels import SEXUAL_EXPOSED_LABELS, UnknownClassError, resolve_classes
from maskveil.core.mosaic import apply_censor
from maskveil.core.pipeline import plan_targets, refine_with_refiner
from maskveil.core.raster import compose_masks, expand_detections
from maskveil.core.refiner import Refiner
from maskveil.core.types import ModelInfo
from maskveil.tools._common import (
    add_refiner_input_size_arg,
    add_threshold_args,
    load_rgb,
    report_write_error,
    save_png,
    validate_input_size,
    validate_thresholds,
    write_json,
)
from maskveil.tools.paths import InvalidPathError, sha256_file, validate_model_path

SCHEMA_VERSION = 1
TOOL_NAME = "censor"
TOOL_VERSION = "1.0.0"
"""レポートの生成元を特定するための CLI 版（`pyproject.toml` と一致させる）。"""

DEFAULT_CLASSES = ",".join(SEXUAL_EXPOSED_LABELS)
"""既定の対象クラス（性器の露出2クラス）。CLI で何も指定しない場合に使う。"""

CENSOR_MODE = "mosaic"
"""CLI が使う加工方法。`blur` / `black` は ComfyUI ノード側の機能（CLI では公開しない）。"""

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
"""入力として受理する拡張子（大小無視）。それ以外は無視する（失敗にしない）。"""

STATUS_MOSAICKED = "mosaicked"
STATUS_CLEAN = "clean"
STATUS_FAILED = "failed"
STATUSES: tuple[str, ...] = (STATUS_MOSAICKED, STATUS_CLEAN, STATUS_FAILED)
"""1枚ごとの処理結果。レポートの `status` はこのいずれか。"""

ROOT_FOLDER = "."
"""入力ディレクトリ直下を表すフォルダ名（レポートの `folder` / `folders[].folder`）。"""

MIN_BLOCK_SIZE, MAX_BLOCK_SIZE = 2, 128
MIN_MASK_EXPAND, MAX_MASK_EXPAND = 0, 256
MIN_COMPRESS_LEVEL, MAX_COMPRESS_LEVEL = 0, 9


class ModelLoadError(RuntimeError):
    """モデルの読み込みに失敗した（引数は正しいが実行できない = 終了コード 1）。"""


@dataclass(frozen=True, slots=True)
class CensorSettings:
    """1回の実行で固定される処理設定（レポートの `run.params` と同一の出所）。"""

    block_size: int
    mask_expand: int
    conf: float
    iou: float
    class_ids: tuple[int, ...]
    refine: bool
    compress_level: int

    def to_params(self) -> dict[str, Any]:
        """レポート `run.params` を返す。"""
        return {
            "mode": CENSOR_MODE,
            "block_size": self.block_size,
            "mask_expand": self.mask_expand,
            "conf": self.conf,
            "iou": self.iou,
            "classes": list(self.class_ids),
            "refine": self.refine,
            "png_compress_level": self.compress_level,
        }


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(
        prog="maskveil",
        description="画像またはディレクトリをモザイク処理し、PNG で保存する。",
    )
    parser.add_argument("input", metavar="INPUT", help="入力画像、または入力ディレクトリ")
    parser.add_argument("--det-model", help="検出モデル(.onnx)の絶対パス（必須）")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="輪郭モデルでマスク形状を精密化する（--refiner-model が必要）",
    )
    parser.add_argument(
        "--refiner-model",
        help="輪郭モデル(.onnx)の絶対パス（--refine のとき必須）",
    )
    add_refiner_input_size_arg(parser)
    parser.add_argument(
        "--classes",
        default=DEFAULT_CLASSES,
        help=f"対象クラス（名前 / ID / all のカンマ区切り。既定 {DEFAULT_CLASSES}）",
    )
    add_threshold_args(parser)
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            f"モザイクのブロック辺（px、{MIN_BLOCK_SIZE}–{MAX_BLOCK_SIZE}。"
            f"既定 {DEFAULT_BLOCK_SIZE}）"
        ),
    )
    parser.add_argument(
        "--mask-expand",
        type=int,
        default=DEFAULT_MASK_EXPAND,
        help=(
            f"マスクの追加膨張量（px、{MIN_MASK_EXPAND}–{MAX_MASK_EXPAND}。"
            f"既定 {DEFAULT_MASK_EXPAND}）"
        ),
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=1,
        help=(f"PNG の圧縮レベル（{MIN_COMPRESS_LEVEL}–{MAX_COMPRESS_LEVEL}。既定 1）"),
    )
    parser.add_argument(
        "--output-dir",
        help="出力ディレクトリ（既定: 画像なら親、ディレクトリなら <入力>_censored）",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="ディレクトリ入力のときサブディレクトリを処理しない",
    )
    parser.add_argument("--report", help="レポート JSON の保存先（省略時は標準出力）")
    parser.add_argument("--pretty", action="store_true", help="JSON を整形して出力")
    return parser


def _validate_ranges(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """数値オプションの範囲を検証する（不正なら exit 2）。"""
    checks = (
        ("--block-size", args.block_size, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE),
        ("--mask-expand", args.mask_expand, MIN_MASK_EXPAND, MAX_MASK_EXPAND),
        (
            "--png-compress-level",
            args.png_compress_level,
            MIN_COMPRESS_LEVEL,
            MAX_COMPRESS_LEVEL,
        ),
    )
    for option, value, low, high in checks:
        if not low <= value <= high:
            parser.error(f"{option} must be within {low}-{high}")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    """引数を検証し、解決済みの入力パスを返す（不正なら exit 2）。

    Returns:
        `Path.resolve()` 済みの入力パス。レポートのパスをすべて絶対に揃えるため。
    """
    if not args.det_model:
        parser.error("--det-model is required (absolute path to a .onnx file)")
    validate_thresholds(parser, conf=args.conf, iou=args.iou)
    _validate_ranges(parser, args)
    if args.refine and not args.refiner_model:
        parser.error("--refiner-model is required when --refine is given")
    if args.refiner_model and not args.refine:
        parser.error("--refiner-model requires --refine (pass --refine to enable refinement)")
    if args.refiner_input_size is not None and not args.refine:
        parser.error("--refiner-input-size requires --refine (pass --refine to enable refinement)")
    validate_input_size(parser, "--refiner-input-size", args.refiner_input_size)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"INPUT: not found: {args.input}")
    if input_path.is_file() and input_path.suffix.lower() not in IMAGE_SUFFIXES:
        parser.error(
            f"INPUT: unsupported extension {input_path.suffix!r} "
            f"(expected one of {', '.join(IMAGE_SUFFIXES)})"
        )
    return input_path


def _model_info(path: Path, input_size: int) -> dict[str, Any]:
    """レポート用のモデル情報（パス・SHA-256・入力サイズ）を作る。"""
    return ModelInfo(
        path=str(path),
        sha256=sha256_file(path),
        input_size=input_size,
    ).to_dict()


def _refiner_error_message(exc: ValueError, input_size: int | None) -> str:
    """輪郭モデルの読込エラーに、引数指定で解決できる場合だけ対処を併記する。

    `core/onnx_utils.py` は「動的入力サイズ」と「非正方形入力」の両方を `ValueError` で
    投げるが、後者は入力サイズを指定しても解決しないため案内を付けない
    （`core/` は本WPの対象外のため、メッセージの内容で判別する）。
    """
    message = str(exc)
    if input_size is None and "dynamic input size" in message:
        return f"{message} (pass --refiner-input-size to set the refiner input size)"
    return message


def _load_models(
    parser: argparse.ArgumentParser,
    *,
    det_model: str,
    refiner_model: str | None,
    refiner_input_size: int | None,
    refine: bool,
) -> tuple[Detector, Refiner | None, dict[str, Any]]:
    """モデルを読み込み、レポート用の情報も返す（パス不正は exit 2、読込失敗は例外）。"""
    try:
        det_path = validate_model_path(det_model, option="--det-model")
        seg_path = validate_model_path(refiner_model, option="--refiner-model") if refine else None
    except InvalidPathError as exc:
        parser.error(str(exc))

    try:
        detector = Detector(det_path)
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - モデル破損などは実行時エラー
        raise ModelLoadError(f"failed to load detector: {exc}") from exc

    models: dict[str, Any] = {
        "det": _model_info(det_path, detector.input_size),
        "seg": None,
    }

    refiner: Refiner | None = None
    if seg_path is not None:
        try:
            refiner = Refiner(seg_path, input_size=refiner_input_size)
        except ValueError as exc:
            parser.error(_refiner_error_message(exc, refiner_input_size))
        except Exception as exc:  # noqa: BLE001 - モデル破損などは実行時エラー
            raise ModelLoadError(f"failed to load refiner: {exc}") from exc
        models["seg"] = _model_info(seg_path, refiner.input_size)

    return detector, refiner, models


def _iter_folders(
    input_path: Path, *, recursive: bool, exclude: Path | None = None
) -> list[tuple[str, list[Path]]]:
    """処理対象をフォルダ単位でまとめる（フォルダ内は名前順）。

    Args:
        input_path: 入力（ファイルまたはディレクトリ）。
        recursive: サブディレクトリも対象にするか。
        exclude: 列挙から除外するディレクトリ（出力先が入力の内側にある場合）。
    """
    if input_path.is_file():
        return [(ROOT_FOLDER, [input_path])]

    pattern = "**/*" if recursive else "*"
    grouped: dict[str, list[Path]] = {}
    for path in sorted(input_path.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if exclude is not None and path.is_relative_to(exclude):
            continue
        relative = path.parent.relative_to(input_path).as_posix()
        grouped.setdefault(ROOT_FOLDER if relative == "." else relative, []).append(path)
    return [(folder, sorted(paths)) for folder, paths in sorted(grouped.items())]


def _resolve_output_root(
    parser: argparse.ArgumentParser, input_path: Path, output_dir: str | None
) -> Path:
    """出力のルートディレクトリを決める（作成はしない）。"""
    if output_dir:
        root = Path(output_dir).expanduser().resolve()
    elif input_path.is_dir():
        root = input_path.parent / f"{input_path.name}_censored"
    else:
        root = input_path.parent

    if input_path.is_dir() and root == input_path:
        parser.error("--output-dir must not be the INPUT directory itself")
    return root


def _folder_dir(output_root: Path, folder: str) -> Path:
    """フォルダ名（`.` はルート）に対応する出力ディレクトリを返す。"""
    return output_root if folder == ROOT_FOLDER else output_root / folder


def _failure_row(
    source: Path, *, folder: str, sequence: int, index: int, error: str
) -> dict[str, Any]:
    """処理できなかった画像の行を作る。"""
    return {
        "index": index,
        "folder": folder,
        "sequence": sequence,
        "input": str(source),
        "output": None,
        "status": STATUS_FAILED,
        "applied": False,
        "detections": 0,
        "bboxes": [],
        "elapsed_sec": 0.0,
        "error": error,
    }


def _render_censored(
    image: np.ndarray, detections: list[Any], settings: CensorSettings
) -> tuple[np.ndarray, bool]:
    """マスクを合成し、必要ならモザイクを掛けた画像を返す。

    Returns:
        `(保存する画像, モザイクを適用したか)`。検出0件なら入力をそのまま返す
        （`uint8` のまま保存するため、無加工コピーが入力と画素一致する）。
    """
    height, width = image.shape[:2]
    mask = compose_masks(detections, height, width, expand_px=settings.mask_expand)
    if not mask.any():
        return image, False
    censored = apply_censor(
        image.astype(np.float32) / 255.0, mask, CENSOR_MODE, settings.block_size
    )
    return censored, True


def _process_one(
    *,
    source: Path,
    target: Path,
    detector: Detector,
    refiner: Refiner | None,
    settings: CensorSettings,
) -> dict[str, Any]:
    """1枚を処理し、レポート行を返す（`Exception` は failed 行へ写像する）。

    行の `status` は既定 `failed` とし、成功経路だけが上書きする（記録漏れ防止）。
    """
    started = time.perf_counter()
    row: dict[str, Any] = {
        "input": str(source),
        "output": None,
        "status": STATUS_FAILED,
        "applied": False,
        "detections": 0,
        "bboxes": [],
        "elapsed_sec": 0.0,
        "error": None,
    }

    def finish() -> dict[str, Any]:
        row["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return row

    if target.exists():
        row["error"] = f"output already exists: {target}"
        return finish()

    try:
        image = load_rgb(source)
        detections = detector.detect(
            image, class_ids=settings.class_ids, conf=settings.conf, iou=settings.iou
        )
        if refiner is not None:
            plan = plan_targets(detections)
            detections = refine_with_refiner(
                image,
                detections,
                plan,
                refiner,
                conf=settings.conf,
                iou=settings.iou,
                seg_crop=False,
            )
        detections = expand_detections(detections, settings.mask_expand)

        output, applied = _render_censored(image, detections, settings)
        save_png(output, target, compress_level=settings.compress_level)
    except Exception as exc:  # noqa: BLE001 - 1枚の失敗で全体を止めない
        row["error"] = f"{type(exc).__name__}: {exc}"
        return finish()

    row["output"] = str(target)
    row["status"] = STATUS_MOSAICKED if applied else STATUS_CLEAN
    row["applied"] = applied
    if applied:
        row["detections"] = len(detections)
        row["bboxes"] = [[float(value) for value in det.bbox] for det in detections]
    return finish()


def _progress_label(folder: str, name: str) -> str:
    """進捗表示用の名前（ルート直下は名前だけ）。"""
    return name if folder == ROOT_FOLDER else f"{folder}/{name}"


def _run_batch(
    groups: list[tuple[str, list[Path]]],
    output_root: Path,
    *,
    detector: Detector,
    refiner: Refiner | None,
    settings: CensorSettings,
) -> list[dict[str, Any]]:
    """フォルダ単位で処理し、画像ごとの行を返す（失敗しても継続する）。"""
    rows: list[dict[str, Any]] = []
    total = sum(len(paths) for _, paths in groups)
    processed = 0

    for folder_index, (folder, paths) in enumerate(groups, start=1):
        folder_dir = _folder_dir(output_root, folder)
        try:
            folder_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error = f"cannot create output directory: {type(exc).__name__}: {exc}"
            print(
                f"[folder {folder_index}/{len(groups)}] {folder}: {error}",
                file=sys.stderr,
            )
            for sequence, source in enumerate(paths, start=1):
                processed += 1
                rows.append(
                    _failure_row(
                        source,
                        folder=folder,
                        sequence=sequence,
                        index=processed,
                        error=error,
                    )
                )
            continue

        for sequence, source in enumerate(paths, start=1):
            processed += 1
            target = folder_dir / f"{sequence:04d}_{source.stem}.png"
            row = _process_one(
                source=source,
                target=target,
                detector=detector,
                refiner=refiner,
                settings=settings,
            )
            rows.append({"index": processed, "folder": folder, "sequence": sequence, **row})
            print(
                f"[folder {folder_index}/{len(groups)}] {processed}/{total} "
                f"{_progress_label(folder, source.name)} → "
                f"{_progress_label(folder, target.name)} "
                f"{rows[-1]['status']} ({rows[-1]['elapsed_sec']:.2f}s)",
                file=sys.stderr,
            )
    return rows


def _count_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    """行の `status` を1パスで集計する。"""
    counts = Counter(row["status"] for row in rows)
    return {status: counts.get(status, 0) for status in STATUSES}


def _summarize_folders(rows: list[dict[str, Any]], folders: list[str]) -> list[dict[str, Any]]:
    """フォルダ別の集計を作る（1パス）。"""
    counts: dict[str, Counter[str]] = {folder: Counter() for folder in folders}
    for row in rows:
        counts[row["folder"]][row["status"]] += 1
    return [
        {
            "folder": folder,
            "total": sum(counts[folder].values()),
            **{status: counts[folder].get(status, 0) for status in STATUSES},
        }
        for folder in folders
    ]


def _timestamp() -> str:
    """レポート用の UTC タイムスタンプ（秒精度）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_report(
    *,
    input_path: Path,
    output_root: Path,
    recursive: bool,
    started_at: str,
    elapsed_sec: float,
    settings: CensorSettings,
    models: dict[str, Any],
    rows: list[dict[str, Any]],
    folders: list[str],
) -> dict[str, Any]:
    """レポート JSON を組み立てる。"""
    counts = _count_status(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "run": {
            "input": str(input_path),
            "output_dir": str(output_root),
            "recursive": recursive,
            "started_at": started_at,
            "finished_at": _timestamp(),
            "elapsed_sec": elapsed_sec,
            "params": settings.to_params(),
        },
        "model": models,
        "folders": _summarize_folders(rows, folders),
        "images": rows,
        "summary": {"total": len(rows), **counts, "elapsed_sec": elapsed_sec},
    }


def _emit_report(payload: dict[str, Any], *, report: str | None, pretty: bool) -> int:
    """レポートをファイルまたは標準出力へ出し、終了コードを返す。"""
    failed = payload["summary"][STATUS_FAILED]
    exit_code = 1 if failed else 0

    if not report:
        try:
            write_json(payload, output=None, pretty=pretty)
        except OSError as exc:
            return report_write_error("stdout", exc)
        return exit_code

    path = Path(report).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(payload, output=str(path), pretty=pretty)
    except OSError as exc:
        return report_write_error(str(path), exc)

    summary = payload["summary"]
    print(
        f"total={summary['total']} mosaicked={summary[STATUS_MOSAICKED]} "
        f"clean={summary[STATUS_CLEAN]} failed={failed} "
        f"elapsed={summary['elapsed_sec']}s report={path}",
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """CLI のエントリポイント。

    Returns:
        0=全件成功（`clean` を含む） / 1=1件以上の失敗（処理は継続済み） /
        2=引数エラー（`parser.error` により `SystemExit`）。
    """
    run_started = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = _validate_args(parser, args)
    try:
        class_ids = tuple(resolve_classes(args.classes))
    except UnknownClassError as exc:
        parser.error(str(exc))

    # 出力先の検証はモデル読込より先に行う（引数エラーでモデルを読み込まない）
    output_root = _resolve_output_root(parser, input_path, args.output_dir)

    try:
        detector, refiner, models = _load_models(
            parser,
            det_model=args.det_model,
            refiner_model=args.refiner_model,
            refiner_input_size=args.refiner_input_size,
            refine=bool(args.refine),
        )
    except ModelLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    exclude = output_root if output_root.is_relative_to(input_path) else None
    groups = _iter_folders(input_path, recursive=not args.no_recursive, exclude=exclude)
    if not groups:
        print(f"warning: no supported images found under {input_path}", file=sys.stderr)

    settings = CensorSettings(
        block_size=args.block_size,
        mask_expand=args.mask_expand,
        conf=args.conf,
        iou=args.iou,
        class_ids=class_ids,
        refine=bool(args.refine),
        compress_level=args.png_compress_level,
    )

    started_at = _timestamp()
    rows = _run_batch(
        groups,
        output_root,
        detector=detector,
        refiner=refiner,
        settings=settings,
    )
    payload = _build_report(
        input_path=input_path,
        output_root=output_root,
        recursive=not args.no_recursive,
        started_at=started_at,
        elapsed_sec=round(time.perf_counter() - run_started, 3),
        settings=settings,
        models=models,
        rows=rows,
        folders=[folder for folder, _ in groups],
    )
    return _emit_report(payload, report=args.report, pretty=args.pretty)


if __name__ == "__main__":
    raise SystemExit(main())
