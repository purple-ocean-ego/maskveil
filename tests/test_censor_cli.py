"""モザイク CLI（`censor`）のテスト。推論は実行しない（Detector / Refiner を差し替える）。"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from _toml import tomllib
from PIL import Image

from maskveil.core.labels import SEXUAL_EXPOSED_LABELS
from maskveil.core.mosaic import apply_censor
from maskveil.core.raster import compose_masks
from maskveil.core.types import Detection
from maskveil.tools import censor as censor_module
from maskveil.tools._common import quantize_rgb, save_png
from maskveil.tools.censor import main

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
"""パッケージのルート（サブプロセスの PYTHONPATH に使う）。"""

BLOCK_SIZE = 48
"""CLI の既定ブロックサイズ（PO 決定 2026-10-31）。"""

BBOX = (16.0, 16.0, 48.0, 48.0)
"""テスト用の検出 bbox（合成画像の中央付近）。"""

POLYGON = ((20.0, 20.0), (30.0, 20.0), (30.0, 44.0), (20.0, 44.0))
"""テスト用の輪郭ポリゴン（bbox の左半分のみ）。

`mosaic` はブロック全体を置換するため、bbox 矩形（右半分のブロックにも掛かる）と
結果が変わる形状にしてある。
"""


def _synthetic_image(path, *, width: int = 64, height: int = 64):
    """グラデーション模様の合成画像を PNG で保存し、uint8 配列を返す。"""
    x = np.arange(width, dtype=np.int32)
    y = np.arange(height, dtype=np.int32)
    grid = ((x[None, :] * 3 + y[:, None] * 5) % 256).astype(np.int32)
    rgb = np.stack([grid, (grid * 2) % 256, (grid * 3) % 256], axis=2).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
    return rgb


def _fake_detector_class(detections):
    """固定の検出を返す `Detector` 代替を作る。"""

    class _FakeDetector:
        def __init__(self, model_path, **kwargs) -> None:
            self.input_size = 640

        def detect(self, image, *, class_ids, conf, iou):
            return list(detections)

    return _FakeDetector


def _spy_detector_class(detections, calls: list[dict]):
    """受け取った引数を記録する `Detector` 代替を作る（配線の検証用）。"""

    class _SpyDetector:
        def __init__(self, model_path, **kwargs) -> None:
            self.input_size = 640
            self.model_path = str(model_path)

        def detect(self, image, *, class_ids, conf, iou):
            calls.append(
                {
                    "model": self.model_path,
                    "class_ids": tuple(class_ids),
                    "conf": conf,
                    "iou": iou,
                }
            )
            return list(detections)

    return _SpyDetector


def _failing_detector_class(error: Exception, *, fail_when_height: int):
    """特定サイズの画像でだけ失敗する `Detector` 代替を作る。"""

    class _FailingDetector:
        def __init__(self, model_path, **kwargs) -> None:
            self.input_size = 640

        def detect(self, image, *, class_ids, conf, iou):
            if image.shape[0] == fail_when_height:
                raise error
            return []

    return _FailingDetector


def _fake_refiner_class(candidates):
    """固定の輪郭候補を返す `Refiner` 代替を作る。"""

    class _FakeRefiner:
        def __init__(self, model_path, **kwargs) -> None:
            self.input_size = 1280

        def detect(self, image, *, class_ids, conf, iou):
            return list(candidates)

    return _FakeRefiner


def _prepare(tmp_path, *, refine: bool = False):
    """入力画像とダミーモデルを用意する。"""
    image_path = tmp_path / "sample.png"
    image = _synthetic_image(image_path)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    seg_model = tmp_path / "seg.onnx"
    if refine:
        seg_model.write_bytes(b"dummy-seg")
    return image_path, image, det_model, seg_model


def _run(argv: list[str]) -> int:
    """CLI を実行する（`--report` を付けないので JSON は stdout に出る）。"""
    return main(argv)


def _run_json(argv: list[str], capsys) -> tuple[int, dict]:
    """CLI を実行し、終了コードと stdout のレポートを返す。"""
    exit_code = main(argv)
    return exit_code, json.loads(capsys.readouterr().out)


def _expected_pixels(
    image: np.ndarray,
    detections: list[Detection],
    block_size: int = BLOCK_SIZE,
    mask_expand: int = 0,
) -> np.ndarray:
    """`compose_masks → apply_censor` の期待画素を CLI とは独立に組み立てる。

    量子化は ComfyUI の `SaveImage` と同じ**切り捨て**（`_common.quantize_rgb` を
    あえて使わず、テスト側で再実装して退行を検出できるようにする）。
    """
    height, width = image.shape[:2]
    mask = compose_masks(detections, height, width, expand_px=mask_expand)
    censored = apply_censor(image.astype(np.float32) / 255.0, mask, "mosaic", block_size)
    return np.clip(censored * 255.0, 0.0, 255.0).astype(np.uint8)


# --- AC1: 1枚経路 ---------------------------------------------------------


def test_single_image_matches_expected_pixels(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 画像1枚と検出モデルだけでモザイク済み PNG を作る
    検証意図: detect → compose_masks → apply_censor → 保存 の配線が正しく、
              保存画素が期待値と完全一致する（AC1）
    """
    image_path, image, det_model, _ = _prepare(tmp_path)
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 0
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["mosaicked"] == 1
    assert payload["summary"]["clean"] == 0
    assert payload["summary"]["failed"] == 0

    output_path = output_dir / "0001_sample.png"
    assert output_path.is_file()
    row = payload["images"][0]
    assert row["status"] == "mosaicked"
    assert row["applied"] is True
    assert row["output"] == str(output_path)
    assert row["input"] == str(image_path)
    assert row["bboxes"] == [list(BBOX)]

    actual = np.asarray(Image.open(output_path).convert("RGB"))
    assert np.array_equal(actual, _expected_pixels(image, [detection]))


def test_refine_changes_shape_and_matches_expected_pixels(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: `--refine` で輪郭モデルの形状を使ってモザイクする
    検証意図: 輪郭のポリゴンがマスクに反映され、bbox 矩形だけの結果と異なる（AC1）
    """
    image_path, image, det_model, seg_model = _prepare(tmp_path, refine=True)
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    candidate = Detection(2, "penis", 0.9, BBOX, masks=(POLYGON,), mask_source="model_seg")
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))
    monkeypatch.setattr(censor_module, "Refiner", _fake_refiner_class([candidate]))

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--refine",
            "--refiner-model",
            str(seg_model),
            "--block-size",
            "16",
            "--output-dir",
            str(output_dir),
        ],
        capsys,
    )

    assert exit_code == 0
    assert payload["run"]["params"]["refine"] is True
    assert payload["run"]["params"]["block_size"] == 16
    assert payload["model"]["seg"] is not None

    actual = np.asarray(Image.open(output_dir / "0001_sample.png").convert("RGB"))
    refined = Detection(
        11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX, masks=(POLYGON,), mask_source="model_seg"
    )
    assert np.array_equal(actual, _expected_pixels(image, [refined], 16))
    assert not np.array_equal(actual, _expected_pixels(image, [detection], 16))


# --- AC2: 再帰バッチ・命名・上書き回避 ------------------------------------


def _build_tree(root):
    """テスト用のディレクトリ構造を作る。"""
    layout = {
        "a.png": (16, 16),
        "note.txt": None,
        "sub/b.png": (16, 16),
        "sub/c.png": (16, 16),
        "sub/deep/d.png": (16, 16),
    }
    for relative, size in layout.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if size is None:
            path.write_text("not an image", encoding="utf-8")
        else:
            _synthetic_image(path, width=size[0], height=size[1])


def _detector_off(monkeypatch) -> None:
    """検出0件の Detector に差し替える（命名・構造の検証に集中するため）。"""
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([]))


def test_recursive_numbering_per_folder_and_structure(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 入れ子のあるディレクトリをまとめてモザイクする
    検証意図: 連番はフォルダ毎に 0001 起点、出力は相対パス構造を保持し、
              画像でないファイルは無視する（AC2）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _build_tree(image_dir)
    (image_dir / "empty").mkdir()  # 空ディレクトリは処理対象にしない
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 0
    assert sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*.png")) == [
        "0001_a.png",
        "sub/0001_b.png",
        "sub/0002_c.png",
        "sub/deep/0001_d.png",
    ]
    assert payload["summary"]["total"] == 4
    assert {row["input"] for row in payload["images"]} == {
        str(image_dir / "a.png"),
        str(image_dir / "sub/b.png"),
        str(image_dir / "sub/c.png"),
        str(image_dir / "sub/deep/d.png"),
    }
    assert [(row["folder"], row["sequence"]) for row in payload["images"]] == [
        (".", 1),
        ("sub", 1),
        ("sub", 2),
        ("sub/deep", 1),
    ]
    assert [(folder["folder"], folder["total"]) for folder in payload["folders"]] == [
        (".", 1),
        ("sub", 2),
        ("sub/deep", 1),
    ]


def test_no_recursive_skips_subdirectories(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 直下の画像だけを処理したい
    検証意図: `--no-recursive` でサブディレクトリを対象にしない（AC2）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _build_tree(image_dir)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    exit_code, payload = _run_json(
        [
            str(image_dir),
            "--det-model",
            str(det_model),
            "--no-recursive",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capsys,
    )

    assert exit_code == 0
    assert [row["input"] for row in payload["images"]] == [str(image_dir / "a.png")]
    assert payload["summary"]["total"] == 1
    assert payload["run"]["recursive"] is False


def test_existing_output_is_recorded_as_failure(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 同じ出力先へ再実行する
    検証意図: 既存ファイルを上書きせず、その画像を失敗として記録して継続する（AC2）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "0001_sample.png"
    existing.write_bytes(b"keep-me")

    exit_code, payload = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 1
    assert payload["images"][0]["status"] == "failed"
    assert "already exists" in payload["images"][0]["error"]
    assert payload["images"][0]["applied"] is False
    assert existing.read_bytes() == b"keep-me"


def test_existing_output_in_directory_continues_other_images(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 一部の出力だけが既に存在する
    検証意図: 既存分だけ failed にし、残りの画像は保存して継続する（AC2）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=16, height=16)
    _synthetic_image(image_dir / "b.png", width=16, height=16)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "0001_a.png").write_bytes(b"occupied")

    exit_code, payload = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 1
    rows = {row["input"].rsplit("/", 1)[-1]: row for row in payload["images"]}
    assert rows["a.png"]["status"] == "failed"
    assert rows["b.png"]["status"] == "clean"
    assert (output_dir / "0002_b.png").is_file()


def test_default_output_dir_for_directory_is_sibling(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: `--output-dir` を省略してディレクトリを処理する
    検証意図: 既定は `<入力名>_censored/` に作られる（AC2）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=16, height=16)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    exit_code, payload = _run_json(
        [str(image_dir), "--det-model", str(det_model)],
        capsys,
    )

    assert exit_code == 0
    assert (tmp_path / "in_censored" / "0001_a.png").is_file()
    assert payload["run"]["output_dir"] == str(tmp_path / "in_censored")


def test_default_output_dir_for_file_is_parent(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: `--output-dir` を省略して画像1枚を処理する
    検証意図: 既定は入力の親ディレクトリ（AC2）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    exit_code, payload = _run_json(
        [str(image_path), "--det-model", str(det_model)],
        capsys,
    )

    assert exit_code == 0
    assert (tmp_path / "0001_sample.png").is_file()
    assert payload["run"]["output_dir"] == str(tmp_path)


def test_output_dir_inside_input_is_excluded_from_scan(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 出力先を入力ディレクトリの内側に置いて再実行する
    検証意図: 前回の出力を入力として拾わず、枚数が増えない（再実行の冪等性）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=16, height=16)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)
    output_dir = image_dir / "censored"

    first_code, first = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )
    second_code, second = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert first_code == 0
    assert second_code == 1  # 既存出力との衝突は失敗として記録する
    assert first["summary"]["total"] == 1
    assert second["summary"]["total"] == 1
    assert not (output_dir / "censored").exists()


def test_output_dir_equal_to_input_is_rejected(tmp_path) -> None:
    """
    ユースケース: 出力先に入力ディレクトリそのものを指定する
    検証意図: 入力を上書きする構成を引数エラーとして拒否する（exit 2）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=16, height=16)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")

    with pytest.raises(SystemExit) as excinfo:
        _run([str(image_dir), "--det-model", str(det_model), "--output-dir", str(image_dir)])
    assert excinfo.value.code == 2


# --- AC3: 失敗継続とサマリ ------------------------------------------------


def test_broken_image_continues_and_summary_counts(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 壊れた画像が1枚混ざったフォルダをまとめて処理する
    検証意図: 失敗した画像だけを failed に計上し、残りは処理を継続して exit 1（AC3）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=32, height=32)
    (image_dir / "b_broken.png").write_bytes(b"not a png")
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 1
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["mosaicked"] == 1
    assert payload["summary"]["clean"] == 0
    assert payload["summary"]["failed"] == 1
    assert payload["summary"]["elapsed_sec"] >= 0.0

    rows = {row["input"].rsplit("/", 1)[-1]: row for row in payload["images"]}
    assert rows["a.png"]["status"] == "mosaicked"
    assert rows["b_broken.png"]["status"] == "failed"
    assert "failed to read image" not in rows["b_broken.png"]["error"]  # 例外型を記録する
    assert rows["b_broken.png"]["output"] is None
    assert (output_dir / "0001_a.png").is_file()
    assert not (output_dir / "0002_b_broken.png").exists()


def test_detector_exception_is_recorded_and_batch_continues(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 特定の画像だけ推論が失敗する
    検証意図: 当該画像のみ failed にし、残りは処理を継続して exit 1（AC3）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=32, height=32)
    _synthetic_image(image_dir / "b.png", width=64, height=64)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    monkeypatch.setattr(
        censor_module,
        "Detector",
        _failing_detector_class(RuntimeError("onnx boom"), fail_when_height=32),
    )

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 1
    rows = {row["input"].rsplit("/", 1)[-1]: row for row in payload["images"]}
    assert rows["a.png"]["status"] == "failed"
    assert "onnx boom" in rows["a.png"]["error"]
    assert rows["b.png"]["status"] == "clean"
    assert (output_dir / "0002_b.png").is_file()


def test_output_directory_failure_is_recorded(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 出力ディレクトリが作れない場所を指定する
    検証意図: 例外で落ちず、レポートを出して該当画像を failed に計上する（AC3）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not a directory")

    exit_code, payload = _run_json(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--output-dir",
            str(blocker / "sub"),
        ],
        capsys,
    )

    assert exit_code == 1
    assert payload["summary"]["failed"] == 1
    assert "cannot create output directory" in payload["images"][0]["error"]


def test_all_success_exits_zero_with_summary(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 全件成功したときのサマリを確認する
    検証意図: 成功時は exit 0、サマリの内訳が一致する（AC3）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=32, height=32)
    _synthetic_image(image_dir / "b.png", width=32, height=32)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    exit_code, payload = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(tmp_path / "out")],
        capsys,
    )

    assert exit_code == 0
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["clean"] == 2
    assert payload["summary"]["failed"] == 0


# --- AC4: 終了コード規約 --------------------------------------------------


def test_exit_code_is_zero_when_all_succeed(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 終了コード規約の 0（全件成功）を確認する
    検証意図: 全件が clean/mosaicked なら 0 を返す（AC4）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    exit_code, _ = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(tmp_path / "out")],
        capsys,
    )
    assert exit_code == 0


def test_exit_code_is_one_when_any_fails(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 終了コード規約の 1（失敗あり）を確認する
    検証意図: 1件でも失敗すれば 1 を返す（AC4）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "0001_sample.png").write_bytes(b"occupied")
    _detector_off(monkeypatch)

    exit_code, _ = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )
    assert exit_code == 1


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"drop_input": True}, "INPUT なし"),
        ({"det_model": "relative.onnx"}, "相対パスの検出モデル"),
        ({"seg_model": "seg.onnx"}, "相対パスの輪郭モデル"),
        ({"extra": ["--block-size", "1"]}, "block_size 下限外"),
        ({"extra": ["--block-size", "129"]}, "block_size 上限外"),
        ({"extra": ["--mask-expand", "-1"]}, "mask_expand 下限外"),
        ({"extra": ["--mask-expand", "257"]}, "mask_expand 上限外"),
        ({"extra": ["--conf", "2.0"]}, "conf 範囲外"),
        ({"extra": ["--iou", "0"]}, "iou 範囲外"),
        ({"extra": ["--refine"]}, "--refine で --refiner-model なし"),
        ({"extra": ["--refiner-model"]}, "--refiner-model だけ指定"),
        ({"extra": ["--classes", "NOT_A_CLASS"]}, "未知のクラス"),
        ({"input": "missing.png"}, "入力が存在しない"),
        ({"input": "note.txt"}, "未対応の拡張子"),
    ],
)
def test_argument_errors_exit_with_2(tmp_path, overrides, reason) -> None:
    """
    ユースケース: 引数が規約に反した状態で実行する
    検証意図: 引数エラーは必ず 2 を返し、処理を開始しない（AC4）
    """
    image_path, _, det_model, seg_model = _prepare(tmp_path, refine=True)
    (tmp_path / "note.txt").write_text("not an image", encoding="utf-8")

    argv = [] if overrides.get("drop_input") else [str(image_path)]
    if "input" in overrides:
        argv = [str(tmp_path / overrides["input"])]
    argv += ["--det-model", str(overrides.get("det_model", det_model))]
    if "seg_model" in overrides:
        argv += ["--refine", "--refiner-model", str(overrides["seg_model"])]
    if overrides.get("extra") == ["--refiner-model"]:
        argv += ["--refiner-model", str(seg_model)]
    else:
        argv += overrides.get("extra", [])

    with pytest.raises(SystemExit) as excinfo:
        _run(argv)
    assert excinfo.value.code == 2, reason


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["input.png"],
    ],
)
def test_required_arguments_exit_with_2(argv: list[str]) -> None:
    """
    ユースケース: 必須引数が欠けた状態で実行する
    検証意図: 引数エラーは終了コード 2（AC4）
    """
    with pytest.raises(SystemExit) as excinfo:
        _run(argv)
    assert excinfo.value.code == 2


# --- AC5: レポート JSON ---------------------------------------------------


def test_report_schema_keys(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: レポート JSON を後段のツール（比較・zip 化）が読む
    検証意図: レポートの必須キーが揃い、型と意味が一致する（AC5）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 0
    assert set(payload) == {
        "schema_version",
        "tool",
        "tool_version",
        "run",
        "model",
        "folders",
        "images",
        "summary",
    }
    assert payload["schema_version"] == 1
    assert payload["tool"] == "censor"
    assert payload["tool_version"] == censor_module.TOOL_VERSION

    run = payload["run"]
    assert set(run) == {
        "input",
        "output_dir",
        "recursive",
        "started_at",
        "finished_at",
        "elapsed_sec",
        "params",
    }
    assert run["input"] == str(image_path)
    assert run["output_dir"] == str(output_dir)
    assert run["recursive"] is True
    assert run["started_at"].endswith("Z")
    assert run["finished_at"].endswith("Z")
    assert run["elapsed_sec"] >= 0.0
    assert run["params"] == {
        "mode": "mosaic",
        "block_size": BLOCK_SIZE,
        "mask_expand": 0,
        "conf": 0.2,
        "iou": 0.45,
        "classes": [3, 11],
        "refine": False,
        "png_compress_level": 1,
    }

    assert set(payload["model"]["det"]) == {"path", "sha256", "input_size"}
    assert payload["model"]["seg"] is None

    row = payload["images"][0]
    assert set(row) == {
        "index",
        "folder",
        "sequence",
        "input",
        "output",
        "status",
        "applied",
        "detections",
        "bboxes",
        "elapsed_sec",
        "error",
    }
    assert row["index"] == 1
    assert row["detections"] == 1
    assert row["bboxes"] == [list(BBOX)]
    assert row["elapsed_sec"] >= 0.0
    assert row["error"] is None

    assert payload["folders"] == [
        {"folder": ".", "total": 1, "mosaicked": 1, "clean": 0, "failed": 0}
    ]
    assert set(payload["summary"]) == {
        "total",
        "mosaicked",
        "clean",
        "failed",
        "elapsed_sec",
    }


def test_model_metadata_matches_actual_files(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: レポートから「どのモデルで処理したか」を後から特定する
    検証意図: det/seg の path・sha256・input_size が実ファイルと一致する（AC5）
    """
    image_path, _, det_model, seg_model = _prepare(tmp_path, refine=True)
    candidate = Detection(2, "penis", 0.9, BBOX, masks=(POLYGON,), mask_source="model_seg")
    monkeypatch.setattr(
        censor_module,
        "Detector",
        _fake_detector_class([Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)]),
    )
    monkeypatch.setattr(censor_module, "Refiner", _fake_refiner_class([candidate]))

    _, payload = _run_json(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--refine",
            "--refiner-model",
            str(seg_model),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capsys,
    )

    assert payload["model"]["det"] == {
        "path": str(det_model),
        "sha256": hashlib.sha256(det_model.read_bytes()).hexdigest(),
        "input_size": 640,
    }
    assert payload["model"]["seg"] == {
        "path": str(seg_model),
        "sha256": hashlib.sha256(seg_model.read_bytes()).hexdigest(),
        "input_size": 1280,
    }


def test_report_option_writes_file_and_prints_summary(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: `--report` でレポートをファイルに残す
    検証意図: ファイルに全文が入り、stdout はサマリ1行だけになる（AC5）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    report_path = tmp_path / "report.json"
    exit_code = _run(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert report_path.is_file()
    summary_line = captured.out.strip()
    assert "\n" not in summary_line
    assert "total=1" in summary_line
    assert "failed=0" in summary_line

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 1
    assert payload["images"][0]["status"] == "clean"


def test_report_writes_into_missing_directory(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: レポートの保存先ディレクトリがまだ無い
    検証意図: 親ディレクトリを作って書き出す（AC5）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    report_path = tmp_path / "nested" / "deep" / "report.json"
    exit_code = _run(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(report_path),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert report_path.is_file()


def test_report_write_failure_exits_1(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: レポートの保存先が書けない
    検証意図: stderr に理由を出して exit 1（画像自体は保存済み）（AC5）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not a directory")

    exit_code = _run(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(blocker / "report.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to write" in captured.err
    assert (tmp_path / "out" / "0001_sample.png").is_file()


def test_pretty_report_is_indented(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 人が読むために整形したレポートが欲しい
    検証意図: `--pretty` で複数行の JSON になる（AC5）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    exit_code = _run(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--output-dir",
            str(tmp_path / "out"),
            "--pretty",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert out.count("\n") > 10
    assert json.loads(out)["summary"]["total"] == 1


def test_pretty_with_report_writes_indented_file(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: `--report` と `--pretty` を併用する
    検証意図: ファイル側が整形される（AC5）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)
    report_path = tmp_path / "report.json"

    exit_code = _run(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(report_path),
            "--pretty",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert report_path.read_text(encoding="utf-8").count("\n") > 10


# --- AC6: PNG 圧縮レベル --------------------------------------------------


def test_png_compress_level_default_is_one(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 圧縮レベルを指定せずに実行する
    検証意図: 既定 1 がレポートに記録され、保存にもその値が渡る（AC6）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)
    recorded: list[int] = []
    original = censor_module.save_png

    def _spy(image, path, *, compress_level=1):
        recorded.append(compress_level)
        return original(image, path, compress_level=compress_level)

    monkeypatch.setattr(censor_module, "save_png", _spy)

    exit_code, payload = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(tmp_path / "out")],
        capsys,
    )

    assert exit_code == 0
    assert payload["run"]["params"]["png_compress_level"] == 1
    assert recorded == [1]


@pytest.mark.parametrize("level", [0, 9])
def test_png_compress_level_is_accepted_and_keeps_pixels(
    tmp_path, monkeypatch, capsys, level: int
) -> None:
    """
    ユースケース: 圧縮レベルを明示指定する
    検証意図: 0 と 9 を受理し、画素は指定に影響されない（AC6）
    """
    image_path, image, det_model, _ = _prepare(tmp_path)
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    output_dir = tmp_path / f"out{level}"
    exit_code, payload = _run_json(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--png-compress-level",
            str(level),
            "--output-dir",
            str(output_dir),
        ],
        capsys,
    )

    assert exit_code == 0
    assert payload["run"]["params"]["png_compress_level"] == level
    actual = np.asarray(Image.open(output_dir / "0001_sample.png").convert("RGB"))
    assert np.array_equal(actual, _expected_pixels(image, [detection]))


@pytest.mark.parametrize("level", [-1, 10])
def test_png_compress_level_out_of_range_exits_2(tmp_path, level: int) -> None:
    """
    ユースケース: 圧縮レベルに範囲外の値を渡す
    検証意図: 引数エラーとして 2 で終了する（AC6）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        _run(
            [
                str(image_path),
                "--det-model",
                str(det_model),
                "--png-compress-level",
                str(level),
            ]
        )
    assert excinfo.value.code == 2


# --- AC7: ブロックサイズ --------------------------------------------------


def test_block_size_default_is_48_and_drives_pixels(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: ブロックサイズを指定せずに実行する
    検証意図: 既定 48 が使われ（core 定数と一致）、画素も 48 のブロックで処理される（AC7）
    """
    image_path, image, det_model, _ = _prepare(tmp_path)
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    assert censor_module.DEFAULT_BLOCK_SIZE == 48
    assert censor_module.build_parser().get_default("block_size") == 48

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 0
    assert payload["run"]["params"]["block_size"] == 48

    actual = np.asarray(Image.open(output_dir / "0001_sample.png").convert("RGB"))
    assert np.array_equal(actual, _expected_pixels(image, [detection], 48))
    assert not np.array_equal(actual, _expected_pixels(image, [detection], 8))


@pytest.mark.parametrize("block_size", [2, 128])
def test_block_size_bounds_are_accepted(tmp_path, monkeypatch, capsys, block_size: int) -> None:
    """
    ユースケース: ブロックサイズの境界値を指定する
    検証意図: 2 と 128 を受理し、その値で画素が決まる（AC7）
    """
    image_path, image, det_model, _ = _prepare(tmp_path)
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    output_dir = tmp_path / f"out{block_size}"
    exit_code, payload = _run_json(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--block-size",
            str(block_size),
            "--output-dir",
            str(output_dir),
        ],
        capsys,
    )

    assert exit_code == 0
    assert payload["run"]["params"]["block_size"] == block_size
    actual = np.asarray(Image.open(output_dir / "0001_sample.png").convert("RGB"))
    assert np.array_equal(actual, _expected_pixels(image, [detection], block_size))


# --- AC8: nodes / torch 非依存 --------------------------------------------


_BLOCKER_SCRIPT = """
import sys

BLOCKED = {"torch", "comfy", "folder_paths", "cv2", "ultralytics"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("blocked: " + name, name=name.split(".")[0])
        return None


sys.meta_path.insert(0, Blocker())

import maskveil.tools.censor as censor

assert hasattr(censor, "main")
assert censor.build_parser().get_default("block_size") == 48
assert censor.build_parser().get_default("mask_expand") == 0
print("ok")
"""
"""torch / ComfyUI をブロックした環境で CLI モジュールを import するスクリプト（AC8）。"""


def test_censor_module_imports_without_torch() -> None:
    """
    ユースケース: ComfyUI も torch も無い環境で CLI を動かす
    検証意図: `tools.censor` が torch 不在でも import・既定値参照できる（AC8）
    """
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCKER_SCRIPT],
        cwd=PACKAGE.parent,
        env={"PYTHONPATH": str(PACKAGE.parent), "PATH": "/usr/local/bin:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_censor_does_not_import_nodes() -> None:
    """
    ユースケース: CLI が ComfyUI ノード層へ依存しないことを保証する
    検証意図: `maskveil.nodes` への import がソースに存在しない（AC8）
    """
    path = PACKAGE / "tools" / "censor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("maskveil.nodes"):
                offenders.append(node.module)
        elif isinstance(node, ast.Import):
            offenders.extend(
                alias.name for alias in node.names if alias.name.startswith("maskveil.nodes")
            )
    assert offenders == []


# --- AC9: 検出0件・出力形式 ------------------------------------------------


def test_no_detections_saves_unmodified_copy(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 検出が0件の画像を渡す
    検証意図: 無加工で保存し、status を clean・applied を false にして入力を保持する（AC9）
    """
    image_path, image, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 0
    row = payload["images"][0]
    assert row["status"] == "clean"
    assert row["applied"] is False
    assert row["detections"] == 0
    assert row["bboxes"] == []

    actual = np.asarray(Image.open(output_dir / "0001_sample.png").convert("RGB"))
    assert np.array_equal(actual, image)


def test_output_is_always_png_for_jpeg_input(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: JPEG を入力する
    検証意図: 出力は常に PNG（拡張子も中身も）で、入力画素が保たれる（AC9）
    """
    image = _synthetic_image(tmp_path / "source.png", width=32, height=32)
    jpeg_path = tmp_path / "sample.jpg"
    Image.fromarray(image, mode="RGB").save(jpeg_path, quality=100)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(jpeg_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 0
    output_path = output_dir / "0001_sample.png"
    assert output_path.is_file()
    assert payload["images"][0]["output"] == str(output_path)

    with Image.open(jpeg_path) as source:
        decoded = np.asarray(source.convert("RGB"))
    with Image.open(output_path) as saved:
        assert saved.format == "PNG"
        assert np.array_equal(np.asarray(saved.convert("RGB")), decoded)


@pytest.mark.parametrize("name", ["UPPER.PNG", "photo.webp"])
def test_extension_matching_is_case_insensitive(tmp_path, monkeypatch, capsys, name: str) -> None:
    """
    ユースケース: 大文字拡張子や webp を入力する
    検証意図: 拡張子の大小を無視し、webp も PNG へ変換して処理する（CLI 仕様）
    """
    image = _synthetic_image(tmp_path / "src.png", width=16, height=16)
    source_path = tmp_path / name
    Image.fromarray(image, mode="RGB").save(source_path)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    output_dir = tmp_path / "out"
    exit_code, payload = _run_json(
        [str(source_path), "--det-model", str(det_model), "--output-dir", str(output_dir)],
        capsys,
    )

    assert exit_code == 0
    assert payload["summary"]["total"] == 1
    assert (output_dir / f"0001_{source_path.stem}.png").is_file()


# --- AC11: マスク膨張 -----------------------------------------------------


@pytest.mark.parametrize("expand", [-1, 257])
def test_mask_expand_out_of_range_exits_2(tmp_path, expand: int) -> None:
    """
    ユースケース: 膨張量に範囲外の値を渡す
    検証意図: 0–256 の外は引数エラーとして 2 で終了する（AC11）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        _run([str(image_path), "--det-model", str(det_model), "--mask-expand", str(expand)])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("expand", [0, 256])
def test_mask_expand_bounds_are_accepted(tmp_path, monkeypatch, capsys, expand: int) -> None:
    """
    ユースケース: 膨張量の境界値を指定する
    検証意図: 0 と 256 を受理し、レポートに反映される（AC11）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    exit_code, payload = _run_json(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--mask-expand",
            str(expand),
            "--output-dir",
            str(tmp_path / f"out{expand}"),
        ],
        capsys,
    )

    assert exit_code == 0
    assert payload["run"]["params"]["mask_expand"] == expand


def test_mask_expand_widens_the_censored_area(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: マスクを膨張させてモザイクを広げる
    検証意図: 膨張量がマスク合成に伝播し、変更画素が単調に増える（AC11・AC1）
    """
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    image_path, image, det_model, _ = _prepare(tmp_path)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    changes = {}
    for expand in (0, 8):
        output_dir = tmp_path / f"out{expand}"
        exit_code, payload = _run_json(
            [
                str(image_path),
                "--det-model",
                str(det_model),
                "--mask-expand",
                str(expand),
                "--output-dir",
                str(output_dir),
            ],
            capsys,
        )
        assert exit_code == 0
        assert payload["run"]["params"]["mask_expand"] == expand
        actual = np.asarray(Image.open(output_dir / "0001_sample.png").convert("RGB"))
        assert np.array_equal(actual, _expected_pixels(image, [detection], mask_expand=expand))
        changes[expand] = int((actual != image).any(axis=2).sum())

    assert changes[8] > changes[0]


# --- 引数の配線・補助関数 -------------------------------------------------


def test_detector_receives_parsed_arguments(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: クラス・閾値を指定して実行する
    検証意図: 解析結果がそのまま Detector へ渡る（取り違えの検出）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    calls: list[dict] = []
    monkeypatch.setattr(censor_module, "Detector", _spy_detector_class([], calls))

    _run_json(
        [
            str(image_path),
            "--det-model",
            str(det_model),
            "--classes",
            "MALE_GENITALIA_EXPOSED",
            "--conf",
            "0.3",
            "--iou",
            "0.5",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capsys,
    )

    assert calls == [
        {
            "model": str(det_model),
            "class_ids": (11,),
            "conf": 0.3,
            "iou": 0.5,
        }
    ]


def test_model_load_failure_returns_1(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: モデルファイルは存在するが壊れている
    検証意図: 引数エラー（2）ではなく実行時エラー（1）として扱い、traceback を出さない
    """
    image_path, _, det_model, _ = _prepare(tmp_path)

    class _BrokenDetector:
        def __init__(self, model_path, **kwargs) -> None:
            raise RuntimeError("onnx boom")

    monkeypatch.setattr(censor_module, "Detector", _BrokenDetector)

    exit_code = main(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(tmp_path / "out")]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "error: failed to load detector" in captured.err
    assert "Traceback" not in captured.err


def test_dot_input_uses_sibling_output(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 画像ディレクトリ内で `censor .` を実行する
    検証意図: カレントディレクトリを入力にしても落ちず、兄弟の `<dir>_censored` へ出力する（R1）
    """
    work = tmp_path / "work"
    work.mkdir()
    _synthetic_image(work / "a.png", width=16, height=16)
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)
    monkeypatch.chdir(work)

    exit_code, payload = _run_json([".", "--det-model", str(det_model)], capsys)

    assert exit_code == 0
    assert payload["run"]["output_dir"] == str(tmp_path / "work_censored")
    assert (tmp_path / "work_censored" / "0001_a.png").is_file()


def test_help_lists_contract_options(capsys) -> None:
    """
    ユースケース: `-h` でオプションを確認する
    検証意図: 主要オプションがヘルプに現れる（終了コード 0）
    """
    with pytest.raises(SystemExit) as excinfo:
        _run(["--help"])
    assert excinfo.value.code == 0

    out = capsys.readouterr().out
    for option in (
        "--det-model",
        "--refine",
        "--refiner-model",
        "--classes",
        "--conf",
        "--iou",
        "--block-size",
        "--mask-expand",
        "--png-compress-level",
        "--output-dir",
        "--no-recursive",
        "--report",
        "--pretty",
    ):
        assert option in out


def test_empty_directory_yields_empty_report(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 画像が1枚も無いディレクトリを処理する
    検証意図: total=0・folders=[] で exit 0、stderr に警告を出す
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    _detector_off(monkeypatch)

    exit_code = main(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(tmp_path / "out")]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["summary"]["total"] == 0
    assert payload["folders"] == []
    assert "no supported images" in captured.err


def test_folders_breakdown_counts_mixed_status(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: zip 化の可否をフォルダ単位で判定する
    検証意図: folders[] に mosaicked/clean/failed の内訳が入る（AC5）
    """
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    _synthetic_image(image_dir / "a.png", width=64, height=64)
    _synthetic_image(image_dir / "b.png", width=32, height=32)
    (image_dir / "c_broken.png").write_bytes(b"not a png")
    det_model = tmp_path / "det.onnx"
    det_model.write_bytes(b"dummy-det")
    detection = Detection(11, "MALE_GENITALIA_EXPOSED", 0.9, BBOX)
    monkeypatch.setattr(censor_module, "Detector", _fake_detector_class([detection]))

    exit_code, payload = _run_json(
        [str(image_dir), "--det-model", str(det_model), "--output-dir", str(tmp_path / "out")],
        capsys,
    )

    assert exit_code == 1
    assert payload["folders"] == [
        {"folder": ".", "total": 3, "mosaicked": 2, "clean": 0, "failed": 1}
    ]


def test_progress_is_reported_on_stderr(tmp_path, monkeypatch, capsys) -> None:
    """
    ユースケース: 大量バッチの進捗を確認する
    検証意図: stdin/stdout の JSON を汚さず、stderr に進捗行を出す（CLI 仕様）
    """
    image_path, _, det_model, _ = _prepare(tmp_path)
    _detector_off(monkeypatch)

    exit_code = main(
        [str(image_path), "--det-model", str(det_model), "--output-dir", str(tmp_path / "out")]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    json.loads(captured.out)
    assert "[folder 1/1]" in captured.err
    assert "sample.png → 0001_sample.png" in captured.err
    assert "clean" in captured.err


def test_save_png_rejects_bad_shape(tmp_path) -> None:
    """
    ユースケース: 形状の壊れた配列を保存しようとする
    検証意図: `ValueError` を投げ、ファイルを残さない（安全網）
    """
    with pytest.raises(ValueError):
        save_png(np.zeros((4, 4), dtype=np.float32), tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()


def test_save_png_removes_temporary_file_on_failure(tmp_path) -> None:
    """
    ユースケース: 保存先にディレクトリを指定して失敗させる
    検証意図: 失敗時に `.tmp` を残さない（原子的書き込みの後始末）
    """
    target = tmp_path / "dir.png"
    target.mkdir()

    with pytest.raises(OSError):
        save_png(np.zeros((4, 4, 3), dtype=np.float32), target)
    assert not (tmp_path / "dir.png.tmp").exists()


def test_quantize_rgb_uses_truncation_like_save_image() -> None:
    """
    ユースケース: ComfyUI の `SaveImage` と同じ画素結果にする
    検証意図: 量子化は四捨五入ではなく切り捨て（WF と画素完全一致させるため、R6 案B）
    """
    image = np.array([[[0.5, 0.9999, 1.0]]], dtype=np.float32)
    assert quantize_rgb(image).tolist() == [[[127, 254, 255]]]


def test_default_classes_match_core_labels() -> None:
    """
    ユースケース: 既定クラスがコアの定義とずれていないことを保証する
    検証意図: CLI 既定が `SEXUAL_EXPOSED_LABELS` と同じ2クラス（AC の再発防止）
    """
    assert ",".join(SEXUAL_EXPOSED_LABELS) == censor_module.DEFAULT_CLASSES


def test_tool_version_matches_pyproject() -> None:
    """
    ユースケース: レポートから CLI 版を特定する
    検証意図: `TOOL_VERSION` が `pyproject.toml` の version と一致する
    """
    with (PACKAGE / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    assert version == censor_module.TOOL_VERSION


def _workflows_dir() -> pathlib.Path | None:
    """既定ワークフローの置場を解決する（無ければ None）。

    `MASKVEIL_WORKFLOWS_DIR` を最優先する（位置を推測しない）。
    未指定の場合は、パッケージ直下 → 1つ上（vendoring された場合のリポジトリ直下）を探す。
    """
    override = os.environ.get("MASKVEIL_WORKFLOWS_DIR")
    if override:
        path = pathlib.Path(override)
        return path if path.is_dir() else None
    candidates = [PACKAGE / "workflows"]
    if len(PACKAGE.parents) > 1:
        candidates.append(PACKAGE.parents[1] / "workflows")
    return next((path for path in candidates if path.is_dir()), None)


def test_workflows_use_core_defaults() -> None:
    """
    ユースケース: 既定WFと core 定数がずれていないことを保証する
    検証意図: 同梱WFの `block_size` / `mask_expand` が core 定数と一致する
    """
    workflows = _workflows_dir()
    if workflows is None:
        pytest.skip(
            "workflows/ を持たないリポジトリ（公開パッケージ単体）では検証しない。"
            " 検証する場合は MASKVEIL_WORKFLOWS_DIR で場所を指定する"
        )

    for name in ("censor-nsfw.json", "censor-nsfw-refiner.json"):
        payload = json.loads((workflows / name).read_text(encoding="utf-8"))
        nodes = {node["type"]: node for node in payload["nodes"]}
        assert nodes["MosaicByMask"]["widgets_values"] == [
            "mosaic",
            censor_module.DEFAULT_BLOCK_SIZE,
        ]
        assert nodes["HotscreenDetector"]["widgets_values"][5] == (
            censor_module.DEFAULT_MASK_EXPAND
        )
        if "HotscreenRefiner" in nodes:
            assert nodes["HotscreenRefiner"]["widgets_values"][5] == (
                censor_module.DEFAULT_MASK_EXPAND
            )
