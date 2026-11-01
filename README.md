# maskveil

NSFW 部位の**検出**（hotscreen YOLO11）と、**形状追従マスクの生成**（YOLO11-Seg）を行う
ComfyUI カスタムノード用パッケージ。実行時依存は `numpy` と `onnxruntime`（CLI は加えて `Pillow`）。
`ultralytics` / `torch` は不要。

| 層 | 役割 | 実行時依存 | ComfyUI 依存 |
| --- | --- | --- | --- |
| `core/` | 推論・前後処理・座標復元・輪郭抽出（純粋ロジック） | numpy / onnxruntime | **なし** |
| `tools/` | CLI（`detect` / `refine` / `censor`）。モデルのパス検証を担当 | 上記 + Pillow | なし |
| `nodes/` | ComfyUI ノード（`folder_paths` によるモデル解決） | 上記 + torch | あり |

- **ノードの入出力契約と既定値は [SPEC.md](./SPEC.md) が正**。CLI・JSON スキーマ・性能の実測値は本 README を正とする。

## 段構成

```
detect : IMAGE + 検出モデル       → 検出JSON（クラス名・スコア・原画像座標bbox）      schema_version 1
refine : 検出JSON + 輪郭モデル    → マスク付きJSON（形状ポリゴン・mask_source）        schema_version 2
censor : 画像/ディレクトリ + 検出モデル（+任意で輪郭）
                                  → モザイク済みPNG + レポートJSON（1コマンド完結）
```

- `detect` は高速（約 0.5 秒/枚・CLI 全体）。`refine` は低速（約 4.8 秒/枚 @ CPU, 1280px）だが任意。
- `detect` に輪郭を内包する便宜フラグは**設けない**（1コマンド1モデル）。

## モデル（フルパス必須）

CLI は**絶対パスのみ**を受け取る。環境変数・既定値・ディレクトリ探索は行わない。

| 用途 | 既定の配置先（例） | SHA-256 |
| --- | --- | --- |
| 検出 | `~/comfyui-local/models/hotscreen/det/hs-real-anime-y11n-640-fp32.onnx` | `fecf08b62cf091df6b46222e163ac1721570d5f0a9cffa93591e5bb46f61bccf` |
| 輪郭 | `~/comfyui-local/models/hotscreen/seg/nsfw-anime-xl-x1280.onnx` | `7d193296684e13d9998f50bdf4bb4696ff826785b6d22f5260313e353e5cae1f` |

ComfyUI ノードからは `folder_paths` のカテゴリ `hotscreen` を選択して解決する。
カテゴリは `extra_model_paths_local.yaml` の `hotscreen: hotscreen` で
`/workspace/models/hotscreen` に割り当てている。

## 使い方

```bash
# リポジトリ直下から（パッケージの親を PYTHONPATH に通す）
export PYTHONPATH=custom_nodes_src

# 1) 検出（高速）
python -m maskveil.tools.detect input.png \
  --classes MALE_GENITALIA_EXPOSED \
  --model "$HOME/comfyui-local/models/hotscreen/det/hs-real-anime-y11n-640-fp32.onnx" \
  > det.json

# 2) 輪郭化（低速・任意）
python -m maskveil.tools.refine < det.json \
  --refiner-model "$HOME/comfyui-local/models/hotscreen/seg/nsfw-anime-xl-x1280.onnx" \
  > refined.json
```

### `detect` の主なオプション

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--classes` | **必須** | 名前 / ID / `all` のカンマ区切り。未指定は `exit 2` |
| `--conf` | 0.2 | スコア閾値（モデル作者の推奨値） |
| `--iou` | 0.45 | class-aware NMS の IoU 閾値 |
| `--input-size` | モデルから取得 | モデル入力の一辺。指定するとモデルと一致するか検証する |
| `--top-k` | 0 | スコア上位N件に絞る（0 は無効） |
| `--list-classes` | - | 16クラスを `id<TAB>name` で表示して終了 |

### `refine` の主なオプション

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--refiner-model` | （必須） | `--mask-source model_seg` のとき必須。絶対パス |
| `--refiner-input-size` | モデルから取得 | 輪郭モデルの入力一辺。指定時はモデルと一致するか検証 |
| `--classes` | 入力JSONの全検出 | 輪郭化する検出クラスを限定 |
| `--mask-source` | `model_seg` | `bbox_ellipse` は**デバッグ用**（内接楕円） |
| `--seg-crop` | false | 検出 bbox 周辺をクロップして推論（計測用。1検出ごとに推論するため遅い） |

- 終了コード: `0` 成功（0件含む） / `1` 実行時エラー（入力JSON不正・画像読込失敗など） / `2` 引数エラー。
- 輪郭モデルは**1画像1回**だけ実行し、対応する全クラスのマスクを一度に取得する（クラスごとに再実行しない）。

### `censor`（モザイクまで1コマンド）

`detect →（--refine 時のみ）refine → モザイク → PNG 保存` を1コマンドで行う。
**ComfyUI・WebUI・コンテナを起動しない**（`tools/` は torch 非依存）。`detect` / `refine` が
「1コマンド1モデル」なのに対し、`censor` は複合ドライバとして例外的に2モデルを扱う。

```bash
export PYTHONPATH=custom_nodes_src

# ディレクトリを再帰処理（既定）
python -m maskveil.tools.censor "$HOME/comfyui-local/output" \
  --det-model "$HOME/comfyui-local/models/hotscreen/det/hs-real-anime-y11n-640-fp32.onnx" \
  --output-dir /abs/out \
  --report /abs/out/report.json

# 輪郭（Refiner）版
python -m maskveil.tools.censor /abs/input.png \
  --det-model /abs/det.onnx --refine \
  --refiner-model "$HOME/comfyui-local/models/hotscreen/seg/nsfw-anime-xl-x1280.onnx"
```

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `INPUT` | （必須） | 画像、またはディレクトリ |
| `--det-model` | （必須） | 検出モデルの**絶対パス** |
| `--refine` | false | 輪郭モデルでマスク形状を精密化する |
| `--refiner-model` | - | `--refine` のとき必須（絶対パス）。`--refine` 無しで渡すと `exit 2` |
| `--classes` | 性器 EXPOSED 2クラス | 名前 / ID / `all` のカンマ区切り（既定WFと同じ） |
| `--conf` / `--iou` | 0.2 / 0.45 | 検出・輪郭の閾値 |
| `--block-size` | **48** | モザイクのブロック辺（2–128） |
| `--mask-expand` | **0** | マスクの追加膨張（0–256）。既定 0 はブロック量子化の余白で足りるため |
| `--png-compress-level` | 1 | PNG 圧縮レベル（0–9） |
| `--output-dir` | 画像→親 / ディレクトリ→`<入力名>_censored` | 出力先（入力ディレクトリと同じ場所は `exit 2`） |
| `--no-recursive` | false | 直下のみ処理する |
| `--report` | 標準出力 | レポート JSON の保存先（指定時は stdout にサマリ1行） |
| `--pretty` | false | JSON を整形する |

**PNG 圧縮レベルとサイズ**（2,560×3,840・モザイク後の6枚・`block_size=48` の実測）

| level | 合計 | 1枚あたり | 画素 |
| --- | --- | --- | --- |
| 1（既定） | 40.4 MB | 5.3–8.3 MB | 基準 |
| 4 | 34.6 MB（−14%） | 4.2–7.2 MB | 1 と同一 |
| 6 | 32.9 MB（−19%） | 4.0–6.9 MB | 1 と同一 |

- 画素は可逆圧縮のため level で変わらない（1/4/6 で実測一致）。サイズは画像内容に依存する
- 用途の目安: 転送重視は既定の 1、保管重視は 6

**出力命名と構造**

- ファイル名は **`NNNN_<元の名前>.png`**（接頭辞連番）。**フォルダ毎に `0001` 起点**。
- 入力の**相対パス構造を保持**する（`in/sub/a.png` → `<out>/sub/0001_a.png`）。
- 出力は**常に PNG**（JPEG / WebP 入力も PNG へ変換）。
- **上書きしない**。出力先に同名ファイルがある場合、その画像は `failed` として記録し継続する。
  → **同じ出力先への再実行は全件 `failed`** になる。再実行時は `--output-dir` を分けるか、既存出力を退避させる。
- 出力先が入力ディレクトリの**内側**にある場合、そのディレクトリは走査から除外する（前回の出力を再処理しない）。
- 対象外の拡張子は無視する（失敗にしない）。検出0件の画像は**無加工コピー**を保存し `clean` と記録する。

**終了コード**: `0`=全件成功（`clean` を含む） / `1`=1件以上の失敗（処理は継続済み） / `2`=引数エラー（処理を開始しない）。

**レポート JSON**（`--report` 省略時はこの全文が stdout に出る）

```json
{
  "schema_version": 1,
  "tool": "censor",
  "tool_version": "0.1.0",
  "run": {
    "input": "/abs/in", "output_dir": "/abs/in_censored", "recursive": true,
    "started_at": "2026-10-31T12:00:00Z", "finished_at": "2026-10-31T12:00:12Z", "elapsed_sec": 12.3,
    "params": {"mode": "mosaic", "block_size": 48, "mask_expand": 0, "conf": 0.2, "iou": 0.45,
               "classes": [3, 11], "refine": false, "png_compress_level": 1}
  },
  "model": {"det": {"path": "/abs/det.onnx", "sha256": "fecf08b6...", "input_size": 640},
            "seg": null},
  "folders": [{"folder": ".", "total": 2, "mosaicked": 1, "clean": 1, "failed": 0}],
  "images": [
    {"index": 1, "folder": ".", "sequence": 1,
     "input": "/abs/in/a.png", "output": "/abs/in_censored/0001_a.png",
     "status": "mosaicked", "applied": true, "detections": 1,
     "bboxes": [[120.0, 60.0, 240.0, 180.0]], "elapsed_sec": 0.42, "error": null}
  ],
  "summary": {"total": 2, "mosaicked": 1, "clean": 1, "failed": 0, "elapsed_sec": 12.3}
}
```

- `status` は `mosaicked`（適用あり） / `clean`（検出0件で無加工コピー） / `failed`（読込・推論・保存の失敗、出力衝突）。
- `images[].input → output` が**対応表**。`NNNN_` 連番で名前が変わるため、後段ツールはこれを使う
  （`compare_censor_runs.py` は `..._00001_.png` 形式を前提とするため、CLI 出力の突合はレポート経由）。
- `folders[]` はフォルダ別集計。**フォルダ単位で zip 化するか**の判断に使える。
- `classes` はクラス**ID**（`detect` の `params.classes` と同じ表現）。
- `started_at` / `finished_at` / `elapsed_sec` は実行ごとに変わるため、レポート差分比較では無視する。
- `params` と `model.*.sha256` が**再現性の記録**（同じ入力・同じモデル・同じ設定かを後から確認できる）。

**契約（互換性の約束）**

利用者・後段ツールが依存してよい範囲。**前提**: リポジトリの `custom_nodes_src/` とワークスペースの
`custom_nodes/` が `install_node.sh` で同一内容に同期されていること（WF 経路はワークスペース側で動く）。

| 項目 | 約束 | 詳細 |
| --- | --- | --- |
| レポート | `schema_version` は `1`。`schema_version` を上げる場合を除き、**キー集合は凍結**（削除・改名・意味変更をしない） | 下記「レポート JSON」 |
| 終了コード | `0`=全件成功（`clean` 含む）/ `1`=1件以上の失敗**または実行時エラー**（モデル読込失敗・レポート書込失敗。この場合レポートが残らないことがある）/ `2`=引数エラー（処理を開始しない） | — |
| 出力 | 常に PNG・`NNNN_<元の名前>.png`（フォルダ毎 `0001` 起点）・既存ファイルは上書きしない。**単一プロセス実行が前提**（同一 `--output-dir` への並行実行は未サポート） | 「出力命名と構造」 |
| 画素（`mosaicked`） | 既定WF（`workflows/censor-nsfw.json`）と Refiner 版WF（`censor-nsfw-refiner.json`）のいずれも、同一入力・モデル・パラメータなら**画素完全一致**（実測: 既定 6/6・Refiner 2/2）。バイト同一ではない | 下記「画素の同一性」 |
| 画素（`clean`） | 入力と画素完全一致（実測 1/1）。既定WF 経由の出力とも一致 | 同上 |
| 依存 | `core/`・`tools/` は ComfyUI・torch 非依存（numpy / onnxruntime / Pillow） | 冒頭の層表 |
| モデル指定 | 絶対パスのみ（相対パスは `exit 2`） | 「モデル（フルパス必須）」 |

**`NNNN` は安定識別子ではない**: 入力フォルダのファイル集合とソート順から毎回算出するため、1枚増減すると
以降の番号が変わる。後段は名前ではなくレポートの `images[].input → output` を使う。

**等価性の検証範囲**: 既定WF（bbox マスク）と Refiner 版（セグメンテーション輪郭）の双方で、実測により
画素一致を確認済み（既定 6/6・Refiner 2/2）。再検証は同一入力・モデル・パラメータで両経路を実行し、
両出力を画素比較する（バイト比較ではない。`--png-compress-level` の差は画素に影響しない）。

**画素の同一性**

`censor` の 8bit 量子化は ComfyUI の `SaveImage` と同じ**切り捨て**に揃えてある。同じ float32 結果から
作った PNG は**画素完全一致**する（モザイク領域を含む）。一方、検出0件の `clean` は `uint8` をそのまま
保存するため、入力と画素完全一致する。

**他スキルとの組み合わせ例**

```bash
# A) 出力群から「生」と「モザイク」の両方を zip 化する
python3 .opencode/skills/strip_metadata/scripts/strip_metadata.py --directory "$OUT"   # 生データのみ
python -m maskveil.tools.censor "$OUT" --det-model /abs/det.onnx \
  --output-dir "$OUT/censored" --report "$OUT/censored/report.json"
(cd "$OUT" && zip -q -r batch_raw.zip ./*.png)
(cd "$OUT/censored" && zip -q -r batch_censored.zip ./*.png)

# B) input の zip を解凍 → フォルダ毎にモザイク → zip
for d in "$WORK"/*/; do
  python -m maskveil.tools.censor "${d%/}" --det-model /abs/det.onnx \
    --output-dir "${d%/}_censored" --report "${d%/}_censored/report.json"
  (cd "${d%/}_censored" && zip -q -r "../$(basename "${d%/}")_censored.zip" .)
done
```

- `censor` の出力は PIL 再エンコードを通るため `tEXt`（`prompt` / `workflow`）は残らない。
  `strip_metadata` が必要なのは**生データ側**だけ。
- `strip_metadata.py --directory` は**直下の PNG のみ**が対象（サブディレクトリ非対応）。入れ子出力へ
  適用しても効果はないが、`censor` 出力は既にメタデータフリーなので実害はない。
- **同じ出力先へ同時実行しない**（`.tmp` 名が衝突し得る）。

**既定WFとの対応**

| WF の設定 | CLI |
| --- | --- |
| `MosaicByMask.mode=mosaic` | 固定（`--mode` は非公開） |
| `MosaicByMask.block_size=48` | `--block-size 48`（既定） |
| `HotscreenDetector.mask_expand=0` | `--mask-expand 0`（既定） |
| `HotscreenRefiner.mask_expand=0` / `seg_crop=false` | `--refine` 時は同じ値・同じ固定 |
| `classes=FEMALE_/MALE_GENITALIA_EXPOSED` | `--classes` の既定 |
| `preview_mode=none` | 相当なし（CLI はプレビューを出力しない） |
| `top_k=0` / `mask_source=model_seg` | 固定（非公開） |

## 出力JSON

**`detect`（schema_version 1）**

```json
{
  "schema_version": 1,
  "image": {"path": "/abs/input.png", "width": 2560, "height": 3840},
  "model": {"path": "/abs/det.onnx", "sha256": "fecf08b6...", "input_size": 640},
  "params": {"conf": 0.2, "iou": 0.45, "classes": [11]},
  "detections": [
    {"class_id": 11, "class_name": "MALE_GENITALIA_EXPOSED", "score": 0.713,
     "bbox": [1587.0, 1247.0, 2258.0, 1637.0]}
  ],
  "count": 1
}
```

**`refine`（schema_version 2）** — `detect` の内容を保ち、`masks` / `mask_source` を追加する。

```json
{
  "schema_version": 2,
  "image": {"path": "/abs/input.png", "width": 2560, "height": 3840},
  "model": {"det": {"path": "...", "sha256": "...", "input_size": 640},
            "seg": {"path": "/abs/seg.onnx", "sha256": "...", "input_size": 1280}},
  "params": {
    "det": {"conf": 0.2, "iou": 0.45, "classes": [11]},
    "refine": {"mask_source": "model_seg", "classes": "all", "conf": 0.2, "iou": 0.45,
               "seg_crop": false}
  },
  "detections": [
    {"class_id": 11, "class_name": "MALE_GENITALIA_EXPOSED", "score": 0.713,
     "bbox": [1587.0, 1247.0, 2258.0, 1637.0],
     "mask_source": "model_seg",
     "masks": [[[1786.0, 1332.0], "..."]]}
  ],
  "count": 1,
  "refine": {"targets": 1, "refined": 1, "preserved": 0, "mask_source": "model_seg"}
}
```

- `masks` は**原画像座標のポリゴンの配列**（連結成分ごと。各要素は閉路）。分断されたマスクも
  すべて保持する（1本だけ採用すると未被覆が生まれるため）。
- `mask_source` は `model_seg` / `bbox_ellipse` / `none`。
- 対応表に無いクラス、または重なる輪郭が無かった検出は `mask_source: "none"`（`masks` なし）。
- `refine.preserved` は「入力JSON（v2）で既にマスクを持っていた検出のうち、今回対象外だった件数」。
  **v2 を再投入しても対象外のマスクは保持される（冪等）**。
- `image.path` は絶対パスに解決される。`image.width/height` が実ファイルと不一致なら `exit 1`。

## クラス対応（検出16 → 輪郭7）

輪郭モデルが持つのは `anus` / `nipple` / `penis` / `vagina` / `female face` / `male face` /
`pubic hair` の7クラスのみ。対応表は `core/class_mapping.py` が持つ。

| 検出クラス | 輪郭クラス | 備考 |
| --- | --- | --- |
| `FEMALE_GENITALIA_EXPOSED` | `vagina` | |
| `MALE_GENITALIA_EXPOSED` | `penis` | |
| `ANUS_EXPOSED` | `anus` | |
| `FEMALE_BREAST_EXPOSED` | `nipple` | **近似**（乳首のみ。乳房全体は覆わない） |
| `FEMALE_FACE` / `MALE_FACE` | `female face` / `male face` | |
| 上記以外（COVERED 系・BUTTOCKS・ARMPITS・BELLY・FEET・EYE） | — | `mask_source: "none"` |

対応する輪郭が複数ある場合は **bbox の IoU が最大**のものを1対1で割り当てる。

## 設計上の注意

- **マスクは検出 bbox で切り取らない。** 検出モデルと輪郭モデルで bbox が一致しないため、
  輪郭側の形状をそのまま使う（実測では輪郭が検出 bbox より 5〜154px 大きい場合があり、
  切り取るとモザイク漏れになる）。
- 輪郭の bbox クロップはプロトタイプ格子（320）で量子化するため、seg 自身の bbox に対しても
  最大 1 格子（入力座標で約 4px）外へ張り出すことがある（安全側の誤差）。
- `--input-size` / `--refiner-input-size` は省略時に ONNX の静的入力から取得する。指定値が
  モデルと食い違う場合は、推論前に明確なエラーで停止する。
- 座標復元は軸別倍率（`scale_x` / `scale_y`）で行い、letterbox の往復誤差を出さない。
- NMS は **class-aware**（同一 bbox でもクラスが違えば残す）。モザイク漏れを避けるため、
  モデル作者実装のクラス非依存 NMS とは意図的に異なる。
- `core` は `comfy` / `torch` / `PIL` / `cv2` を import しない（`tests/test_imports.py` が AST で検証）。
  リサイズは numpy のバイリニア実装。

## ComfyUI ノード

`hotscreen` カテゴリに3ノードを公開する。**入出力の正は [SPEC.md](./SPEC.md) §4**。

| ノード | 入力 | 出力 | 役割 |
| --- | --- | --- | --- |
| `Hotscreen Detector` | `IMAGE` + 検出モデル + `classes` / `conf` / `iou` / `top_k` / `mask_expand` / `preview_mode` | `HOTSCREEN` / `MASK` / `SEGS` / `detection`(JSON) / `preview` | 検出とマスク生成 |
| `Hotscreen Refiner` | `HOTSCREEN` + 輪郭モデル + `refiner_classes` / `conf` / `iou` / `seg_crop` / `mask_expand` / `preview_mode` | `HOTSCREEN` / `MASK` / `refine`(JSON) / `preview` | マスク形状を輪郭へ置換 |
| `Mosaic by Mask` | `mode` / `block_size` ＋（任意）`hotscreen` / `image` / `mask` | `IMAGE` / `HOTSCREEN` | マスク領域の隠蔽 |

### 配線

| 構成 | 配線 | 用途 |
| --- | --- | --- |
| 最短 | `Detector → Mosaic`（`HOTSCREEN` 1本） | 輪郭なし・最速 |
| 標準 | `Detector → Refiner → Mosaic` | 形状追従 |
| 汎用 | `IMAGE` + `MASK` を `Mosaic` へ直接 | hotscreen 非依存のモザイク |
| 多段 | 1段目の `hotscreen` を2段目へ、**2段目の `mask` は別 `Detector` から明示接続** | 部位ごとに異なる加工 |

- `Mosaic by Mask` の既定は **`mode=mosaic` / `block_size=48`**。48px は 2560×3840 で
  モザイクと分かる粗さ（8px では細かすぎてほぼ隠れないことを実機で確認済み。PO 決定 2026-10-31 で
  既定WF・CLI と揃えて 32 → 48 へ統一）。
  細かくしたい場合は `block_size` を下げる（`blur` では半径として働く）
- `Mosaic` のベース画像は **明示 `image` > `hotscreen.mosaic_image` > `hotscreen.image`** の順。
  多段では明示 `image` を繋がない（繋ぐと元画像へ戻り連鎖が切れる）
- `mask` は **明示 `mask` > `hotscreen.mask`**。明示指定で「検出ごとに異なる加工」を組める
- 全ゼロマスクのときは入力をそのまま返し、`mosaic_image` と `meta.processing` を更新しない

### `mask_expand`（追加の膨張）

- 形状を外側へ N px 広げる（**既定 0px**）。`bbox` は変わらない
- 既定 0 の理由: `mosaic` はブロック量子化のため実効的な外側余白（最大 `block_size-1` px）を持つ。
  そのため追加の膨張は不要（PO 決定 2026-10-31 で 4 → 0）。`blur` / `black` は境界が厳密なので必要なら明示的に広げる
- `masks` を持たない検出（`mask_source="none"`）は **bbox 矩形を外挿**してマスク化する
- 膨張させた場合、膨張後の形状が正であり、`HOTSCREEN.detections[].masks`・`MASK`・`SEGS.cropped_mask` は同じラスタから作られる

### `SEGS`（互換ビュー）の注意

- `SEGS` は **先頭バッチ（`batch_index == 0`）の画像だけ**を表す。全バッチの検出は
  `HOTSCREEN.detections` と `MASK`（`[B,H,W]`）が保持する
- `SEG.bbox` / `crop_region` は **`mask_expand` を含むマスク全体を囲む**（膨張分が欠落しない）。
  画像端では画像内へクリップされるため、検出の `bbox` と一致しないことがある
- `SEGS` は Impact Pack 互換の**構造のみ**を提供する（実接続は未検証）

### モデルの解決

- `folder_paths` の `hotscreen` カテゴリ（`extra_model_paths.yaml`）から **`.onnx` のみ**を候補にする
- 入力サイズは det 640 / seg 1280 を要求し、取り違えると `ValueError` で停止する
- `InferenceSession` はプロセス内でキャッシュする（パス + mtime + size + 入力サイズ）

## 性能（実測 / CPU, onnxruntime 1.31.0）

| 処理 | 時間 | 備考 |
| --- | --- | --- |
| `detect`（2560×3840, CLI全体） | 0.52〜0.55 s | 推論のみ約 0.09 s |
| `refine`（全体1回, 12検出） | 4.79〜4.98 s | 輪郭推論 約 2.9 s（検出数に依存しない） |
| `refine --seg-crop`（1検出） | 4.37〜4.46 s | 検出ごとに推論するため検出数に比例 |
| `censor`（2560×3840, 検出のみ） | 0.74〜1.00 s/枚 | 検出＋モザイク＋PNG保存。100枚で約1.5分 |
| `censor`（同, `--refine`） | 約 4〜5 s/枚 | 輪郭推論の支配的コストが加算される |

## 開発・テスト

ホストに `onnxruntime` は無いため、**コンテナ `comfyui-local-dev` 内**で実行する。

```bash
docker cp custom_nodes_src/maskveil comfyui-local-dev:/tmp/
docker exec -w /tmp/maskveil comfyui-local-dev python -m pytest tests/
docker exec -w /tmp/maskveil comfyui-local-dev uvx ruff check .

# クラス選択 UI（web/hotscreen_classes.js）の挙動検証（deno が必要・pytest 対象外）
deno run -A custom_nodes_src/maskveil/tests/js/check_classes.ts
```

- `tests/js/check_classes.ts` は JS を**実際に import** して `parseSelection` / `serializeSelection`
  を実行する（ラベル配列の一致だけでは「`all` の展開」や「定義順の正規化」の退行を検出できないため）
- `tests/test_comfy_loader.py` は ComfyUI と同じ手順（マングル名・`sys.path` に親を入れない）で
  パッケージを読み込む。`nodes/` が絶対 import に戻るとここで落ちる

## 制約

- 輪郭 ONNX は `ultralytics 8.4.189` で書き出した**静的入力（1280）**版。動的入力版は未作成。
- 本パッケージは AGPL コードを同梱しない（書き出し済み ONNX の推論のみ）。
- 輪郭が対応しない部位（臀部・脇・腹・足・眼・COVERED 系）は bbox 形状しか持たない。
