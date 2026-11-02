# maskveil

**領域を検出し、輪郭に沿ったマスクを作り、モザイク／ぼかし／黒塗りで隠すエンジン。**
ComfyUI ノードと CLI は、そのエンジンに載せたアダプタです。

| 層 | 役割 | 実行時依存 | ComfyUI 依存 |
| --- | --- | --- | --- |
| `core/` | 推論・前後処理・座標復元・輪郭抽出（純粋ロジック） | numpy / onnxruntime | **なし** |
| `tools/` | CLI（`detect` / `refine` / `censor`）。モデルのパス検証を担当 | 上記 + Pillow | なし |
| `nodes/` | ComfyUI ノード（`folder_paths` によるモデル解決） | 上記 + ComfyUI 提供の torch | あり |

- 対象は NSFW 部位に限りません。**対応モデルのクラス契約（16クラス）に含まれるクラス**であれば、
  顔・文字などの墨塗りにも使えます（任意の ONNX をそのまま受け付けるわけではありません）。
- 隠蔽方法は **`mosaic` / `blur` / `black`** の3モード（モザイクは選択肢の1つ）。
- `ultralytics`（AGPL）は**同梱せず、import もしません**。書き出し済み ONNX の推論だけを行います。

> **名前について**: `maskveil` はパッケージ（エンジン）の名前です。
> ノード名が `Hotscreen*` なのは、**同梱するモデルではなく「対応モデルのクラス契約」に紐づくノードだから**です（[ノード名の方針](#ノード名の方針)）。

## 段構成

```
detect : IMAGE + 検出モデル       → 検出JSON（クラス名・スコア・原画像座標bbox）      schema_version 1
refine : 検出JSON + 輪郭モデル    → マスク付きJSON（形状ポリゴン・mask_source）        schema_version 2
censor : 画像/ディレクトリ + 検出モデル（+任意で輪郭）
                                  → 隠蔽済みPNG + レポートJSON（1コマンド完結）
```

- `detect` は高速（約 0.5 秒/枚・CLI 全体）。`refine` は低速（約 4.8 秒/枚 @ CPU, 1280px）だが任意。
- `detect` に輪郭を内包する便宜フラグは**設けない**（1コマンド1モデル）。

## 対応モデル（**非同梱**）

**モデルは本パッケージに含まれません。** 利用者が取得し、**絶対パス**で指定します。
環境変数・既定値・ディレクトリ探索は行いません（ComfyUI ノードのみ `folder_paths` で解決します）。

### 入手元とライセンス

下記は**入手元・ライセンス・参照リビジョン**の一覧です。取得手順は「モデルの入手」にあります。

| 用途 | モデル（ONNX） | 入手元（Hugging Face） | ライセンス |
| --- | --- | --- | --- |
| 検出 | `hs-real-anime-y11n-640-fp32.onnx`（YOLO11・16クラス・入力640） | [`Perfectfox256/hotscreen-detection-models`](https://huggingface.co/Perfectfox256/hotscreen-detection-models) の `yolo-07-2025/hs-real-anime-y11n-640-fp32.onnx` | **AGPL-3.0** |
| 輪郭（任意） | `nsfw-anime-xl-x1280.onnx`（YOLO26-Seg・7クラス・入力1280） | [`01miku/anime-nsfw-segm-yolo26`](https://huggingface.co/01miku/anime-nsfw-segm-yolo26) | **MIT** |

> **検出モデルのライセンスは系列で分かれています。** 同じリポジトリでも `yolo-07-2025/` は **AGPL-3.0**、`yolox-07-2025/` は **MIT** です。
> 本パッケージが前提とするのは `yolo-07-2025/`（AGPL-3.0）です。`yolox-07-2025/` は別系列のため、本 README の手順では扱いません。

- 参照リビジョン（HF revision・取得時点）: 検出 `8134f9ea866820684cdb9f022a5ce3d306bed0a2` / 輪郭 `1697d5d1827b6a818b350b44bf3ec27f08837a2a`
  モデルは更新されうるため、再現性が必要な場合はこの revision を指定して取得します。

### モデルの入手

検出モデルは配布済み ONNX を取得します。輪郭モデルは **ONNX を取得する（経路A）** か、
**`.pt` から ONNX へ変換する（経路B）** のどちらかです。

#### 検出モデル（ONNX を取得）

```bash
DET_DIR=~/comfyui-local/models/hotscreen/det
mkdir -p "$DET_DIR"

curl -L -o "$DET_DIR/hs-real-anime-y11n-640-fp32.onnx" \
  "https://huggingface.co/Perfectfox256/hotscreen-detection-models/resolve/8134f9ea866820684cdb9f022a5ce3d306bed0a2/yolo-07-2025/hs-real-anime-y11n-640-fp32.onnx"

# クラス名の並び（16行）も同じ revision から取得しておく
curl -L -o "$DET_DIR/labels.txt" \
  "https://huggingface.co/Perfectfox256/hotscreen-detection-models/resolve/8134f9ea866820684cdb9f022a5ce3d306bed0a2/yolo-07-2025/labels.txt"
```

`hf` CLI（`pip install -U huggingface_hub`）でも同じことができます。
`--local-dir` は**リポジトリ内のパスを保持する**ため、`yolo-07-2025/` を平坦化します。

```bash
DET_DIR=~/comfyui-local/models/hotscreen/det

hf download Perfectfox256/hotscreen-detection-models \
  yolo-07-2025/hs-real-anime-y11n-640-fp32.onnx yolo-07-2025/labels.txt \
  --revision 8134f9ea866820684cdb9f022a5ce3d306bed0a2 \
  --local-dir "$DET_DIR"

# --local-dir/yolo-07-2025/... に落ちるので、配置先の例に合わせて平坦化する
mv "$DET_DIR"/yolo-07-2025/* "$DET_DIR"/ && rmdir "$DET_DIR/yolo-07-2025"
```

#### 輪郭モデル: 経路A（ONNX をダウンロード）

```bash
SEG_DIR=~/comfyui-local/models/hotscreen/seg
mkdir -p "$SEG_DIR"

curl -L -o "$SEG_DIR/nsfw-anime-xl-x1280.onnx" \
  "https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/1697d5d1827b6a818b350b44bf3ec27f08837a2a/nsfw-anime-xl-x1280.onnx"
```

> **配布元の ONNX は動的入力です。** このファイルは `dynamic=True` / `half=True`（ultralytics 8.4.52）で
> 書き出されており、入力形状が `[batch, 3, height, width]` です。
> **`maskveil-refine` は `--refiner-input-size 1280` を明示しないと「動的入力サイズ」のエラーで停止します。**
> 統合 CLI `maskveil`（`censor`）はこのフラグを持たないため、**経路A のモデルでは `--refine` を使えません**。
> ComfyUI ノードは既定で 1280 を渡すため、そのまま動作します。
> 静的・fp32 のモデルが必要な場合（統合 CLI で `--refine` したい場合）は、経路B で `.pt` から再変換してください。

#### 輪郭モデル: 経路B（`.pt` から ONNX へ変換）

**変換は利用者環境で行う行為です。** 本パッケージは変換スクリプトを同梱せず、
`ultralytics`（AGPL-3.0）を依存にも含みません（`tests/test_license.py` が機械検証）。

```bash
# 変換する環境にだけ入れる（本パッケージの依存には足さない）
python -m pip install ultralytics
```

`.pt` は ONNX と同じリポジトリから取得できます（約 141MB）。

```bash
SEG_DIR=~/comfyui-local/models/hotscreen/seg

curl -L -o "$SEG_DIR/nsfw-anime-xl-x1280.pt" \
  "https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/1697d5d1827b6a818b350b44bf3ec27f08837a2a/nsfw-anime-xl-x1280.pt"
```

```python
from ultralytics import YOLO

model = YOLO("nsfw-anime-xl-x1280.pt")  # 上で取得した .pt
model.export(
    format="onnx",
    imgsz=1280,
    opset=17,
    simplify=True,
    dynamic=False,
    half=False,  # fp32 のまま
)
# → 同じディレクトリに nsfw-anime-xl-x1280.onnx ができる
```

本リポジトリで動作確認した書き出し条件（生成物の ONNX メタデータ）:

| 項目 | 値 |
| --- | --- |
| モデル | `YOLO26x-seg`（`task=segment` / `head=Segment26`） |
| ultralytics | `8.4.189` |
| 入力 | `images` = `[1, 3, 1280, 1280]` float32 |
| 出力 | `output0` = `[1, 43, 33600]`（`4 + 7クラス + 32`）/ `output1` = `[1, 32, 320, 320]` |
| opset / simplify / dynamic | `17` / `True` / `False` |

> **合否は形状で判定します。** 書き出し条件が違うと `output0` のクラス次元（43）や
> `output1` のプロトタイプ数（32）が変わります。`core/onnx_utils.py` が検証するのは
> 入力サイズだけなので、**変換後は「機能検証」の出力形状チェックまで実施してください**。

#### AI への指示テンプレート

`.pt` の変換を AI に任せる場合は、次のテンプレートの `<...>` を差し替えて渡します。

```text
次の .pt を ONNX に変換してください。

- 入力（.pt）: <例: ~/Downloads/nsfw-anime-xl-x1280.pt>
- 出力先: <例: ~/comfyui-local/models/hotscreen/seg/nsfw-anime-xl-x1280.onnx>
- 入力サイズ: 1280（正方形・静的）
- クラス数: 7（anus / nipple / penis / vagina / female face / male face / pubic hair）
- 出力ファイル名: nsfw-anime-xl-x1280.onnx
- 書き出し条件: opset=17 / simplify=True / dynamic=False / fp32

受け入れ条件:
1. onnxruntime で読み込めること
2. 入力が images=[1,3,1280,1280] float32
3. output0 が [1, 4+7+32, N]、output1 が [1, 32, 320, 320]
4. 元の .pt は変更しない。本パッケージのリポジトリへモデルをコミットしない
```

> `ultralytics` は **AGPL-3.0** です。変換は利用者環境の行為であり、本パッケージは
> 変換スクリプトも `ultralytics` も同梱・import しません。

### 配置先の例

| 用途 | 配置先の例 |
| --- | --- |
| 検出 | `~/comfyui-local/models/hotscreen/det/hs-real-anime-y11n-640-fp32.onnx` |
| 輪郭（任意） | `~/comfyui-local/models/hotscreen/seg/nsfw-anime-xl-x1280.onnx` |

`hotscreen` カテゴリ配下に置くと、ComfyUI ノードは `folder_paths` で解決できます（CLI は常に絶対パス指定）。

### モデルの同一性をどう確認するか

ONNX のバイト列は**変換環境で変わります**（ultralytics のバージョン、opset、fp16/fp32、simplify の有無など）。
同じモデルでも、配布元が公開している ONNX と、手元で `.pt` から変換した ONNX は**ハッシュ値が一致しません**。
固定値を載せても同一性の保証にならないため、本 README ではハッシュ値を記載しません。

代わりに、**形状とクラス契約**で「使えるモデルか」を判定します。手順は「機能検証」を参照してください。

> CLI / ノードの JSON 出力には実行時の `sha256` が含まれます。これは「その実行で読んだファイルは何か」を
> 記録するためのもので、配布物の同一性を保証する値ではありません。

### 機能検証（入手・変換したモデルが使えるか）

**形状とクラス契約**で判定します。上から順に、**同じシェル**で実行してください。

前提: `maskveil-detect` / `maskveil-refine` が使える環境（`pipx install maskveil` または `pip install -e .`）と、
その Python で `onnxruntime` を import できること。

```bash
DET=~/comfyui-local/models/hotscreen/det/hs-real-anime-y11n-640-fp32.onnx
SEG=~/comfyui-local/models/hotscreen/seg/nsfw-anime-xl-x1280.onnx
```

#### 1. 入力サイズと出力形状

```bash
python - <<PY
import onnxruntime as ort

for p in ["$DET", "$SEG"]:
    session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    print(p.rsplit("/", 1)[-1])
    print("  input: ", session.get_inputs()[0].shape)
    for out in session.get_outputs():
        print("  output:", out.name, out.shape)
PY
```

期待値:

| モデル | 入力 | 出力 | 次元の意味 |
| --- | --- | --- | --- |
| 検出 | `[1, 3, 640, 640]` | `output0` = `[1, 20, 8400]` | `4 + 16クラス` |
| 輪郭（経路B・静的） | `[1, 3, 1280, 1280]` | `output0` = `[1, 43, 33600]` / `output1` = `[1, 32, 320, 320]` | `4 + 7クラス + 32プロトタイプ` |
| 輪郭（経路A・動的） | `['batch', 3, 'height', 'width']` | `output0` = `['batch', 43, 'anchors']` / `output1` = `['batch', 32, 'mask_height', 'mask_width']` | 2次元目が **43** / **32** なら可（N・H/W は動的） |

> 経路A は記号次元を返すため、**数値の一致は要求しません**。合否は
> `output0` の2次元目（検出 20 / 輪郭 43）と `output1` の2次元目（32）で判定します。

#### 2. ラベル順の一致（16クラス）

`--list-classes` は `0<TAB>FEMALE_FACE` の形式で出力するため、**2列目だけ**を比較します。
配布元の `labels.txt` は **CRLF** なので `tr -d '\r'` で正規化します。

```bash
maskveil-detect --list-classes | cut -f2 > /tmp/cli_labels.txt
tr -d '\r' < "$(dirname "$DET")/labels.txt" > /tmp/hf_labels.txt
diff /tmp/cli_labels.txt /tmp/hf_labels.txt && echo "labels: OK"
```

`diff` が空なら、**クラス数16・名前・順序**が一致しています（`core/labels.py` の `LABELS` と配布元が同一）。

#### 3. スモーク推論

対象部位を含む画像を1枚使い、**1件以上の検出**が返ることを確認します。
`detect` は検出0件でも終了コード 0 を返すため、**件数を `assert` で判定**します。

```bash
IMG=/path/to/image.png
maskveil-detect "$IMG" --model "$DET" --classes all --conf 0.25 --output /tmp/det.json --pretty

python - <<'PY'
import json

data = json.load(open("/tmp/det.json"))
print("count:", data["count"])
assert data["count"] >= 1, "検出が0件です（画像かモデルを確認してください）"
print("smoke: OK")
PY
```

#### 実行結果（2026-11-02・開発環境）

| 検証 | 結果 |
| --- | --- |
| 1. 入力・出力形状 | 検出 `[1,3,640,640]` → `[1,20,8400]` / 輪郭（経路B）`[1,3,1280,1280]` → `[1,43,33600]`・`[1,32,320,320]` |
| 2. ラベル順 | `diff` **空**（16行一致） |
| 3. スモーク推論 | 3枚すべてで `assert` 通過（下表） |

| 画像 | 検出件数 | 最高スコア |
| --- | --- | --- |
| `heroine1_00001_.png` | 3 | 0.912 |
| `heroine1_00016_.png` | 7 | 0.861 |
| `heroine1_00032_.png` | 5 | 0.889 |

（3枚とも `FEMALE_FACE` を検出。`FEMALE_GENITALIA_EXPOSED` / `ANUS_EXPOSED` などの EXPOSED クラスを含む）

#### 配布元 ONNX（126MB）の検証結果

経路A のファイル（`nsfw-anime-xl-x1280.onnx` / 126,350,117 B）を実際に配置して確認しました（2026-11-02）。

| 実行 | 結果 |
| --- | --- |
| CLI `refine`（入力サイズ未指定） | **エラー**: `refiner model has a dynamic input size; specify it explicitly` |
| CLI `refine --refiner-input-size 1280` | 成功。検出7件のうち **3件**に `mask_source=model_seg`（FEMALE_FACE / ANUS_EXPOSED / FEMALE_GENITALIA_EXPOSED）。ローカル変換版と同じ3件 |
| ComfyUI `censor-nsfw-refiner.json`（1枚） | **1枚入力 → 1枚出力**で完走。モザイク適用を画素差分で確認（最大差 209） |

→ 動作しますが、**`maskveil-refine` では入力サイズの明示が必須**、**統合 CLI `maskveil`（`--refine`）では使えません**。
統合 CLI で輪郭を使いたい場合は、経路B の静的モデルを用意してください。

### モデルの差し替え可能性

エンジンはモデル名に依存しませんが、**クラス契約**には依存します。

| 前提 | 根拠 | 満たさない場合 |
| --- | --- | --- |
| YOLO 系の出力 `(1, 4+nc, N)` | `core/postprocess.py` | デコード処理の追加が必要 |
| クラス数 16・クラス名と順序が一致 | `core/labels.py` の `LABELS` | `LABELS` の差し替え（データ修正） |
| 静的・正方形入力 | `core/onnx_utils.py` | 入力サイズの明示指定、非正方形は非対応 |
| NMS 非内蔵 | 自前の class-aware NMS | 二重適用に注意 |
| 輪郭は YOLO-Seg 系（プロトタイプ 32） | `core/refiner.py` | SAM 等はデコード実装の追加が必要 |

## インストール

### ComfyUI にノードとして入れる

```bash
# ComfyUI Manager を使う場合: 「maskveil」を検索してインストール
# または Comfy Registry CLI
comfy node install maskveil

# 手動の場合（ComfyUI/custom_nodes/ 配下に clone する）
cd ComfyUI/custom_nodes
git clone https://github.com/purple-ocean-ego/maskveil.git
# 手動 clone では依存が自動で入らないため、必ず入れる
python -m pip install -r maskveil/requirements.txt
```

- **`onnxruntime` は ComfyUI 本体の依存に含まれません。** 手動 clone の経路では
  `requirements.txt` を入れるまでノードが読み込まれません（`IMPORT FAILED` になります）。
- ComfyUI Manager / Registry 経由のインストールでは `requirements.txt` が自動で使われます。
- ノードの追加は **ComfyUI の再起動**で反映されます。

### CLI として入れる

```bash
pipx install maskveil     # 推奨（隔離環境）
# または
pip install maskveil
```

## ComfyUI で使う

`hotscreen` カテゴリに3ノードを公開します。

| ノード（クラス名） | 入力 | 出力 | 役割 |
| --- | --- | --- | --- |
| `Hotscreen Detector`（`HotscreenDetector`） | `IMAGE` + 検出モデル + `classes` / `conf` / `iou` / `top_k` / `mask_expand` / `preview_mode` | `HOTSCREEN` / `MASK` / `SEGS` / `detection`(JSON) / `preview` | 検出とマスク生成 |
| `Hotscreen Refiner`（`HotscreenRefiner`） | `HOTSCREEN` + 輪郭モデル + `refiner_classes` / `conf` / `iou` / `seg_crop` / `mask_expand` / `preview_mode` | `HOTSCREEN` / `MASK` / `refine`(JSON) / `preview` | マスク形状を輪郭へ置換 |
| `Mosaic by Mask`（`MosaicByMask`） | `mode` / `block_size` ＋（任意）`hotscreen` / `image` / `mask` | `IMAGE` / `HOTSCREEN` | マスク領域の隠蔽 |

### 配線

| 構成 | 配線 | 用途 |
| --- | --- | --- |
| 最短 | `Detector → Mosaic`（`HOTSCREEN` 1本） | 輪郭なし・最速 |
| 標準 | `Detector → Refiner → Mosaic` | 形状追従 |
| 汎用 | `IMAGE` + `MASK` を `Mosaic` へ直接 | 検出モデルに依存しない隠蔽 |
| 多段 | 1段目の `hotscreen` を2段目へ、**2段目の `mask` は別 `Detector` から明示接続** | 部位ごとに異なる加工 |

- `Mosaic by Mask` の既定は **`mode=mosaic` / `block_size=48`**。48px は 2560×3840 で
  モザイクと分かる粗さです（8px では細かすぎてほぼ隠れません）。
  細かくしたい場合は `block_size` を下げる（`blur` では半径として働く）
- `Mosaic` のベース画像は **明示 `image` > `hotscreen.mosaic_image` > `hotscreen.image`** の順。
  多段では明示 `image` を繋がない（繋ぐと元画像へ戻り連鎖が切れる）
- `mask` は **明示 `mask` > `hotscreen.mask`**。明示指定で「検出ごとに異なる加工」を組める
- 全ゼロマスクのときは入力をそのまま返し、`mosaic_image` と `meta.processing` を更新しない

### `mask_expand`（追加の膨張）

- 形状を外側へ N px 広げる（**既定 0px**）。`bbox` は変わらない
- 既定 0 の理由: `mosaic` はブロック量子化のため実効的な外側余白（最大 `block_size-1` px）を持つ。
  `blur` / `black` は境界が厳密なので、必要なら明示的に広げる
- `masks` を持たない検出（`mask_source="none"`）は **bbox 矩形を外挿**してマスク化する
- 膨張後の形状が正であり、`HOTSCREEN.detections[].masks`・`MASK`・`SEGS.cropped_mask` は同じラスタから作られる

### `SEGS`（互換ビュー）の注意

- `SEGS` は **先頭バッチ（`batch_index == 0`）の画像だけ**を表す。全バッチの検出は
  `HOTSCREEN.detections` と `MASK`（`[B,H,W]`）が保持する
- `SEG.bbox` / `crop_region` は **`mask_expand` を含むマスク全体を囲む**。画像端では画像内へ
  クリップされるため、検出の `bbox` と一致しないことがある
- `SEGS` は Impact Pack 互換の**構造のみ**を提供する（実接続は未検証）

### モデルの解決

- `folder_paths` の `hotscreen` カテゴリ（`extra_model_paths.yaml`）から **`.onnx` のみ**を候補にする
  ```yaml
  # extra_model_paths.yaml（例）
  hotscreen: hotscreen
  ```
- 入力サイズは det 640 / seg 1280 を要求し、取り違えると `ValueError` で停止する
- `InferenceSession` はプロセス内でキャッシュする（パス + mtime + size + 入力サイズ）

### クラス選択 UI

`web/hotscreen_classes.js` が `classes` ウィジェットの選択 UI（`all` の展開・定義順の正規化）を提供します。
モデル固有 UI のため、ノード名と同じく `Hotscreen*` の命名を維持しています。

## CLI で使う

```bash
# 1) 検出（高速）
maskveil-detect input.png \
  --classes MALE_GENITALIA_EXPOSED \
  --model /abs/det/hs-real-anime-y11n-640-fp32.onnx \
  > det.json

# 2) 輪郭化（低速・任意）
maskveil-refine < det.json \
  --refiner-model /abs/seg/nsfw-anime-xl-x1280.onnx \
  > refined.json

# 3) 隠蔽まで1コマンド（ディレクトリを再帰処理）
maskveil "$HOME/pictures/input" \
  --det-model /abs/det/hs-real-anime-y11n-640-fp32.onnx \
  --output-dir /abs/out \
  --report /abs/out/report.json

# 輪郭（Refiner）版
maskveil /abs/input.png --det-model /abs/det.onnx --refine \
  --refiner-model /abs/seg/nsfw-anime-xl-x1280.onnx
```

> **統合 CLI の `--refine` は動的入力の輪郭モデルに未対応です。** `maskveil` には `--refiner-input-size` が無いため、
> 経路A の ONNX（動的入力）を渡すと `refiner model has a dynamic input size` で `exit 2` になります。
> 経路A を使う場合は上の 1) → 2) の2コマンド（`maskveil-refine --refiner-input-size 1280`）に分けるか、
> 経路B の静的モデルを使ってください。

`python -m maskveil.tools.detect` / `.refine` / `.censor` でも同じことができます。

### `detect` の主なオプション

> 正は `maskveil-detect --help`。以下の表は既定値の要約です。

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--classes` | **必須** | 名前 / ID / `all` のカンマ区切り。未指定は `exit 2` |
| `--conf` | 0.2 | スコア閾値（モデル作者の推奨値） |
| `--iou` | 0.45 | class-aware NMS の IoU 閾値 |
| `--input-size` | モデルから取得 | モデル入力の一辺。指定するとモデルと一致するか検証する |
| `--top-k` | 0 | スコア上位N件に絞る（0 は無効） |
| `--list-classes` | - | 16クラスを `id<TAB>name` で表示して終了 |

### `refine` の主なオプション

> 正は `maskveil-refine --help`。以下の表は既定値の要約です。

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--refiner-model` | （必須） | `--mask-source model_seg` のとき必須。絶対パス |
| `--refiner-input-size` | モデルから取得 | 輪郭モデルの入力一辺。指定時はモデルと一致するか検証 |
| `--classes` | 入力JSONの全検出 | 輪郭化する検出クラスを限定 |
| `--mask-source` | `model_seg` | `bbox_ellipse` は**デバッグ用**（内接楕円） |
| `--seg-crop` | false | 検出 bbox 周辺をクロップして推論（計測用。1検出ごとに推論するため遅い） |

- 終了コード: `0` 成功（0件含む） / `1` 実行時エラー（入力JSON不正・画像読込失敗など） / `2` 引数エラー。
- 輪郭モデルは**1画像1回**だけ実行し、対応する全クラスのマスクを一度に取得する（クラスごとに再実行しない）。

### `censor`（隠蔽まで1コマンド）

`detect →（--refine 時のみ）refine → 隠蔽 → PNG 保存` を1コマンドで行います。
**ComfyUI・WebUI・コンテナを起動しません**（`tools/` は torch 非依存）。`detect` / `refine` が
「1コマンド1モデル」なのに対し、`censor` は複合ドライバとして例外的に2モデルを扱います。

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `INPUT` | （必須） | 画像、またはディレクトリ |
| `--det-model` | （必須） | 検出モデルの**絶対パス** |
| `--refine` | false | 輪郭モデルでマスク形状を精密化する |
| `--refiner-model` | - | `--refine` のとき必須（絶対パス）。`--refine` 無しで渡すと `exit 2` |
| `--classes` | 性器 EXPOSED 2クラス | 名前 / ID / `all` のカンマ区切り |
| `--conf` / `--iou` | 0.2 / 0.45 | 検出・輪郭の閾値 |
| `--block-size` | **48** | モザイクのブロック辺（2–128） |
| `--mask-expand` | **0** | マスクの追加膨張（0–256）。既定 0 はブロック量子化の余白で足りるため |
| `--png-compress-level` | 1 | PNG 圧縮レベル（0–9） |
| `--output-dir` | 画像→親 / ディレクトリ→`<入力名>_censored` | 出力先（入力ディレクトリと同じ場所は `exit 2`） |
| `--no-recursive` | false | 直下のみ処理する |
| `--report` | 標準出力 | レポート JSON の保存先（指定時は stdout にサマリ1行） |
| `--pretty` | false | JSON を整形する |

ファイル名はフォルダごとに `0001` 起点の `NNNN_<元ファイル名>.png` で、入力の相対パス構造を保ちます。
検出0件の画像は無加工でコピーされ、レポートでは `clean` として記録されます。

## 出力JSON

**`detect`（schema_version 1）**

```json
{
  "schema_version": 1,
  "image": {"path": "/abs/input.png", "width": 2560, "height": 3840},
  "model": {"path": "/abs/det.onnx", "sha256": "…", "input_size": 640},
  "params": {"conf": 0.2, "iou": 0.45, "classes": [11]},
  "detections": [
    {"class_id": 11, "class_name": "MALE_GENITALIA_EXPOSED", "score": 0.713,
     "bbox": [1587.0, 1247.0, 2258.0, 1637.0]}
  ],
  "count": 1
}
```

**`refine`（schema_version 2）** — `detect` の内容を保ち、`masks` / `mask_source` を追加します。

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
  すべて保持する（1本だけ採用すると未被覆が生まれるため）
- `mask_source` は `model_seg` / `bbox_ellipse` / `none`
- 対応表に無いクラス、または重なる輪郭が無かった検出は `mask_source: "none"`（`masks` なし）
- `refine.preserved` は「入力JSON（v2）で既にマスクを持っていた検出のうち、今回対象外だった件数」。
  **v2 を再投入しても対象外のマスクは保持される（冪等）**
- `image.path` は絶対パスに解決される。`image.width/height` が実ファイルと不一致なら `exit 1`

## クラス対応（検出16 → 輪郭7）

輪郭モデルが持つのは `anus` / `nipple` / `penis` / `vagina` / `female face` / `male face` /
`pubic hair` の7クラスのみ。対応表は `core/class_mapping.py` が持ります。

| 検出クラス | 輪郭クラス | 備考 |
| --- | --- | --- |
| `FEMALE_GENITALIA_EXPOSED` | `vagina` | |
| `MALE_GENITALIA_EXPOSED` | `penis` | |
| `ANUS_EXPOSED` | `anus` | |
| `FEMALE_BREAST_EXPOSED` | `nipple` | **近似**（乳首のみ。乳房全体は覆わない） |
| `FEMALE_FACE` / `MALE_FACE` | `female face` / `male face` | |
| 上記以外（COVERED 系・BUTTOCKS・ARMPITS・BELLY・FEET・EYE） | — | `mask_source: "none"` |

対応する輪郭が複数ある場合は **bbox の IoU が最大**のものを1対1で割り当てます。

## 設計上の注意

- **マスクは検出 bbox で切り取らない。** 検出モデルと輪郭モデルで bbox が一致しないため、
  輪郭側の形状をそのまま使う（実測では輪郭が検出 bbox より 5〜154px 大きい場合があり、
  切り取ると隠蔽漏れになる）。
- 輪郭の bbox クロップはプロトタイプ格子（320）で量子化するため、seg 自身の bbox に対しても
  最大 1 格子（入力座標で約 4px）外へ張り出すことがある（安全側の誤差）。
- `--input-size` / `--refiner-input-size` は、**静的入力**のモデルでは省略時に ONNX から取得する。
  動的入力のモデルでは**明示指定が必須**（省略するとエラー）。静的入力のモデルで指定値が
  食い違う場合は、推論前に明確なエラーで停止する。
- 統合 CLI `maskveil` は `--refiner-input-size` を持たないため、動的入力の輪郭モデルには未対応。
- 座標復元は軸別倍率（`scale_x` / `scale_y`）で行い、letterbox の往復誤差を出さない。
- NMS は **class-aware**（同一 bbox でもクラスが違えば残す）。隠蔽漏れを避けるため、
  モデル作者実装のクラス非依存 NMS とは意図的に異なる。
- `core` は `comfy` / `torch` / `PIL` / `cv2` を import しない（`tests/test_imports.py` が AST で検証）。
  リサイズは numpy のバイリニア実装。

## 性能（実測 / CPU, onnxruntime 1.31.0）

| 処理 | 時間 | 備考 |
| --- | --- | --- |
| `detect`（2560×3840, CLI全体） | 0.52〜0.55 s | 推論のみ約 0.09 s |
| `refine`（全体1回, 12検出） | 4.79〜4.98 s | 輪郭推論 約 2.9 s（検出数に依存しない） |
| `refine --seg-crop`（1検出） | 4.37〜4.46 s | 検出ごとに推論するため検出数に比例 |
| `censor`（2560×3840, 検出のみ） | 0.74〜1.00 s/枚 | 検出＋隠蔽＋PNG保存。100枚で約1.5分 |
| `censor`（同, `--refine`） | 約 4〜5 s/枚 | 輪郭推論の支配的コストが加算される |

**参考値であり保証しません。** 計測は CPU 推論（GPU 不使用）の開発環境、onnxruntime 1.31.0 時点の実測です。
実行環境・モデル・画像サイズで変動します。

## ノード名の方針

**パッケージ名は汎用（`maskveil`）、ノード名はモデル契約ごと（`Hotscreen*`）** という2層の命名です。

- `HotscreenDetector` / `HotscreenRefiner` は **hotscreen モデルのクラス契約（16クラス／7クラス）に紐づく**ため、
  ノード名を変更しません。既存のワークフローが壊れないことを優先しています。
- **別のモデルに対応するときは、既存ノードを改名せず新しいノードクラスを追加します。**
  これにより、モデルを増やしても既存ワークフローは影響を受けません。
- カテゴリ文字列 `hotscreen` と `web/hotscreen_classes.js` も同じ理由で維持しています。
- **ただし現時点では、別モデルの追加に `core/` の変更も必要です。** クラス名の一覧
  （`core/labels.py`）と検出→輪郭の対応表（`core/class_mapping.py`）がグローバル定数のため、
  新しいモデルを足すときは「ノードの追加」に加えてクラス表の追加が要ります
  （既存ノードの挙動は変えません）。

## ライセンス

| 対象 | ライセンス | 配布 |
| --- | --- | --- |
| **本パッケージのコード**（`core` / `nodes` / `tools` / `web`） | **MIT** | する |
| **検出モデル**（`hs-real-anime-y11n-640-fp32.onnx`） | **AGPL-3.0**（`yolo-07-2025/LICENSE`） | **しない**（利用者が取得し、絶対パスで指定） |
| **輪郭モデル**（`nsfw-anime-xl-x1280.onnx`） | **MIT**（`01miku/anime-nsfw-segm-yolo26` のモデルカード） | **しない**（同上） |
| `ultralytics`（学習・ONNX 書き出しに使われるライブラリ） | AGPL-3.0 | **しない**（import もしない） |

本パッケージが MIT である根拠は、**モデルを同梱・再配布せず、AGPL ライブラリを import・同梱しない**ことです。
モデルのライセンスは**モデルごとに異なり**（検出=AGPL-3.0 / 輪郭=MIT、詳細は[入手元とライセンス](#入手元とライセンス)）、
いずれも再配布しないため本パッケージの MIT には影響しません。
モデルを同梱した時点でこの前提は崩れるため、リポジトリにも配布物にも `*.onnx` を含めません
（`tests/test_license.py` が機械検証）。

## 開発

```bash
git clone https://github.com/purple-ocean-ego/maskveil.git
cd maskveil
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

pytest
ruff check .
ruff format --check .
```

- **clone 先のディレクトリ名は `maskveil` にしてください。** `tests/conftest.py` は親ディレクトリを
  `sys.path` へ追加して `import maskveil` を通すため、別名だと収集に失敗します
  （その場合は `pip install -e .` で解決できます）。
- `tests/js/check_classes.ts` はクラス選択 UI の検証で、**deno** が必要です（pytest 対象外）。
  JS を実際に import して `parseSelection` / `serializeSelection` を実行します。
- `tests/test_comfy_loader.py` は ComfyUI と同じ手順（マングル名・`sys.path` に親を入れない）で
  パッケージを読み込みます。`nodes/` が絶対 import に戻るとここで落ちます。

## 制約

- 動作確認している輪郭 ONNX は、`ultralytics 8.4.189` で `.pt` から書き出した**静的入力（1280）**版です（経路B）。
  HF 配布元の ONNX（経路A）は**動的入力**のため、CLI では入力サイズの明示が必要で、統合 CLI の `--refine` は使えません。
- 輪郭が対応しない部位（臀部・脇・腹・足・眼・COVERED 系）は bbox 形状しか持ちません。
- ComfyUI ノードは GPU 推論を前提にしていません（onnxruntime の CPU 実行で動作します）。
