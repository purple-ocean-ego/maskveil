"""パッケージ全体で共有する定数（マジックナンバーの一元化）。"""

from __future__ import annotations

DEFAULT_CONF = 0.2
"""スコア閾値の既定値（モデル作者の推奨値）。"""

DEFAULT_IOU = 0.45
"""NMS の IoU 閾値の既定値（モデル作者の推奨値）。"""

DEFAULT_BLOCK_SIZE = 48
"""モザイクのブロック辺の既定値（px）。

既定ワークフロー（`workflows/censor-nsfw.json`）と CLI（`tools/censor.py`）で
同一値を使うため、PO 決定（2026-10-31）により 32 から 48 へ統一した。
"""

DEFAULT_MASK_EXPAND = 0
"""マスク形状の膨張量の既定値（px）。

モザイクはブロック量子化により実効的な外側余白（最大 `DEFAULT_BLOCK_SIZE - 1` px）を
持つため、追加の膨張は不要と判断した（PO 決定 2026-10-31、4 → 0）。
`blur` / `black` は境界が厳密なので、必要なら明示的に広げる。
"""

DET_INPUT_SIZE = 640
"""検出モデルの入力一辺。"""

SEG_INPUT_SIZE = 1280
"""輪郭モデルの入力一辺（静的入力）。"""

LETTERBOX_PAD_VALUE = 114
"""letterbox のパディング埋め値（YOLO 系の慣例）。"""

MASK_CHANNELS = 32
"""輪郭モデルのプロトタイプ係数次元。"""

MASK_SOURCE_NONE = "none"
MASK_SOURCE_MODEL_SEG = "model_seg"
MASK_SOURCE_BBOX_ELLIPSE = "bbox_ellipse"

MASK_SOURCES: tuple[str, ...] = (
    MASK_SOURCE_MODEL_SEG,
    MASK_SOURCE_BBOX_ELLIPSE,
)
"""CLI で選択できるマスク生成方法（`none` は「形状なし」を表す出力値）。"""
