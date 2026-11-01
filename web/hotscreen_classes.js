// maskveil — クラス選択チェックボックス
//
// STRING widget が唯一の真実であり、この JS は**表示と入力の手段**にすぎない。
// JS が無効でも STRING を直接編集すれば同じ結果になる。
//
// ラベル配列は Python から読めないため二重管理になる。
// `tests/test_web_labels.py` が本ファイルを解析して `core` と一致することを検証する。

import { app } from "../../scripts/app.js";

// core/labels.py の LABELS と同一順序（16件）
const DETECTION_LABELS = [
  "FEMALE_FACE",
  "MALE_FACE",
  "FEMALE_GENITALIA_COVERED",
  "FEMALE_GENITALIA_EXPOSED",
  "BUTTOCKS_COVERED",
  "BUTTOCKS_EXPOSED",
  "FEMALE_BREAST_COVERED",
  "FEMALE_BREAST_EXPOSED",
  "MALE_BREAST_EXPOSED",
  "ARMPITS_EXPOSED",
  "BELLY_EXPOSED",
  "MALE_GENITALIA_EXPOSED",
  "ANUS_EXPOSED",
  "FEET_COVERED",
  "FEET_EXPOSED",
  "EYE",
];

// core/class_mapping.py の DET_TO_SEG に対応のある検出クラス（6件・LABELS の定義順）
const REFINER_LABELS = [
  "FEMALE_FACE",
  "MALE_FACE",
  "FEMALE_GENITALIA_EXPOSED",
  "FEMALE_BREAST_EXPOSED",
  "MALE_GENITALIA_EXPOSED",
  "ANUS_EXPOSED",
];

const NODE_SETTINGS = {
  HotscreenDetector: { widget: "classes", labels: DETECTION_LABELS },
  HotscreenRefiner: { widget: "refiner_classes", labels: REFINER_LABELS },
};

/** STRING をクラス名の集合へ解釈する（`core.labels.resolve_classes` と同じ受理形式）。 */
function parseSelection(value, labels, vocabulary = DETECTION_LABELS) {
  const text = (value ?? "").toString().trim();
  const selected = new Set();
  if (!text) return selected;
  if (text.toLowerCase() === "all") {
    labels.forEach((label) => selected.add(label));
    return selected;
  }

  // 名前・ID は「検出16クラス語彙」で解決してから、このノードの表示対象へ絞る
  // （Refiner でも `11` = MALE_GENITALIA_EXPOSED が解決できる）
  const byUpper = new Map(
    vocabulary.map((label) => [label.toUpperCase(), label]),
  );
  text.split(",").forEach((token) => {
    const item = token.trim();
    if (!item) return;
    const upper = item.toUpperCase();
    let resolved = byUpper.get(upper);
    if (!resolved && /^\d+$/.test(item)) {
      resolved = vocabulary[Number(item)];
    }
    if (resolved && labels.includes(resolved)) selected.add(resolved);
  });
  return selected;
}

/** 選択状態を STRING へ戻す（順序は labels の定義順、全選択は `all`）。 */
function serializeSelection(selected, labels) {
  const names = labels.filter((label) => selected.has(label));
  if (names.length === labels.length) return "all";
  return names.join(",");
}

function setupNode(node, settings) {
  const widget = node.widgets?.find((item) => item.name === settings.widget);
  if (!widget) return;

  const state = { syncing: false };

  const container = document.createElement("div");
  // 16件でもノードの箱からはみ出さないよう、折り返し＋行間を詰める。
  // 実際の高さは描画後に測ってノードの高さへ反映する（measureAndFit）。
  container.style.cssText =
    "display:flex;flex-wrap:wrap;gap:0 10px;padding:2px 4px;font-size:10px;line-height:14px;";

  const boxes = settings.labels.map((label) => {
    const row = document.createElement("label");
    row.style.cssText =
      "display:flex;align-items:center;gap:3px;white-space:nowrap;";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.label = label;
    input.addEventListener("change", () => writeBack());
    row.appendChild(input);
    row.appendChild(document.createTextNode(label));
    container.appendChild(row);
    return { label, input };
  });

  const originalCallback = widget.callback;

  function applyValue(value) {
    if (state.syncing) return;
    const selected = parseSelection(value ?? widget.value, settings.labels);
    boxes.forEach(({ label, input }) => {
      input.checked = selected.has(label);
    });
  }

  function writeBack() {
    const selected = new Set(
      boxes.filter((box) => box.input.checked).map((box) => box.label),
    );
    const value = serializeSelection(selected, settings.labels);
    state.syncing = true;
    widget.value = value;
    try {
      originalCallback?.call(widget, value);
    } finally {
      state.syncing = false;
    }
    node.setDirtyCanvas?.(true, true);
  }

  // STRING を直接編集した場合もチェック状態を再 parse する
  widget.callback = function (value, ...rest) {
    const result = originalCallback?.apply(this, [value, ...rest]);
    applyValue(value ?? widget.value);
    return result;
  };

  state.sync = () => applyValue(widget.value);
  node.__hotscreenClasses = state;

  if (typeof node.addDOMWidget === "function") {
    const domWidget = node.addDOMWidget("hotscreen_classes", "div", container, {
      serialize: false,
      hideOnZoom: false,
    });
    let widgetHeight = 56;

    // 折り返し後の実高さを測り、DOM widget とノード本体の高さへ反映する
    // （measure しないと 16件のチェックボックスがノードの箱からはみ出す）
    function measureAndFit() {
      widgetHeight = Math.max(container.scrollHeight, 20) + 6;
      if (
        typeof node.computeSize === "function" &&
        typeof node.setSize === "function"
      ) {
        const computed = node.computeSize();
        node.setSize([Math.max(computed[0], 320), Math.max(computed[1], 90)]);
      }
      node.setDirtyCanvas?.(true, true);
    }

    domWidget.computeSize = function (width) {
      return [width ?? node.size?.[0] ?? 320, widgetHeight];
    };

    // DOM がレイアウトされた後に測る（onNodeCreated 時点では高さが 0 のため）
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(measureAndFit);
    } else {
      measureAndFit();
    }
    state.fit = measureAndFit;
  } else {
    // 古いフロントエンド向けフォールバック（トグル widget で表現する）
    settings.labels.forEach((label) => {
      node.addWidget("toggle", label, false, (value) => {
        const box = boxes.find((item) => item.label === label);
        if (box) box.input.checked = Boolean(value);
        writeBack();
      });
    });
  }

  applyValue(widget.value);
}

app.registerExtension({
  name: "hotscreen.classes",
  beforeRegisterNodeDef(nodeType, nodeData) {
    const settings = NODE_SETTINGS[nodeData.name];
    if (!settings) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setupNode(this, settings);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.__hotscreenClasses?.sync?.();
      // ワークフロー読込時は保存されたノードサイズが復元されるため、高さを測り直す
      this.__hotscreenClasses?.fit?.();
      return result;
    };
  },
});

// ComfyUI は副作用 import で読み込むため、追加の export は動作に影響しない。
// 単体検証（deno）から STRING の解釈・生成を直接呼べるようにしておく。
export { DETECTION_LABELS, parseSelection, REFINER_LABELS, serializeSelection };
