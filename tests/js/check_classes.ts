// クラス選択 UI の挙動検証ハーネス
//
// `web/hotscreen_classes.js` は ComfyUI の `../../scripts/app.js` を import するため、
// 実行時にだけ `scripts/app.js` のスタブを持つ一時ディレクトリへ複製して読み込む。
//
// 実行（deno が必要。pytest の対象外）:
//   deno run -A tests/js/check_classes.ts
//
// 検証内容は STRING の解釈・生成と、旧ハーネス（使い捨て）の
// golden ケースを引き継いだもの。

const here = new URL(".", import.meta.url);
const source = await Deno.readTextFile(
  new URL("../../web/hotscreen_classes.js", here),
);

const tmp = await Deno.makeTempDir({ prefix: "hotscreen-js-" });
await Deno.mkdir(`${tmp}/scripts`, { recursive: true });
await Deno.mkdir(`${tmp}/web`, { recursive: true });
await Deno.writeTextFile(
  `${tmp}/scripts/app.js`,
  "export const app = { registerExtension() {} };\n",
);
await Deno.writeTextFile(`${tmp}/web/hotscreen_classes.js`, source);

const mod = await import(`file://${tmp}/web/hotscreen_classes.js`);
const { DETECTION_LABELS, REFINER_LABELS, parseSelection, serializeSelection } =
  mod;

let failures = 0;
function check(name: string, actual: unknown, expected: unknown) {
  const left = JSON.stringify(actual);
  const right = JSON.stringify(expected);
  const ok = left === right;
  if (!ok) failures += 1;
  console.log(
    `${ok ? "PASS" : "FAIL"} ${name}: ${left}${
      ok ? "" : ` (expected ${right})`
    }`,
  );
}

const names = (set: Set<string>) => [...set];

check("detector label count", DETECTION_LABELS.length, 16);
check("refiner label count", REFINER_LABELS.length, 6);
check(
  "all selects 16",
  names(parseSelection("all", DETECTION_LABELS)).length,
  16,
);
check(
  "all selects 6 for refiner",
  names(parseSelection("all", REFINER_LABELS)).length,
  6,
);
check("empty selects nothing", names(parseSelection("", DETECTION_LABELS)), []);
check(
  "mixed name + id",
  names(parseSelection("FEMALE_FACE,11", DETECTION_LABELS)),
  ["FEMALE_FACE", "MALE_GENITALIA_EXPOSED"],
);
check(
  "case insensitive",
  names(parseSelection("female_face", DETECTION_LABELS)),
  ["FEMALE_FACE"],
);
check(
  "unknown name ignored",
  names(parseSelection("NOT_A_CLASS", DETECTION_LABELS)),
  [],
);
check(
  "out of range id ignored",
  names(parseSelection("99", DETECTION_LABELS)),
  [],
);
check(
  "refiner resolves id 11 via detection vocabulary",
  names(parseSelection("11", REFINER_LABELS)),
  ["MALE_GENITALIA_EXPOSED"],
);
check(
  "refiner drops non-mapped class (EYE)",
  names(parseSelection("EYE", REFINER_LABELS)),
  [],
);
check(
  "duplicate tokens collapse",
  names(parseSelection("EYE,EYE", DETECTION_LABELS)),
  ["EYE"],
);
check(
  "serialize all",
  serializeSelection(new Set(DETECTION_LABELS), DETECTION_LABELS),
  "all",
);
check(
  "serialize keeps definition order",
  serializeSelection(new Set(["EYE", "FEMALE_FACE"]), DETECTION_LABELS),
  "FEMALE_FACE,EYE",
);
check(
  "round trip default",
  serializeSelection(
    parseSelection(
      "FEMALE_GENITALIA_EXPOSED,MALE_GENITALIA_EXPOSED",
      DETECTION_LABELS,
    ),
    DETECTION_LABELS,
  ),
  "FEMALE_GENITALIA_EXPOSED,MALE_GENITALIA_EXPOSED",
);
check(
  "round trip all",
  serializeSelection(parseSelection("all", DETECTION_LABELS), DETECTION_LABELS),
  "all",
);

await Deno.remove(tmp, { recursive: true });
console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURES`);
if (failures > 0) Deno.exit(1);
