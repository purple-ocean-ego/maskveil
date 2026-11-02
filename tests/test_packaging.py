"""配布範囲の整合（公開物に何が入るか）を検証する。

公開範囲は次の4箇所で制御されている。ここでは**互いに矛盾していないこと**を機械検証する。

- `pyproject.toml` の `[tool.setuptools] packages`（wheel に入る Python パッケージ）
- `MANIFEST.in`（sdist に入るファイル）
- `.comfyignore`（ComfyUI へインストールされるファイル）
- `requirements.txt`（ComfyUI 経路で導入される依存）

新しいファイルやディレクトリを足したときに「どこにも登録されないまま配布される」
「必要なのに配布から漏れる」のどちらも起きないようにするのが目的。
"""

from __future__ import annotations

import fnmatch
import pathlib

import pytest
from _toml import tomllib

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

# ComfyUI へ配る（＝`.comfyignore` で除外してはいけない）もの
RUNTIME_ENTRIES = {
    "__init__.py",
    "core",
    "nodes",
    "tools",
    "web",
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
}

# 配布しない（＝リポジトリ運用のための）もの。`.comfyignore` に書かれているべき
NON_RUNTIME_ENTRIES = {
    ".gitignore",
    ".comfyignore",
    ".session",
    ".local",
    "assets",
    "tests",
    "MANIFEST.in",
    "uv.lock",
}


def _comfyignore_patterns() -> list[str]:
    path = PACKAGE / ".comfyignore"
    assert path.is_file(), ".comfyignore が無い（ComfyUI へ tests 等が配布されてしまう）"
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _is_ignored(entry: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            # ディレクトリ指定（`tests/`）は存在しないエントリにも一致させる
            if fnmatch.fnmatch(entry, pattern.rstrip("/")):
                return True
        elif fnmatch.fnmatch(entry, pattern):
            return True
    return False


def _normalized_requirements() -> set[str]:
    names = set()
    for line in (PACKAGE / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        names.add(line.split(">=")[0].split("==")[0].split("[")[0].strip().lower())
    return names


def test_comfyignore_keeps_runtime_entries() -> None:
    """
    ユースケース: ComfyUI にインストールしたパッケージが実際に動く
    検証意図: 実行時に必要なファイル・ディレクトリが `.comfyignore` で除外されていない
    """
    patterns = _comfyignore_patterns()
    leaked = sorted(entry for entry in RUNTIME_ENTRIES if _is_ignored(entry, patterns))
    assert leaked == [], f"ComfyUI へ配布されない実行時必須エントリ: {leaked}"


def test_comfyignore_excludes_non_runtime_entries() -> None:
    """
    ユースケース: ComfyUI の custom_nodes を汚さない
    検証意図: テスト・運用資材が `.comfyignore` で除外されている
    """
    patterns = _comfyignore_patterns()
    missing = sorted(entry for entry in NON_RUNTIME_ENTRIES if not _is_ignored(entry, patterns))
    assert missing == [], f".comfyignore に除外設定が無い: {missing}"


def test_every_top_level_entry_is_classified() -> None:
    """
    ユースケース: 将来ファイルを追加した人が配布範囲を決め忘れない
    検証意図: リポジトリ直下の全エントリが「配る」「配らない」のどちらかに分類されている
    """
    unclassified = []
    for entry in sorted(path.name for path in PACKAGE.iterdir()):
        if entry in {".git"}:
            continue
        if entry in RUNTIME_ENTRIES or entry in NON_RUNTIME_ENTRIES:
            continue
        if _is_ignored(entry, _comfyignore_patterns()):
            continue
        unclassified.append(entry)
    assert unclassified == [], (
        "配布範囲が未定義のエントリがある。"
        f" RUNTIME_ENTRIES / NON_RUNTIME_ENTRIES か .comfyignore に登録すること: {unclassified}"
    )


def test_setuptools_packages_cover_all_subpackages() -> None:
    """
    ユースケース: wheel に全サブパッケージが入る
    検証意図: `[tool.setuptools] packages` が実在する全サブパッケージを列挙している
    """
    data = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(data["tool"]["setuptools"]["packages"])

    actual = {"maskveil"}
    for directory in PACKAGE.iterdir():
        if directory.is_dir() and (directory / "__init__.py").is_file():
            if directory.name.startswith("."):
                continue
            actual.add(f"maskveil.{directory.name}")

    assert actual <= declared, f"packages に載っていないサブパッケージ: {sorted(actual - declared)}"


def test_package_dir_mapping_covers_declared_packages() -> None:
    """
    ユースケース: パッケージ名と実ディレクトリの対応が壊れない
    検証意図: `packages` の各要素に `package-dir` の対応があり、実ディレクトリが存在する
    """
    data = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    mapping = data["tool"]["setuptools"]["package-dir"]
    for package in data["tool"]["setuptools"]["packages"]:
        assert package in mapping, f"{package} の package-dir が無い"
        assert (PACKAGE / mapping[package]).is_dir(), f"{package} の実ディレクトリが無い"


def test_requirements_txt_matches_project_dependencies() -> None:
    """
    ユースケース: ComfyUI 経路（requirements.txt）と PyPI 経路（pyproject）で依存がずれない
    検証意図: 両者の依存名の集合が一致する（片方だけ更新して公開する事故を防ぐ）
    """
    data = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        dependency.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        for dependency in data["project"]["dependencies"]
    }
    assert _normalized_requirements() == declared, (
        "requirements.txt と pyproject.toml の依存が一致しない"
        f"（requirements={sorted(_normalized_requirements())} / pyproject={sorted(declared)}）"
    )


def test_requirements_do_not_include_torch() -> None:
    """
    ユースケース: ComfyUI の環境を壊さない
    検証意図: `torch` を requirements.txt に足していない（ComfyUI が提供する）
    """
    assert not any(name.startswith("torch") for name in _normalized_requirements())


@pytest.mark.parametrize("entry", ["tests", "assets", ".github"])
def test_manifest_prunes_non_runtime(entry: str) -> None:
    """
    ユースケース: sdist にテストや運用資材を入れない
    検証意図: `MANIFEST.in` が配布不要なディレクトリを prune している
    """
    manifest = (PACKAGE / "MANIFEST.in").read_text(encoding="utf-8")
    assert f"prune {entry}" in manifest
