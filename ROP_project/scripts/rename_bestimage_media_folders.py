"""
bestimage_validation 配下の各caseフォルダで、
  画像 -> images
  動画 -> movie
へ統一するための一括リネーム/統合スクリプト。

- Windowsの日本語フォルダ名でも安全に動くようにPythonで処理します。
- 既にターゲット側(images/movie)が存在する場合は中身をマージします。
  - 同名衝突が発生した場合は *_dupN を付けて退避します。

実行例（ropenv）:
  .\\ropenv\\Scripts\\python.exe ROP_project\\scripts\\rename_bestimage_media_folders.py
"""

from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path("ROP_project/bestimage_validation")


def merge_move(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                merge_move(item, target)
                try:
                    item.rmdir()
                except OSError:
                    pass
            else:
                stem = target.stem
                suf = target.suffix
                k = 1
                while True:
                    alt = dst / f"{stem}_dup{k}{suf}"
                    if not alt.exists():
                        shutil.move(str(item), str(alt))
                        print(f"[WARN] collision: moved {item} -> {alt}")
                        break
                    k += 1
        else:
            shutil.move(str(item), str(target))


def rename_or_merge(case_dir: Path, jp: str, en: str) -> None:
    src = case_dir / jp
    dst = case_dir / en
    if not src.exists() or not src.is_dir():
        return

    if dst.exists() and dst.is_dir():
        merge_move(src, dst)
        shutil.rmtree(src)
        print(f"[OK] {case_dir.name}: merged {jp} -> {en}")
    else:
        src.rename(dst)
        print(f"[OK] {case_dir.name}: {jp} -> {en}")


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Not found: {ROOT.resolve()}")

    case_dirs = [p for p in ROOT.iterdir() if p.is_dir() and p.name.isdigit()]
    for case in sorted(case_dirs, key=lambda p: p.name):
        rename_or_merge(case, "画像", "images")
        rename_or_merge(case, "動画", "movie")

    print("[DONE]")


if __name__ == "__main__":
    main()



