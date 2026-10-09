"""Refresh generated case runners and documentation without copying raw logs."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "metadata/manifest.json"
    manifest = {item["path"]: item for item in json.loads(manifest_path.read_text(encoding="utf-8"))}
    data = json.loads((root / "metadata/data.json").read_text(encoding="utf-8"))
    changed = []
    variants = [variant for case in data["cases"] for variant in case["variants"]]
    for variant in variants:
        case = root / variant["camodel_case"]["path"]
        runner = case / "run.py"
        shutil.copy2(root / "scripts/run_camodel_case.py", runner)
        changed.append(runner)
        readme = case / "README.txt"
        note = readme.read_text(encoding="utf-8")
        if "source ../../../set_env.sh" not in note:
            note = note.replace("直接复现（Linux/WSL，安装对应 CANN）：\n",
                                "直接复现（Linux/WSL，安装对应 CANN）：\n"
                                "  source ../../../set_env.sh  # 首次使用前编辑结果根目录中的环境配置\n")
        note = note.replace("python3 run.py --toolkit <CANN目录> --out-dir", "python3 run.py --out-dir")
        readme.write_text(note, encoding="utf-8")
        changed.append(readme)
    changed.extend(root / name for name in (
        "set_env.sh", "scripts/run_camodel_case.py", "scripts/bundle_camodel_cases.py",
        "scripts/sync_case_runners.py", "scripts/validate_archive.py", "使用说明.txt",
    ))
    for path in changed:
        name = str(path.relative_to(root))
        entry = manifest.setdefault(name, dict(path=name))
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(list(manifest.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {len(variants)} case runners and {len(changed)} file checksums")


if __name__ == "__main__":
    main()
