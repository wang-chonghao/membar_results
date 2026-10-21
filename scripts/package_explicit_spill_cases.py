"""Make validated explicit-spill cases runnable without their parent archive."""
import argparse
import json
from pathlib import Path
import shutil


def package_case(root, case, variant):
    folder = root / case / variant / "explicit_spill_camodel_case"
    config = json.loads((folder / "case.json").read_text())
    comparison = json.loads((folder / "comparison.json").read_text())
    if (not comparison["golden"]["passed"]
            or not comparison["normalized_dynamic_dataflow_and_addresses_equal"]
            or not comparison["output_checks"]
            or not all(item["byte_equal"] for item in comparison["output_checks"])):
        raise ValueError(f"Cannot package unverified case: {case}/{variant}")
    for name in (config["kernel_source"], "host.cpp", "run.py",
                 "source/original_pc_mapping.json", "run_output/validation.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)

    shutil.copyfile(root / "set_env.sh", folder / "set_env.sh")
    (folder / "run.sh").write_text(
        '#!/usr/bin/env bash\n'
        'set -e\n'
        'case_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        'source "${case_dir}/set_env.sh"\n'
        'exec python3 "${case_dir}/run.py" "$@"\n', encoding="utf-8")
    evidence = folder / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "summary.json").write_text(json.dumps(dict(
        case=case, variant=variant, kernel_name=config["kernel_name"],
        host_args=config["host_args"], host_defines=config["host_defines"],
        ccec_flags=config["ccec_flags"], original_cycles=comparison["original_cycles"],
        explicit_cycles=comparison["explicit_cycles"],
        golden=comparison["golden"], output_checks=comparison["output_checks"],
        dynamic_dataflow_and_addresses_equal=True,
        comparison_file="../comparison.json",
        historical_commands="../run_output/commands.json",
        environment_file="../set_env.sh",
    ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / "README.txt").write_text(
        f"{case} / {variant.upper()} 显式 Spill 完整 CAModel 用例\n\n"
        "本目录可以单独复制到新机器，不需要相邻的原版用例、上层 set_env.sh 或 VfSimulator。\n"
        "依赖 Linux/WSL、Bash、Python 3.11+，以及带匹配模拟器的 CANN。\n"
        "原验证环境：CANN 9.0.0-beta.1 / Ascend950PR_9599 / dav-c310-vec。\n\n"
        "目录内容\n"
        "  source/kernel.cce：完整内核，含 VF、GM/UB 搬运、同步及可调用的 kernel 入口。\n"
        "  source/original_pc_mapping.json：与原硬件日志的 PC 对应关系。\n"
        "  host.cpp：设备初始化、内存分配、输入构造、kernel 调用、golden 计算及精度校验。\n"
        "  case.json：kernel 名称、host 参数、编译选项、输出文件和精度容限。\n"
        "  set_env.sh：本用例的独立环境配置；换机器时在此修改 CANN 路径等四项配置。\n"
        "  run.py / run.sh：编译 host 和 CCE、链接、执行 CAModel、检查 golden。\n"
        "  comparison.json / evidence/summary.json：原版与显式版的核验结果和运行配置。\n"
        "  run_output/：实际编译源码快照、命令、构建日志、输入/golden/输出、kernel.o 和主要 CAModel 日志。\n\n"
        "运行方法（先进入本目录）\n"
        "  1. 编辑本目录 set_env.sh 的 MEMBAR_CANN_ROOT、MEMBAR_HOST_ARCH、\n"
        "     MEMBAR_NPU_TYPE、MEMBAR_CORE_ARCH。额外 include/lib 和工具覆盖通常留空。\n"
        "  2. 加载环境并运行：\n"
        "       source ./set_env.sh\n"
        "       python3 run.py --dry-run\n"
        f"       python3 run.py --out-dir /tmp/{case}_{variant}_explicit_rerun\n"
        "     或由脚本加载本地环境后运行：\n"
        f"       bash ./run.sh --out-dir /tmp/{case}_{variant}_explicit_rerun\n"
        "  两种方式任选一种；每次必须换用为空或尚不存在的输出目录，且位于本归档目录之外。\n"
        "  精度通过时终端显示 passed: true（JSON 格式）。\n\n"
        "只检查归档数值（不需要 CANN，也无需 source 环境）\n"
        "  python3 run.py --check-archived\n\n"
        f"原版 / 显式版 VF：{comparison['original_cycles']} / {comparison['explicit_cycles']} cycles。\n"
        "已核对完整动态计算依赖、谓词依赖、UB 字节地址、mode 和屏障顺序，输出逐字节相同。\n"
        "不保证机器编码、标量指令时序或周期完全相同；重新编译后仍须校验，不能仅凭周期接近判定等效。\n"
        "历史 run_output/commands.json 中的绝对路径仅用于追溯；运行入口使用当前用例位置。\n"
        "不复制带历史绝对路径的 original_scripts，也不归档 VCD 或 VfSim history 大文件。\n",
        encoding="utf-8")
    readme = folder / "README.md"
    if readme.exists() and "## 单目录独立运行" not in readme.read_text(encoding="utf-8"):
        with readme.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## 单目录独立运行\n\n"
                "可只复制本目录到新机器。完整文件清单和配置步骤见 [README.txt](README.txt)。"
                "编辑本目录 `set_env.sh` 后，在本目录执行：\n\n"
                f"```bash\nbash ./run.sh --out-dir /tmp/{case}_{variant}_explicit_rerun\n```\n")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    results = json.loads((args.root / "metadata/explicit_spill_roundtrip.json").read_text())
    for result in results:
        print(package_case(args.root, result["case"], result["variant"]))


if __name__ == "__main__":
    main()
