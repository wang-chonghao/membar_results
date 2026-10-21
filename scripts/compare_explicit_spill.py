"""Compare recompiled explicit CCE against archived compiler-spill hardware logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from explicit_spill_roundtrip import CASES, LOG, ROOT, address, relevant
from prepare_hardware_equivalent import field, scalar_bits
from package_explicit_spill_cases import package_case


ALIASES = {"RV_VLD": "VLDS", "RV_VLDI": "VLDS", "RV_VLDS": "VLDS",
           "RV_VST": "VSTS", "RV_VSTI": "VSTS", "RV_VSTS": "VSTS",
           "RV_PLDI": "PLDS", "RV_PLDS": "PLDS", "RV_PLD": "PLDS",
           "RV_PSTI": "PSTS", "RV_PSTS": "PSTS", "RV_PST": "PSTS"}


def dynamic_rows(path):
    # decode_log sorts each PC's events by original instruction ID. Merge by IDs
    # directly instead: popped timestamps are not architectural stream order.
    pattern = re.compile(r"\(PC: (0x[0-9a-f]+)\).*?\(Binary: (0x[0-9a-f]+)\).*?\(ID:\s*(\d+)\)\s+(RV_\w+)(.*)")
    rows = {}
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            pc, binary, iid, op, detail = match.groups()
            rows[int(iid)] = dict(pc=pc, binary=binary, opcode=op, detail=detail)
    return [rows[i] for i in sorted(rows)]


def semantic_trace(rows):
    values = {}
    result = []
    for row in rows:
        op, detail = row["opcode"], row["detail"]
        if not relevant(op):
            continue

        def key(name):
            item = field(detail, name)
            return (name[0], item[0]) if item else None

        def read(name):
            resource = key(name)
            if resource not in values:
                raise ValueError((row["pc"], op, "Undefined register", resource))
            return values[resource]

        if op == "RV_PSET":
            values[key("Pd")] = ("PAT", re.search(r"Dtype:\s*(\w+)", detail)[1],
                                 int(re.search(r"#pattern=(\d+)", detail)[1]))
            continue
        if op == "RV_VMOV":
            values[key("Vd")] = read("Vn")
            continue
        signature = dict(op=ALIASES.get(op, op.removeprefix("RV_")), inputs=[], attributes={})
        dst = None
        if op.startswith(("RV_VLD", "RV_VST", "RV_PLD", "RV_PST")):
            load = op.startswith(("RV_VLD", "RV_PLD"))
            pred = op.startswith(("RV_PLD", "RV_PST"))
            operand = "Pd" if pred else "Vd"
            if load:
                dst = key(operand)
            else:
                signature["inputs"].append(read(operand))
            if field(detail, "Pg"):
                signature["inputs"].append(read("Pg"))
            signature["attributes"] = dict(byte_address=address(row)[0],
                mode=re.search(r"dist:(\w+)", detail)[1].replace("NORAML", "NORM"))
        elif op == "RV_SMEM_BAR":
            signature["attributes"]["type"] = detail.split(":")[-1].strip()
        elif op == "RV_VDUPS":
            dst = key("Vd")
            signature["attributes"]["bits"] = scalar_bits(detail, "Sn")
            signature["inputs"].append(read("Pg"))
        else:
            dst = key("Pd") if op.startswith("RV_VCMP") else key("Vd")
            if op == "RV_VAXPY":
                signature["inputs"].append(read("Vd"))
            for name in ("Vn", "Vm", "Pg"):
                if field(detail, name):
                    signature["inputs"].append(read(name))
            if field(detail, "Sm"):
                signature["attributes"]["bits"] = scalar_bits(detail, "Sm")
            if dst is None:
                raise ValueError((row["pc"], "Unhandled modeled instruction", op))
        if dst:
            values[dst] = ("def", len(result))
        result.append(dict(pc=row["pc"], signature=signature))
    return result


def metrics(rows):
    counts = Counter(r["opcode"] for r in rows if relevant(r["opcode"]))
    spill = Counter()
    for row in rows:
        op = row["opcode"]
        if op.startswith(("RV_VLD", "RV_VST", "RV_PLD", "RV_PST")) and address(row)[0] >= 0x30000:
            kind = ("predicate_" if op.startswith(("RV_PLD", "RV_PST")) else "vector_")
            kind += "load" if op.startswith(("RV_VLD", "RV_PLD")) else "store"
            spill[kind] += 1
    return dict(opcode_counts=dict(counts), spill_counts=dict(spill),
                scalar_spill_counts=dict(Counter(r["opcode"] for r in rows
                    if r["opcode"] in ("RV_SSTI", "RV_SLDI", "RV_SSTS", "RV_SLDS"))))


def cycles(path):
    found = re.findall(r"vf_execute_time:\s*(\d+)", path.read_text())
    if len(found) != 1:
        raise ValueError((path, "Expected one VF", found))
    return int(found[0])


def compare(case, variant, output, archive=False):
    base = ROOT / case / variant
    target = base / "explicit_spill_camodel_case"
    old_rows = dynamic_rows(base / "camodel" / LOG)
    new_rows = dynamic_rows(output / LOG)
    old, new = semantic_trace(old_rows), semantic_trace(new_rows)
    differences = [dict(index=i, original=a, recompiled=b)
        for i, (a, b) in enumerate(zip(old, new)) if a["signature"] != b["signature"]]
    config = json.loads((target / "case.json").read_text())
    checks = []
    for _, filename in config["golden_pairs"]:
        original = base / "camodel_case/run_output" / filename
        actual = output / filename
        checks.append(dict(output=filename, byte_equal=original.read_bytes() == actual.read_bytes(),
                           original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                           explicit_sha256=hashlib.sha256(actual.read_bytes()).hexdigest()))
    original_cycles = cycles(base / "camodel/core0.veccore0.instr_log.dump")
    explicit_cycles = cycles(output / "core0.veccore0.instr_log.dump")
    old_metrics, new_metrics = metrics(old_rows), metrics(new_rows)
    result = dict(case=case, variant=variant, original_cycles=original_cycles,
        explicit_cycles=explicit_cycles, cycle_delta=explicit_cycles - original_cycles,
        golden=json.loads((output / "validation.json").read_text()), output_checks=checks,
        original=old_metrics, recompiled=new_metrics,
        original_semantic_events=len(old), recompiled_semantic_events=len(new),
        normalized_dynamic_dataflow_and_addresses_equal=not differences and len(old) == len(new),
        difference_count=len(differences) + abs(len(old) - len(new)), first_differences=differences[:12],
        comparison_policy="All dynamic vector/predicate dataflow, scalar constants, effective UB byte addresses, memory modes and barrier order; transparent VMOV and constant PAT_ALL PSET normalized; scalar instruction timing and exact machine encoding excluded.")
    result["source_sha256"] = hashlib.sha256((target / "source/kernel.cce").read_bytes()).hexdigest()
    result["log_sha256"] = {"original": hashlib.sha256((base / "camodel" / LOG).read_bytes()).hexdigest(),
                            "explicit": hashlib.sha256((output / LOG).read_bytes()).hexdigest()}
    if archive:
        archived = target / "run_output"
        source_snapshot = output / "source_used.cce"
        if not source_snapshot.exists() or source_snapshot.read_bytes() != (target / "source/kernel.cce").read_bytes():
            raise ValueError("Missing/mismatched actual compilation source snapshot")
        for path in output.iterdir():
            if path.name in (LOG, "core0.veccore0.instr_log.dump", "core0.veccore0.rvec.IDU.dump",
                             "core0.veccore0.rvec.EXU.dump", "core0.veccore0.rvec.LSU.dump",
                             "compile.log", "host_build.log", "link.log", "camodel.log", "commands.json",
                             "validation.json", "source_used.cce", "kernel.o") or path.suffix == ".bin":
                shutil.copyfile(path, archived / path.name)
        (target / "comparison.json").write_text(json.dumps(result, indent=2))
        config.update(rerun_performed=True, expected_vf_cycles=explicit_cycles)
        (target / "case.json").write_text(json.dumps(config, indent=2))
        (target / "README.md").write_text(
            "# 显式 Spill CCE 重编译验证\n\n"
            "`source/kernel.cce` 是完整可编译内核，保留原 host、GM/UB 布局和输入。"
            "每条 VF 指令标注原日志 PC。原用例在相邻 `camodel_case/`，未修改。\n\n"
            "这是对应固定 shape/unroll 的硬件流回放，不是通用算法实现；标量地址效果已折入访存表达式，"
            "没有逐条重放原来的标量指令。\n\n"
            f"原版 / 显式版：{original_cycles} / {explicit_cycles} cycles。"
            f"完整动态数据流、地址、模式及屏障序列一致：{result['normalized_dynamic_dataflow_and_addresses_equal']}。"
            f"输出逐字节一致：{all(c['byte_equal'] for c in checks)}。\n\n"
            "核查不要求物理寄存器编号或立即数/寄存器寻址编码相同，透明 VMOV 和全真 PSET 归一化；"
            "不把周期接近或 golden 通过作为指令流一致的证明。详细差异见 `comparison.json`。\n\n"
            "在仓库根目录配置并运行：\n\n```bash\nsource ./set_env.sh\n"
            f"python3 {case}/{variant}/explicit_spill_camodel_case/run.py "
            f"--out-dir /tmp/{case}_{variant}_explicit_rerun\n```\n\n"
            "新输出目录必须为空，不能覆盖归档目录。仅保留主要日志、输入输出和运行命令，不归档 VCD 等大文件。\n")
        package_case(ROOT, case, variant)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, action="append")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    results = []
    for case, variant in CASES:
        output = next((directory / case / variant for directory in reversed(args.work_dir)
                       if (directory / case / variant / "validation.json").exists()), None)
        if output is None:
            print("Not validated yet:", case, variant)
            continue
        result = compare(case, variant, output, args.archive)
        results.append(result)
        print(case, variant, result["original_cycles"], result["explicit_cycles"],
              "trace_equal=", result["normalized_dynamic_dataflow_and_addresses_equal"],
              "diffs=", result["difference_count"], flush=True)
    if args.archive:
        (ROOT / "metadata/explicit_spill_roundtrip.json").write_text(json.dumps(results, indent=2))
        lines = ["# 显式 Spill CCE 重编译验证", "", "日期：2026-10-20。A5 / Ascend950PR_9599，CANN 9.0.0-beta.1，O2、vec-misched=0。", "",
            "原 CCE 不动；在每个版本下增加 `explicit_spill_camodel_case/`，包含完整 kernel、host、运行配置、PC 对照、输出及两份主要 CAModel 日志。", "",
            "| Case | 原版 cycles | 显式版 cycles | 差值 | Vector 写/读 | Predicate 写/读 | 原版/显式 Membar | 动态流核对 | 数值校验 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for r in results:
            c, v = r["case"], r["variant"]
            a, b = r["original"], r["recompiled"]
            counts = a["spill_counts"]
            memory = lambda stem: f"{counts.get(stem + '_store', 0)} / {counts.get(stem + '_load', 0)}"
            lines.append(f"| [{c} {v}]({c}/{v}/explicit_spill_camodel_case/source/kernel.cce) | {r['original_cycles']} | "
                f"{r['explicit_cycles']} | {r['cycle_delta']:+d} | {memory('vector')} | {memory('predicate')} | "
                f"{a['opcode_counts'].get('RV_SMEM_BAR', 0)} / {b['opcode_counts'].get('RV_SMEM_BAR', 0)} | "
                f"{'一致' if r['normalized_dynamic_dataflow_and_addresses_equal'] else '不一致，不能认定等效'} | "
                f"{'golden 通过，原输出逐字节相同' if all(c['byte_equal'] for c in r['output_checks']) else 'golden 通过，非逐字节相同'} |")
        pending = [(c, v) for c, v in CASES if not any(r["case"] == c and r["variant"] == v for r in results)]
        if pending:
            lines += ["", "## 未完成项", "",
                "本轮尚未通过重编译验证：" + "、".join(f"{c} {v}" for c, v in pending) + "。不能把这些版本视为等效实现。",
                "AdamApplyOne U8 的重建源码会被 CCEC 再次插入 EPS spill，覆盖原显式 spill 槽，导致 golden 失败；"
                "已尝试共享基址、16 位循环、补码偏移、移除 bitcast、SSA 临时变量和寄存器分配开关，尚未解决。"
                "失败源码不作为可用等效用例发布；原 U8 及其旧结果不变。"]
        lines += ["", "## 核对口径", "",
            "- 不是只数指令：比较整个 VF 所有动态迭代的计算 RAW、谓词依赖、VAXPY 旧 destination、标量常数位模式、实际 UB 字节地址、访存 mode 与 Membar 顺序。",
            "- 寄存器重编号、VLD/VLDS/VLDI 与 VST/VSTS/VSTI 的寻址编码、透明 VMOV 和相同 PAT_ALL PSET 归一化；B8 spill 拷贝不改成 B32 模式，PSTS/PLDS 保留真实谓词搬运。",
            "- 表中写/读是整个 VF 的动态 spill 指令次数，不是寄存器个数；根据本组 UB 布局的高地址 spill 区分类，不能当作任意 kernel 的通用识别规则。",
            "- **数据流/地址等效不等于周期或机器编码逐位一致**。地址表达式的重新物化、标量初始化/标量 spill 和编译器分配仍可能变化；不改模型参数，不覆盖旧 CAModel/VfSim 记录。GeLU Poly 原版含标量 spill，该部分只保留地址效果，不逐条重放时序。",
            "- 每个 `comparison.json` 提供原/新动态 opcode 数量、spill 数量、源码和日志 SHA256、数值检查及差异。失败试跑不作为等效结果归档。", "",
            "## 复现", "", "在结果仓库根目录修改 `set_env.sh` 的 CANN 路径后：", "", "```bash", "source ./set_env.sh",
            "python3 adam_apply_one/u6/explicit_spill_camodel_case/run.py --out-dir /tmp/adam_u6_explicit_check", "```", "",
            "其他版本替换路径即可，输出目录必须为空。精度通过后，将新目录交给核查脚本：", "", "```bash",
            "python3 scripts/compare_explicit_spill.py --work-dir /tmp/your_runs --require-all", "```", "",
            "`--work-dir` 下按 `<case>/<uN>/` 放运行目录，可以重复指定多个工作目录；`--require-all` 要求本轮六组均已通过数值和动态流核对。",
            "生成工具为 `scripts/explicit_spill_roundtrip.py`，只接受本轮已验证的单层、固定循环、仿射地址模式；未覆盖 opcode/模式直接报错。",
            "归档不包含全量 VCD 或 VfSim history 大文件。"]
        (ROOT / "explicit_spill_roundtrip.md").write_text("\n".join(lines) + "\n")
    if args.require_all and (len(results) != len(CASES) or
            any(not r["normalized_dynamic_dataflow_and_addresses_equal"] or
                not all(c["byte_equal"] for c in r["output_checks"]) for r in results)):
        raise SystemExit("Not all six cases passed full round-trip validation")


if __name__ == "__main__":
    main()
