"""Bundle PC-ordered CCE representations of the recorded hardware VF streams."""
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path


def modeled(op):
    return op == "RV_SMEM_BAR" or op.startswith(("RV_PLD", "RV_PST")) or (op.startswith("RV_V") and not op.startswith(("RV_VLOOP", "RV_VAG")))


def decode_log(path):
    pattern = re.compile(r"\(PC: (0x[0-9a-f]+)\).*?\(Binary: (0x[0-9a-f]+)\).*?\(ID:\s*(\d+)\)\s+(RV_\w+)(.*)")
    rows = {}
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            pc, binary, iid, op, detail = match.groups()
            rows[int(iid)] = dict(pc=pc, binary=binary, opcode=op, detail=detail)
    static = {}
    occurrences = defaultdict(list)
    for iid, row in sorted(rows.items()):
        static.setdefault(row["pc"], row)
        occurrences[row["pc"]].append(row)
    return sorted(static.values(), key=lambda r: int(r["pc"], 16)), occurrences


def field(detail, name):
    match = re.search(r"\b" + name + r"\[(\d+)\](?:=0x([0-9a-f]+))?", detail)
    return (int(match[1]), int(match[2], 16) if match[2] else None) if match else None


def scalar_bits(detail, name):
    value = field(detail, name)[1]
    upper = field(detail, name + "2")
    return value | ((upper[1] << 16) if upper else 0)


def literal(bits):
    value = struct.unpack("<f", struct.pack("<I", bits & 0xffffffff))[0]
    text = format(value, ".9g")
    return text + ("f" if "." in text or "e" in text else ".0f")


def memory_address(row):
    detail = row["detail"]
    base = scalar_bits(detail, "Sn")
    immediate = re.search(r"#offset=(\d+)", detail)
    if immediate:
        offset = int(immediate[1])
        offset = (offset if offset < 128 else offset - 256) * 32
    else:
        offset = scalar_bits(detail, "Sm")
    post = "#p=1" in detail
    return (base - offset if post else base + offset), offset, post


def gelu_hardware(rows, occurrences):
    loop_index = next(i for i, row in enumerate(rows) if row["opcode"] == "RV_VLOOPv2")
    loop = rows[loop_index]
    count = scalar_bits(loop["detail"], "Sn")
    size = int(re.search(r"instr_num\s*:\s*(\d+)", loop["detail"])[1])
    assert count == 8 and size == 192
    prefix, body = rows[:loop_index], rows[loop_index + 1:loop_index + 1 + size]
    assert len(body) == 192
    pointers = []
    mapping = []

    def translate(row):
        op, detail, pc = row["opcode"], row["detail"], row["pc"]
        register = lambda name: ("p" if name.startswith("P") else "r") + str(field(detail, name)[0])
        memory = op.startswith(("RV_VLD", "RV_VST", "RV_PLD", "RV_PST"))
        addresses = None
        if memory:
            address, increment, post = memory_address(row)
            addresses = [memory_address(item)[0] for item in occurrences[pc]]
            if post:
                assert all(value == address + i * increment for i, value in enumerate(addresses)), pc
                if address < 0x4000:
                    origin, name = 0, "dy"
                elif address < 0x8000:
                    origin, name = 0x4000, "x"
                elif address < 0xc000:
                    origin, name = 0x8000, "out"
                else:
                    raise ValueError((pc, address))
                assert (address - origin) % 4 == increment % 4 == 0
                pointer = "ptr_" + pc[2:]
                pointers.append(f"    __ubuf__ float *{pointer} = {name} + {(address - origin) // 4};")
                if op.startswith("RV_VLD"):
                    code = f"vlds({register('Vd')}, {pointer}, {increment // 4}, NORM, POST_UPDATE);"
                else:
                    code = f"vsts({register('Vd')}, {pointer}, {increment // 4}, NORM_B32, {register('Pg')}, POST_UPDATE);"
            else:
                assert len(set(addresses)) == 1, pc
                offset = address - 0x40000
                if op.startswith("RV_PLD"):
                    code = f"plds({register('Pd')}, spill, {offset}, NORM);"
                elif op.startswith("RV_PST"):
                    code = f"psts({register('Pd')}, spill, {offset}, NORM);"
                elif op.startswith("RV_VLD"):
                    assert offset % 4 == 0
                    code = f"vlds({register('Vd')}, (__ubuf__ float *)spill, {offset // 4}, NORM);"
                else:
                    code = f"vsts((vector_u8 &){register('Vd')}, (__ubuf__ uint8_t *)spill, {offset}, NORM_B8, {register('Pg')});"
        elif op == "RV_PSET":
            assert re.search(r"#pattern=(\d+)", detail)[1] == "0"
            width = re.search(r"Dtype:\s*B(\d+)", detail)[1]
            code = f"{register('Pd')} = pset_b{width}(PAT_ALL);"
        elif op == "RV_VDUPS":
            code = f"vdup({register('Vd')}, {literal(scalar_bits(detail, 'Sn'))}, {register('Pg')}, MODE_ZEROING);"
        elif op == "RV_VMOV":
            code = f"vmov({register('Vd')}, {register('Vn')});"
        elif op in {"RV_VAXPY", "RV_VADDS"}:
            code = f"{op[3:].lower()}({register('Vd')}, {register('Vn')}, {literal(scalar_bits(detail, 'Sm'))}, {register('Pg')}, MODE_ZEROING);"
        elif op == "RV_VEXP":
            code = f"vexp({register('Vd')}, {register('Vn')}, {register('Pg')}, MODE_ZEROING);"
        elif op == "RV_VCMP_EQ":
            code = f"vcmp_eq({register('Pd')}, {register('Vn')}, {register('Vm')}, {register('Pg')});"
        elif op == "RV_VSEL":
            code = f"vsel({register('Vd')}, {register('Vn')}, {register('Vm')}, {register('Pg')});"
        elif op in {"RV_VMUL", "RV_VDIV", "RV_VADD"}:
            code = f"{op[3:].lower()}({register('Vd')}, {register('Vn')}, {register('Vm')}, {register('Pg')}, MODE_ZEROING);"
        elif op == "RV_SMEM_BAR":
            code = f"mem_bar({detail.split(':')[-1].strip()});"
        elif not modeled(op):
            return None
        else:
            raise ValueError((pc, op))
        mapping.append({"pc": pc, "opcode": op, "binary": row["binary"], "cce": code,
                        "dynamic_count": len(occurrences[pc]), "byte_addresses": addresses})
        return f"// PC {pc}: {op}\n" + code

    before = [code for row in prefix if (code := translate(row)) is not None]
    inner = [code for row in body if (code := translate(row)) is not None]
    lines = ["__attribute__((always_inline)) inline [aicore] void hardware_equivalent_vf(",
             "    __ubuf__ float *dy, __ubuf__ float *x, __ubuf__ float *out,",
             "    __ubuf__ uint32_t *spill) {", "    // 原日志 UB 起点：dy=0，x=0x4000，out=0x8000，spill=0x40000。",
             *pointers, "    __VEC_SCOPE__ {",
             "        vector_bool " + ", ".join(f"p{i}" for i in range(8)) + ";",
             "        vector_f32 " + ", ".join(f"r{i}" for i in range(32)) + ";"]
    for text in before:
        lines.extend("        " + part for part in text.splitlines())
    lines.extend([f"        // PC {loop['pc']}: RV_VLOOPv2，8 路已展开，硬件重复 8 次。",
                  "        #pragma unroll(1)", "        for (int iteration = 0; iteration < 8; ++iteration) {"])
    for text in inner:
        lines.extend("            " + part for part in text.splitlines())
    lines.extend(["        }", "    }", "}"])
    return "\n".join(lines) + "\n", mapping


def bundle(root):
    data = json.loads((root / "data.json").read_text())
    generated = []
    for case in data["cases"]:
        for v in case["variants"]:
            if not v["vector_store"] and not v["predicate_store"]:
                continue
            folder = root / case["id"] / v["id"]
            rows, occurrences = decode_log(folder / "camodel/core0.veccore0.instr_popped_log.dump")
            if case["id"] == "gelu_grad":
                code, mapping = gelu_hardware(rows, occurrences)
            else:
                original = folder / "source/compiled_pc_replay.cce"
                code = original.read_text()
                static = {r["pc"]: r for r in rows}
                annotations = re.findall(r"// PC (0x[0-9a-f]+):\s*(RV_\w+)", code)
                mapping = []
                for pc, op in annotations:
                    assert static[pc]["opcode"] == op, (case["id"], pc)
                    mapping.append({"pc": pc, "opcode": op, "binary": static[pc]["binary"],
                                    "dynamic_count": len(occurrences[pc])})
            covered = {row["pc"] for row in mapping}
            ordered_pcs = [int(row["pc"], 16) for row in mapping]
            assert len(mapping) == len(covered) and ordered_pcs == sorted(ordered_pcs)
            expected = {row["pc"] for row in rows if modeled(row["opcode"])}
            optional = {row["pc"] for row in rows if row["opcode"] == "RV_PSET"}
            assert expected <= covered <= expected | optional, (case["id"], v["id"], expected - covered, covered - expected - optional)
            counts = Counter()
            for item in mapping:
                counts[item["opcode"]] += item["dynamic_count"]
            assert counts["RV_SMEM_BAR"] == v["membars"]
            header = (f"// {case['title']} {v['label']}：实际底层硬件执行指令流的等效 CCE。\n"
                      "// 依据同目录 CAModel instr_popped_log / instr_log 按静态 PC 还原。\n"
                      "// 已显式补入编译器因寄存器溢出生成的保存、重载、mem_bar；保持硬件顺序与循环次数。\n"
                      f"// 动态 vector spill：写 {v['vector_store']} / 读 {v['vector_load']}；"
                      f"predicate spill：写 {v['predicate_store']} / 读 {v['predicate_load']}；mem_bar {v['membars']}。\n"
                      "// 这是硬件 VF 的计算/UB 访存/同步指令流的等价表达；标量地址效果折入指针和立即数，\n"
                      "// 不表达标量流水时序或 GM wrapper。再次编译时编译器仍可能重新分配寄存器。\n")
            target = folder / "source/hardware_equivalent.cce"
            target.write_text(header + code)
            audit = {"basis": "对应原始 CAModel 的实际底层 VF 指令流", "camodel_cycles": v["camodel"],
                     "modeled_pc_coverage_complete": True, "dynamic_instruction_counts": dict(counts),
                     "unrepresented_opcodes": dict(Counter(item["opcode"] for pc, items in occurrences.items()
                         if pc not in covered for item in items)), "pc_mapping": mapping,
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            mapping_path = folder / "source/hardware_pc_mapping.json"
            mapping_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
            link = {"label": "实际硬件执行指令流 · 等效 CCE（含 spill）", "path": str(target.relative_to(root))}
            v["sources"] = [link] + [s for s in v["sources"] if not s["path"].endswith("/hardware_equivalent.cce")]
            for s in v["sources"]:
                if s["path"].endswith("/kernel.cce"):
                    s["label"] = "显式 spill 的可编译实验源码" if v["id"] == "u8_explicit" else "编译前实验源码"
                elif s["path"].endswith("/compiled_pc_replay.cce"):
                    s["label"] = "原预测使用的 PC 回放输入"
            v["hardware_equivalent"] = {"path": str(target.relative_to(root)), "mapping": str(mapping_path.relative_to(root)),
                                         "pc_coverage": True, "camodel_cycles": v["camodel"]}
            if str(mapping_path.relative_to(root)) not in v["evidence"]:
                v["evidence"].append(str(mapping_path.relative_to(root)))
            (folder / "result.json").write_text(json.dumps(v, ensure_ascii=False, indent=2))
            generated.append({"case": case["title"], "variant": v["label"], "file": str(target.relative_to(root)),
                              "membars": v["membars"], "covered_pcs": len(covered)})
        (root / case["id"] / "summary.json").write_text(json.dumps(case, ensure_ascii=False, indent=2))
    (root / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (root / "data.js").write_text("window.MEMBAR_REPORT=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    (root / "hardware_equivalent_index.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2))
    print(json.dumps(generated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    bundle(parser.parse_args().root)
