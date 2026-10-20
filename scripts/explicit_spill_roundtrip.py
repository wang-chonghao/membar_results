"""Recompile PC-reconstructed spill CCEs, retaining the original GM/host contract."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from prepare_hardware_equivalent import decode_log, field, literal, memory_address, modeled, scalar_bits


ROOT = Path(__file__).resolve().parents[1]
CASES = [("adam_apply_one", "u6"), ("adam_apply_one", "u8"),
         ("gelu_poly", "u12"), ("swiglu_grad", "u6"),
         ("swiglu_grad", "u8"), ("gelu_grad", "u8")]
LOG = "core0.veccore0.instr_popped_log.dump"


def relevant(op):
    return modeled(op) or op == "RV_PSET"


def affine(values):
    step = values[1] - values[0] if len(values) > 1 else 0
    if values != [values[0] + step * i for i in range(len(values))]:
        raise ValueError("Non-affine dynamic address; cannot reconstruct a counted loop")
    return values[0], step


def address(row):
    detail = row["detail"]
    if field(detail, "Am"):
        return (scalar_bits(detail, "Sn") + field(detail, "Am")[1]) & 0xffffffff, 0, False
    effective, offset, post = memory_address(row)
    if offset & 0x80000000 and offset >= 0:
        offset -= 0x100000000
    return effective & 0xffffffff, offset, post


def reconstruct(log, buffer_origins=(), ssa_body=False):
    rows, occurrences = decode_log(log)
    loops = [r for r in rows if r["opcode"] == "RV_VLOOPv2"]
    if len(loops) != 1:
        raise ValueError("Expected exactly one original hardware loop")
    loop = loops[0]
    # VLOOP packs unrelated fields in adjacent scalar halves; Sn2 is not count[31:16].
    count = field(loop["detail"], "Sn")[1] & 0xffff
    start = int(loop["pc"], 16) + 4
    stop = start + 4 * int(re.search(r"instr_num\s*:\s*(\d+)", loop["detail"])[1])
    pointers = {}
    mapping = []

    def translate(row):
        op, detail, pc = row["opcode"], row["detail"], row["pc"]
        reg = lambda name: ("p" if name.startswith("P") else "r") + str(field(detail, name)[0])
        if op.startswith(("RV_VLD", "RV_VST", "RV_PLD", "RV_PST")):
            addresses = [address(r)[0] for r in occurrences[pc]]
            initial, step = affine(addresses)
            # Preserve the hardware access width/mask, not just the copied FP32 values.
            pred = op.startswith(("RV_PLD", "RV_PST"))
            load = op.startswith(("RV_VLD", "RV_PLD"))
            dist = re.search(r"dist:(\w+)", detail)[1].replace("NORAML", "NORM")
            # NORM_B8 is a hardware distribution mode, independent of the source
            # vector's C++ element type. Keeping FP32 avoids extra bitcast lives.
            byte_mode = pred
            ctype, width = ("unsigned char", 1) if byte_mode else ("float", 4)
            if initial % width or step % width:
                raise ValueError((pc, "Unaligned address", initial, step))
            _, increment, post = address(row)
            if post:
                pointer = "ptr_" + pc[2:]
                pointer_base = initial
                if step != increment and len(addresses) > 1:
                    raise ValueError((pc, "POST_UPDATE increment mismatch"))
                offset = str(increment // width)
                tail = ", POST_UPDATE"
            else:
                pointer_base = 0x40000 if initial >= 0x30000 else max(b for b in buffer_origins if b <= initial)
                pointer = f"base_{pointer_base:x}_{width}"
                offset = str((initial - pointer_base) // width)
                if step:
                    offset += f" + {step // width} * iteration"
                tail = ""
            pointers[pointer] = f"    __ubuf__ {ctype} *{pointer} = (__ubuf__ {ctype} *)get_imm({pointer_base:#x});"
            if pred:
                # Predicate memory intrinsics require an integer pointer, byte offsets.
                call = f"{'plds' if load else 'psts'}({reg('Pd')}, (__ubuf__ unsigned int *){pointer}, {offset}, NORM);"
            elif load:
                if dist not in ("NORM", "BRC_B32"):
                    raise ValueError((pc, "Unsupported load mode", dist))
                call = f"vlds({reg('Vd')}, {pointer}, {offset}, {dist}{tail});"
            else:
                if dist not in ("NORM_B8", "NORM_B32"):
                    raise ValueError((pc, "Unsupported store mode", dist))
                source = f"(vector_u8 &){reg('Vd')}" if byte_mode else reg("Vd")
                call = f"vsts({source}, {pointer}, {offset}, {dist}, {reg('Pg')}{tail});"
        elif op == "RV_PSET":
            width = re.search(r"Dtype:\s*B(\d+)", detail)[1]
            if re.search(r"#pattern=(\d+)", detail)[1] != "0":
                raise ValueError("Only measured PAT_ALL PSET is supported")
            call = f"{reg('Pd')} = pset_b{width}(PAT_ALL);"
        elif op == "RV_SMEM_BAR":
            call = f"mem_bar({detail.split(':')[-1].strip()});"
        elif op == "RV_VDUPS":
            call = f"vdup({reg('Vd')}, {literal(scalar_bits(detail, 'Sn'))}, {reg('Pg')}, MODE_ZEROING);"
        elif op == "RV_VMOV":
            call = f"vmov({reg('Vd')}, {reg('Vn')});"
        else:
            name = op.removeprefix("RV_").lower()
            binary = {"vadd", "vsub", "vmul", "vdiv", "vmax", "vmin", "vcmp_eq"}
            scalar = {"vadds", "vsubs", "vmuls", "vmins", "vmaxs", "vaxpy"}
            if name in binary:
                args = [reg("Pd" if name.startswith("vcmp") else "Vd"), reg("Vn"), reg("Vm"), reg("Pg")]
            elif name in scalar:
                args = [reg("Vd"), reg("Vn"), literal(scalar_bits(detail, "Sm")), reg("Pg")]
            elif name in ("vsqrt", "vexp"):
                args = [reg("Vd"), reg("Vn"), reg("Pg")]
            elif name == "vsel":
                args = [reg("Vd"), reg("Vn"), reg("Vm"), reg("Pg")]
            else:
                raise ValueError((pc, "Unsupported opcode", op))
            call = f"{name}({', '.join(args)});"
        mapping.append(dict(original_pc=pc, opcode=op, dynamic_count=len(occurrences[pc]), cce=call))
        return f"            {call} // original PC {pc}: {op}"

    prefix, body, suffix = [], [], []
    for row in rows:
        if not relevant(row["opcode"]):
            continue
        pc = int(row["pc"], 16)
        target = prefix if pc < start else body if pc < stop else suffix
        if start <= pc < stop and len(occurrences[row["pc"]]) != count:
            raise ValueError((row["pc"], "Dynamic count differs from loop count"))
        target.append(translate(row))
    if ssa_body:
        bindings = {}
        rewritten = []
        for line in body:
            call, comment = line.split(" // original PC ", 1)
            match = re.search(r"(\w+)\((.*)\);", call)
            op, operands = match.groups()
            output = re.match(r"\s*(r\d+)\s*,", operands)
            destination = output[1] if output and op not in ("vsts", "psts", "mem_bar") else None
            resolved = re.sub(r"\br\d+\b", lambda m: bindings.get(m[0], m[0]), operands)
            if destination:
                if op == "vaxpy":
                    raise ValueError("SSA body rewriting of read/write operands is not implemented")
                name = "value_" + comment.split(":")[0].removeprefix("0x")
                resolved = name + resolved[resolved.index(","):]
                rewritten.append(f"            vector_f32 {name};")
                bindings[destination] = name
            rewritten.append(f"            {op}({resolved}); // original PC {comment}")
        if suffix:
            raise ValueError("SSA body trial requires all results consumed within the loop")
        body = rewritten
    code = "\n".join([
        "// Explicit spill reconstruction; byte-copy/predicate modes retained.",
        "__attribute__((always_inline)) inline [aicore] void explicit_spill_vf() {",
        *pointers.values(), "    __VEC_SCOPE__ {",
        "        vector_bool " + ", ".join(f"p{i}" for i in range(8)) + ";",
        "        vector_f32 " + ", ".join(f"r{i}" for i in range(32)) + ";",
        *prefix, "        #pragma unroll(1)",
        f"        for (uint16_t iteration = 0; iteration < uint16_t({count}); ++iteration) {{",
        *body, "        }", *suffix, "    }", "}", ""])
    return code, mapping


def prepare(case, variant, ssa_body=False):
    base = ROOT / case / variant
    source_case = base / "camodel_case"
    target = base / "explicit_spill_camodel_case"
    if (target / "comparison.json").exists():
        return target
    config = json.loads((source_case / "case.json").read_text())
    original = (source_case / config["kernel_source"]).read_text()
    origins = [int(x, 0) for x in re.findall(r"get_imm\((0x[\da-fA-F]+|\d+)\)", original)]
    vf, mapping = reconstruct(base / "camodel" / LOG, origins, ssa_body)
    # Keep the original launch arguments, DMA, input/output layout and host golden.
    wrapper = original[original.index('extern "C"'):]
    name = {"adam_apply_one": "adam_fused_vf", "gelu_poly": "gelu_simd_ub",
            "swiglu_grad": "swiglu_grad_vf", "gelu_grad": "gelu_grad_vf"}[case]
    wrapper, substitutions = re.subn(r"\b" + name + r"\([^;]*\);", "explicit_spill_vf();", wrapper)
    if substitutions != 1:
        raise ValueError((case, "Missing or ambiguous VF call"))
    (target / "source").mkdir(parents=True, exist_ok=True)
    (target / "run_output").mkdir(exist_ok=True)
    (target / "source/kernel.cce").write_text("#define __aicore__ [aicore]\n\n" + vf + wrapper)
    (target / "source/original_pc_mapping.json").write_text(json.dumps(mapping, indent=2))
    for filename in ("host.cpp", "run.py"):
        shutil.copyfile(source_case / filename, target / filename)
    for path in (source_case / "run_output").glob("*.bin"):
        shutil.copyfile(path, target / "run_output" / path.name)
    config.update(kernel_source="source/kernel.cce", expected_vf_cycles=None,
                  explicit_spill=True, baseline_case="../camodel_case",
                  source_generation_policy=dict(ssa_body=ssa_body, loop_index="uint16_t",
                                                normalize_ub_address_bits=32),
                  baseline_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  rerun_performed=False)
    (target / "case.json").write_text(json.dumps(config, indent=2))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted({c for c, _ in CASES}))
    parser.add_argument("--variant")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip-run-existing", action="store_true")
    parser.add_argument("--ssa-body", action="store_true")
    parser.add_argument("--toolkit", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/membar-explicit-spill"))
    args = parser.parse_args()
    failed = []
    for case, variant in CASES:
        if args.case and case != args.case or args.variant and variant != args.variant:
            continue
        if args.run and args.skip_run_existing and (args.work_dir / case / variant).exists():
            continue
        target = prepare(case, variant, args.ssa_body)
        print("Prepared", target.relative_to(ROOT), flush=True)
        if args.run:
            if args.skip_run_existing and (args.work_dir / case / variant).exists():
                continue
            command = [sys.executable, str(target / "run.py"), "--out-dir",
                       str(args.work_dir / case / variant)]
            if args.toolkit:
                command.extend(["--toolkit", str(args.toolkit)])
            compiled_source = (target / "source/kernel.cce").read_bytes()
            result = subprocess.run(command)
            output = args.work_dir / case / variant
            if output.is_dir():
                (output / "source_used.cce").write_bytes(compiled_source)
            if result.returncode:
                failed.append(f"{case}/{variant}")
                print(f"FAILED {case}/{variant}: inspect compile.log/camodel.log", flush=True)
    if failed:
        raise SystemExit("Failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
