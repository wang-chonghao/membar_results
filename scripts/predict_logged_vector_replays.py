"""Replay archived vector/predicate execution; scalar timing is explicitly excluded.

This experiment never recompiles CCE. Effective addresses come from the original
CAModel log, including scalar restores and POST_UPDATE effects.
"""
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

CASES = {
    "history-adam_apply_one-u8": "adam_apply_one/u8",
    "history-flash_attention_grad_front-u2_concentrated_spill": "flash_attention_grad_front/u2_concentrated_spill",
}


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def scalar_spill(event):
    return event.op.startswith(("RV_SST", "RV_SLD"))


def audit_scalar_operands(events):
    """Freezing scalar arithmetic inputs is allowed only if each PC is constant."""
    observed = defaultdict(set)
    for event in events:
        if not event.op.startswith("RV_V") or event.op.startswith(("RV_VLD", "RV_VST", "RV_VLOOP")):
            continue
        item = event.scalar("Sm")
        if item:
            high = event.scalar("Sm2")
            observed[event.pc].add(item[1] | ((high[1] if high else 0) << 16))
    if any(len(values) != 1 for values in observed.values()):
        raise ValueError("A vector instruction has a changing scalar operand; cannot freeze it")
    return {hex(pc): next(iter(values)) for pc, values in observed.items()}


def freeze_addresses(vf, events):
    from api.common.schema import AffineExpression, CanonicalInstruction, CanonicalLoop, CanonicalMembar
    from multisoc_log_replay import affine_fit, expanded_instructions, memory_address, operation
    expanded = list(expanded_instructions(vf.context))
    observed = [e for e in events if operation(e) is not None]
    if len(expanded) != len(observed):
        raise ValueError("Dynamic stream length differs from the original projected log")
    samples, geometries = defaultdict(list), {}
    addresses = []
    for (node, indices), event in zip(expanded, observed):
        if isinstance(node, CanonicalMembar):
            if operation(event) != "MEMBAR" or node.barrier != event.detail.split("type:", 1)[1].strip():
                raise ValueError("Barrier sequence changed")
            continue
        if node.attributes["rv_pc"] != hex(event.pc) or node.attributes["rv_opcode"] != event.op:
            raise ValueError("Dynamic PC/opcode order differs from the original log")
        memory = [o for o in (*node.inputs, *node.outputs) if o.memory_access is not None]
        if not memory:
            continue
        if len(memory) != 1:
            raise ValueError("Expected one memory operand in this replay")
        operand = memory[0]
        unit = operand.memory_access.address_unit_bytes or {
            "fp32": 4, "fp16": 2, "uint8": 1, "uint32": 4,
        }[vf.values[operand.value_id].dtype]
        address = memory_address(event)
        if address % unit:
            raise ValueError("Byte address is not expressible in the operand units")
        samples[node.instruction_id].append((indices, address // unit))
        geometries[node.instruction_id] = unit
        addresses.append(dict(dynamic_id=event.ident, pc=hex(event.pc), opcode=event.op,
                              byte_start=address, span_bytes=operand.memory_access.span * unit
                              if operand.memory_access.span is not None else None))
    expressions = {key: affine_fit(values, [v for v, _ in values[0][0]]) for key, values in samples.items()}

    def visit(nodes):
        result = []
        for node in nodes:
            if isinstance(node, CanonicalLoop):
                node = replace(node, body=tuple(visit(node.body)))
            elif isinstance(node, CanonicalInstruction) and node.instruction_id in expressions:
                def update(operand):
                    if operand.memory_access is None:
                        return operand
                    return replace(operand, memory_access=replace(operand.memory_access,
                        offset=expressions[node.instruction_id], address_state_id=None,
                        update_mode="none", post_update_delta_bytes=None))
                node = replace(node, inputs=tuple(map(update, node.inputs)), outputs=tuple(map(update, node.outputs)),
                               attributes={**node.attributes, "logged_effective_address_replay": True})
            result.append(node)
        return result

    frozen = replace(vf, context=tuple(visit(vf.context)), source={**vf.source,
        "scalar_spill_policy": "exclude_scalar_timing_use_logged_effective_addresses",
        "post_update_policy": "folded_into_verified_per_PC_affine_addresses"})
    # Verify every inferred affine address, not only the first loop iteration.
    for node, indices in expanded_instructions(frozen.context):
        if not isinstance(node, CanonicalInstruction) or node.instruction_id not in expressions:
            continue
        expr = expressions[node.instruction_id]
        value = expr.constant + sum(t.coefficient * dict(indices)[t.variable_id] for t in expr.terms)
        original = samples[node.instruction_id].pop(0)
        if value != original[1]:
            raise AssertionError("Affine replay does not reproduce the recorded address")
    return frozen, addresses


def diagnose_mhc(root, folder):
    from api.common.validator import validate_canonical_vf_info
    from api.json.serialization import canonical_vf_info_from_dict
    from refresh_multisoc_predictions import migrate_payload
    paths = ["mhc_sinkhorn/m128_element_major_u2/evidence/8_canonical_input.json",
             "spill_candidate_alignment/followup/history-mhc_sinkhorn-m128_element_major_u2/vfsim/canonical_input.json"]
    records = []
    for name in paths:
        payload, _ = migrate_payload(read(root / name))
        validation = validate_canonical_vf_info(canonical_vf_info_from_dict(payload))
        nodes = {}
        def visit(items, path="context"):
            for index, node in enumerate(items):
                key = f"{path}[{index}]"
                nodes[key] = node
                if node["kind"] == "loop":
                    visit(node["body"], key + ".body")
        visit(payload["context"])
        errors = []
        for error in validation.errors:
            node = nodes.get(error.context.get("path"), {})
            errors.append(dict(code=error.code, path=error.context.get("path"),
                opcode=node.get("opcode"), form=node.get("form"),
                compiled_pc=node.get("attributes", {}).get("compiled_pc")))
        normalized = copy.deepcopy(payload)
        fixes = Counter()
        def normalize_forms(items):
            for node in items:
                if node["kind"] == "loop":
                    normalize_forms(node["body"])
                elif node.get("opcode") in {"VGATHER2", "VSCATTER"} and node["form"] == "b32":
                    memory = next(o for o in (*node["inputs"], *node["outputs"]) if o.get("memory_access"))
                    node["form"] = normalized["values"][memory["value_id"]]["dtype"]
                    fixes[node["opcode"]] += 1
        normalize_forms(normalized["context"])
        after = validate_canonical_vf_info(canonical_vf_info_from_dict(normalized))
        records.append(dict(path=name, error_counts=dict(Counter(e["code"] for e in errors)),
                            opcode_counts=dict(Counter(e["opcode"] for e in errors)), errors=errors,
                            diagnostic_only_dtype_form_normalization=dict(fixes),
                            errors_after_dtype_form_normalization=len(after.errors),
                            normalized_payload_used_for_prediction=False,
                            note="Indexed per-lane byte footprints must also be restored; do not predict local dependencies from the old single memory interval."))
    save(folder / "mhc_diagnostics.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="Replace only these two unsupported original predictions")
    args = parser.parse_args()
    root, model = args.root.absolute(), args.model_root.absolute()
    baseline = root / "multisoc_refresh_20261025"
    provenance = read(baseline / "provenance.json")
    commit = subprocess.check_output(["git", "-C", str(model), "rev-parse", "HEAD"], text=True).strip()
    snapshot = None
    if commit != provenance["model_commit"]:
        # Do not switch or reset the user's advancing worktree.
        snapshot = tempfile.TemporaryDirectory(prefix="vfsim-replay-model-")
        archive = subprocess.check_output(["git", "-C", str(model), "archive", provenance["model_commit"], "api", "core", "configs"])
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(snapshot.name, filter="data")
        model = Path(snapshot.name)
        commit = provenance["model_commit"]
        print("Using archived report model", commit, flush=True)
    sys.path.insert(0, str(model))
    from api.json.adapter import CanonicalJsonVfInfoAdapter
    from api.json.serialization import canonical_vf_info_to_dict
    from core.params.loader import load_parameters
    import refresh_multisoc_predictions as refresh
    from multisoc_log_replay import Replay, vf_streams, operation
    refresh.MODEL = model
    for name, expected in provenance["configuration_sha256"].items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model configuration differs: {name}")
    output = root / "logged_vector_replay_20261026"
    diagnose_mhc(root, output)
    parameters = {arch: load_parameters(model, soc=soc) for arch, soc in (("A5", "dav-3510"), ("A6", "dav-9201"))}
    results = read(baseline / "summary.json")
    totals = {}
    for key, relative in CASES.items():
        folder, case = output / key, root / relative
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vfsim-logged-replay-") as temporary:
            view = Path(temporary)
            (view / "cce").symlink_to(case / "camodel", target_is_directory=True)
            streams = vf_streams(view)
        inputs, audits = [], []
        scalar_counts = Counter()
        for index, (_, _, address, words, events) in enumerate(streams):
            scalar = [e for e in events if scalar_spill(e)]
            scalar_counts.update(e.op for e in scalar)
            selected = [e for e in events if not scalar_spill(e)]
            constants = audit_scalar_operands(selected)
            raw = Replay(key, index, case / "camodel/core0.veccore0.instr_popped_log.dump").build(address, words, selected)
            vf, memory = freeze_addresses(raw, selected)
            payload, migrations = refresh.migrate_payload(canonical_vf_info_to_dict(vf))
            vf = CanonicalJsonVfInfoAdapter.from_payload(payload)
            inputs.append(vf)
            save(folder / f"vf_{index:03d}.json", payload)
            save(folder / f"address_audit_{index:03d}.json", memory)
            save(folder / f"scalar_events_{index:03d}.json", [asdict(e) for e in scalar])
            expected = Counter(e.op for e in selected if operation(e) is not None)
            from multisoc_log_replay import expanded_instructions
            actual = Counter(n.attributes["rv_opcode"] if hasattr(n, "attributes") else "RV_SMEM_BAR"
                             for n, _ in expanded_instructions(vf.context))
            if expected != actual:
                raise AssertionError((key, expected, actual))
            audits.append(dict(dynamic_order_matches=True, affine_addresses_match_every_iteration=True,
                               original_modeled_counts=dict(expected), replay_modeled_counts=dict(actual),
                               frozen_compute_scalar_operands=constants, migrations=migrations))
        original_cce = case / "source/hardware_equivalent.cce"
        # This is the existing PC-ordered vector/predicate CCE, not newly compiled code.
        source = ("// LOG-BASED VFSim INPUT ONLY: no recompile or numeric validation in this run.\n"
                  "// Scalar SSTI/SLDI timing is intentionally excluded by user request.\n"
                  "// Prediction uses the attached Canonical JSON; logged byte addresses are audited.\n") + original_cce.read_text()
        (folder / "vector_predicate_replay.cce").write_text(source)
        with gzip.open(folder / "original_dynamic_stream.jsonl.gz", "wt") as stream:
            for index, (_, _, _, _, events) in enumerate(streams):
                for event in events:
                    stream.write(json.dumps(dict(vf_index=index, scalar_spill_excluded=scalar_spill(event), **asdict(event))) + "\n")
        record = copy.deepcopy(results[key])
        for arch in ("A5", "A6"):
            evidence = dict(kind="logged_vector_predicate_projection", scalar_timing_modeled=False,
                scalar_spill_opcodes_excluded=dict(scalar_counts), recompiled=False,
                address_policy="logged effective bytes, per-PC exact affine fit",
                path=str((folder / "audit.json").relative_to(root)),
                cce=str((folder / "vector_predicate_replay.cce").relative_to(root)))
            entry = dict(status="predicted", evidence=evidence, vfs=[])
            for index, vf in enumerate(inputs):
                metadata = refresh.memory_metadata(vf)
                entry["vfs"].append(refresh.run_one(vf, parameters[arch], arch, metadata,
                                                   folder / arch / f"vf_{index:03d}"))
            entry.update(global_cycles=sum(v["global_cycles"] for v in entry["vfs"]),
                         local_cycles=sum(v["local_cycles"] for v in entry["vfs"]))
            if "previous_original_input_diagnostic" not in record["arches"][arch]:
                record["arches"][arch]["previous_original_input_diagnostic"] = record["arches"][arch]["original"]
            record["arches"][arch]["original"] = entry
        save(folder / "audit.json", dict(scalar_timing_modeled=False, scalar_spill_opcodes_excluded=dict(scalar_counts),
            model_commit=commit, user_authorized_scalar_exclusion=True, vf_audits=audits,
            source_sha256=hashlib.sha256(original_cce.read_bytes()).hexdigest(),
            source_logs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (case / "camodel").glob("core0.veccore0.instr*_log.dump")}))
        save(folder / "comparison.json", record)
        totals[key] = record
        if args.publish:
            results[key] = record
            save(baseline / key / "comparison.json", record)
        print(key, {a: (record["arches"][a]["original"]["global_cycles"],
                        record["arches"][a]["original"]["local_cycles"]) for a in ("A5", "A6")}, flush=True)
    save(output / "summary.json", totals)
    lines = ["# 原始日志的向量 / 谓词回放", "",
             f"固定模型提交 `{commit}`，与原汇总表相同。未修改模型资源或时序参数，未重新编译回放 CCE。", "",
             "按用户确认口径：忽略所有标量流水和标量 spill 时序，只预测向量、谓词及保留的 Membar。标量恢复后的指针数值直接从原硬件日志读取，POST_UPDATE 效果折入逐 PC 仿射有效地址，并逐动态访问核对。", "",
             "保留全部向量/谓词 spill、原动态 PC/opcode 顺序和循环结构。全局和局部使用同一输入，只改变同步方式，检查动态顺序一致和 RAW/WAR producer.done+1 边界。", "",
             "预测实际输入是每个 case 的 `vf_000.json`。`vector_predicate_replay.cce` 是既有 PC 对照 CCE 的阅读参考，不是本轮重新编译或验证过的二进制；原始完整指令和排除的标量指令另有记录。", "",
             "| Case | A5 全局 | A5 局部 | A6 全局 | A6 局部 |", "|---|---:|---:|---:|---:|"]
    for key, record in totals.items():
        a5, a6 = (record["arches"][a]["original"] for a in ("A5", "A6"))
        lines.append(f"| {key} | {a5['global_cycles']} | {a5['local_cycles']} | {a6['global_cycles']} | {a6['local_cycles']} |")
    lines += ["", "这些是忽略标量时序后的模型值，不是完整标量流水的精确预测。已在主表和分类统计中标记，并保留此前拒绝预测的诊断。", "",
              "## MHC 为什么仍未补入", "",
              "MHC 128 矩阵 Element-major U2 的原始及软件回放 JSON，各有 34 条 VGATHER2 和 32 条 VSCATTER 使用旧 b32 form，而当前 Catalog 使用具体类型 form。按 memory operand 的实际 dtype 归一化后，66 个校验错误可以消除；`mhc_diagnostics.json` 记录了每条错误及该诊断实验。", "",
              "但不能只替换字符串就直接跑局部预测：gather/scatter 访问的是逐 lane 非连续地址。旧 Canonical 中同一 memory operand 只保留了单个区间，例如首个 gather 的 offset=63、span=1；旁路 `mhc_sinkhorn/m128_element_major_u2/evidence/14_memory_records.json` 实际记录该指令 64 个 4B 地址（0x3400 到 0x34fc）。这份逐 lane 记录必须接入当前 UB 元数据，才能避免漏掉依赖。", "",
              "因此 MHC 仍单列缺失，不是把已有的 gather/scatter 指令判为不支持，也不是简单的 scalar spill 阻碍。", "",
              "## 复现", "", "```bash", "python3 scripts/predict_logged_vector_replays.py --model-root /path/to/MultiSoC-membar-refactor --publish", "```", "",
              "当模型分支已向前开发，脚本自动从其 Git 历史提取表格基准提交到临时目录，不切换、不重置原 worktree。", ""]
    (output / "README.md").write_text("\n".join(lines))
    if args.publish:
        save(baseline / "summary.json", results)
    if snapshot is not None:
        snapshot.cleanup()


if __name__ == "__main__":
    main()
