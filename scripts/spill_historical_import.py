"""Normalize existing archive results without inventing new measurements."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def spill_present(variant):
    return any(variant.get(key, 0) for key in ("vector_store", "vector_load", "predicate_store", "predicate_load"))


def unroll_of(variant, root):
    for text in (variant.get("id", ""), variant.get("label", "")):
        found = re.search(r"(?:^|[_\s])u(\d+)(?:[_\s]|$)", text, re.I)
        if found:
            return int(found[1])
    for link in variant.get("evidence", []):
        if not re.search(r"(?:summary|optimization)\.json$", link):
            continue
        record = json.loads((root / link).read_text(encoding="utf-8"))
        if record.get("requested_unroll") is not None:
            return int(record["requested_unroll"])
        found = re.search(r"(?:^|_)u(\d+)(?:_|$)", record.get("case", ""), re.I)
        if found:
            return int(found[1])
    for source in variant.get("sources", []):
        if not source["path"].endswith(".cce"):continue
        text=(root/source["path"]).read_text(encoding="utf-8")
        pragmas=re.findall(r"#pragma\s+unroll\s*\(\s*(\d+)\s*\)",text)
        if pragmas and set(pragmas)=={"1"}:return 1
    return None


def loop_metadata(row, root):
    if row.get("origin")=="supplemental_search":
        count,label,detail=48,"48","未展开前向量数据循环：3072 元素 / 64 元素每迭代 = 48 次。"
    elif row.get("origin")=="fixed_unroll_search":
        count,label,detail=16,"16","未展开前的向量数据循环：1024 元素 / 64 元素每迭代 = 16 次。"
    elif row["case"] in {"gelu_poly","swiglu_grad","adam_apply_one","gelu_grad","gelu_grad_three_stage"}:
        count={"gelu_poly":96,"swiglu_grad":96,"adam_apply_one":48,"gelu_grad":64,"gelu_grad_three_stage":64}[row["case"]]
        label=str(count)
        detail=f"未展开前的数据循环 {count} 次；源码 repeat_times 或固定循环上界与归档 shape 对齐。"
        if row.get("variant")=="three_stage_post":detail+="三段循环各执行 64 次。"
    elif row["case"]=="flash_attention_grad_front":
        count,label,detail=8,"行 8","未展开前的行循环 8 次（unroll 作用于行）；每行 12 个 64 元素向量块在循环体中显式展开。"
    elif row["case"]=="softmax_three_loop":
        count,label,detail=8,"行 8；内层 3/7/16","MAX、SUM、NORM 三个阶段各有 8 次行循环；其内部 blk 循环分别为 3、7、16 次，均按源码实际 for 上界记录。"
    elif row["case"]=="mhc_sinkhorn":
        summary=None
        for link in row["historical_record"].get("evidence",[]):
            if not link.endswith("summary.json"):continue
            candidate=json.loads((Path(root)/link).read_text(encoding="utf-8"))
            if "matrices" in candidate and "num_iters" in candidate:
                summary=candidate;break
        if summary is None:return dict(loop_count=None,loop_count_label=None,loop_count_detail="未找到带矩阵数和归一化轮数的归档记录。")
        matrices,rounds=int(summary["matrices"]),int(summary["num_iters"])
        subsequent=int(summary.get("additional_iters",rounds-1))
        if "element_major" in row["variant"]:
            count=(matrices+63)//64
            label=f"批次 {count}；迭代 {subsequent}"
            detail=f"未展开前 SIMD batch 循环 ceil({matrices}/64) = {count} 次；每批先做初始化轮，再执行 {subsequent} 次归一化循环，总计 {rounds} 轮。"
            if "split4_15" in row["variant"]:
                label=f"批次 {count}；迭代 4+15"
                detail=f"两段的 SIMD batch 循环各 {count} 次；初始化后归一化循环拆为 4 次和 15 次，总计 {rounds} 轮。"
        else:
            count=matrices
            label=f"矩阵 {count}；迭代 {subsequent}"
            detail=f"未展开前矩阵循环 {matrices} 次（unroll 作用层）；每矩阵先初始化，再执行 {subsequent} 次归一化循环，总计 {rounds} 轮；初始化另有 4 行循环。"
    else:
        return dict(loop_count=None,loop_count_label=None,loop_count_detail="原循环次数未确认。")
    return dict(loop_count=count,loop_count_label=label,loop_count_detail=detail)


def groups(case, variant):
    if case["id"] != "mhc_sinkhorn":
        return "history-" + case["id"], case["shape"]
    found = re.match(r"m(\d+)_", variant["id"])
    matrices = int(found[1]) if found else 36
    calibrated = "element_major" in variant["id"]
    profile = "新校准 GSU" if calibrated else "历史冻结参数"
    return f"mhc-{matrices}-{'calibrated' if calibrated else 'frozen'}", f"{matrices} 矩阵 / {profile}"


def references(variant):
    return {
        "original_cce": list(variant.get("sources", [])),
        "comparison": [dict(label=Path(path).name, path=path) for path in variant.get("logs", []) + variant.get("evidence", [])],
    }


def comparable_software_pairs(cases, root):
    variants = {(case["id"], variant["id"]): variant for case in cases for variant in case["variants"]}
    pairs = {
        ("gelu_grad", "u1"): (("gelu_grad_three_stage", "three_stage_post"), "已有三段切分 U1 对照"),
        ("gelu_grad_three_stage", "single_u1"): (("gelu_grad_three_stage", "three_stage_post"), "已有三段切分 U1 对照"),
        ("mhc_sinkhorn", "m128_element_major"): (("mhc_sinkhorn", "m128_element_major_split4_15"), "已有 4+15 切分重排 U1 对照"),
    }
    candidates = [(key, value) for key, value in variants.items()
                  if key[0] == "flash_attention_grad_front" and unroll_of(value, root) == 2
                  and not spill_present(value) and value.get("global_cycles") is not None]
    if candidates:
        best, _ = min(candidates, key=lambda item: (item[1]["camodel"], item[1]["global_cycles"], item[0]))
        pairs[("flash_attention_grad_front", "u2_concentrated_spill")] = (
            best, "同 U2 无 spill、CAModel 实测最快：Bank-aware 两行交错；比值仍用 VfSim")
    result = {}
    for original, (optimized, strategy) in pairs.items():
        if original not in variants or optimized not in variants:
            continue
        source, target = variants[original], variants[optimized]
        before, after = unroll_of(source, root), unroll_of(target, root)
        if before is None or before != after:
            raise ValueError(f"Software comparison changes unroll: {original} -> {optimized}")
        if spill_present(target) or target.get("global_cycles") is None or target.get("camodel") is None:
            raise ValueError(f"Software counterpart lacks a no-spill measured result: {optimized}")
        if target.get("golden") != "已通过":
            raise ValueError(f"Software counterpart lacks recorded numerical validation: {optimized}")
        equivalence = {}
        source_bundle=source.get("camodel_case",{}).get("path")
        target_bundle=target.get("camodel_case",{}).get("path")
        if source_bundle and target_bundle:
            for name in ("input.bin","output.bin"):
                first=root/source_bundle/"run_output"/name
                second=root/target_bundle/"run_output"/name
                if first.exists() and second.exists():
                    left=hashlib.sha256(first.read_bytes()).hexdigest()
                    right=hashlib.sha256(second.read_bytes()).hexdigest()
                    if left!=right:raise ValueError(f"Historical comparison input/output differs: {original} -> {optimized}")
                    equivalence[name]=left
        result[original] = dict(case=optimized[0], variant=optimized[1], unroll=after,
                                cycles=target["global_cycles"], camodel=target["camodel"], strategy=strategy,
                                label=target["label"], files=references(target), shared_binary_sha256=equivalence)
    return result


def import_archive(data, root):
    root = Path(root)
    pairs = comparable_software_pairs(data["cases"], root)
    rows = []
    for case in data["cases"]:
        for index, variant in enumerate(case["variants"]):
            if variant.get("camodel") is None:
                continue
            global_cycles, local_cycles = variant.get("global_cycles"), variant.get("local_cycles")
            spill = spill_present(variant)
            counterpart = pairs.get((case["id"], variant["id"]))
            software_cycles = counterpart["cycles"] if counterpart else global_cycles if not spill else None
            software_camodel = counterpart["camodel"] if counterpart else variant["camodel"] if software_cycles is not None else None
            group, group_label = groups(case, variant)
            files = references(variant)
            if counterpart:
                files["software"] = counterpart["files"]["original_cce"] + counterpart["files"]["comparison"]
            row = dict(id=f"history-{case['id']}-{variant['id']}", origin="historical_membar",
                       case=case["id"], title=case["title"], variant=variant["id"], variant_label=variant["label"],
                       variant_order=index, unroll=unroll_of(variant, root), shape=case["shape"],
                       chart_label=variant["label"], comparison_group=group, comparison_label=group_label,
                       status="predicted" if global_cycles is not None else "unsupported", spill=spill,
                       reason="" if global_cycles is not None else variant.get("note", "历史版本没有完整预测"),
                       camodel=variant["camodel"], global_cycles=global_cycles, local_cycles=local_cycles,
                       source_cycles=variant.get("source_cycles"), software_cycles=software_cycles,
                       software_camodel=software_camodel, software_accuracy=None, original_accuracy=None,
                       local_gain=None, software_gain=None, plan=None, software_validation=None,
                       software_comparison=counterpart, candidate_count=0, finalists=0,
                       strategy=counterpart["strategy"] if counterpart else "保留历史无已记录 spill 版本" if software_cycles is not None else "尚无同 unroll 软件对照",
                       files=files, note=variant.get("note", ""), case_notes=case.get("notes", []),
                       spill_counts={key: variant.get(key, 0) for key in ("vector_store", "vector_load", "predicate_store", "predicate_load", "membars")},
                       historical_record=variant, historical_case_path=f"{case['id']}/summary.json",
                       evidence_path=f"spill_candidate_alignment/imported/{case['id']}/{variant['id']}/result.json")
            rows.append(row)
    note = dict(source="metadata/data.json", source_date=data["generated"],
                source_sha256=hashlib.sha256((root / "metadata/data.json").read_bytes()).hexdigest(),
                groups=len(data["cases"]), versions=len(rows), status_counts=dict(Counter(row["status"] for row in rows)), software_pairs=pairs_to_json(pairs),
                limitations=["只归档已有结果，不重新预测或校准。",
                             "保留各版本的模型快照、scalar/VMOV/GSU 限制；不能把历史结果声称为当前模型的完整指令回放。",
                             "未执行的局部预测不由全局结果填充。源码预测不包含 compiler spill 时，不充当软件优化结果。",
                             "MHC 按矩阵数和冻结/新校准 GSU 参数分组画图；不跨参数或 unroll 选软件对照。"])
    return rows, note


def pairs_to_json(pairs):
    return [dict(original_case=key[0], original_variant=key[1], **value) for key, value in pairs.items()]


def validate_compiled_replay_update(record, provenance, row):
    audit = record["original_stream_audit"]
    current, experiment = record["current_global"], record["experiment"]
    memory_count = experiment.get("memory_access_count")
    if (record["id"] != row["id"] or record["camodel_cycles"] != row["camodel"]
            or current.get("status") != "predicted" or experiment.get("status") != "predicted"
            or record["global_cycles"] != current["global_cycles"]
            or record["global_cycles"] != experiment["global_cycles"]
            or record["local_cycles"] != experiment["local_cycles"]
            or not experiment.get("dynamic_order_match")
            or not experiment.get("raw_war_release_boundary_verified")
            or type(memory_count) is not int or memory_count <= 0
            or experiment.get("ub_log_verified_accesses") != memory_count
            or not all(audit.get(k) for k in ("dynamic_stream_match", "vf_cycles_match", "output_bitwise_match"))
            or not audit["numeric_validation"].get("passed") or not provenance.get("model_stable")):
        raise ValueError(f"Compiled historical replay audit failed: {row['id']}")
    counts = audit["spill_counts"]
    for field, opcode in (("vector_store", "RV_VSTI"), ("vector_load", "RV_VLDI"),
                          ("predicate_store", "RV_PSTI"), ("predicate_load", "RV_PLDI"),
                          ("membars", "RV_SMEM_BAR")):
        if counts.get(opcode) != row["spill_counts"].get(field):
            raise ValueError(f"Historical spill count differs: {row['id']} {field}")


def apply_compiled_replay_updates(data, root, archive, manifest, copy_tree):
    """Overlay audited new predictions without mutating the historical source."""
    root = Path(root)
    for row in data["rows"]:
        if row.get("origin") != "historical_membar":
            continue
        source = root / row["case"] / row["variant"] / "model/compiled_replay_current"
        if not (source / "prediction.json").exists():
            continue
        record = json.loads((source / "prediction.json").read_text())
        provenance = json.loads((source / "provenance.json").read_text())
        validate_compiled_replay_update(record, provenance, row)
        source_code = root / row["case"] / row["variant"] / "source/kernel.cce"
        if hashlib.sha256(source_code.read_bytes()).hexdigest() != provenance["original_source_sha256"]:
            raise ValueError(f"Original historical CCE changed: {row['id']}")
        counts = record["original_stream_audit"]["spill_counts"]
        target = archive / "imported" / row["case"] / row["variant"] / "compiled_replay_current"
        files = copy_tree(source, target, manifest)
        row["files"]["comparison"].extend(files)
        row["previous_global_cycles"] = row.get("global_cycles")
        row.update(status="predicted", reason="", global_cycles=record["global_cycles"],
                   local_cycles=record["local_cycles"], compiled_replay_update=record,
                   compiled_replay_provenance=provenance)
        row["note"] = ("原历史源码预测没有 compiler spill，未使用该值。现按原 CAModel 编译后完整 PC 流补跑，"
                       f"保留 vector 写/读 {counts['RV_VSTI']}/{counts['RV_VLDI']}、predicate 写/读 "
                       f"{counts['RV_PSTI']}/{counts['RV_PLDI']} 和 {counts['RV_SMEM_BAR']} 条 Membar；原 CCE 复跑数值、"
                       "动态流和 VF 周期一致，全部 UB 地址及局部 RAW/WAR done+1 通过审计。"
                       "使用当前模型，scalar 地址算术仍不计时；局部时间为实验预测，不是 CAModel 实测。")
