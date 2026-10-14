"""Import round-three measured evidence without changing existing report rows."""

import hashlib
import json
from pathlib import Path
import shutil


TITLES = {
    "groupnorm_swish_dx": "GroupNormSwishGrad dx（双块）",
    "logsigmoid": "LogSigmoid Forward",
    "layernorm_dx": "LayerNormGrad V3 dx",
    "softplus": "Softplus DAG / 融合 CCE",
    "moe_softplus": "MoE 单专家 Softplus",
    "causal_conv": "FusedCausalConv1D Residual",
    "rope_grad": "RoPE Grad Half",
    "rmsnorm_dx": "RMSNormGrad dx",
    "softmax_ce": "SoftmaxCrossEntropy 输出子 VF",
    "mhc_partial": "MHC Pre Partial Merge",
    "compressor_softmax": "Compressor DND Softmax",
}

NOTES = {
    "groupnorm_swish_dx": "原实现每轮处理两个 VL，U4 是继续展开四个 pair，即每轮八个 VL。",
    "softplus": "原 AscendC DAG 为 12 个 VF，总计 1869 cycles；等效融合 CCE 是 1 个 VF、515 cycles。主表精度与 CCE 比较，不将 VF 融合收益称为 spill 收益。",
    "logsigmoid": "原 SDK 零常量采用 VBR，CCE 为 PSET+VDUP；精度与对应 CCE CAModel 周期比较。",
    "causal_conv": "保持 FP16 输入/输出、FP32 计算、原滑窗及残差。U2 还发生数据重用与 Cast 消除；stack 配对写/读数不等于总访存数相对 U1 的差值。",
    "rope_grad": "保持双输出；只展开独立行，保留行内顺序。较大 U 同时增加 scalar 指针/控制开销，不仅是 vector 压力。",
    "softmax_ce": "统计量来自真实 48×128 行；当前子 VF 处理后 64 列，旧 loss partial 来自前 64 列。仅测试输出子 VF，不代表完整交叉熵 kernel。",
    "mhc_partial": "保留原 unaligned VLDAS/VLDUS/VSTUS/VSTAS 和 K 累加顺序；使用编译后日志对照的显式 Canonical JSON，按封存 generation 核对缓冲写出。scalar 运算暂未计时，预测与 CAModel 仍有偏差。",
    "compressor_softmax": "保持 DND [Sc,Reduce,Row] 布局及原 4 路分块。U1 已有 vector spill 和源码阶段同步，局部收益包含原阶段 Membar 局部化，不是纯 spill 消除收益。",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def loop_fields(case, constants):
    count = constants["iterations"]
    detail = f"未展开前主循环 {count} 次，不除以 unroll。"
    if case == "groupnorm_swish_dx":
        detail += "每轮原 pair 处理 2×64 个 FP32 元素，共 6144 元素。"
    elif case == "causal_conv":
        detail += "xLoop 每轮处理两个时间行；还有排空行和 channel 尾块，输出形状 49×192。"
    elif case == "compressor_softmax":
        detail += "Sc 外层一次处理 4 路，Reduce=8 的真实规约回边保留，不做独立 AABB unroll。"
    elif case == "mhc_partial":
        detail += "独立 row=48，groupK=4，输出宽度 64；保留行内 partial 累加。"
    elif case == "rope_grad":
        detail += "独立 sequence row=48，D=128，双半输出；B/N 和 half-D block 均为 1。"
    elif case == "softmax_ce":
        detail += "48 行，完整 R=128；当前 VF 为后 64 列的 loss/backprop 输出。"
    else:
        detail += "每轮 64 个 FP32 元素，共 3072 元素。"
    return dict(loop_count=count, loop_count_label=str(count), loop_count_detail=detail)


def import_round3(repo, dest, archive, manifest, copy_tree):
    root = repo / "cce_code/spill_candidates_round3"
    if not (root / "summary.json").exists():
        return [], {}, None
    base = {r["case"]: r for r in load(root / "summary.json")}
    versions = [{**r, "factor": 1} for r in base.values()] + load(root / "unroll_summary.json")
    predictions = {(r["case"], r["factor"]): r for r in load(root / "prediction_summary.json")}
    baseline_spills = {r["case"]: r["baseline_spill"] for r in load(root / "baseline_spill_summary.json")}
    evidence_manifest = load(root / "manifest.json")["files"]
    rows = []
    excluded_path = dest / "metadata/spill_candidate_excluded_versions.json"
    excluded = {r["id"]: r for r in load(excluded_path)} if excluded_path.exists() else {}
    references = {}

    def copy_file(original, target):
        relative = original.relative_to(root).as_posix()
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        expected = evidence_manifest.get(relative)
        if (expected is None and relative != "manifest.json") or (expected is not None and digest != expected):
            raise ValueError(f"Round-three evidence changed: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.chmod(target.stat().st_mode | 0o200)
        shutil.copyfile(original, target)
        manifest.append(dict(path=target.relative_to(dest).as_posix(), source=original.relative_to(repo).as_posix(),
                             sha256=digest))
        return dict(label=original.name, path=target.relative_to(dest).as_posix())

    for case in TITLES:
        reference = archive / "round3" / case / "reference"
        references[case] = copy_tree(root / "cases" / case / "ascendc", reference / "ascendc", manifest)
        for name in ("reference_original.h", "provenance.json", "golden_fp64.bin", "layout.json", "baseline_spill_audit.json"):
            references[case].append(copy_file(root / "cases" / case / name, reference / name))

    for version in sorted(versions, key=lambda r: (list(TITLES).index(r["case"]), r["factor"])):
        case, factor = version["case"], version["factor"]
        source = root / "cases" / case
        if factor != 1:
            source /= f"u{factor}"
        target = archive / "round3" / case / f"u{factor}"
        ident = f"round3-{case}-u{factor}"
        if version["status"] != "passed":
            log = (source / "cce/compile.log").read_text()
            if "exceeded vf stack size" not in log:
                raise ValueError(f"Unexpected round-three test failure: {ident}")
            copy_tree(source / "cce", target / "compile_failure", manifest)
            excluded[ident] = dict(id=ident, case=case, unroll=factor, status="vf_stack_exceeded",
                origin="round3_search", reason=version.get("error"))
            continue
        check = version["validation"]
        if not (check["bitwise_identical"] and check["guard_ok"] and
                all(check[k]["ok"] for k in ("cce_vs_ascendc", "ascendc_vs_formula", "cce_vs_formula"))):
            raise ValueError(f"Round-three numerical checks did not pass: {ident}")
        prediction = predictions[(case, factor)]
        global_cycles = prediction.get("global_cycles") if prediction["status"] == "predicted" else None
        local_cycles = None
        if prediction.get("local_audit_accepted"):
            experiment = prediction["experiment"]
            if experiment.get("zero_mask_nominal_conservative_count", 0) and not experiment.get("effective_address_audit_complete"):
                raise ValueError(f"Round-three conservative address audit failed: {ident}")
            if not (experiment["global_cycles"] == global_cycles and experiment["dynamic_order_match"]
                    and experiment["raw_war_release_boundary_verified"]
                    and experiment["ub_log_verified_accesses"] + experiment.get("zero_mask_nominal_conservative_count", 0)
                        == experiment["memory_access_count"]):
                raise ValueError(f"Round-three local A/B audit failed: {ident}")
            local_cycles = prediction["local_cycles"]
        stats = version["stream"]
        counts = baseline_spills[case] if factor == 1 else version["spill"]
        vector_store = counts["vector_stores"] if factor == 1 else counts["spill_vector_stores"]
        vector_load = counts["vector_loads"] if factor == 1 else counts["spill_vector_loads"]
        spill = any((vector_store, vector_load, stats["predicate_stores"], stats["predicate_loads"],
                     stats["scalar_spill_stores"], stats["scalar_spill_loads"]))
        constants = base[case]["constants"]
        dtype = constants["dtype"].upper()
        shape = "×".join(map(str, constants.get("shape", [constants["elements"]])))
        measured = (version["engines"]["cce"] if factor == 1 else version["camodel"])["total_vf_cycles"]
        files = dict(original_cce=copy_tree(source / "cce", target / "original/cce", manifest),
            comparison=copy_tree(source / "prediction", target / "original/comparison", manifest),
            reference=references[case])
        files["spill_audit"] = [copy_file(source / "summary.json", target / "original/summary.json")]
        if factor == 1:
            files["spill_audit"].append(copy_file(source / "baseline_spill_audit.json", target / "original/baseline_spill_audit.json"))
        note = NOTES.get(case, "所选 FP32 子 VF；前置统计与完整 kernel 不计入 VF 周期。")
        if prediction.get("fixture_kind") == "manual_compiled_log_contract":
            note += " 已用原编译动态日志构造手工 JSON，循环结构及全部 spill/Membar 保留；不是源码级忽略 spill 的预测。"
        if prediction.get("zero_mask_nominal_conservative_count"):
            note += f" {prediction['zero_mask_nominal_conservative_count']} 条全零 mask store 保留名义范围作保守阻塞；不是硬件实际发生的写区间。"
        if global_cycles is not None and abs(global_cycles - measured) / measured > 0.15:
            note += " 全局预测存在明显偏差，未校准时序，不把局部模型收益作为硬件实测证明。"
        reason = ""
        if global_cycles is None:
            reason = "真实编译指令流回放拒绝：" + str(prediction.get("reason", "未知拒绝点"))
        elif local_cycles is None:
            reason = "局部结果未通过与当前全局结果一致的 A/B 审计，不填入局部周期。"
        row = dict(id=ident, origin="round3_search", case=case, title=TITLES[case],
            variant=f"u{factor}", variant_label=f"pair-U{factor}" if case == "groupnorm_swish_dx" else f"U{factor}",
            variant_order=factor, unroll=factor, shape=f"{shape} / {dtype}", chart_label=f"U{factor}",
            comparison_group=f"round3-{case}", comparison_label=f"{shape} / {dtype} / 第三批 A5",
            status="predicted" if global_cycles is not None else "unsupported", spill=spill, reason=reason,
            camodel=measured, ascendc_camodel=base[case]["engines"]["ascendc"]["total_vf_cycles"],
            global_cycles=global_cycles, local_cycles=local_cycles, software_cycles=None, software_camodel=None,
            software_accuracy=None, original_accuracy=None, local_gain=None, software_gain=None,
            plan=None, software_validation=None, software_comparison=None, candidate_count=0, finalists=0,
            strategy="未开展软件寻优" if spill else "保留原无 spill 版本", files=files,
            evidence_path=(target / "result.json").relative_to(dest).as_posix(),
            spill_counts=dict(vector_store=vector_store, vector_load=vector_load,
                predicate_store=stats["predicate_stores"], predicate_load=stats["predicate_loads"]),
            scalar_stack_stores=stats["scalar_spill_stores"], scalar_stack_loads=stats["scalar_spill_loads"],
            other_extra_vector_loads=counts.get("other_extra_vector_loads", 0),
            spill_count_basis="第三批 CAModel 动态次数：V 按 stack 配对地址统计，P 按 PST/PLD，S 按 SST/SLD；不是寄存器个数。",
            numeric_validation=check, membar_count=stats["membars"], note=note, **loop_fields(case, constants))
        rows.append(row)
    excluded_path.write_text(json.dumps(list(excluded.values()), ensure_ascii=False, indent=2) + "\n")
    metadata = dest / "metadata" / "round3"
    for name in ("summary.json", "unroll_summary.json", "prediction_summary.json", "prediction_provenance.json",
                 "baseline_spill_summary.json", "manual_json_provenance.json", "manifest.json", "README.md"):
        if not (root / name).exists():
            continue
        copy_file(root / name, metadata / name)
    note = dict(cases=len(TITLES), archived_successful_versions=len(rows),
        selected_versions=sum(r["spill"] or r["unroll"] == 1 for r in rows),
        selected_spill_versions=sum(r["spill"] for r in rows),
        stack_failures=sum(r.get("origin") == "round3_search" for r in excluded.values()),
        selection="U1 plus actual vector/predicate/scalar spill; never fabricate unsupported or software cycles")
    return rows, TITLES, note
