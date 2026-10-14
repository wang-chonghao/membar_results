"""Import verified follow-up evidence without altering historical measurements."""

import json
from pathlib import Path
import shutil


def load(path):
    return json.loads(path.read_text())


def import_round2(repo, dest, archive, manifest, copy_tree):
    root = repo / "cce_code/spill_candidates_round2"
    titles = {"mish": "Mish", "logsigmoid_grad": "LogSigmoidGrad",
              "binary_cross_entropy": "BinaryCrossEntropy（概率输入）"}
    if not (root / "summary.json").exists():
        return [], {}
    predictions = {(r["case"], r["unroll"]): r for r in load(root / "vfsim_comparison/summary.json")}
    rows = []
    for case in titles:
        copy_tree(root / "cases" / case / "ascendc", archive / case / "reference/ascendc", manifest)
        for name in ("layout.json", "golden_fp64.bin", "reference_original.h", "provenance.json"):
            original = root / "cases" / case / name
            target = archive / case / "reference" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    excluded_path = dest / "metadata/spill_candidate_excluded_versions.json"
    excluded = {r["id"]: r for r in load(excluded_path)} if excluded_path.exists() else {}
    for version in load(root / "summary.json"):
        case, factor = version["case"], version["factor"]
        if version["status"] != "passed":
            excluded[f"round2-{case}-u{factor}"] = dict(id=f"round2-{case}-u{factor}", case=case, unroll=factor,
                status="vf_stack_exceeded", origin="supplemental_search", reason=version.get("error"))
            continue
        p = predictions[(case, factor)]
        target = archive / case / f"u{factor}"
        source = root / "cases" / case
        if factor != 1:
            source /= f"u{factor}"
        files = dict(original_cce=copy_tree(source / "cce", target / "original/cce", manifest),
            comparison=copy_tree(root / "vfsim_comparison" / case / f"u{factor}", target / "original/comparison", manifest))
        spill, stats = version.get("spill", {}), version["stream"]
        row = dict(id=f"round2-{case}-u{factor}", origin="supplemental_search", case=case, title=titles[case],
            variant=f"u{factor}", variant_label=f"U{factor}", variant_order=factor, unroll=factor,
            shape="3072 元素 / FP32 / 48 次向量迭代", chart_label=f"U{factor}",
            comparison_group=f"round2-{case}", comparison_label="3072 元素 / 当前 A5 参数",
            status="predicted" if p["status"] == "predicted" else "unsupported",
            spill=bool(spill.get("detected")), reason="" if p["status"] == "predicted" else str(p.get("reason")),
            camodel=p["camodel_cycles"], global_cycles=p.get("global_cycles"), local_cycles=p.get("local_cycles"),
            software_cycles=None, software_camodel=None, software_accuracy=None, original_accuracy=None, local_gain=None,
            software_gain=None, plan=None, software_validation=None, software_comparison=None,
            candidate_count=0, finalists=0, strategy="未搜索" if spill.get("detected") else "保留原无 spill 版本",
            files=files, evidence_path=(target / "result.json").relative_to(dest).as_posix(),
            spill_counts=dict(vector_store=spill.get("spill_vector_stores", 0), vector_load=spill.get("spill_vector_loads", 0),
                              predicate_store=stats["predicate_stores"], predicate_load=stats["predicate_loads"]),
            scalar_stack_stores=stats["scalar_spill_stores"], scalar_stack_loads=stats["scalar_spill_loads"],
            other_extra_vector_loads=spill.get("other_extra_vector_loads", 0),
            spill_count_basis="第二批 CAModel 编译后动态次数；vector 按新增写读共同地址，predicate 按 PST/PLD 统计。",
            note="同流全局/局部 A/B 已核对当前主线全局周期、动态顺序及 core0 UB 地址；含 scalar stack 的版本明确拒绝。")
        rows.append(row)
    excluded_path.write_text(json.dumps(list(excluded.values()), ensure_ascii=False, indent=2) + "\n")
    return rows, titles


def apply_followup(data, repo, dest, archive, manifest, copy_tree):
    root = repo / "cce_code/spill_software_followup"
    plans = {}
    for file in root.glob("*_plans.json"):
        for plan in load(file):
            plans[(plan["name"], plan["unroll"])] = plan
    results = []
    for file in sorted((root / "software_optimization").glob("*/u*/result.json")):
        result = load(file)
        results.append(result)
        plan = plans.get((result["case"], result["unroll"]), {})
        ids = plan.get("report_row_ids", [f"round2-{result['case']}-u{result['unroll']}"])
        for row in data["rows"]:
            if row["id"] not in ids:
                continue
            row.update(candidate_count=result["candidate_count"], finalists=len(result["validated"]))
            best = result.get("best")
            if not best:
                row["strategy"] = "已搜索，未找到通过验证的无 spill 方案"
                row["note"] = row.get("note", "") + f" 已尝试 {result['candidate_count']} 个候选。"
                continue
            if best["status"] != "validated_no_spill" or not best["validation"]["ok"] or best["compiler_stack_instruction_count"]:
                raise ValueError("Follow-up software candidate is not validated spill-free")
            row.update(software_cycles=best["global_cycles"], software_camodel=best["camodel_cycles"],
                plan=best["plan"], software_validation=best["validation"], software_comparison=None,
                strategy="切分＋重排" if best["plan"]["stages"] > 1 else "仅重排")
            folder = repo / best["artifact_dir"]
            source = next((folder / p for p in ("cce/kernel.cce", "mhc/source/kernel.cce", "kernel.cce")
                           if (folder / p).exists()), None)
            import hashlib
            if source is None or hashlib.sha256(source.read_bytes()).hexdigest() != best["plan"]["source_sha256"]:
                raise ValueError("Validated follow-up source changed")
            target = archive / "followup" / row["id"]
            row["files"]["software"] = copy_tree(folder, target, manifest)
            row["note"] = row.get("note", "") + " 软件对照为当前 A5 主线全局模型；历史原始/局部周期保留旧归档及原参数说明，不覆盖原记录。"
    old = load(repo / "cce_code/spill_candidate_alignment/software_optimization/summary.json")
    data["candidate_count"] = sum(r["candidate_count"] for r in old + results)
    data["finalists"] = sum(len(r["validated"]) for r in old + results)
