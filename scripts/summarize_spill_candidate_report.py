"""Summarize the HTML dataset without treating retained U1 baselines as gains."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANDS = ("≥100%", "90%–<100%", "80%–<90%", "70%–<80%", "<70%")


def band(value):
    for minimum, label in zip((100, 90, 80, 70), BANDS):
        if value >= minimum:
            return label
    return BANDS[-1]


def performance(row):
    software, local = row.get("software_cycles"), row.get("local_cycles")
    reported = row.get("membar_vs_software_performance")
    if software is None or local is None:
        if reported is not None:
            raise ValueError(f"Missing prediction has a fabricated ratio: {row['id']}")
        return None
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (software, local)):
        raise ValueError(f"Invalid cycle count: {row['id']}")
    result = 100 * software / local
    if reported is None or abs(result - reported) > 1e-9:
        raise ValueError(f"Ratio differs from HTML: {row['id']}")
    return result


def distribution(values):
    counts = Counter(band(v) for v in values)
    return {label: counts[label] for label in BANDS}


def statistics(data, all_versions, excluded):
    rows = data["rows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate HTML case IDs")
    spill = [r for r in rows if r["spill"]]
    baseline = [r for r in rows if not r["spill"]]
    comparable = [r for r in spill if performance(r) is not None]
    comparable_operators = len({r['case'] for r in comparable})
    missing = [r for r in spill if performance(r) is None]
    values = [performance(r) for r in comparable]
    return dict(
        operator_count=len({r["case"] for r in rows}), comparison_groups=len({(r["case"], r["comparison_group"]) for r in rows}),
        archived_successful_versions=len(all_versions), excluded_versions=len(excluded),
        displayed_cases=len(rows), spill_cases=len(spill), baseline_cases=len(baseline),
        comparable_spill_cases=len(comparable), missing_spill_comparisons=len(missing),
        comparable_operators=comparable_operators, unclassified_operators=len({r["case"] for r in rows})-comparable_operators,
        spill_distribution=distribution(values),
        all_distribution=distribution(performance(r) for r in rows if performance(r) is not None),
        baseline_distribution=distribution(performance(r) for r in baseline if performance(r) is not None),
        all_comparable_cases=sum(performance(r) is not None for r in rows),
        strict_better_spill_cases=sum(v > 100 for v in values), tied_spill_cases=sum(v == 100 for v in values),
        below_70_subdistribution={"60%–<70%":sum(60 <= v < 70 for v in values),
                                 "50%–<60%":sum(50 <= v < 60 for v in values), "<50%":sum(v < 50 for v in values)},
        missing_local=sum(r["local_cycles"] is None for r in missing),
        missing_software=sum(r["software_cycles"] is None for r in missing),
        missing_both=sum(r["local_cycles"] is None and r["software_cycles"] is None for r in missing),
        missing_case_ids=[r["id"] for r in missing])


def cell(value):
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def number(value, percent=False):
    return "--" if value is None else f"{value:.2f}%" if percent else str(value)


def source_link(row):
    return f"[{cell(row['title'])}](./{row['source_cce_path']})"


def spill_description(row):
    labels = {"vector_predicate":"Vector + Predicate spill", "vector":"Vector spill", "predicate":"Predicate spill",
              "scalar":"Scalar spill", "none":"无 Vector / Predicate spill"}
    lines = [labels[row["spill_kind"]],
        f"Vector 写 {row['vector_spill_store']} / 读 {row['vector_spill_load']}",
        f"Predicate 写 {row['predicate_spill_store']} / 读 {row['predicate_spill_load']}"]
    if row.get("scalar_stack_stores") or row.get("scalar_stack_loads"):
        lines.append(f"Scalar 写 {row.get('scalar_stack_stores',0)} / 读 {row.get('scalar_stack_loads',0)}")
    return "<br>".join(lines)


def markdown(data, stats, dataset_sha):
    rows = sorted(data["rows"], key=lambda r: (data["case_order"].index(r["case"]), r["variant_order"]))
    count = stats["comparable_spill_cases"]
    sections = ["# Register Spill 与 Membar 优化实验总结", "",
        f"数据日期：{data['date']}。平台：A5 / DV100；CANN 9.0.0-beta.1。",
        "来源：[HTML 汇总页](./spill_candidate_alignment_summary.html)、[完整展示数据](./metadata/spill_candidate_alignment.json)。",
        f"数据快照 SHA256：`{dataset_sha}`。本文件按该快照生成，未重新运行预测或修改测量结果。", "",
        "## 测试范围", "",
        f"共 {stats['operator_count']} 个算子/子 VF 分组、{stats['comparison_groups']} 个 shape/参数对照组。主表 {stats['displayed_cases']} 个 case，包含 {stats['spill_cases']} 个含 spill case 和 {stats['baseline_cases']} 个无 spill 对照。",
        f"成功归档的原始版本共 {stats['archived_successful_versions']} 个；另有 {stats['excluded_versions']} 个 VF stack 超限版本，不进入主表或性能统计。软件搜索候选不计入原始 case 数。",
        f"可比较的含 spill case 为 {count} 个，覆盖 {stats['comparable_operators']} 个算子分组；不同 U、shape 和实现算不同 case，不等同于不同算子。", "",
        "## 软件性能与比较口径", "",
        "**软件性能**指软件改写方案的预测性能：保持该行原 shape、算术和名义 unroll，通过循环切分、指令重排降低活跃值数量、避免编译器 spill，仍采用全局 Membar 同步。对应主表的 **VfSim 软件优化全局** 周期，不是 CAModel 周期，也不是 Python/C++ 运行耗时。",
        "软件方案是有限搜索中已编译并通过数值校验的最好候选，不是理论最优。无 spill 对照行沿用原版本，不代表额外做了软件优化。",
        "局部依赖侧保留原编译后指令流及 spill 搬运，只将全局 Membar 改为局部 UB 地址依赖同步。两侧统一用 VfSim 数据比较。",
        "相同工作量下性能与周期数成反比，因此：", "",
        "`Membar / 软件性能 = VfSim 软件优化全局周期 / VfSim 局部周期 × 100%`。", "",
        "- 100% 表示两侧一样快；大于 100% 表示局部依赖侧更快。",
        "- 例如软件方案 100 cycle，局部依赖侧 110 cycle，比值为 90.91%，即达到软件方案性能的 90.91%；不是周期减少 90.91%。",
        "- 比值按每行独立计算，不将不同 U/shape 的周期混合相除；缺失项不按 0 处理。", "",
        "## 含 Spill Case 的统计", ""]
    for label in BANDS:
        amount = stats["spill_distribution"][label]
        sections.append(f"- {label}：{amount} 个 case，占 {100*amount/count:.2f}%。")
    sections += ["",
        f"其中严格超过 100% 的有 {stats['strict_better_spill_cases']} 个，恰好持平 {stats['tied_spill_cases']} 个。达到 90% 及以上的共 {stats['spill_distribution'][BANDS[0]]+stats['spill_distribution'][BANDS[1]]}/{count} 个。",
        f"另有 {stats['missing_spill_comparisons']} 个含 spill case 暂不能比较：缺局部预测 {stats['missing_local']} 个，缺软件优化结果 {stats['missing_software']} 个，其中两项都缺 {stats['missing_both']} 个。",
        f"无 spill 对照中有比值的 {sum(stats['baseline_distribution'].values())} 行均为 100%，不纳入上述 spill 效果统计。", "",
        "## 结论边界", "",
        "- 局部依赖列是假设硬件具备 UB 地址依赖能力的预测，不是该硬件上的 CAModel 实测。",
        "- 历史记录可能使用不同冻结模型/GSU 参数，部分仍为源码级预测。具体参数、限制和证据保留在详情与结果 JSON，不声称全部记录完全同口径。",
        "- 未建模的标量 spill 不静默忽略；部分普通标量地址运算仍未计时。MHC 等低精度记录需谨慎解释。",
        "- 分布使用未舍入比值，90%、80%、70% 分别归入以该值为下界的区间，100% 归入 ≥100%。", "",
        "## 表头说明", ""]
    fields = [
        ("Case", "算子/子 VF 分组名称；点击跳转该行 CAModel 实际使用的 CCE 源码"),
        ("版本 / Shape 与来源", "U/实现变体、元素形状、参数组及首批/第二批/第三批/历史归档来源；不同组不直接混合周期"),
        ("循环次数", "未展开前的主循环次数，不除以 U；多层循环按原记录标注批次和迭代"),
        ("U", "名义 unroll 倍数；pair 等原实现每轮可处理多个 VL，不能将 U 等同于向量指令数量"),
        ("原 CAModel", "原版本实际编译并运行的 VF 周期；包含编译器插入的 spill/Membar，不是完整 kernel 的 GM/MTE 耗时"),
        ("VfSim 全局", "相同编译后实际流在全局 Membar 同步下的预测周期"),
        ("VfSim 局部", "相同流在局部 UB 地址依赖同步下的预测周期；保留 spill，按 RAW/WAR done+1 放行"),
        ("全局精度", "100 × (1 − abs(VfSim 全局 − 原 CAModel) / 原 CAModel)，不是输出数值正确率"),
        ("局部周期减少", "100 × (1 − VfSim 局部 / VfSim 全局)，负数表示变慢"),
        ("VfSim 软件优化全局", "循环切分/指令重排后的源码重新编译，在全局 Membar 下的预测；无 spill 对照沿用原版本"),
        ("优化后 CAModel", "与软件优化列对应源码的 CAModel 实测周期；不是局部依赖假设硬件的实测"),
        ("优化后精度", "100 × (1 − abs(VfSim 软件优化全局 − 优化后 CAModel) / 优化后 CAModel)"),
        ("Membar / 软件性能", "100 × VfSim 软件优化全局 / VfSim 局部；越大越好，100% 持平"),
        ("软件策略", "已采用的切分/重排方案，或保留原无 spill 版本、未开展软件寻优"),
        ("原版本 / Spill 写读次数", "原流中的 Vector/Predicate/Scalar spill 动态写出、读回次数；不是超出容量的寄存器个数，写读可以不相等"),
        ("证据", "HTML 中可打开详情；本 Markdown 链接对应 result.json，保留原始日志、数值校验、地址审计、SHA 与模型版本")]
    sections += [*(f"- **{name}**：{description}。" for name, description in fields), "", "`--` 表示缺少/不支持，不是 0；精度显示两位小数，分布统计使用原始未舍入值。", "",
        "## HTML 完整汇总表", "",
        f"下表复制 HTML 默认完整视图的全部 {stats['displayed_cases']} 行。数据与主表相同；Case 链接指向 CCE，证据链接指向该行 JSON。", ""]
    origins = {"fixed_unroll_search":"首批固定 unroll", "supplemental_search":"第二批", "round3_search":"第三批", "historical_membar":"历史归档"}
    columns = ["Case", "版本 / Shape 与来源", "循环次数", "U", "原 CAModel", "VfSim 全局", "VfSim 局部", "全局精度",
               "局部周期减少", "VfSim 软件优化全局", "优化后 CAModel", "优化后精度", "Membar / 软件性能", "软件策略", "原版本 / Spill 写读次数", "证据"]
    values = []
    for row in rows:
        values.append([source_link(row), f"{row['variant_label']}<br>{row['comparison_label']}<br>{origins[row['origin']]}",
            row["loop_count_label"], number(row["unroll"]), number(row["camodel"]), number(row["global_cycles"]), number(row["local_cycles"]),
            number(row["original_accuracy"], True), number(row["local_gain"], True), number(row["software_cycles"]),
            number(row["software_camodel"]), number(row["software_accuracy"], True), number(row["membar_vs_software_performance"], True),
            row["strategy"], spill_description(row), f"[详情](./{row['evidence_path']})"])
    sections.append(table(columns, values))
    sections += ["", "## 更新方法", "", "HTML 数据更新后，在结果仓库根目录重新生成本总结：", "", "```bash",
        "python3 scripts/summarize_spill_candidate_report.py", "```", "",
        "发布前应通过报告生成脚本同步更新文件校验清单；单独生成文档不会刷新清单。", ""]
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.root / "metadata/spill_candidate_alignment.json"
    raw = source.read_bytes()
    data = json.loads(raw)
    all_versions = json.loads((args.root / "metadata/spill_candidate_all_versions.json").read_text())
    excluded = json.loads((args.root / "metadata/spill_candidate_excluded_versions.json").read_text())
    result = statistics(data, all_versions, excluded)
    sha = hashlib.sha256(raw).hexdigest()
    output = args.root / "spill_candidate_alignment_summary.md"
    output.write_text(markdown(data, result, sha), encoding="utf-8")
    (args.root / "metadata/spill_candidate_statistics.json").write_text(json.dumps(
        dict(source_dataset_sha256=sha, date=data["date"], **result), ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(document=str(output), operators=result["operator_count"], cases=result["displayed_cases"],
        comparable_spill_cases=result["comparable_spill_cases"], bands=result["spill_distribution"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
