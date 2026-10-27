"""Archive fixed-unroll experiments and build a portable, offline HTML report."""

import csv
import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


DEST = Path(__file__).resolve().parents[1]
REPO = DEST.parents[1]
SOURCE = REPO / "cce_code/spill_candidate_alignment"
ARCHIVE = DEST / "spill_candidate_alignment"
PAGE = DEST / "spill_candidate_alignment_summary.html"
CASE_ORDER = ["swiglu", "focal", "swiglu_quant", "bce", "adam_decay",
              "groupnorm", "logit", "lstm", "syncbn", "deepnorm"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(predicted, measured):
    if predicted is None or measured is None or measured <= 0:
        return None
    return 100 * (1 - abs(predicted - measured) / measured)


def membar_vs_software_performance(software_cycles, local_cycles):
    if software_cycles is None or local_cycles is None or local_cycles <= 0:
        return None
    return 100 * software_cycles / local_cycles


def complete_retained_baseline(row):
    if row.get("plan") is None and not row.get("software_comparison") and row["status"] == "predicted" and not row["spill"]:
        row["software_cycles"] = row["global_cycles"]
        row["software_camodel"] = row["camodel"]
        row["software_accuracy"] = accuracy(row["global_cycles"], row["camodel"])
    return row


def report_rows(rows):
    return [row for row in rows if row["status"] != "vf_stack_exceeded"]


def spill_and_baseline_rows(rows):
    groups = {}
    for row in report_rows(rows):
        key = (row["case"],row["comparison_group"])
        groups.setdefault(key,[]).append(row)
    selected = {row["id"] for row in rows if row["spill"] and row["status"]!="vf_stack_exceeded"}
    for group in groups.values():
        candidates = [row for row in group if not row["spill"] and row.get("unroll")==1]
        if candidates:
            selected.add(min(candidates,key=lambda row:row["variant_order"])["id"])
        else:
            originals = [row for row in group if not row["spill"] and row.get("unroll") is None]
            if len(originals)==1:selected.add(originals[0]["id"])
    result = []
    for row in rows:
        if row["id"] not in selected:continue
        role = "register spill" if row["spill"] else "U1 基线" if row.get("unroll")==1 else "原始基线（unroll 未记录）"
        result.append(dict(row,selection_role=role))
    return result


def populate_spill_statistics(row):
    keys=("vector_store","vector_load","predicate_store","predicate_load")
    if row["origin"] in {"historical_membar","supplemental_search","round3_search"}:
        counts={key:row.get("spill_counts",{}).get(key) for key in keys}
        source=row.get("spill_count_basis","历史归档动态统计；保留原记录，未重新推导。")
        extra=row.get("other_extra_vector_loads")
    elif row["unroll"]==1:
        counts=dict.fromkeys(keys,0)
        source="U1 编译后流已核对，未发现 Vector/Predicate spill。"
        extra=0
    else:
        original=SOURCE/"unroll_manual"/row["case"]/f"u{row['unroll']}"/"summary.json"
        archived=ARCHIVE/row["case"]/f"u{row['unroll']}"/"original/spill_summary.json"
        if original.exists():
            archived.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,archived)
        if archived.exists():
            summary=read(archived)
            spill,audit=summary["spill"],summary["instruction_audit"]
            counts=dict(vector_store=spill.get("spill_vector_stores"),vector_load=spill.get("spill_vector_loads"),
                        predicate_store=audit.get("predicate_stores"),predicate_load=audit.get("predicate_loads"))
            extra=spill.get("other_extra_vector_loads")
            row["scalar_stack_stores"]=audit.get("scalar_spill_stores")
            row["scalar_stack_loads"]=audit.get("scalar_spill_loads")
            source="编译后动态统计：向量按已配对 spill 地址统计；谓词按 PLD/PST 统计。额外只读系数重载不计入 spill。"
            row["files"]["spill_audit"]=[dict(label="Spill 分类与动态次数原记录",path=archived.relative_to(DEST).as_posix())]
        else:
            counts=dict.fromkeys(keys,None)
            extra=None
            source="没有已核对的 spill 分类/数量记录。"
    row["spill_counts"]=dict(row.get("spill_counts",{}),**counts)
    row["other_extra_vector_loads"]=extra
    row["spill_count_basis"]=source
    for key,value in counts.items():row[key.replace("_","_spill_",1)]=value
    vector=any(counts[key] is not None and counts[key]>0 for key in ("vector_store","vector_load"))
    predicate=any(counts[key] is not None and counts[key]>0 for key in ("predicate_store","predicate_load"))
    scalar=bool(row.get("scalar_stack_stores") or row.get("scalar_stack_loads"))
    row["spill_kind"]="vector_predicate" if vector and predicate else "vector" if vector else "predicate" if predicate else "scalar" if scalar else "unknown" if row["spill"] else "none"


def spill_status_html(row):
    counts=row.get("spill_counts",{})
    kind=row.get("spill_kind","unknown" if row["spill"] else "none")
    labels={"vector_predicate":"Vector + Predicate spill","vector":"Vector spill","predicate":"Predicate spill",
            "none":"无 Vector / Predicate spill","scalar":"Scalar spill","unknown":"含 spill，分类待核对"}
    lines=[]
    for title,stem in (("Vector","vector"),("Predicate","predicate")):
        stores,loads=counts.get(stem+"_store"),counts.get(stem+"_load")
        lines.append(f'<small>{title} 写 {stores if stores is not None else "--"} / 读 {loads if loads is not None else "--"}</small>')
    if row.get("scalar_stack_stores") or row.get("scalar_stack_loads"):
        lines.append(f'<small>Scalar 写 {row.get("scalar_stack_stores",0)} / 读 {row.get("scalar_stack_loads",0)}</small>')
    return f'<span>{labels[kind]}</span>'+''.join(lines)


def cce_links(row):
    originals=[file for file in row.get("files",{}).get("original_cce",[]) if file["path"].endswith(".cce")]
    primary=next((file for file in originals if Path(file["path"]).name=="kernel.cce"),originals[0] if originals else None)
    candidates=originals+[file for file in row.get("files",{}).get("comparison",[]) if file["path"].endswith(".cce")]
    secondary=None
    for name,label in (("hardware_equivalent.cce","硬件等效 CCE（含 spill）"),
                       ("compiled_vf_replay.cce","编译后动态回放"),("compiled_pc_replay.cce","编译后 PC 回放")):
        found=next((file for file in candidates if Path(file["path"]).name==name),None)
        if found and (primary is None or found["path"]!=primary["path"]):
            secondary=dict(found,label=label);break
    return primary,secondary


def merge_historical_results(data):
    from spill_historical_import import import_archive

    historical = read(DEST / "metadata/data.json")
    base = [row for row in data["rows"] if row.get("origin") not in {"historical_membar","supplemental_search","round3_search"}]
    for row in base:
        row.update(origin="fixed_unroll_search", variant=f"u{row['unroll']}", variant_label=f"U{row['unroll']}",
                   variant_order=row["unroll"], shape="1024 元素 / 16 个向量迭代",
                   chart_label=f"U{row['unroll']}", comparison_group="fixed-"+row["case"],
                   comparison_label="1024 元素 / 本轮参数", note=row.get("note", ""))
    rows, note = import_archive(historical, DEST)
    data["rows"] = base + rows
    data["case_order"] = CASE_ORDER + [case["id"] for case in historical["cases"]]
    data["titles"].update({case["id"]:case["title"] for case in historical["cases"]})
    data["historical_import"] = note
    target = DEST / "metadata/spill_historical_import.json"
    target.write_text(json.dumps(note, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    data["evidence"] = [item for item in data["evidence"] if item["path"] not in
                        {"metadata/data.json", "metadata/spill_historical_import.json"}]
    data["evidence"] += [dict(label="历史结果与建模口径",path="metadata/data.json"),
                         dict(label="历史导入与同 unroll 软件对照",path="metadata/spill_historical_import.json")]


def merge_supplemental_results(data, manifest):
    from spill_supplemental_import import import_round2, apply_followup
    rows, titles = import_round2(REPO, DEST, ARCHIVE, manifest, copy_tree)
    data["rows"].extend(rows)
    data["titles"].update(titles)
    data["case_order"].extend(titles)
    from spill_round3_import import import_round3
    new_rows, new_titles, note = import_round3(REPO, DEST, ARCHIVE, manifest, copy_tree)
    data["rows"].extend(new_rows)
    data["titles"].update(new_titles)
    data["case_order"].extend(new_titles)
    if note:
        data["round3_import"]=note
        for name,label in (("prediction_provenance.json","第三批模型与 A/B 口径"),
                           ("summary.json","第三批数值验证与 CAModel")):
            path=f"metadata/round3/{name}"
            data["evidence"]=[item for item in data["evidence"] if item["path"]!=path]
            data["evidence"].append(dict(label=label,path=path))
    apply_followup(data, REPO, DEST, ARCHIVE, manifest, copy_tree)
    source=REPO/'cce_code/spill_software_followup/provenance.json'
    if source.exists():
        target=DEST/'metadata/spill_followup_provenance.json'
        shutil.copyfile(source,target)
        data['evidence']=[entry for entry in data['evidence'] if entry['path']!=target.relative_to(DEST).as_posix()]
        data['evidence'].append(dict(label='补充软件寻优：模型、计划与候选数量',path=target.relative_to(DEST).as_posix()))
    data["date"] = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def copy_tree(source, target, manifest):
    if not source.exists():
        return []
    links = []
    for original in sorted(source.rglob("*")):
        if not original.is_file() or original.name == "sim_history.json":
            continue
        if "__pycache__" in original.parts or original.suffix in {".pyc", ".o", ".so"}:
            continue
        final = target / original.relative_to(source)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists() and not final.is_symlink():
            final.chmod(final.stat().st_mode | 0o200)
        shutil.copyfile(original, final)
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if digest != hashlib.sha256(final.read_bytes()).hexdigest():
            raise ValueError(f"Archive checksum differs: {final}")
        relative = final.relative_to(DEST).as_posix()
        manifest.append(dict(path=relative, source=original.relative_to(REPO).as_posix(), sha256=digest))
        links.append(dict(label=original.relative_to(source).as_posix(), path=relative))
    return links


def build_dataset():
    originals = read(SOURCE / "ub_dependency_comparison/summary.json")
    titles = {r["case"]: r["title"] for r in read(SOURCE / "summary.json")}
    software = {(r["case"], r["unroll"]): r for r in read(SOURCE / "software_optimization/summary.json")}
    rows, manifest = [], []
    for case in CASE_ORDER:
        copy_tree(SOURCE / "cases" / case / "ascendc", ARCHIVE / case / "reference/ascendc", manifest)
        for name in ("golden_fp64.bin", "layout.json"):
            original = SOURCE / "cases" / case / name
            if original.exists():
                temporary = ARCHIVE / case / "reference"
                temporary.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, temporary / name)
                manifest.append(dict(path=(temporary / name).relative_to(DEST).as_posix(),
                                     source=original.relative_to(REPO).as_posix(),
                                     sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    for original in sorted(originals, key=lambda r: (CASE_ORDER.index(r["case"]), r["unroll"])):
        case, unroll = original["case"], original["unroll"]
        target = ARCHIVE / case / f"u{unroll}"
        source = SOURCE / "cases" / case if unroll == 1 else SOURCE / "unroll_manual" / case / f"u{unroll}"
        files = dict(original_cce=copy_tree(source / "cce", target / "original/cce", manifest),
                     comparison=copy_tree(SOURCE / "ub_dependency_comparison" / case / f"u{unroll}", target / "original/comparison", manifest))
        result = software.get((case, unroll))
        best = result["best"] if result else None
        if best:
            if best["status"] != "validated_no_spill" or not best["validation"]["ok"] or best["compiler_stack_instruction_count"]:
                raise ValueError("Only validated, spill-free software candidates may be published")
            trial = SOURCE / best["artifact_dir"]
            digest = hashlib.sha256((trial / "cce/kernel.cce").read_bytes()).hexdigest()
            if digest != best["plan"]["source_sha256"]:
                raise ValueError("Validated software source changed")
            files["software"] = copy_tree(trial, target / "software", manifest)
        target.mkdir(parents=True, exist_ok=True)
        measured = original.get("camodel_cycles")
        if original["status"] == "predicted":
            global_cycles, local_cycles = original["global_cycles"], original["local_cycles"]
            reason = ""
        else:
            global_cycles, local_cycles = None, None
            reason = ("CCEC 编译失败：VF stack 超过默认 6144B 容量。" if original["status"] == "vf_stack_exceeded"
                      else "编译器插入了标量 spill 指令 RV_SSTI/SLDI，当前 VfSim 不建模其执行；不忽略这些指令生成不完整预测。")
        plan = best["plan"] if best else None
        software_cycles = best["global_cycles"] if best else global_cycles if not original.get("spill") else None
        row = dict(id=f"{case}-u{unroll}", case=case, title=titles[case], unroll=unroll,
                   status=original["status"], spill=bool(original.get("spill")), reason=reason,
                   camodel=measured, global_cycles=global_cycles, local_cycles=local_cycles,
                   original_accuracy=accuracy(global_cycles, measured),
                   local_gain=100 * (1 - local_cycles / global_cycles) if global_cycles and local_cycles is not None else None,
                   software_cycles=software_cycles, software_camodel=best["camodel_cycles"] if best else None,
                   software_accuracy=accuracy(best["global_cycles"], best["camodel_cycles"]) if best else None,
                   software_gain=100 * (1 - software_cycles / global_cycles) if best and global_cycles else None,
                   strategy=("切分＋重排" if plan["stages"] > 1 else "仅重排") if plan else "保留原无 spill 版本" if not original.get("spill") else "未搜索",
                   plan=plan, software_validation=best["validation"] if best else None,
                   candidate_count=result["candidate_count"] if result else 0,
                   finalists=len(result["validated"]) if result else 0, files=files,
                   evidence_path=(target / "result.json").relative_to(DEST).as_posix())
        if original["status"] == "vf_stack_exceeded":
            row["strategy"] = "编译失败，未搜索"
        complete_retained_baseline(row)
        (target / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
    evidence = []
    for name, folder in (("ub", "ub_dependency_comparison"), ("software", "software_optimization")):
        for filename in ("provenance.json", "baseline_audit.json"):
            original = SOURCE / folder / filename
            if not original.exists():
                continue
            final = DEST / "metadata" / f"spill_{name}_{filename}"
            shutil.copyfile(original, final)
            evidence.append(dict(label=f"{name} / {filename}", path=final.relative_to(DEST).as_posix()))
    return dict(rows=rows, case_order=CASE_ORDER, titles=titles, evidence=evidence,
                shape="1024 元素 / 16 个向量迭代", hardware="A5 / DV100 / Ascend950PR_9599",
                date="2026-10-12", candidate_count=sum(r["candidate_count"] for r in software.values()),
                finalists=sum(len(r["validated"]) for r in software.values()), manifest=manifest)


def metric(value, digits=0):
    return "--" if value is None else f"{value:.{digits}f}" + ("%" if digits else "")


def row_html(row):
    values = [row.get("unroll"), row["camodel"], row["global_cycles"], row["local_cycles"],
              metric(row["original_accuracy"], 2), metric(row["local_gain"], 2), row["software_cycles"],
              row["software_camodel"], metric(row["software_accuracy"], 2),
              metric(row.get("membar_vs_software_performance"), 2)]
    keys = ("unroll", "camodel", "global_cycles", "local_cycles", "original_accuracy", "local_gain",
            "software_cycles", "software_camodel", "software_accuracy", "membar_vs_software_performance")
    cells = "".join(f'<td class="num" data-metric="{key}">{html.escape(str(v)) if v is not None else "--"}</td>' for key, v in zip(keys, values))
    cells += '<td class="num" data-metric="a6_vs_a5_local_performance">--</td>'
    status = "栈超限" if row["status"] == "vf_stack_exceeded" else spill_status_html(row)
    variant = html.escape(row.get("variant_label",f"U{row['unroll']}"))
    shape = html.escape(row.get("comparison_label","1024 元素"))
    origin = "第三批实测" if row.get("origin")=="round3_search" else "历史归档" if row.get("origin")=="historical_membar" else "本轮寻优"
    primary,secondary=cce_links(row)
    title=html.escape(row["title"])
    if primary:
        title=f'<a class="cce-link" href="{html.escape(primary["path"])}" target="_blank" rel="noopener" title="查看此版本的原始 CCE 源码">{title}</a>'
    replay=(f'<small><a href="{html.escape(secondary["path"])}" target="_blank" rel="noopener">{html.escape(secondary["label"])}</a></small>' if secondary else "")
    return (f'<tr data-id="{row["id"]}"><th scope="row">{title}<small>{row["case"]}</small>{replay}</th>'
            + f'<td class="variant">{variant}<small>{shape} · {origin}</small></td>'
            + f'<td class="num" title="{html.escape(row.get("loop_count_detail",""))}">{html.escape(row.get("loop_count_label") or "--")}</td>'
            + cells + f'<td class="strategy">{html.escape(row["strategy"])}</td><td class="status" title="{html.escape(row.get("spill_count_basis",""))}">{status}</td>'
            + f'<td><button class="detail-button" data-id="{row["id"]}" aria-label="查看 {row["case"]} {variant} 的证据">详情</button></td></tr>')


def build_page(data):
    replay_scripts = ('<script defer src="metadata/replay_validation.js"></script>\n'
                      '<script defer src="assets/replay_validation_view.js"></script>') if data.get("replay_validation_available") else ""
    columns = [("unroll", "U"), ("camodel", "原 CAModel"), ("global_cycles", "VfSim 全局"),
               ("local_cycles", "VfSim 局部"), ("original_accuracy", "全局精度"), ("local_gain", "局部周期减少"),
               ("software_cycles", "VfSim 软件优化全局"), ("software_camodel", "优化后 CAModel"), ("software_accuracy", "优化后精度"),
               ("membar_vs_software_performance", "Membar / 软件性能"),
               ("a6_vs_a5_local_performance", "A6 / A5 局部性能")]
    headers = '<th scope="col" class="num"><button data-sort="loop_count" title="按未展开前的主循环次数排序">循环次数 &#8597;</button></th>'
    headers += ''.join(f'<th scope="col" class="num"><button data-sort="{key}" title="按{label}排序">{label} &#8597;</button></th>' for key,label in columns)
    operators = ''.join(f'<option value="{case}">{html.escape(data["titles"][case])}</option>' for case in data["case_order"])
    evidence = ' · '.join(f'<a href="{item["path"]}">{html.escape(item["label"])}</a>' for item in data["evidence"])
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A5 / A6 Spill 与同步优化 · 统一汇总</title>
  <link rel="stylesheet" href="assets/spill_report.css">
  <script defer src="assets/chart.umd.js"></script>
  <script defer src="metadata/spill_candidate_alignment.js"></script>
  <script defer src="metadata/multisoc_predictions.js"></script>
  <script defer src="metadata/spill_performance_summary.js"></script>
  <script defer src="metadata/software_cce_links.js"></script>
  <script defer src="assets/multisoc_view.js"></script>
  <script defer src="assets/spill_report.js"></script>
  <script defer src="assets/spill_performance_summary.js"></script>
  {replay_scripts}
</head>
<body>
<header><div><p id="arch-heading">A5 / DV100 · A6 / DV121 · MultiSoC-membar-refactor</p><h1>Spill 与同步优化结果</h1></div>
<nav aria-label="归档导航"><a href="multisoc_refresh_20261025/comparison.md">A5/A6 对比说明</a><a href="spill_candidate_alignment_summary.md">历史实验总结与字段说明</a><a href="spill_candidate_alignment_multisoc.csv" download>双架构结果 CSV</a><a href="metadata/spill_candidate_manifest.json">文件校验清单</a><a href="a6_softmax_fusion_results_summary.html">A6 Softmax 融合与多 Shape</a><a href="a6_spill_dag_comparison_20261027.html">A6 Spill DAG 对比</a><a href="explicit_spill_roundtrip.md">历史显式 Spill 验证</a>{'<a href="replay_recompile_20261025/README.md">回放重编译验证状态</a>' if data.get('replay_validation_available') else ''}</nav></header>
<main>
<section class="totals" aria-label="实验统计">
<div><span>Case 分组</span><strong>{len(data['case_order'])}</strong></div><div><span>Spill / 无 Spill 归档版本</span><strong>{sum(row['spill'] for row in data['rows'])} / {sum(not row['spill'] for row in data['rows'])}</strong></div><div><span>历史验证无 spill 软件方案</span><strong>{sum(row.get('plan') is not None for row in data['rows'])}</strong></div><div><span>历史搜索候选 / 编译验证</span><strong>{data['candidate_count']} / {data['finalists']}</strong></div>
</section>
<section aria-labelledby="table-title">
<div class="section-title"><h2 id="table-title">统一汇总</h2><span id="row-count" role="status">{len(data['rows'])} 个版本</span></div>
<div class="filters"><label>Architecture<select id="arch-filter"><option value="A5">A5 arch · DV100</option><option value="A6">A6 arch · DV121</option></select></label>
<label>显示范围<select id="version-scope"><option value="main">Spill / U1 基线</option><option value="all">全部成功 unroll</option></select></label>
<label>算子<select id="case-filter"><option value="all">全部算子</option>{operators}</select></label>
<label>来源<select id="origin-filter"><option value="all">全部来源</option><option value="fixed_unroll_search">首批固定 unroll 寻优</option><option value="supplemental_search">第二批新增例子</option><option value="round3_search">第三批 11 个入口</option><option value="historical_membar">历史 Membar 归档</option></select></label>
<label>版本<select id="status-filter"><option value="all">Spill 与基线</option><option value="spill">register spill</option><option value="no-spill">无 register spill</option><option value="unsupported">原版本未预测</option></select></label>
<label class="search-label">搜索<input id="search" type="search" placeholder="算子名称" autocomplete="off"></label></div>
<p id="architecture-note" role="status"></p>
{'<p>回放验证标记仅针对 A5 重编译：允许等价编码变化，但数据流、实际 UB 字节区间、屏障和 VF 总周期必须完全一致。未通过验证的回放仍是审计清单，不能当作同周期可编译源码；本轮不覆盖原有预测和 CAModel 数字。</p>' if data.get('replay_validation_available') else ''}
<div class="table-scroll" tabindex="0" role="region" aria-label="周期与精度汇总表">
<table><caption>单位：VF cycles。CAModel 均为 A5 实测参考；A6 没有同代实测，精度列留空。架构切换同时更新表格、详情和图表。</caption><thead><tr><th scope="col"><button data-sort="case">Case &#8597;</button></th><th scope="col">版本 / Shape 与来源</th>{headers}<th scope="col">软件策略</th><th scope="col">原版本 / Spill 写读次数</th><th scope="col">证据</th></tr></thead>
<tbody id="results-body">{''.join(row_html(row) for row in data['rows'])}</tbody></table></div>
<p id="empty-state" hidden>没有匹配的版本。</p>
</section>
<section id="spill-performance-section" aria-labelledby="spill-performance-title">
<h2 id="spill-performance-title">硬件 Membar 优化达到软件优化的性能比例</h2>
<p>固定历史 35 个 spill 版本：Vector 14、Predicate 13、两者都有 8。不是 35 个不同算子。统计跟随 Architecture 选择，不随主表筛选改变。</p>
<p>性能比 = <strong>软件优化全局 VfSim 周期 / 原始 spill 流局部依赖 VfSim 周期 × 100%</strong>。100% 为持平，高于 100% 表示仅做硬件 Membar 优化更快。括号内占比以有效比较数为分母。</p>
<div class="table-scroll"><table><caption>A5 保留历史 35 项，并另列同模型复算；A6 使用本轮 A6 重新寻优结果。缺失不按零值补齐。</caption>
<thead><tr><th>数据口径</th><th>Spill 类型</th><th>范围</th><th>有效</th><th>缺失</th><th>≥100%</th><th>90–&lt;100%</th><th>80–&lt;90%</th><th>70–&lt;80%</th><th>&lt;70%</th></tr></thead>
<tbody id="spill-performance-body"></tbody></table></div>
<p id="spill-performance-missing" role="status"></p>
<p id="spill-performance-scalar-note" role="status"></p>
<p>A6 候选保持 shape、算术、名义 unroll 和 tail，通过重排或最多五段切分寻优，保留全局屏障及分段搬运。A6 target 展开后完整动态流的活跃值上限：<strong>Vector ≤32、Predicate ≤8</strong>，包含前缀、循环不变量、exit 和 V2 临时值。不改变物理资源或时序参数。</p>
<p>这是有限候选中的最优，不是理论最优。A5 软件预测来自通过 CAModel 数值/无 spill 验证的编译后日志；A6 软件方案已按原周期排名做 VF 编译期寄存器 spill 检查，首个通过者入选。A6 软件周期仍是源级 Canonical 预测，不是二进制回放，也没有 A6 数值验证；编译探针不含外围 DMA/AscendC 包装。硬件侧保留原始 A5 编译后 spill 流，只改同步方式。<a href="a5_software_search_20261026/README.md">A5 重新寻优与实测</a> · <a href="a6_software_search_20261025/README.md">A6 寻优记录</a> · <a href="a6_compiler_validation_20261027/README.md">A6 编译期检查</a> · <a href="metadata/spill_performance_summary.json">完整数据</a></p>
</section>
<section class="chart-section" aria-labelledby="chart-title"><div class="section-title"><h2 id="chart-title">版本周期对比</h2><div class="chart-pickers"><label>算子<select id="chart-case">{operators}</select></label><label>对照组<select id="chart-group"></select></label></div></div>
<div class="chart"><canvas id="cycles-chart" role="img" aria-label="同一算子的 unroll 周期对比"></canvas></div></section>
<section class="scope" aria-labelledby="scope-title"><h2 id="scope-title">比较口径</h2>
<p>架构选择使用同一 MultiSoC-membar-refactor 提交重跑的 A5/A6 结果；两侧复用 A5 编译日志还原的指令流、spill、Membar 和字节地址，再应用各自的 SoC 指令展开。没有重新使用 A6 编译器；Spill 次数仍描述该 A5 编译指令流。只去掉方向屏障的局部实验仍保留 UB 地址依赖。完整结果和配置哈希见 <a href="multisoc_refresh_20261025/summary.json">双架构运行记录</a>与 <a href="multisoc_refresh_20261025/provenance.json">模型来源</a>。旧表数据保留在详情中，不作为新架构的兜底值。</p>
<p>A6 / A5 局部性能 = 同一版本 A5 局部周期 / A6 局部周期 × 100%，高于 100% 表示 A6 更快。历史 35 项范围的 A6 软件列改用本轮资源约束寻优结果，其他行仍是历史选定实现的 A6 预测。新 A6 方案没有 CAModel 实测，旧软件实测不得当作新方案的精度依据。未支持输入显示空值和诊断，不视为 0 cycles。</p>
<details><summary>历史归档的测量、寻优与比较口径（以下旧周期不作为架构切换的兜底数据）</summary>
<p>原全局和局部列使用同一份编译后指令流，仅改变同步机制；软件优化列使用重新编译后的另一份源码，保持原 shape、算术及名义 unroll，保留全局 Membar。软件列是本轮有限搜索的最佳已验证结果，不是理论最优。</p>
<p>“软件性能”指该软件改写方案在全局 Membar 下的 VfSim 预测性能，按相同工作量的周期数倒数比较；不是 CAModel 周期或预测程序运行耗时。例如软件为 100 cycle、局部依赖为 110 cycle，Membar / 软件性能为 90.91%，不是周期减少 90.91%。</p>
<p>Logit 和 MHC Partial Merge 的残留 AscendC 接口已改为显式 CCE 并重新验证。Case 链接指向本行 CAModel 实际编译源码，预测来自该版本新编译日志，源代码及输入、二进制、日志 SHA256 在详情中可查；转换前结果独立保留为历史证据。标量 spill 版本仍明确拒绝预测，不因换成纯 CCE 而忽略指令。</p>
<p>第二批 Mish、LogSigmoidGrad、概率输入 BinaryCrossEntropy 使用 FP32、3072 元素、48 次循环。其局部实验快照全局周期与当前主线逐项一致，动态顺序、访存字节地址与 RAW/WAR 解除边界已核对；LogSigmoidGrad U4 全局精度约 80%，保留该偏差，不声称已校准。第二批 U12 标量栈指令仍明确拒绝。</p>
<p>第三批 11 个新入口的基准均通过 AscendC 逐位一致、独立 golden 和 guard；另测 30 个 U2/U4/U8 版本，26 个通过、4 个 VF stack 超限。主表保留 U1 和真实 V/P/S spill，未开展软件寻优的列留空。局部结果只采用与当前全局周期一致且动态顺序、字节地址及 done+1 边界通过审计的结果；全零 mask store 按名义范围保守阻塞并单列，不声称其真实写入。它是实验预测，不是局部依赖硬件的 CAModel 实测。</p>
<p>第三批 GroupNormSwishGrad 的 U 按原双块 pair 继续展开，U4 每轮是八个 VL。Compressor 保留 DND 布局，U1 已有 spill 和原阶段 Membar，其局部收益不只来自 spill。Softplus 原 DAG 的 12 个 VF 与融合 CCE 不比较整体性能；表中 CAModel 是对应 CCE 测量。MHC 的 unaligned 状态和 CausalConv 的指针 reset 已用编译后日志对照的手工 JSON 补齐，未删 spill 和 Membar；MHC 仍有明显时序偏差。scalar spill 不静默忽略，未预测原因在详情中保留。</p>
<p>补充寻优覆盖第二批与历史缺项共 16 行：保持原 unroll，不以回退 U1 代替优化。软件值来自重新编译后的真实指令流，已通过数值和无 compiler spill 检查；部分历史模型及参数版本与当前软件模型不同，原记录不覆盖，跨版本比例只能参考。cohort 限制同时推进的 lane，不改变展开组工作量和循环步长。MHC 的真实 19 次回边迭代不当作独立 lane。</p>
<p>主表、CSV 与图表只展示发生 register spill 的版本，以及每个算子/shape/参数组的 U1 基线。其他无 spill 的 U2/U4 等版本不单独列出，但同 unroll 软件优化对照及其证据仍保留。Softmax 原记录没有标注 unroll，经源码核对所有循环均使用 unroll(1)，因此作为 U1 基线。</p>
<p>原版本标明 Vector、Predicate 和 Scalar spill。写/读数字是整个 VF 的动态 spill 搬运次数：Vector 对应 VST/VLD，Predicate 对应 PST/PLD，Scalar 对应 SST/SLD；不是溢出寄存器的个数，也不是普通输入输出访存总数。写读次数不一定相等，同一个保存值可以多次重载；额外未配对读取另记，不混入 spill。</p>
<p>循环次数记录未展开前的主循环，不是 unroll 后的循环趟数。多层循环同时标明行、矩阵或 SIMD batch 与内层次数；MHC 的 20 轮包含首轮初始化和后续 19 次循环。完整层级说明在循环次数单元格及版本详情中保留。</p>
<p>历史页面的 8 个分组、51 个版本按原记录合入；GeLU Grad U8 已另外按原编译后完整 PC 流补跑当前全局/局部预测，原归档记录仍保留。其他历史版本没有在本次重新仿真，可能使用冻结模型、源码级预测或不同 GSU 校准参数，具体限制在版本详情中保留。MHC 图表按矩阵数和参数版本分组，不将不同 shape 或参数版本连成同一条曲线。</p>
<p>历史软件对照仅使用已存在、同 unroll、数值通过且无已记录 spill 的版本；按 VfSim 周期选对照，不按 CAModel 周期选。GeLU Grad U1 有三段切分对照，FlashAttention 集中加载 U2 有同 U2 重排对照，MHC 128 矩阵 U1 有 4+15 切分对照。MHC U2 不能使用 U1 切分结果；尚无同 unroll 对照时软件列留空，不用忽略 compiler spill 的源码预测代填。</p>
<p>汇总表、CSV 和图表只包含成功编译的版本；VF stack 超限的版本不参与展示，其原始编译失败记录仍保留在归档中。</p>
<p>原无 spill 版本未做本轮软件寻优，软件侧的 VfSim、CAModel 和精度均沿用原版本，并标记“保留原无 spill 版本”；这些不是新测量。缺少预测时显示“--”，不视为 0 cycles。Adam U7、GroupNorm U16、Logit U8 的原始流含未建模的标量 spill；优化后的流已消除 spill，因此仍有软件预测。</p>
<p>入选软件方案通过 AscendC/U1 逐位一致、独立 golden 与输出 guard 检查，并确认无编译器 stack spill；分段新增的显式临时 UB 搬运不算编译器 spill。51 组原始全局预测已逐项复核一致。现有 forwarding/II fallback 仍保留，不将所有时序参数声称为已实测。</p>
<p>周期预测精度 = (1 − |VfSim − CAModel| / CAModel) × 100%；周期减少 = (1 − 局部 / 全局) × 100%。图表中缺失数据保留缺口。</p>
<p>Membar / 软件性能 = VfSim 软件优化全局周期 / VfSim 局部依赖周期 × 100%，两侧统一使用 VfSim。100% 表示持平，大于 100% 表示 Membar 优化更快，小于 100% 表示软件优化更快。任一预测缺失时不计算；原无 spill 行仍以保留的原全局版本作为软件侧对照。</p>
</details><p class="provenance">{evidence}</p></section>
</main>
<dialog id="detail-dialog" aria-labelledby="detail-title"><div class="dialog-heading"><h2 id="detail-title"></h2><button id="close-detail" aria-label="关闭详情" title="关闭详情">&#215;</button></div><div id="detail-body"></div></dialog>
<footer>报告更新：{data['date']}。原测量来源与参数版本分别保留，未调整核心时序参数。全部引用资源在本目录内。</footer>
</body></html>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-page-only", action="store_true", help="Reuse archived evidence; regenerate page and checksums")
    parser.add_argument("--preserve-imported", action="store_true", help="Keep already imported historical/software rows; apply only audited prediction/source updates")
    parser.add_argument("--refresh-historical-software", action="store_true",
                        help="Refresh historical software references from validated archived measurements")
    args = parser.parse_args()
    if args.preserve_imported and not args.refresh_page_only:
        parser.error("--preserve-imported requires --refresh-page-only")
    if args.refresh_historical_software and not (args.refresh_page_only and args.preserve_imported):
        parser.error("--refresh-historical-software requires --refresh-page-only --preserve-imported")
    (DEST / "metadata").mkdir(parents=True, exist_ok=True)
    if args.refresh_page_only:
        data = read(DEST / "metadata/spill_candidate_alignment.json")
        all_versions_path = DEST / "metadata/spill_candidate_all_versions.json"
        if all_versions_path.exists():data["rows"] = read(all_versions_path)
        manifest = [entry for entry in read(DEST / "metadata/spill_candidate_manifest.json")
                    if entry["source"] != "generated report or local supporting asset"]
    else:
        data = build_dataset()
        manifest = data.pop("manifest")
    excluded_path = DEST / "metadata/spill_candidate_excluded_versions.json"
    excluded = {row["id"]: row for row in read(excluded_path)} if excluded_path.exists() else {}
    excluded.update({row["id"]: row for row in data["rows"] if row["status"] == "vf_stack_exceeded"})
    excluded_path.write_text(json.dumps(list(excluded.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data["rows"] = report_rows(data["rows"])
    if not args.preserve_imported:
        merge_historical_results(data)
        merge_supplemental_results(data, manifest)
    elif args.refresh_historical_software:
        from spill_historical_import import comparable_software_pairs, pairs_to_json
        pairs = comparable_software_pairs(read(DEST / "metadata/data.json")["cases"], DEST)
        for row in data["rows"]:
            counterpart = pairs.get((row["case"], row.get("variant")))
            if row.get("origin") == "historical_membar" and counterpart:
                row.update(software_comparison=counterpart, software_cycles=counterpart["cycles"],
                           software_camodel=counterpart["camodel"], strategy=counterpart["strategy"])
                row["files"]["software"] = (counterpart["files"]["original_cce"]
                                            + counterpart["files"]["comparison"])
        data["historical_import"]["software_pairs"] = pairs_to_json(pairs)
        (DEST / "metadata/spill_historical_import.json").write_text(
            json.dumps(data["historical_import"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    from spill_historical_import import apply_compiled_replay_updates
    if not args.refresh_historical_software:
        apply_compiled_replay_updates(data, DEST, ARCHIVE, manifest, copy_tree)
    from spill_pure_cce_import import apply_pure_cce_updates
    if not args.refresh_historical_software:
        apply_pure_cce_updates(data, REPO, DEST, ARCHIVE, manifest, copy_tree)
    from spill_historical_import import loop_metadata
    for row in data["rows"]:
        if row["origin"]!="round3_search":
            row.update(loop_metadata(row,DEST))
        populate_spill_statistics(row)
        original_cce,compiled_cce=cce_links(row)
        if original_cce is None:raise ValueError(f"No CCE source for {row['id']}")
        row["source_cce_path"]=original_cce["path"]
        row["compiled_cce_path"]=compiled_cce["path"] if compiled_cce else None
        complete_retained_baseline(row)
        row["original_accuracy"] = accuracy(row["global_cycles"],row["camodel"])
        row["software_accuracy"] = accuracy(row["software_cycles"],row["software_camodel"])
        row["local_gain"] = 100*(1-row["local_cycles"]/row["global_cycles"]) if row["local_cycles"] is not None and row["global_cycles"] else None
        row["software_gain"] = 100*(1-row["software_cycles"]/row["global_cycles"]) if row["software_cycles"] is not None and row["global_cycles"] else None
        row["membar_vs_software_performance"] = membar_vs_software_performance(row["software_cycles"], row["local_cycles"])
        row["software_reference"] = (f"{row['software_comparison']['case']}/{row['software_comparison']['variant']}" if row.get("software_comparison") else "")
        (DEST / row["evidence_path"]).parent.mkdir(parents=True,exist_ok=True)
        (DEST / row["evidence_path"]).write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_versions_path = DEST / "metadata/spill_candidate_all_versions.json"
    all_versions_path.write_text(json.dumps(data["rows"],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    data["archive_version_count"] = len(data["rows"])
    data["rows"] = spill_and_baseline_rows(data["rows"])
    data["case_order"] = [case for case in data["case_order"] if any(row["case"]==case for row in data["rows"])]
    for row in data["rows"]:
        (DEST/row["evidence_path"]).write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    (DEST / "metadata/spill_candidate_alignment.json").write_text(text + "\n", encoding="utf-8")
    (DEST / "metadata/spill_candidate_alignment.js").write_text("window.SPILL_REPORT = " + text + ";\n", encoding="utf-8")
    fields = ["case", "title", "origin", "variant", "variant_label", "selection_role", "shape", "comparison_group", "loop_count", "loop_count_label", "loop_count_detail", "unroll", "status", "spill", "spill_kind", "vector_spill_store", "vector_spill_load", "predicate_spill_store", "predicate_spill_load", "scalar_stack_stores", "scalar_stack_loads", "other_extra_vector_loads", "spill_count_basis", "camodel", "global_cycles", "local_cycles",
              "original_accuracy", "local_gain", "software_cycles", "software_camodel", "software_accuracy",
              "membar_vs_software_performance", "software_reference", "source_cce_path", "compiled_cce_path", "strategy", "reason", "note"]
    with (DEST / "spill_candidate_alignment.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["rows"])
    PAGE.write_text(build_page(data), encoding="utf-8")
    import summarize_spill_candidate_report as summary
    dataset_sha = hashlib.sha256((DEST / "metadata/spill_candidate_alignment.json").read_bytes()).hexdigest()
    stats = summary.statistics(data, read(all_versions_path), read(excluded_path))
    (DEST / "spill_candidate_alignment_summary.md").write_text(summary.markdown(data, stats, dataset_sha), encoding="utf-8")
    (DEST / "metadata/spill_candidate_statistics.json").write_text(json.dumps(
        dict(source_dataset_sha256=dataset_sha, date=data["date"], **stats), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    generated = [PAGE, DEST / "spill_candidate_alignment.csv", DEST / "metadata/spill_candidate_alignment.json",
                 DEST / "metadata/spill_candidate_alignment.js", DEST / "assets/spill_report.css",
                 DEST / "assets/spill_report.js", DEST / "assets/chart.umd.js", Path(__file__),
                 DEST / "scripts/check_spill_candidate_report.cjs", DEST / "scripts/validate_spill_candidate_archive.py", excluded_path,
                 DEST / "scripts/spill_historical_import.py",all_versions_path]
    generated += [DEST / "spill_candidate_alignment_summary.md", DEST / "metadata/spill_candidate_statistics.json",
                  DEST / "scripts/summarize_spill_candidate_report.py"]
    generated.append(DEST / "scripts/spill_supplemental_import.py")
    generated.append(DEST / "scripts/spill_round3_import.py")
    generated.append(DEST / "scripts/spill_pure_cce_import.py")
    generated += [DEST / item["path"] for item in data["evidence"]]
    generated += list(ARCHIVE.glob("*/u*/result.json"))
    generated += list((ARCHIVE/"imported").glob("*/*/result.json"))
    generated += list((ARCHIVE/"round3").glob("*/u*/result.json"))
    generated += list(ARCHIVE.glob("*/u*/original/spill_summary.json"))
    generated += [DEST / row["evidence_path"] for row in data["rows"]]
    linked = {file["path"] for row in data["rows"] if row["origin"]=="historical_membar" for files in row["files"].values() for file in files}
    linked.update(row["historical_case_path"] for row in data["rows"] if row["origin"]=="historical_membar")
    original_manifest = {entry["path"]:entry for entry in read(DEST/"metadata/manifest.json")}
    for relative in sorted(linked):
        entry = original_manifest.get(relative)
        if entry:
            manifest.append(dict(path=relative,source="existing Membar archive",sha256=entry["sha256"]))
        else:
            generated.append(DEST/relative)
    for path in generated:
        manifest.append(dict(path=path.relative_to(DEST).as_posix(), source="generated report or local supporting asset",
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest=list({entry["path"]:entry for entry in manifest}.values())
    (DEST / "metadata/spill_candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(page=str(PAGE), rows=len(data["rows"]), archived_files=len(manifest)), ensure_ascii=False))


if __name__ == "__main__":
    main()
