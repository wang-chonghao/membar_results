"""Draw archived Logit Grad U1/U4 compiled streams in program order."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import textwrap

from plot_flash_attention_spill_dag import records


ROOT = Path(__file__).resolve().parents[1]
CASE = Path("spill_candidate_alignment/pure_cce/logit-u4")
STEM = "logit_grad_u4_predicate_spill_order"
LANES = (
    ("vector_load", "输入 VLD", "#facc15"),
    ("predicate", "谓词生成 / 计算", "#60a5fa"),
    ("predicate_store", "Predicate spill PSTS", "#22c55e"),
    ("predicate_load", "Predicate reload PLDS", "#facc15"),
    ("compute", "向量计算 / VSEL", "#60a5fa"),
    ("vector_store", "输出 VST", "#22c55e"),
    ("barrier", "Membar", "#fca5a5"),
    ("scalar", "标量地址 / 循环辅助", "#d1d5db"),
)


def flatten(nodes):
    for node in nodes:
        if node["kind"] == "loop":
            yield from flatten(node["body"])
        else:
            yield node


def build_graph(root, unroll=4):
    folder = root / CASE if unroll == 4 else root / "spill_candidate_alignment/pure_cce/logit-u1"
    paths = dict(log=folder / "cce/core0.veccore0.instr_popped_log.dump",
                 canonical=folder / "prediction/current/canonical/vf_000.json",
                 addresses=folder / "prediction/experiment/memory_address_audit.json",
                 result=folder / "result.json")
    raw = records(paths["log"])
    canonical = json.loads(paths["canonical"].read_text())
    result = json.loads(paths["result"].read_text())
    if Counter(n["op"] for n in raw) != Counter(result["measurements"]["dynamic_rv_opcounts"]):
        raise ValueError("Archived RV instructions do not match the measured case")
    if not result["provenance"]["numerical_validation"]["bitwise_matches_original"]:
        raise ValueError("The compiled case lacks numerical validation")
    static = {}
    for node in flatten(canonical["context"]):
        pc = node.get("attributes", {}).get("rv_pc")
        if pc is None:
            pc = "0x" + re.search(r"\.pc([0-9a-f]+)\.", node["instruction_id"])[1]
        if pc in static:
            raise ValueError("Expected one canonical node per static PC")
        static[pc] = node
    accesses = json.loads(paths["addresses"].read_text())["expected"]
    memory = {item["dynamic_id"]: item for item in accesses}
    if len(memory) != len(accesses):
        raise ValueError("Duplicate dynamic memory identity")
    loop, = [n for n in raw if n["op"] == "RV_VLOOPv2"]
    loop_pc = int(loop["pc"], 16)
    words = int(re.search(r"instr_num\s*:\s*(\d+)", loop["detail"])[1])
    occurrences = Counter()
    nodes, edges, last_write, memory_writes = [], [], {}, {}
    memory_reads = defaultdict(set)
    prior_lsu = {"read": [], "write": []}
    last_barrier, gates = None, {}
    for item in raw:
        node = dict(item)
        iid, op = node["id"], node["op"]
        inside = loop_pc < int(node["pc"], 16) <= loop_pc + 4*words
        node["iteration"] = occurrences[node["pc"]] if inside else None
        occurrences[node["pc"]] += 1
        node["section"] = "body" if inside else "prefix" if iid <= loop["id"] else "suffix"
        semantic = static.get(node["pc"])
        node["reads"], node["writes"] = [], []
        if semantic and semantic["kind"] == "instruction":
            for field, target in (("inputs", "reads"), ("outputs", "writes")):
                for operand in semantic[field]:
                    value = canonical["values"][operand["value_id"]]
                    if value["storage"] in ("Register", "PredicateRegister"):
                        node[target].append(value["logical_id"])
            field_registers = {prefix.lower()+index for prefix, index in
                               re.findall(r"\b([VP])[dnmg]\[(\d+)\]", node["detail"])}
            if not set(node["reads"]+node["writes"]) <= field_registers:
                raise ValueError(f"Canonical register operands disagree with raw log: {node['pc']}")
            instruction_class = semantic["instruction_class"]
            node["lane"] = ("predicate_load" if op == "RV_PLDI" else
                            "predicate_store" if op == "RV_PSTI" else
                            "vector_load" if instruction_class == "load" else
                            "vector_store" if instruction_class == "store" else
                            "predicate" if any(v.startswith("p") for v in node["writes"]) else "compute")
        elif op == "RV_SMEM_BAR":
            node["lane"] = "barrier"
        elif op.startswith("RV_S") and op != "RV_SEND":
            node["lane"] = "scalar"
        elif op in ("RV_VLOOPv2", "RV_SEND"):
            node["lane"] = "control"
        else:
            raise ValueError(f"Instruction semantics missing: {op}")
        for value in node["reads"]:
            if value not in last_write:
                raise ValueError(f"Vector/predicate live-in was not decoded: {value}")
            edges.append(dict(src=last_write[value], dst=iid, value=value,
                              kind="predicate" if value.startswith("p") else "vector"))
        # Scalar dependencies are retained in the evidence, but not overlaid on the chart.
        scalar_fields = {name: "s"+index for name, index in
                         re.findall(r"\b(Sd|Sn|Sm|Sn2|Sm2)\[(\d+)\]", node["detail"])}
        scalar_reads = [v for k, v in scalar_fields.items() if k != "Sd"]
        scalar_writes = [scalar_fields["Sd"]] if "Sd" in scalar_fields else []
        if op == "RV_SMOVK":
            scalar_reads += scalar_writes
        for value in scalar_reads:
            if value in last_write:
                edges.append(dict(src=last_write[value], dst=iid, value=value, kind="scalar"))
        last_write.update({v: iid for v in node["writes"]+scalar_writes})
        node["memory"] = memory.get(iid)
        if node["lane"] in ("predicate_load", "predicate_store", "vector_load", "vector_store"):
            access = node["memory"]
            if access is None or access["pc"] != node["pc"]:
                raise ValueError("Memory address audit is incomplete")
            key = access["byte_start"], access["byte_end"]
            previous = memory_writes.get(key)
            if previous is not None:
                edges.append(dict(src=previous, dst=iid, kind="memory_raw" if access["kind"] == "read"
                                  else "memory_waw", value=f"UB[{key[0]},{key[1]})"))
            elif node["lane"] == "predicate_load":
                raise ValueError("Predicate reload has no earlier store at its audited address")
            if access["kind"] == "write":
                edges.extend(dict(src=p, dst=iid, kind="memory_war", value=f"UB[{key[0]},{key[1]})")
                             for p in sorted(memory_reads[key]))
                memory_reads[key].clear()
                memory_writes[key] = iid
            else:
                memory_reads[key].add(iid)
            gate = gates.get("VST_VLD" if access["kind"] == "read" else "VLD_VST")
            if gate is not None:
                edges.append(dict(src=gate, dst=iid, kind="barrier_release", value=access["kind"]))
            prior_lsu[access["kind"]].append(iid)
        if node["lane"] == "barrier":
            barrier = node["detail"].split(":")[-1].strip()
            if barrier not in ("VLD_VST", "VST_VLD"):
                raise ValueError(f"Unsupported barrier: {barrier}")
            node["barrier_type"] = barrier
            wait = "read" if barrier == "VLD_VST" else "write"
            edges.extend(dict(src=p, dst=iid, kind="barrier_wait", value=wait) for p in prior_lsu[wait])
            if last_barrier is not None:
                edges.append(dict(src=last_barrier, dst=iid, kind="barrier_order", value="stream order"))
            last_barrier = gates[barrier] = iid
        nodes.append(node)
    return dict(nodes=nodes, edges=edges, counts=dict(Counter(n["op"] for n in nodes)),
                sources={key: dict(path=path.relative_to(root).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key, path in paths.items()},
                measurements={key: result[key] for key in ("camodel", "global_cycles", "local_cycles")},
                scalar_dependency_scope="Explicit RV scalar register fields only; scalar timings and implicit loop updates are not modeled",
                memory_dependency_scope="Audited equal byte intervals; register RAW, memory RAW/WAR/WAW and global barrier edges, not a cycle schedule")


def prepare(graph, unroll=4):
    nodes = [dict(n) for n in graph["nodes"] if n["iteration"] == 0]
    expected = dict(vector_load=8, predicate=24, predicate_store=13, predicate_load=14,
                    compute=44, vector_store=4, barrier=27, scalar=10)
    if unroll == 1:
        expected = dict(vector_load=2, predicate=6, compute=11, vector_store=1, scalar=2)
    if Counter(n["lane"] for n in nodes) != Counter(expected):
        raise ValueError(f"Unexpected compiled U{unroll} body; expected {sum(expected.values())} instructions")
    for sequence, node in enumerate(nodes, 1):
        node["sequence"] = sequence
    by_id = {n["id"]: n for n in nodes}
    incoming = [dict(e) for e in graph["edges"] if e["kind"] in ("vector", "predicate", "memory_raw")
                and e["dst"] in by_id]
    for edge in incoming:
        edge["src_sequence"] = by_id[edge["src"]]["sequence"] if edge["src"] in by_id else None
        edge["dst_sequence"] = by_id[edge["dst"]]["sequence"]
        if edge["src"] >= edge["dst"]:
            raise ValueError("A data dependency points backwards in program order")
    spill_edges = [e for e in incoming if e["kind"] == "memory_raw"
                   and by_id[e["dst"]]["lane"] == "predicate_load"]
    expected_loads, expected_stores = (14, 13) if unroll == 4 else (0, 0)
    if len(spill_edges) != expected_loads or len({e["src"] for e in spill_edges}) != expected_stores:
        raise ValueError("Predicate saves and reloads do not match the compiled variant")
    pages = [dict(start=start, end=start+47) for start in (1, 49, 97)] if unroll == 4 else []
    return nodes, incoming, pages


def render(nodes, edges, pages, output, metadata, font, unroll=4):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    stem_prefix = STEM if unroll == 4 else "logit_grad_u1_instruction_order"
    lanes = [entry for entry in LANES if any(n["lane"] == entry[0] for n in nodes)]
    y_by_lane = {key: len(lanes)-i for i, (key, _, _) in enumerate(lanes)}
    for part, page in enumerate([None, *pages]):
        start, end = (1, len(nodes)) if page is None else (page["start"], page["end"])
        detail = page is not None or unroll == 1
        shown = [n for n in nodes if start <= n["sequence"] <= end]
        positions = {n["id"]: (n["sequence"], y_by_lane[n["lane"]]) for n in shown}
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=0.20, right=0.975, top=0.81, bottom=0.30)
        for y in y_by_lane.values():
            ax.axhspan(y-0.36, y+0.36, color="#f3f5f7", zorder=0)
        for edge in edges:
            if edge["src"] not in positions or edge["dst"] not in positions:
                continue
            a, b = positions[edge["src"]], positions[edge["dst"]]
            color = {"vector": "#64748b", "predicate": "#9333ea", "memory_raw": "#15803d"}[edge["kind"]]
            ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={-0.17 if a[1] == b[1] else 0.045}",
                arrowstyle="-|>", mutation_scale=9 if detail else 6,
                shrinkA=8 if detail else 3, shrinkB=8 if detail else 3,
                linewidth=1.0 if detail else 0.65, color=color,
                linestyle="--" if edge["kind"] == "memory_raw" else "-", alpha=0.65, zorder=1))
        colors = {key: color for key, _, color in LANES}
        counts = Counter(n["lane"] for n in shown)
        for node in shown:
            x, y = positions[node["id"]]
            ax.scatter([x], [y], s=260 if detail else 35, color=colors[node["lane"]],
                       edgecolors="#334155", linewidths=0.6,
                       marker="D" if node["lane"] == "barrier" else "o", zorder=3)
            if detail:
                ax.text(x, y, str(node["sequence"]), ha="center", va="center", fontsize=8, zorder=4)
        ax.set_yticks([y_by_lane[key] for key, _, _ in lanes],
                      [f"{label} · {counts[key]}" for key, label, _ in lanes], fontsize=12)
        ax.tick_params(axis="y", length=0, pad=10)
        step = 5 if detail else 20
        ax.set_xticks(sorted({start, end, *(i for i in range(start, end+1) if i % step == 0)}))
        ax.set_xlim(start-1.2, end+1.2)
        ax.set_ylim(0.35, len(lanes)+0.6)
        ax.set_xlabel("编译后首轮循环体内的指令顺序编号（不是执行 cycle）", fontsize=15, labelpad=12)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        title = "U4 Predicate Spill" if unroll == 4 else "U1 无 Spill"
        fig.suptitle(f"Logit Grad · {title} 指令顺序与数据依赖", fontsize=22, y=0.96)
        subtitle = "首组 144 条指令，重复 4 次；每组 13 次 PSTS、14 次 PLDS、27 条 Membar" if page is None else f"细节 {part}/3：第 {start}–{end} 条指令"
        if unroll == 1:
            subtitle = "首轮 22 条指令，重复 16 次；无 PSTS、PLDS 或 Membar"
        fig.text(0.59, 0.88, subtitle, ha="center", fontsize=15)
        if detail:
            crossing = defaultdict(set)
            for edge in edges:
                if start <= edge["dst_sequence"] <= end and (edge["src_sequence"] is None or edge["src_sequence"] < start):
                    source = str(edge["src_sequence"]) if edge["src_sequence"] is not None else f"初始化ID{edge['src']}"
                    crossing[edge["dst_sequence"]].add(source)
            text = "跨段输入（目标←来源）：" + ("；".join(f"{dst}←{','.join(sorted(srcs))}"
                for dst, srcs in sorted(crossing.items())) or "无")
            fig.text(0.20, 0.205, textwrap.fill(text, 110, break_long_words=True, break_on_hyphens=False),
                     fontsize=9, va="top", color="#475569")
        legend = [Line2D([0], [0], color=color, linestyle=style, label=label) for color, style, label in (
            ("#64748b", "-", "Vector RAW"), ("#9333ea", "-", "Predicate RAW"), ("#15803d", "--", "Predicate spill UB RAW"))]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.59, 0.075), ncol=3, frameon=False, fontsize=12)
        footer = ("红菱形为真实 Membar；为便于阅读，不叠加屏障扇出和标量依赖。完整节点与依赖见 JSON。"
                  if unroll == 4 else "循环外 PSET、VDUP 常量未放入横轴；相关初始化输入列在上方，完整数据见 JSON。")
        fig.text(0.59, 0.035, footer,
                 ha="center", fontsize=10, color="#475569")
        stem = stem_prefix if page is None else f"{stem_prefix}_part{part:02d}"
        for ext in ("png", "svg", "pdf"):
            fig.savefig((metadata if ext == "pdf" else output) / f"{stem}.{ext}", dpi=160, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--unroll", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    output, metadata = args.root / "figure/DAG", args.root / "metadata/figure_sources/DAG"
    stem_prefix = STEM if args.unroll == 4 else "logit_grad_u1_instruction_order"
    if not args.overwrite and any(list(path.glob(stem_prefix+"*")) for path in (output, metadata)):
        parser.error("Logit diagrams already exist; use --overwrite to refresh")
    graph = build_graph(args.root, args.unroll)
    nodes, edges, pages = prepare(graph, args.unroll)
    output.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    graph.update(displayed_nodes=nodes, displayed_edges=edges, pages=pages,
                 displayed_dependency_counts=dict(Counter(e["kind"] for e in edges)))
    (metadata / f"{stem_prefix}.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (metadata / f"{stem_prefix}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "dynamic_id", "pc", "opcode", "lane", "reads", "writes", "byte_start", "byte_end", "detail"))
        for node in nodes:
            memory = node["memory"] or {}
            writer.writerow((node["sequence"], node["id"], node["pc"], node["op"], node["lane"],
                " ".join(node["reads"]), " ".join(node["writes"]), memory.get("byte_start"), memory.get("byte_end"), node["detail"]))
    render(nodes, edges, pages, output, metadata, args.font, args.unroll)
    lines = ["# Logit Grad U4 Predicate Spill DAG", "",
        "依据纯 CCE 复测的实际编译日志、Canonical 操作数及动态地址核验记录绘制；不是源码级无 spill 示意图。",
        "首组 144 条指令：8 输入 VLD、24 谓词生成/计算、13 PSTS、14 PLDS、44 向量计算、4 输出 VST、27 Membar、10 标量辅助。重复四次，总计 52 PSTS、56 PLDS、108 Membar。",
        "保留完整首组的指令顺序；四路展开不是四张独立图。细节图按连续 48 条指令分页，跨段输入在图下方给出。",
        "紫线为谓词 RAW，灰线为向量 RAW，绿虚线为 spill UB RAW。红菱形保留全部屏障节点，屏障扇出不叠加于图片，避免掩盖数据依赖。",
        "图中节点是架构寄存器动态读写及 UB 保存/恢复，不是物理寄存器分配图，也不是执行时间轴。",
        "循环外 PSET、VDUP 常量和标量初始化不包含在 144 条循环体内，相关输入依赖及完整 590 条 RV 记录保存在 JSON 中。",
        f"本次未重新预测周期。既有测量：CAModel {graph['measurements']['camodel']}、VfSim 全局 {graph['measurements']['global_cycles']}、局部 {graph['measurements']['local_cycles']} cycles。", "",
        "| 范围 | PNG | SVG |", "| --- | --- | --- |"]
    if args.unroll == 1:
        lines = ["# Logit Grad U1 指令顺序 DAG", "",
            "基于纯 CCE 复测的实际编译日志及 Canonical 操作数重建；横轴是程序顺序，不是执行 cycle。",
            "首轮全部 22 条指令：2 条输入 VLD、6 条谓词生成/计算、11 条向量计算、1 条输出 VST、2 条标量辅助。循环执行 16 次；无 spill、无 Membar。",
            "紫线为谓词 RAW，灰线为向量 RAW。循环外的 PSET、4 条 VDUP 常量及其输入依赖保存在 JSON 中，并在图片底部列出。",
            f"既有测量：CAModel {graph['measurements']['camodel']}、VfSim 全局 {graph['measurements']['global_cycles']} cycles。本次没有重跑预测。", "",
            "| 范围 | PNG | SVG |", "| --- | --- | --- |"]
    labels = ("总览 1–144", "细节 1–48", "细节 49–96", "细节 97–144") if args.unroll == 4 else ("首轮 1–22",)
    for number, label in enumerate(labels):
        stem = stem_prefix if number == 0 else f"{stem_prefix}_part{number:02d}"
        png, svg = (os.path.relpath(output / f"{stem}.{ext}", metadata) for ext in ("png", "svg"))
        lines.append(f"| {label} | [查看]({png}) | [查看]({svg}) |")
    lines.extend(["", f"[编号、PC、操作数与字节地址]({stem_prefix}.csv) · [依赖及来源校验值]({stem_prefix}.json)", ""])
    (metadata / f"{stem_prefix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(directory=str(output), instructions=len(nodes), dependency_counts=graph["displayed_dependency_counts"], images=2*len(labels))))


if __name__ == "__main__":
    main()
