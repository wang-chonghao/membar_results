"""Draw recorded FlashAttention-front U1/U2 dataflow and hardware-stream DAGs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"\(PC: (0x[0-9a-f]+)\).*?\(ID:\s*(\d+)\)\s+(RV_\w+)(.*)")
FIELD = re.compile(r"\b(V[dnm]|P[dnmg]|S[dnm]|Sn2|Sm2|Ust)\[(\d+)\]")
COLORS = {"load": "#facc15", "store": "#22c55e", "compute": "#60a5fa",
          "scalar": "#d1d5db", "barrier": "#fca5a5", "control": "#e5e7eb", "external": "#ffffff"}


def records(path):
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE.search(line)
        if match:
            pc, iid, op, detail = match.groups()
            row = dict(id=int(iid), pc=pc, op=op, detail=detail)
            if int(iid) in by_id and by_id[int(iid)] != row:
                raise ValueError(f"Conflicting dynamic instruction ID: {iid}")
            by_id[int(iid)] = row
    return [row for _, row in sorted(by_id.items())]


def decode_registers(row):
    fields = {name: ("u" if name == "Ust" else name[0].lower()) + index
              for name, index in FIELD.findall(row["detail"])}
    op = row["op"]
    reads, writes = [], []
    if op == "RV_PSET":
        writes = ["Pd"]
    elif op in ("RV_VMUL", "RV_VADD"):
        reads, writes = ["Vn", "Vm", "Pg"], ["Vd"]
    elif op == "RV_VCADD":
        reads, writes = ["Vn", "Pg"], ["Vd"]
    elif op.startswith("RV_VLD"):
        reads, writes = ["Sn", "Sn2", "Sm", "Sm2"], ["Vd"]
    elif op.startswith("RV_VSTU"):
        reads = ["Vd", "Sn", "Sn2"]
    elif op.startswith("RV_VSTA"):
        reads = ["Sn", "Sn2"]
    elif op.startswith("RV_VST"):
        reads = ["Vd", "Pg", "Sn", "Sn2", "Sm", "Sm2"]
    elif op == "RV_SMOV":
        reads, writes = ["Sn", "Sn2"], ["Sd"]
    elif op == "RV_SMOVI":
        writes = ["Sd"]
    elif op == "RV_SMOVK":
        reads, writes = ["Sd"], ["Sd"]
    elif op == "RV_SSTI":
        reads = ["Sd", "Sn"]
    elif op == "RV_SLDI":
        reads, writes = ["Sn"], ["Sd"]
    elif op == "RV_VLOOPv2":
        reads = ["Sn", "Sn2"]
    elif op not in ("RV_SMEM_BAR", "RV_SEND"):
        raise ValueError(f"Unimplemented DAG operand semantics: {op}")
    if "#p=1" in row["detail"] and op.startswith(("RV_VLD", "RV_VST")):
        writes.append("Sn")
    return fields, [fields[k] for k in reads if k in fields], [fields[k] for k in writes if k in fields]


def build_graph(folder):
    source = folder / "camodel/core0.veccore0.instr_popped_log.dump"
    raw = records(source)
    loop, = [r for r in raw if r["op"] == "RV_VLOOPv2"]
    loop_pc = int(loop["pc"], 16)
    words = int(re.search(r"instr_num\s*:\s*(\d+)", loop["detail"])[1])
    mapping_path = folder / "camodel_case/run_output/pc_mapping.json"
    mapping = {r["pc"]: r for r in json.loads(mapping_path.read_text())}
    occurrences = Counter()
    nodes, edges, last_write, memory_writes, memory_reads = [], [], {}, {}, defaultdict(set)
    pending_align = defaultdict(list)
    prior_lsu = {"read": [], "write": []}
    barriers = {}
    last_barrier = None
    external = {}
    for item in raw:
        row = dict(item)
        row["iteration"] = occurrences[row["pc"]] if loop_pc < int(row["pc"], 16) <= loop_pc + 4 * words else None
        occurrences[row["pc"]] += 1
        row["section"] = ("body" if row["iteration"] is not None else
                          "prefix" if row["id"] <= loop["id"] else "suffix")
        row["fields"], row["reads"], row["writes"] = decode_registers(row)
        op, iid = row["op"], row["id"]
        if op in ("RV_VSTI", "RV_VSTS") and "Pg" not in row["fields"]:
            mask, = re.findall(r"\bp(\d+)\b", mapping[row["pc"]]["statement"])
            row["fields"]["Pg"] = "p" + mask
            row["reads"].append("p" + mask)
            row["implicit_mask_evidence"] = "archived PC-to-CCE mapping"
        row["category"] = ("scalar" if op.startswith("RV_S") and op not in ("RV_SMEM_BAR", "RV_SEND") else
                           "load" if op.startswith("RV_VLD") else "store" if op.startswith("RV_VST") else
                           "barrier" if op == "RV_SMEM_BAR" else
                           "control" if op in ("RV_VLOOPv2", "RV_SEND") else "compute")
        row["address"] = None
        row["spill"] = False
        if op.startswith(("RV_VLD", "RV_VST")) and op != "RV_VSTAI":
            address_rows = mapping[row["pc"]]["byte_addresses"]
            row["address"] = address_rows[occurrences[row["pc"]] - 1]
            row["spill"] = row["address"] >= 0x30000
        for reg in row["reads"]:
            producer = last_write.get(reg)
            if producer is None:
                if reg.startswith(("v", "p")):
                    raise ValueError(f"Unexpected live-in vector/predicate: {reg} at {row['pc']}")
                producer = "live_" + reg
                external[producer] = dict(id=producer, op="LIVE-IN", label=reg, category="external")
            kind = "vector" if reg.startswith("v") else "predicate" if reg.startswith("p") else "scalar"
            edges.append(dict(src=producer, dst=iid, kind=kind, value=reg))
        row["memory_key"] = None
        if row["address"] is not None:
            width = 4 if op == "RV_VSTUI" else 256
            row["memory_key"] = f"UB:{row['address']:x}:{width}"
        elif op in ("RV_SSTI", "RV_SLDI"):
            # Preserve the raw scalar slot expression instead of guessing its encoding scale.
            base = re.search(r"Sn\[\d+\]=0x([0-9a-f]+)", row["detail"])[1]
            imm = re.search(r"IMM:0x([0-9a-f]+)", row["detail"])[1]
            row["memory_key"] = f"scalar_slot:base={base}:raw_IMM={imm}"
        if row["memory_key"] is not None:
            key = row["memory_key"]
            is_read = op.startswith("RV_VLD") or op == "RV_SLDI"
            previous = memory_writes.get(key)
            if previous is not None:
                edges.append(dict(src=previous, dst=iid, kind="memory_raw" if is_read else "memory_waw", value=key))
            if is_read:
                memory_reads[key].add(iid)
            else:
                edges.extend(dict(src=reader, dst=iid, kind="memory_war", value=key) for reader in sorted(memory_reads[key]))
                memory_reads[key].clear()
                memory_writes[key] = iid
        if op == "RV_VSTUI":
            pending_align[row["fields"]["Ust"]].append(iid)
        elif op == "RV_VSTAI":
            producers = pending_align.pop(row["fields"]["Ust"])
            edges.extend(dict(src=p, dst=iid, kind="align_state", value=row["fields"]["Ust"]) for p in producers)
        if row["category"] in ("load", "store"):
            kind = "read" if row["category"] == "load" else "write"
            gating = barriers.get("VST_VLD" if kind == "read" else "VLD_VST")
            if gating is not None:
                edges.append(dict(src=gating, dst=iid, kind="barrier_release", value=kind))
            prior_lsu[kind].append(iid)
        elif op == "RV_SMEM_BAR":
            row["barrier_type"] = row["detail"].split(":")[-1].strip()
            wait = "read" if row["barrier_type"] == "VLD_VST" else "write"
            edges.extend(dict(src=p, dst=iid, kind="barrier_wait", value=wait) for p in prior_lsu[wait])
            if last_barrier is not None:
                edges.append(dict(src=last_barrier, dst=iid, kind="barrier_order", value="stream order"))
            barriers[row["barrier_type"]] = last_barrier = iid
        last_write.update({reg: iid for reg in row["writes"]})
        nodes.append(row)
    order = {n["id"]: i for i, n in enumerate(nodes)}
    assert all(isinstance(e["src"], str) or order[e["src"]] < order[e["dst"]] for e in edges)
    spill_stores = [n for n in nodes if n["spill"] and n["category"] == "store"]
    spill_loads = [n for n in nodes if n["spill"] and n["category"] == "load"]
    paired = [e for e in edges if e["kind"] == "memory_raw" and any(n["id"] == e["dst"] for n in spill_loads)]
    assert len(paired) == len(spill_loads)
    return dict(source=str(source.relative_to(folder.parent.parent)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                pc_mapping_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
                nodes=nodes, external=list(external.values()), edges=edges,
                counts=dict(Counter(n["op"] for n in nodes)),
                vector_spill_store=len(spill_stores), vector_spill_load=len(spill_loads),
                loop_iterations=max(n["iteration"] for n in nodes if n["iteration"] is not None) + 1)


def data_view(graph, output, stem, unroll, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Patch
    from matplotlib.lines import Line2D

    selected = {n["id"]: n for n in graph["nodes"] if n["iteration"] == 0 and n["op"].startswith("RV_V")}
    links = [e for e in graph["edges"] if e["src"] in selected and e["dst"] in selected
             and e["kind"] in ("vector", "memory_raw")]
    incoming = defaultdict(list)
    for edge in links:
        incoming[edge["dst"]].append(edge["src"])
    support, depth = {}, {}
    for iid, node in selected.items():
        if node["category"] == "load" and not node["spill"]:
            addr = node["address"]
            origin = 0x8000 if addr >= 0x8000 else 0
            support[iid] = {(addr-origin)//3072,}
            node["data_row"] = (addr-origin)//3072
            node["chunk"] = ((addr-origin) % 3072)//256
            node["input_buffer"] = "grad" if origin else "src"
        else:
            support[iid] = set().union(*(support[p] for p in incoming[iid]))
            node["data_row"], = support[iid]
        depth[iid] = (1 + max((depth[p] for p in incoming[iid]), default=0)
                      if node["op"] == "RV_VADD" else 0)
    leaf_order = {}
    def leaves(iid):
        node = selected[iid]
        if node["op"] == "RV_VMUL":
            return [iid]
        return [leaf for p in incoming[iid] for leaf in leaves(p)]
    for node in selected.values():
        if node["op"] == "RV_VCADD":
            row = node["data_row"]
            leaf_order.update({iid: offset for offset, iid in enumerate(leaves(node["id"]))})
            assert len(leaves(node["id"])) == 12
    positions = {}
    for iid, node in selected.items():
        if node["op"] == "RV_VMUL":
            positions[iid] = (10, -node["data_row"] * 22 - leaf_order[iid] * 1.6)
    def place_inputs(iid, y):
        if iid in positions:
            return
        node = selected[iid]
        if node["category"] == "load":
            x = 7.5 if node["spill"] else (2.5 if node["input_buffer"] == "grad" else 0)
        elif node["category"] == "store":
            x = 5
        else:
            raise ValueError(node)
        positions[iid] = x, y
        for parent in incoming[iid]:
            place_inputs(parent, y)
    for iid, node in selected.items():
        if node["op"] == "RV_VMUL":
            for parent in incoming[iid]:
                place_inputs(parent, positions[iid][1])
    for iid, node in selected.items():
        if iid in positions:
            continue
        y = sum(positions[p][1] for p in incoming[iid])/len(incoming[iid])
        x = 10 + 3 * depth[iid] if node["op"] == "RV_VADD" else 25 if node["op"] == "RV_VCADD" else 28
        positions[iid] = x, y
    assert len(positions) == len(selected)
    fig, ax = plt.subplots(figsize=(24, 21 if unroll == 1 else 42))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.88, bottom=0.10)
    for e in links:
        a, b = positions[e["src"]], positions[e["dst"]]
        memory = e["kind"] == "memory_raw"
        arc = -0.18 if selected[e["src"]].get("input_buffer") == "grad" else 0.14
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={arc}",
                                    arrowstyle="-|>", mutation_scale=14, shrinkA=25, shrinkB=25,
                                    color="#15803d" if memory else "#64748b",
                                    linewidth=1.6, linestyle="--" if memory else "-", zorder=1))
    for iid, node in selected.items():
        x, y = positions[iid]
        color = COLORS[node["category"]]
        ax.add_patch(Circle((x, y), 0.63, facecolor=color, edgecolor="#334155", linewidth=1.2, zorder=2))
        op = "VLDS" if node["category"] == "load" else "VSTS" if node["spill"] else node["op"][3:]
        extra = (f"reload\n{node['address']:x}" if node["spill"] and node["category"] == "load" else
                 f"spill\n{node['address']:x}" if node["spill"] else
                 f"{node['input_buffer']}[{node['chunk']}]" if node["category"] == "load" else
                 node["fields"].get("Vd", ""))
        ax.text(x, y, f"{op}\n{extra}", fontsize=9.5, weight="bold", ha="center", va="center", zorder=3)
        ax.text(x, y-0.84, node["pc"][-4:], fontsize=8, ha="center", color="#475569")
    for row in range(unroll):
        ax.text(-0.8, -row * 22 + 1.8, f"ROW {row}", weight="bold", fontsize=16)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 29.5)
    ax.set_ylim(-(unroll-1)*22 - 19.2, 3.1)
    ax.axis("off")
    spill = sum(n["spill"] and n["category"] == "store" for n in selected.values())
    fig.suptitle(f"FlashAttention Grad Front: U{unroll} Compiled Vector Dataflow", fontsize=25, y=0.98)
    fig.text(0.5, 0.94, f"First hardware loop body: {unroll} row(s), {spill} spill store/reload pairs; "
             f"body repeats {graph['loop_iterations']} times", ha="center", fontsize=16)
    legend = [Patch(facecolor=COLORS[k], edgecolor="#334155", label=label) for k, label in
              (("load", "Vector load / reload"), ("store", "Vector store / spill"), ("compute", "Compute"))]
    legend += [Line2D([0], [0], color="#15803d", linestyle="--", label="Spill memory RAW")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False, fontsize=14)
    fig.text(0.5, 0.018, "Curved arrows = true dataflow, not issue order or a cycle axis. "
             "PC suffix below each node. Predicate/scalar/barrier edges and final VSTAS: see hardware/full-VF DAGs.",
             ha="center", fontsize=12, color="#475569")
    for ext in ("png", "svg", "pdf"):
        folder = metadata if ext == "pdf" else output
        fig.savefig(folder / f"{stem}_dataflow.{ext}", dpi=130, facecolor="white")
    plt.close(fig)
    return dict(nodes=len(selected), vector_spill_pairs=spill, rows=unroll)


def dot_view(graph, output, stem, dot, full, metadata):
    selected = {n["id"]: n for n in graph["nodes"] if full or n["section"] == "prefix" or n["iteration"] == 0}
    edges = [e for e in graph["edges"] if e["dst"] in selected and
             (e["src"] in selected or isinstance(e["src"], str))]
    # Previous barriers already carry earlier waits through barrier_order edges.
    # Remove only those transitive wait arrows from the drawing, not the JSON.
    wait_sources = defaultdict(set)
    for e in edges:
        if e["kind"] == "barrier_wait":
            wait_sources[e["dst"]].add(e["src"])
    redundant_waits, covered = set(), set()
    for barrier in sorted(wait_sources):
        redundant_waits.update((p, barrier) for p in wait_sources[barrier] & covered)
        covered.update(wait_sources[barrier])
    edges = [e for e in edges if e["kind"] != "barrier_wait" or (e["src"], e["dst"]) not in redundant_waits]
    externals = {n["id"]: n for n in graph["external"] if any(e["src"] == n["id"] for e in edges)}
    title = f"FlashAttention Grad Front / {stem} / " + ("complete VF" if full else "first compiled loop body + setup")
    lines = ["digraph G {", "graph [rankdir=LR, splines=spline, overlap=false, nodesep=0.18, ranksep=0.45, mclimit=0.1, nslimit=2, nslimit1=2,",
             f"label={json.dumps(title)}, labelloc=t, fontsize=22, bgcolor=white];",
             'node [shape=circle, style=filled, fontname="DejaVu Sans", fontsize=8, width=0.8];',
             'edge [arrowsize=0.5, color="#64748b", penwidth=0.7];']
    for n in [*selected.values(), *externals.values()]:
        label = n.get("label", n["op"][3:])
        if "pc" in n:
            label += f"\nPC {n['pc'][-4:]} / ID {n['id']}"
            if n.get("address") is not None:
                label += f"\nUB {n['address']:x}"
        shape = "diamond" if n["category"] == "barrier" else "box" if n["category"] == "external" else "circle"
        if n["category"] == "barrier": label = n["barrier_type"] + f"\nID {n['id']}"
        lines.append(f'{json.dumps(str(n["id"]))} [label={json.dumps(label)}, shape={shape}, '
                     f'fillcolor="{COLORS[n["category"]]}"];')
    for e in edges:
        color = ("#15803d" if e["kind"].startswith("memory") else "#a855f7" if e["kind"] == "predicate" else
                 "#ef4444" if e["kind"].startswith("barrier") else "#0891b2" if e["kind"] == "align_state" else "#94a3b8")
        style = "dashed" if e["kind"].startswith(("memory", "barrier")) else "solid"
        lines.append(f'{json.dumps(str(e["src"]))} -> {json.dumps(str(e["dst"]))} '
                     f'[color="{color}", style="{style}", tooltip={json.dumps(e["kind"]+": "+e["value"])}];')
    lines.append("}")
    dot_path = metadata / f"{stem}_{'full_vf' if full else 'hardware_group'}.dot"
    dot_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for ext in ("svg", "pdf"):
        folder = metadata if ext == "pdf" else output
        target = folder / dot_path.with_suffix('.'+ext).name
        subprocess.run([str(dot), f"-T{ext}", str(dot_path), "-o", str(target)], check=True, timeout=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.root / "figure/DAG"
    metadata = args.metadata_dir or args.root / "metadata/figure_sources/DAG"
    if output.exists() and not args.overwrite:
        parser.error("Output directory exists; pass a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant, unroll in (("u1", 1), ("u2_concentrated_spill", 2)):
        graph = build_graph(args.root / "flash_attention_grad_front" / variant)
        (metadata / f"{variant}_graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        view = data_view(graph, output, variant, unroll, metadata)
        for full in (False, True):
            dot_view(graph, output, variant, args.dot, full, metadata)
        summary[variant] = dict(counts=graph["counts"], vector_spill_store=graph["vector_spill_store"],
                                vector_spill_load=graph["vector_spill_load"], first_group=view,
                                source_sha256=graph["source_sha256"], nodes=len(graph["nodes"]), edges=len(graph["edges"]))
    (metadata / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# FlashAttention Grad 前段 U1 / U2 DAG", "",
             "来源为各版本保存的 CAModel instr_popped_log，按动态 ID 重建指令顺序、寄存器 RAW 与 spill 内存边；不是根据编译前变量名猜测。",
             "U2 使用集中加载、实际发生 compiler spill 的版本，不是原序直接展开但没有 spill 的 u2_original。", "",
             "## 图的范围", "",
             "- dataflow.png/svg/pdf：首轮编译后向量数据流。U1 是一行，U2 是两行；全部 21 对向量 spill 搬运均保留。",
             "- hardware_group.svg/pdf：首轮编译后循环组及初始化，保留全部记录的标量、谓词、Membar 指令与相关边。",
             "- full_vf.svg/pdf：完整 VF，包含 U1 的 8 轮或 U2 的 4 个双行组，以及最后的 VSTAS。没有截断节点。",
             "- graph.json：全部动态指令、PC、寄存器读写、真实字节地址及分类型依赖边。dot 文件可用 Graphviz 重新布局。", "",
             "黄色圆为向量 load/reload，绿色为 store/spill，蓝色为计算；详细图中灰色为标量/控制，红色菱形为 Membar。",
             "向量主图绿虚线是 spill UB 数据边；其余为寄存器数据流。详细图中红虚线为屏障顺序/等待/放行，紫色为谓词，青色为 align state。",
             "全部连线采用曲线；布局不是执行时间轴，边不表示统一 latency。相同向量寄存器名称的不同写入按动态版本识别，不增加虚假的寄存器 WAR/WAW。",
             "详细图删除了可由前序屏障及屏障顺序边传递的冗余 wait 箭头，所有节点仍保留；原始全部依赖边保存在 JSON。",
             "全图保留 RAW 和同槽内存 WAR/WAW；标量 SSTI/SLDI 的槽以日志原始 base/IMM 表达式配对，不猜测其字节编码。",
             "Vector 主图省略了全真 mask、标量地址及同步边，以免遮蔽主体；它们保留在详细图和 JSON 中。最后 VSTAS 消费全部 8 个 VSTUS 的 state，不只首个循环组。", "",
             "## 日志核查补充", "",
             "U1：192 条输入 VLD、96 VMUL、88 VADD、8 VCADD、8 VSTUS、1 VSTAS；无向量 spill、无 Membar。",
             "U2：192 条输入 VLD，另有 84 条 spill VLD、84 条 spill VST 和 168 条 Membar。每个展开组是 21 对保存/重载和 42 条 Membar，重复 4 次。",
             "**本次还发现旧表未单列的标量保存/恢复：U2 日志存在 48 条 RV_SSTI、48 条 RV_SLDI，用于复用标量地址临时寄存器。**",
             "因此旧表的“仅 Vector spill”并不等价于没有任何标量保存/恢复。这些指令已纳入详细图；本次不改变旧周期或重新校准预测，不能据旧表把其偏差全部归因于向量 spill。", "",
             "## 查看", "",
             "| 版本 | 数据流 PNG | 完整首组硬件图 | 完整 VF |", "| --- | --- | --- | --- |"]
    for variant in summary:
        lines.append(f"| {variant} | [PNG]({variant}_dataflow.png) / [PDF]({variant}_dataflow.pdf) | "
                     f"[SVG]({variant}_hardware_group.svg) | [SVG]({variant}_full_vf.svg) / [PDF]({variant}_full_vf.pdf) |")
    lines += ["", "## 重建", "", "需要 matplotlib 与 Graphviz；在结果仓根目录执行：", "", "```bash",
              "python3 scripts/plot_flash_attention_spill_dag.py --dot /path/to/dot --output-dir figure/flash_attention_dag_new", "```", ""]
    import os
    document = re.sub(r"\]\(([^)]+\.(?:png|svg))\)",
                      lambda match: "](" + os.path.relpath(output / match[1], metadata) + ")",
                      "\n".join(lines))
    (metadata / "README.md").write_text(document, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
