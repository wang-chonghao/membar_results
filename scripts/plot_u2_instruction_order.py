"""Draw fixed-width overview/detail charts for the actual spilling U2 body."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
LANES = (
    ("input", "输入 VLDS", "#facc15"),
    ("spill_store", "Spill VSTS", "#22c55e"),
    ("reload", "Reload VLDS", "#facc15"),
    ("mul", "VMUL", "#60a5fa"),
    ("add", "VADD", "#60a5fa"),
    ("reduce", "VCADD", "#60a5fa"),
    ("output", "输出 VSTUS", "#22c55e"),
    ("barrier", "Membar", "#fca5a5"),
    ("scalar", "标量辅助 / 保存恢复", "#d1d5db"),
)


def lane(node):
    if node["category"] == "load":
        return "reload" if node["spill"] else "input"
    if node["category"] == "store":
        return "spill_store" if node["spill"] else "output"
    if node["category"] in ("barrier", "scalar"):
        return node["category"]
    return {"RV_VMUL": "mul", "RV_VADD": "add", "RV_VCADD": "reduce"}[node["op"]]


def prepare(graph):
    nodes = sorted((dict(n) for n in graph["nodes"] if n["iteration"] == 0), key=lambda n: n["id"])
    for seq, node in enumerate(nodes, 1):
        node["sequence"] = seq
        node["lane"] = lane(node)
    expected = dict(input=48, spill_store=21, reload=21, mul=24, add=22,
                    reduce=2, output=2, barrier=42, scalar=48)
    if Counter(n["lane"] for n in nodes) != Counter(expected):
        raise ValueError("Unexpected U2 hardware body; refusing to use stale chart labels")
    index = {n["id"]: n for n in nodes}
    edges = [dict(e, src_sequence=index[e["src"]]["sequence"], dst_sequence=index[e["dst"]]["sequence"])
             for e in graph["edges"] if e["src"] in index and e["dst"] in index
             and (e["kind"] == "vector" or e["kind"] == "memory_raw"
                  and index[e["src"]]["lane"] == "spill_store" and index[e["dst"]]["lane"] == "reload")]
    if sum(e["kind"] == "memory_raw" for e in edges) != 21:
        raise ValueError("Spill RAW edges are incomplete")
    pages = []
    for start in range(1, len(nodes)+1, 46):
        end = min(start+45, len(nodes))
        pages.append(dict(start=start, end=end,
                          incoming=[e for e in edges if e["src_sequence"] < start <= e["dst_sequence"] <= end],
                          outgoing=[e for e in edges if start <= e["src_sequence"] <= end < e["dst_sequence"]]))
    return nodes, edges, pages


def render(nodes, edges, pages, output, font, metadata):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    for page_number, page in enumerate([None, *pages]):
        overview = page is None
        start, end = (1, len(nodes)) if overview else (page["start"], page["end"])
        shown = [n for n in nodes if start <= n["sequence"] <= end]
        counts = Counter(n["lane"] for n in shown)
        active = [entry for entry in LANES if overview or counts[entry[0]]]
        y_by_lane = {key: len(active)-i for i, (key, _, _) in enumerate(active)}
        positions = {n["id"]: (n["sequence"], y_by_lane[n["lane"]]) for n in shown}
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=0.19, right=0.975, top=0.81, bottom=0.30 if not overview else 0.25)
        for key, _, _ in active:
            y = y_by_lane[key]
            ax.axhspan(y-0.38, y+0.38, color="#f4f6f8", zorder=0)
        for edge in edges:
            if edge["src"] not in positions or edge["dst"] not in positions:
                continue
            a, b = positions[edge["src"]], positions[edge["dst"]]
            memory = edge["kind"] == "memory_raw"
            radius = -0.18 if a[1] == b[1] else 0.025 if overview else 0.055
            ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={radius}", arrowstyle="-|>",
                                        shrinkA=3 if overview else 9, shrinkB=3 if overview else 9,
                                        mutation_scale=5 if overview else 10, linewidth=0.6 if overview else 1.1,
                                        color="#15803d" if memory else "#64748b", alpha=0.45 if overview else 0.7,
                                        linestyle="--" if memory else "-", zorder=1))
        for node in shown:
            x, y = positions[node["id"]]
            color = dict((k, c) for k, _, c in LANES)[node["lane"]]
            ax.scatter([x], [y], s=18 if overview else 285, color=color, edgecolors="#334155",
                       marker="D" if node["lane"] == "barrier" else "o", linewidths=0.5 if overview else 0.8, zorder=3)
            if not overview:
                ax.text(x, y, str(node["sequence"]), ha="center", va="center", fontsize=8, zorder=4)
        labels = [f"{label} · {counts[key]}" for key, label, _ in active]
        ax.set_yticks([y_by_lane[key] for key, _, _ in active], labels, fontsize=12)
        ax.tick_params(axis="y", length=0, pad=10)
        step = 20 if overview else 5
        ticks = sorted({start, end, *(i for i in range(start, end+1) if i % step == 0)})
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xlim(start-1.2, end+1.2)
        ax.set_ylim(0.35, len(active)+0.70)
        ax.set_xlabel("编译后循环体内的指令序号（不是执行 cycle）", fontsize=15, labelpad=14)
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        suffix = "首个双行展开组 · 230 条指令 · 硬件重复 4 次" if overview else f"首组细节 {page_number}/5 · 第 {start}–{end} 条指令"
        fig.suptitle("FlashAttention Grad 前段 · U2 含 Spill 指令顺序", fontsize=22, y=0.96)
        fig.text(0.58, 0.88, suffix, ha="center", fontsize=16)
        if overview:
            fig.text(0.58, 0.12, "48 条输入 load + 21 次 spill 保存 + 21 次重载 + 48 条计算 + 2 条输出 + 42 条 Membar + 48 条标量辅助。",
                     ha="center", fontsize=10.5, color="#475569")
        else:
            incoming = defaultdict(list)
            for edge in page["incoming"]:
                incoming[edge["dst_sequence"]].append(edge["src_sequence"])
            text = "跨段输入（目标←来源）：" + ("； ".join(
                f"{target}←{','.join(map(str, sorted(set(sources))))}" for target, sources in sorted(incoming.items())) or "无")
            wrapped = textwrap.fill(text, width=125, break_long_words=False, break_on_hyphens=False)
            fig.text(0.19, 0.15, wrapped, ha="left", va="top", fontsize=10, linespacing=1.7, color="#475569")
        fig.text(0.58, 0.055, "圆点编号为程序顺序，红色菱形为屏障；灰线画向量 RAW，绿虚线画 spill UB RAW。完整扇出与标量依赖见硬件 DAG。",
                 ha="center", fontsize=10.5, color="#475569")
        fig.text(0.58, 0.025, "仅展示首个双行展开组；循环外 PSET、地址初始化和最后 VSTAS 未包含在本图中。",
                 ha="center", fontsize=9.5, color="#475569")
        name = "u2_instruction_order" if overview else f"u2_instruction_order_part{page_number:02d}"
        for ext in ("png", "svg", "pdf"):
            folder = metadata if ext == "pdf" else output
            fig.savefig(folder / f"{name}.{ext}", dpi=160, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    folder = args.root / "figure/DAG"
    metadata = args.root / "metadata/figure_sources/DAG"
    source = metadata / "u2_concentrated_spill_graph.json"
    raw = source.read_bytes()
    graph = json.loads(raw)
    nodes, edges, pages = prepare(graph)
    if not args.overwrite and any(list(path.glob("u2_instruction_order*")) for path in (folder, metadata)):
        parser.error("U2 sequence outputs already exist; refusing to overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    render(nodes, edges, pages, folder, args.font, metadata)
    (metadata / "u2_instruction_order.json").write_text(json.dumps(dict(
        source=source.name, source_sha256=hashlib.sha256(raw).hexdigest(), nodes=nodes,
        displayed_dependencies=edges, pages=pages, width=2560, height=1440), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (metadata / "u2_instruction_order.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "dynamic_id", "pc", "opcode", "lane", "register_reads", "register_writes", "ub_byte_address", "raw_detail"))
        for n in nodes:
            writer.writerow((n["sequence"], n["id"], n["pc"], n["op"], n["lane"], " ".join(n["reads"]),
                             " ".join(n["writes"]), n["address"], n["detail"]))
    lines = ["# U2 含 Spill 指令顺序图", "", "[总览 PNG](u2_instruction_order.png) · [总览 PDF](u2_instruction_order.pdf)", "",
             "沿用 U1 的泳道形式，横轴严格保持首轮双行展开组的程序顺序，不按依赖重新排序，也不是执行 cycle。",
             "首组共有 230 条真实指令，重复 4 次；全部 48 条标量辅助/保存恢复及 42 条 Membar 均保留，不折叠计数。",
             "每组有 48 条输入 load、21 次向量 spill store、21 次 reload、24 VMUL、22 VADD、2 VCADD、2 VSTUS。",
             "黄色为输入/reload，绿色为保存/输出，蓝色为计算，红色菱形为 Membar，灰色为标量辅助。",
             "曲线仅绘制向量寄存器 RAW 和 spill UB RAW；没有把所有屏障扇出、标量、谓词依赖叠在图中。完整依赖见 [原始图数据](u2_concentrated_spill_graph.json)。",
             "总览保留全部节点但不在小圆内挤入编号；细节图每段 46 条，节点写全局连续编号，没有重新从 1 编号。",
             "跨段依赖在目标所在图下方列出“目标←来源”；跨段输出可在后续目标段找到，对照 [JSON](u2_instruction_order.json) 保留全部边。",
             "循环外 PSET、地址初始化及最后 VSTAS 不包含在 230 条循环体中，完整记录保存在原始图数据中。",
             "每张图均为 2560×1440（16:9），另附 SVG/PDF。原始 PC、动态 ID、读写寄存器和地址见 [指令对照表](u2_instruction_order.csv)。", "",
             "| 分段 | 指令编号 | PNG | PDF |", "| --- | --- | --- | --- |"]
    for number, page in enumerate(pages, 1):
        stem = f"u2_instruction_order_part{number:02d}"
        lines.append(f"| {number} | {page['start']}–{page['end']} | [查看]({stem}.png) | [下载]({stem}.pdf) |")
    document = re.sub(r"\]\(([^)]+\.(?:png|svg))\)",
                      lambda match: "](" + os.path.relpath(folder / match[1], metadata) + ")",
                      "\n".join(lines)+"\n")
    (metadata / "u2_instruction_order.md").write_text(document, encoding="utf-8")
    print(json.dumps(dict(overview=str(folder / "u2_instruction_order.png"), instructions=len(nodes),
                          dependencies=len(edges), parts=len(pages), image_size=[2560, 1440]), ensure_ascii=False))


if __name__ == "__main__":
    main()
