"""Draw a fixed-width program-order view of the recorded U1 loop body."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LANES = {
    "RV_VLDI": (5, "输入 VLDS · 24 条", "#facc15"),
    "RV_VMUL": (4, "VMUL · 12 条", "#60a5fa"),
    "RV_VADD": (3, "VADD · 11 条", "#60a5fa"),
    "RV_VCADD": (2, "VCADD · 1 条", "#60a5fa"),
    "RV_VSTUI": (1, "输出 VSTUS · 1 条", "#22c55e"),
}


def prepare(graph):
    nodes = sorted((dict(n) for n in graph["nodes"] if n["iteration"] == 0), key=lambda n: n["id"])
    if len(nodes) != 49 or any(n["op"] not in LANES for n in nodes):
        raise ValueError("Expected the verified U1 body: 24 loads, 24 computes, 1 store")
    for seq, node in enumerate(nodes, 1):
        node["sequence"] = seq
    positions = {n["id"]: (n["sequence"], LANES[n["op"]][0]) for n in nodes}
    edges = [dict(e) for e in graph["edges"] if e["kind"] == "vector"
             and e["src"] in positions and e["dst"] in positions]
    if len(edges) != 48:
        raise ValueError("Unexpected vector dependency count")
    return nodes, edges, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    folder = args.root / "figure/DAG"
    metadata = args.root / "metadata/figure_sources/DAG"
    source = metadata / "u1_graph.json"
    raw = source.read_bytes()
    graph = json.loads(raw)
    nodes, edges, positions = prepare(graph)
    prefix = folder / "u1_instruction_order"
    metadata_prefix = metadata / prefix.name
    targets = [prefix.with_suffix('.'+ext) for ext in ("png", "svg")]
    targets += [metadata_prefix.with_suffix('.'+ext) for ext in ("pdf", "csv", "json", "md")]
    if not args.overwrite and any(p.exists() for p in targets):
        parser.error("Instruction-order outputs already exist; refusing to overwrite")
    folder.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.16, right=0.975, top=0.81, bottom=0.25)
    for y, _, _ in LANES.values():
        ax.axhspan(y-0.40, y+0.40, color="#f4f6f8", zorder=0)
    for boundary in (24.5, 36.5, 47.5):
        ax.axvline(boundary, color="#d1d5db", linewidth=1, linestyle="--", zorder=0)
    for edge in edges:
        start, end = positions[edge["src"]], positions[edge["dst"]]
        radius = -0.18 if start[1] == end[1] else 0.065
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={radius}",
                                    arrowstyle="-|>", mutation_scale=10, shrinkA=8, shrinkB=8,
                                    linewidth=1.0, color="#64748b", alpha=0.65, zorder=1))
    for node in nodes:
        x, y = positions[node["id"]]
        ax.scatter([x], [y], s=210, color=LANES[node["op"]][2], edgecolors="#334155",
                   linewidths=0.8, zorder=3)
        ax.text(x, y, str(node["sequence"]), ha="center", va="center", fontsize=8.5, zorder=4)
    ax.set_yticks([y for y, _, _ in LANES.values()], [label for _, label, _ in LANES.values()], fontsize=13)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xticks([1, 5, 10, 15, 20, 24, 30, 36, 40, 45, 49])
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlim(0, 50)
    ax.set_ylim(0.35, 5.8)
    ax.set_xlabel("编译后循环体内的指令序号（不是执行 cycle）", fontsize=16, labelpad=15)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    fig.suptitle("FlashAttention Grad 前段 · U1 指令顺序与数据依赖", fontsize=23, y=0.95)
    fig.text(0.565, 0.875, "先集中加载 → 逐项乘法 → 加法树 → 行内规约 → 输出", ha="center", fontsize=16)
    fig.text(0.565, 0.11, "每个圆点是一条真实指令，数字为顺序编号；曲线箭头保留全部 48 条向量 RAW 依赖。",
             ha="center", fontsize=12, color="#475569")
    fig.text(0.565, 0.065, "展示第 0 行的 49 条循环体指令，硬件重复 8 轮；循环外 PSET、地址初始化及最终 VSTAS 未画在此视图。",
             ha="center", fontsize=11, color="#475569")
    for ext in ("png", "svg", "pdf"):
        target_prefix = metadata_prefix if ext == "pdf" else prefix
        fig.savefig(target_prefix.with_suffix('.'+ext), dpi=160, facecolor="white")
    plt.close(fig)
    with metadata_prefix.with_suffix('.csv').open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "dynamic_id", "pc", "opcode", "register_reads", "register_writes", "ub_byte_address"))
        for n in nodes:
            writer.writerow((n["sequence"], n["id"], n["pc"], n["op"], " ".join(n["reads"]),
                             " ".join(n["writes"]), n["address"]))
    metadata_prefix.with_suffix('.json').write_text(json.dumps(dict(
        source="u1_graph.json", source_sha256=hashlib.sha256(raw).hexdigest(), nodes=nodes,
        vector_dependencies=edges, width=2560, height=1440), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    document = (
        "# U1 指令顺序图\n\n"
        "[PNG](u1_instruction_order.png) · [SVG](u1_instruction_order.svg) · [PDF](u1_instruction_order.pdf)\n\n"
        "横轴为第一轮编译后循环体的程序顺序编号，不是发射顺序、硬件 ID 或 cycle。节点不按依赖重新排列。\n\n"
        "左侧按指令类别分泳道，圆点内是编号；真实 PC、动态 ID、寄存器读写及 UB 地址见 [完整对照表](u1_instruction_order.csv)。\n\n"
        "1–24：24 条输入 VLDS；25–36：12 条 VMUL；37–47：11 条 VADD；48：VCADD；49：VSTUS。\n\n"
        "图中保留循环体内全部 49 条指令和 48 条向量 RAW 依赖，连线用曲线。循环实际重复 8 次。\n\n"
        "循环外 PSET、地址初始化、最终 VSTAS 及其依赖未展示在本图；完整记录保存在 [原始图数据](u1_graph.json)。\n")
    document = re.sub(r"\]\(([^)]+\.(?:png|svg))\)",
                      lambda match: "](" + os.path.relpath(folder / match[1], metadata) + ")", document)
    metadata_prefix.with_suffix('.md').write_text(document, encoding="utf-8")
    print(json.dumps(dict(png=str(prefix.with_suffix('.png')), instructions=len(nodes), dependencies=len(edges),
                          image_size=[2560, 1440]), ensure_ascii=False))


if __name__ == "__main__":
    main()
