"""Draw the compiled Bank-aware U2 body in program order, with curved RAW edges."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from plot_flash_attention_spill_dag import build_graph


ROOT = Path(__file__).resolve().parents[1]
STEM = "u2_bank_aware_instruction_order"
LANES = (
    ("RV_VLDI", "VLDS", "#facc15"),
    ("RV_VMUL", "VMUL", "#60a5fa"),
    ("RV_VADD", "VADD", "#60a5fa"),
    ("RV_VCADD", "VCADD", "#60a5fa"),
    ("RV_VSTUI", "VSTUS", "#22c55e"),
)


def prepare(graph):
    nodes = sorted((dict(n) for n in graph["nodes"] if n["iteration"] == 0), key=lambda n: n["id"])
    expected = {"RV_VLDI": 48, "RV_VMUL": 24, "RV_VADD": 22, "RV_VCADD": 2, "RV_VSTUI": 2}
    if Counter(n["op"] for n in nodes) != Counter(expected) or graph["loop_iterations"] != 4:
        raise ValueError("Expected the verified Bank-aware U2 body: 98 instructions repeated four times")
    ids = {n["id"] for n in nodes}
    edges = [dict(e) for e in graph["edges"] if e["kind"] == "vector" and e["src"] in ids and e["dst"] in ids]
    if len(edges) != 96 or graph["vector_spill_store"] or graph["vector_spill_load"]:
        raise ValueError("Unexpected vector dependencies or spill in Bank-aware U2")
    rows = {}
    for sequence, node in enumerate(nodes, 1):
        node["sequence"] = sequence
        if node["op"] == "RV_VLDI":
            address = node["address"]
            if address is None or not 0 <= address < 0x10000:
                raise ValueError("Load is outside the archived src/grad buffers")
            row = (address % 0x8000) // (768 * 4)
        else:
            parents = {rows[e["src"]] for e in edges if e["dst"] == node["id"]}
            if len(parents) != 1:
                raise ValueError("Expected single-row arithmetic with no cross-row data dependency")
            row, = parents
        if row not in (0, 1):
            raise ValueError("First U2 group must process rows zero and one")
        node["row"] = rows[node["id"]] = row
    if Counter(rows.values()) != Counter({0: 49, 1: 49}):
        raise ValueError("Rows have unequal instruction coverage")
    by_id = {n["id"]: n for n in nodes}
    for edge in edges:
        edge.update(src_sequence=by_id[edge["src"]]["sequence"], dst_sequence=by_id[edge["dst"]]["sequence"])
        if edge["src_sequence"] >= edge["dst_sequence"]:
            raise ValueError("Dependency is not ordered before its consumer")
    pages = []
    for start, end in ((1, 49), (50, 98)):
        pages.append(dict(start=start, end=end, incoming=[e for e in edges
            if e["src_sequence"] < start <= e["dst_sequence"] <= end]))
    return nodes, edges, pages


def render(nodes, edges, pages, output, metadata, font):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    lanes = [(op, row, label, color) for op, label, color in LANES for row in (0, 1)]
    y_by_lane = {(op, row): len(lanes)-i for i, (op, row, _, _) in enumerate(lanes)}
    for page_number, page in enumerate([None, *pages]):
        start, end = (1, 98) if page is None else (page["start"], page["end"])
        shown = [n for n in nodes if start <= n["sequence"] <= end]
        positions = {n["id"]: (n["sequence"], y_by_lane[n["op"], n["row"]]) for n in shown}
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=0.15, right=0.975, top=0.82, bottom=0.23)
        for op, row, _, _ in lanes:
            y = y_by_lane[op, row]
            ax.axhspan(y-0.35, y+0.35, color="#f3f5f7" if row == 0 else "#e9eef3", zorder=0)
        for edge in edges:
            if edge["src"] in positions and edge["dst"] in positions:
                a, b = positions[edge["src"]], positions[edge["dst"]]
                ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={-0.18 if a[1] == b[1] else 0.07}",
                    arrowstyle="-|>", mutation_scale=9, shrinkA=7, shrinkB=7,
                    linewidth=0.9, color="#64748b", alpha=0.65, zorder=1))
        colors = {op: color for op, _, color in LANES}
        for node in shown:
            x, y = positions[node["id"]]
            ax.scatter([x], [y], s=150 if page is None else 240, color=colors[node["op"]],
                       edgecolors="#334155", linewidths=0.7, zorder=3)
            ax.text(x, y, str(node["sequence"]), ha="center", va="center",
                    fontsize=6.5 if page is None else 8, zorder=4)
        ax.set_yticks([y_by_lane[op, row] for op, row, _, _ in lanes],
                      [f"{label} · 行 {row}" for _, row, label, _ in lanes], fontsize=12)
        ax.tick_params(axis="y", length=0, pad=10)
        step = 10 if page is None else 5
        ax.set_xticks(sorted({start, end, *(i for i in range(start, end+1) if i % step == 0)}))
        ax.set_xlim(start-1.3, end+1.3)
        ax.set_ylim(0.4, len(lanes)+0.65)
        ax.set_xlabel("编译后首个双行组的指令顺序编号（不是执行 cycle）", fontsize=15, labelpad=12)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        fig.suptitle("FlashAttention Grad · Bank-aware U2 指令顺序与数据依赖", fontsize=22, y=0.96)
        detail = "完整首组：98 条指令，硬件重复 4 次" if page is None else f"细节 {page_number}/2：第 {start}–{end} 条指令"
        fig.text(0.56, 0.88, detail, ha="center", fontsize=16)
        if page is not None:
            incoming = "；".join(f"{e['dst_sequence']}←{e['src_sequence']}" for e in page["incoming"]) or "无"
            fig.text(0.15, 0.125, "跨段输入（目标←来源）："+incoming, fontsize=10, color="#475569")
        fig.text(0.56, 0.075, "黄色为加载，蓝色为计算，绿色为输出；两行分泳道展示，曲线保留 96 条向量 RAW 依赖。",
                 ha="center", fontsize=11, color="#475569")
        fig.text(0.56, 0.035, "未改变原加法树；无 spill、无 Membar。循环外 PSET、标量准备和最终 VSTAS 见原始图数据。",
                 ha="center", fontsize=10, color="#475569")
        stem = STEM if page is None else f"{STEM}_part{page_number:02d}"
        for ext in ("png", "svg", "pdf"):
            fig.savefig((metadata if ext == "pdf" else output) / f"{stem}.{ext}", dpi=160, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.root / "figure/DAG"
    metadata = args.root / "metadata/figure_sources/DAG"
    if not args.overwrite and any(list(path.glob(STEM+"*")) for path in (output, metadata)):
        parser.error("Bank-aware diagrams already exist; pass --overwrite to refresh")
    graph = build_graph(args.root / "flash_attention_grad_front/u2_bank_window1")
    nodes, edges, pages = prepare(graph)
    output.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(graph, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
    (metadata / "u2_bank_aware_graph.json").write_bytes(raw)
    render(nodes, edges, pages, output, metadata, args.font)
    (metadata / f"{STEM}.json").write_text(json.dumps(dict(source="u2_bank_aware_graph.json",
        source_sha256=hashlib.sha256(raw).hexdigest(), nodes=nodes, vector_dependencies=edges,
        pages=pages, width=2560, height=1440), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (metadata / f"{STEM}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "row", "dynamic_id", "pc", "opcode", "register_reads", "register_writes", "ub_byte_address"))
        for node in nodes:
            writer.writerow((node["sequence"], node["row"], node["id"], node["pc"], node["op"],
                             " ".join(node["reads"]), " ".join(node["writes"]), node["address"]))
    lines = ["# Bank-aware U2 指令顺序 DAG", "",
        "根据同版本 CAModel 日志按动态 ID、PC 重建，未按依赖重新排序。保留首组全部 98 条指令和 96 条向量 RAW；两行各 49 条，硬件重复四次。",
        "行号由实际 UB 地址和向量数据流推导；PNG/SVG 在 figure/DAG，PDF 和编号对照在本目录。",
        "循环外 PSET、标量准备和最终 VSTAS 不在顺序图中，完整动态依赖保存在 [原始图数据](u2_bank_aware_graph.json)。",
        "实测 228 cycles，VfSim 全局预测 204 cycles；数据来自既有验证记录，本次未重新仿真。", "",
        "| 范围 | PNG | SVG |", "| --- | --- | --- |"]
    for number, label in enumerate(("总览 1–98", "前半 1–49", "后半 50–98")):
        stem = STEM if number == 0 else f"{STEM}_part{number:02d}"
        links = [os.path.relpath(output / f"{stem}.{ext}", metadata) for ext in ("png", "svg")]
        lines.append(f"| {label} | [查看]({links[0]}) | [查看]({links[1]}) |")
    lines.extend(["", f"[指令编号与 PC]({STEM}.csv) · [分段依赖]({STEM}.json)", ""])
    (metadata / f"{STEM}.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(images=6, instructions=len(nodes), dependencies=len(edges), directory=str(output))))


if __name__ == "__main__":
    main()
