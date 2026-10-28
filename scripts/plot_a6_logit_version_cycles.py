"""Export the current A6 Logit Grad report chart at a fixed 16:9 ratio."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).absolute().parents[1]
STEM = "logit_grad_a6_version_cycles_20261028"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    # Reuse the browser's projection so software-search overrides stay identical.
    program = """
const fs = require('fs');
const path = require('path');
const root = process.argv[1];
const read = name => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'));
const data = read('metadata/spill_candidate_alignment.json');
const refreshed = read('metadata/multisoc_predictions.json');
globalThis.SPILL_PERFORMANCE_SUMMARY = read('metadata/spill_performance_summary.json');
const {project} = require(path.join(root, 'assets/multisoc_view.js'));
const rows = data.rows.filter(row => row.case === 'logit')
  .sort((a, b) => a.variant_order - b.variant_order)
  .map(row => project(row, 'A6', refreshed));
console.log(JSON.stringify({model_commit: refreshed.model_commit, rows}));
"""
    data = json.loads(subprocess.check_output(
        ["node", "-e", program, str(ROOT)], text=True))
    keys = ("global_cycles", "local_cycles", "software_cycles", "a5_local_cycles")
    rows = [{key: row.get(key) for key in ("id", "chart_label", "unroll", *keys)}
            for row in data["rows"]]
    assert len({row["id"] for row in rows}) == len(rows)
    assert rows and all(row["id"].startswith("logit-u") for row in rows)
    assert len({row["comparison_group"] for row in data["rows"]}) == 1

    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font_path).get_name(),
        "font.size": 17, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    output = ROOT / "figure/cycles"
    metadata = ROOT / "metadata/figure_sources/cycles"
    output.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    targets = [output / f"{STEM}.png", output / f"{STEM}.svg",
               metadata / f"{STEM}.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Dated outputs already exist; preserve the archived figures")

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.32)
    fig.suptitle("Logit Grad · A6 版本周期对比", fontsize=28, y=0.96)
    fig.text(0.5, 0.89, "DV121 · FP32 · 1024 元素 · 16 次向量迭代",
             ha="center", fontsize=19)
    specs = (
        ("global_cycles", "A6 VfSim 全局 Membar", "#2376b8", "s", "-"),
        ("local_cycles", "A6 VfSim 局部 UB 依赖", "#18846d", "^", "-"),
        ("software_cycles", "A6 VfSim 软件优化（全局 Membar）", "#a94f69", "D", "-"),
        ("a5_local_cycles", "A5 VfSim 局部 UB 依赖（对照）", "#68737d", "o", "--"),
    )
    x = list(range(len(rows)))
    for key, label, color, marker, style in specs:
        ax.plot(x, [float("nan") if row[key] is None else row[key] for row in rows],
                label=label, color=color, marker=marker, linestyle=style,
                linewidth=2.6, markersize=9, markerfacecolor="white",
                markeredgewidth=2)
    for index, row in enumerate(rows):
        value = row["global_cycles"]
        if value is not None and value > 500:
            ax.annotate(f"{value:,}", (index, value), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=17,
                        color="#2376b8")
    ax.set_xticks(x, [row["chart_label"] for row in rows], fontsize=19)
    ax.set_xlim(-0.25, len(rows) - 0.75)
    ax.set_ylim(0, 3300)
    ax.set_xlabel("版本 / Unroll（分类轴）", fontsize=20, labelpad=12)
    ax.set_ylabel("VF cycles", fontsize=21, labelpad=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.095), frameon=False, fontsize=17)
    fig.text(0.5, 0.055, "沿用 HTML 当前 A6 数据；U8 全局 / 局部缺失保留断点；未重跑预测。",
             ha="center", fontsize=13, color="#555555")
    fig.text(0.5, 0.025, "A6 软件方案通过 VF 编译期无 spill 检查；无 A6 CAModel 数值验证。",
             ha="center", fontsize=13, color="#555555")
    fig.savefig(targets[0], dpi=160, facecolor="white")
    fig.savefig(targets[1], facecolor="white")
    plt.close(fig)
    sources = ["metadata/spill_candidate_alignment.json",
               "metadata/multisoc_predictions.json",
               "metadata/spill_performance_summary.json", "assets/multisoc_view.js"]
    record = dict(model_commit=data["model_commit"], rows=rows,
                  width=2560, height=1440, source_sha256={
                      source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
                      for source in sources})
    targets[2].write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
