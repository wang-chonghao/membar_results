"""Export the HTML report's per-group cycle charts without rerunning predictions."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SERIES = (
    ("camodel", "原 CAModel", "#344251", "o", "-"),
    ("global_cycles", "VfSim 全局", "#2376b8", "s", "-"),
    ("local_cycles", "VfSim 局部", "#18846d", "^", "--"),
    ("software_cycles", "VfSim 软件优化全局 / 保留原版本", "#a94f69", "D", ":"),
)


def build_groups(data):
    groups = []
    for case in data["case_order"]:
        candidates = sorted(
            (row for row in data["rows"] if row["case"] == case),
            key=lambda row: row["variant_order"],
        )
        by_group = {}
        for row in candidates:
            for key, *_ in SERIES:
                value = row[key]
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                ):
                    raise ValueError(f"Invalid {key}: {row['id']}")
            by_group.setdefault(row["comparison_group"], []).append(row)
        for group_id, rows in by_group.items():
            stem = f"{case}__{group_id}"
            if re.fullmatch(r"[a-zA-Z0-9_-]+", stem) is None:
                raise ValueError(f"Unsafe figure filename: {stem}")
            groups.append(dict(
                case=case, title=data["titles"][case], comparison_group=group_id,
                comparison_label=rows[0]["comparison_label"], stem=stem,
                rows=[dict(id=row["id"], label=row["chart_label"],
                           **{key: row[key] for key, *_ in SERIES}) for row in rows],
            ))
    return groups


def render(groups, output, font_path, metadata):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 16, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    for group in groups:
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.81, bottom=0.28)
        fig.suptitle(textwrap.fill(group["title"], 55, break_long_words=False),
                     fontsize=24, y=0.96)
        fig.text(0.5, 0.845, group["comparison_label"], ha="center", fontsize=17)
        rows = group["rows"]
        x = list(range(len(rows)))
        for key, label, color, marker, style in SERIES:
            # NaN breaks the line at missing predictions, matching spanGaps=false.
            y = [math.nan if row[key] is None else row[key] for row in rows]
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=style,
                    linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=1.8)
        ax.set_xticks(x, [textwrap.fill(row["label"], 16, break_long_words=False)
                         for row in rows], fontsize=14)
        ax.set_xlim(-0.35, max(0.35, len(rows) - 0.65))
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylabel("VF cycles", fontsize=19, labelpad=12)
        ax.set_xlabel("版本 / Unroll（同 shape、同参数组）", fontsize=18, labelpad=14)
        ax.grid(axis="y", color="#dedede", linewidth=0.8)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
                   ncol=2, frameon=False, fontsize=15, columnspacing=2.0)
        fig.text(0.5, 0.025, "A5 / DV100 · 缺失数据保留断点，不按 0 或插值处理",
                 ha="center", fontsize=12, color="#555555")
        # Do not use bbox_inches='tight': exported images must remain exactly 16:9.
        fig.savefig(output / f"{group['stem']}.png", dpi=160, facecolor="white")
        fig.savefig(metadata / f"{group['stem']}.pdf", facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    font = args.font
    if font is None:
        font = next((path for path in (
            Path("/mnt/c/Windows/Fonts/msyh.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ) if path.is_file()), None)
    if font is None or not font.is_file():
        parser.error("A Chinese font is required; pass --font /path/to/font.ttf")
    source = args.root / "metadata/spill_candidate_alignment.json"
    raw = source.read_bytes()
    data = json.loads(raw)
    groups = build_groups(data)
    output = args.output_dir or args.root / "figure/cycles"
    metadata = args.metadata_dir or args.root / "metadata/figure_sources/cycles"
    targets = [folder / f"{group['stem']}.{ext}" for group in groups
               for ext, folder in (("png", output), ("pdf", metadata))]
    targets += [metadata / "README.md", metadata / "index.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        parser.error("Output files already exist; use another --output-dir or explicit --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    render(groups, output, font, metadata)
    for group in groups:
        group["files"] = {
            ext: dict(path=os.path.relpath(folder / f"{group['stem']}.{ext}", metadata),
                      sha256=hashlib.sha256((folder / f"{group['stem']}.{ext}").read_bytes()).hexdigest())
            for ext, folder in (("png", output), ("pdf", metadata))
        }
    index = dict(source="metadata/spill_candidate_alignment.json", source_date=data["date"],
                 source_sha256=hashlib.sha256(raw).hexdigest(), width=2560, height=1440,
                 series=[dict(key=key, label=label) for key, label, *_ in SERIES], groups=groups)
    (metadata / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 版本周期对比图", "",
             f"与 [HTML 汇总页]({os.path.relpath(args.root / 'spill_candidate_alignment_summary.html', metadata)}) 默认完整数据一致，共 {len(groups)} 张图、{len(data['rows'])} 行记录。",
             "每个算子和 shape/参数对照组独立绘图，不跨组连接。横轴沿用 HTML 的分类版本轴，不补入未展示的无 spill U2/U4。",
             "四条曲线分别为原 CAModel、VfSim 全局、VfSim 局部、VfSim 软件优化全局 / 保留原版本。缺失值保留断点，不设为零，也不插值。",
             "PNG 为 2560 × 1440（16:9），PDF 为相同比例的矢量图。未重跑预测或修改周期。",
             "软件侧仍采用全局 Membar；无 spill 对照沿用原版本。历史参数差异及限制参见 HTML 详情。",
             f"数据 SHA256：`{index['source_sha256']}`。各图的数据、行 ID 和文件校验值见 [index.json](./index.json)。", "",
             "| 算子 | Shape / 参数组 | PNG | PDF |", "| --- | --- | --- | --- |"]
    for group in groups:
        lines.append(f"| {group['title']} | {group['comparison_label']} | "
                     f"[查看]({group['files']['png']['path']}) | [下载]({group['files']['pdf']['path']}) |")
    lines += ["", "## 重新生成", "", "在结果仓库根目录执行（需要 matplotlib 和中文字体）：", "", "```bash",
              "python3 scripts/plot_version_cycles.py --font /path/to/chinese-font.ttf --output-dir figure_new", "```", "",
              "默认不覆盖已有文件；确认需要刷新时可显式添加 `--overwrite`。", ""]
    (metadata / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(directory=str(output), groups=len(groups), rows=len(data["rows"]),
                          png_size=[2560, 1440]), ensure_ascii=False))


if __name__ == "__main__":
    main()
