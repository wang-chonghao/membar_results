"""将历史 A6 Softmax 多形状实验整理为独立离线报告，不重新运行模型。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SHAPES = (8, 12, 16, 24, 32)
VARIANTS = ("three_loop", "gap2")
NAMES = {"three_loop": "三循环", "gap2": "两行错位融合"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_root, archive):
    experiment = source_root / "results/ub_address_dependency_experiment"
    source = experiment / "softmax_fp16_row_scaling"
    target = archive / "a6_softmax_fp16_row_scaling"
    records = read_json(source / "summary.json")
    indexed = {(r["rows"], r["variant"]): r for r in records}
    if len(indexed) != 10 or len(records) != 10:
        raise ValueError("应有五种 shape、两个版本，共十条结果")
    files = []

    def copy(src, relative):
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if digest(src) != digest(dst):
            raise ValueError(f"归档校验失败：{relative}")
        files.append({"path": relative, "sha256": digest(dst)})

    copy(source / "summary.json", "evidence/summary.json")
    copy(source / "README.md", "evidence/original_experiment.md")
    plots = ["row_scaling.png", "ipc_five_rows_window10.png",
             "ipc_five_rows_window10.json"]
    plots += [f"ipc_rows{rows}_window10.csv" for rows in SHAPES]
    for name in plots:
        copy(source / "plots" / name, f"plots/{name}")
    ipc = read_json(target / "plots/ipc_five_rows_window10.json")
    for panel in ipc["panels"]:
        for series in panel["series"]:
            record = indexed[panel["rows"], series["variant"]]
            if (series["cycles"], series["events"]) != (
                record["vf_end_cycle"], record["compute_events"]
            ):
                raise ValueError("IPC 图元数据与周期/事件数不一致")

    table = []
    source_rows = []
    for rows in SHAPES:
        pair = {variant: indexed[rows, variant] for variant in VARIANTS}
        for variant, record in pair.items():
            if (record["cols"], record["soc"], record["soc_selector"]) != (
                1024, "DV121", "dav-9201"
            ) or not record["output_address_coverage_pass"]:
                raise ValueError("shape、SoC 或输出覆盖校验不符合归档口径")
            folder = source / f"rows{rows}" / variant
            sources = list((folder / "source").glob("*.cce"))
            if len(sources) != 1 or digest(sources[0]) != record["source_sha256"]:
                raise ValueError(f"源码校验失败：rows{rows}/{variant}")
            prefix = f"rows{rows}/{variant}"
            copy(sources[0], f"{prefix}/source/kernel.cce")
            copy(folder / "summary.json", f"{prefix}/summary.json")
            if read_json(folder / "summary.json") != record:
                raise ValueError(f"单例结果与汇总不同：{prefix}")
            copy(folder / "vfsim/local/model_warnings.json",
                 f"{prefix}/model_warnings.json")
            source_rows.append(
                f'<tr><th scope="row">{rows} × 1024 · {NAMES[variant]}</th>'
                f'<td><a href="a6_softmax_fp16_row_scaling/{prefix}/source/kernel.cce">CCE 源码</a></td>'
                f'<td><a href="a6_softmax_fp16_row_scaling/{prefix}/summary.json">预测记录</a></td>'
                f'<td><a href="a6_softmax_fp16_row_scaling/{prefix}/model_warnings.json">参数与模型警告</a></td></tr>'
            )
        baseline = pair["three_loop"]["vf_end_cycle"]
        fused = pair["gap2"]["vf_end_cycle"]
        saved = baseline - fused
        trend_only = rows == 32
        row_class = ' class="trend-only"' if trend_only else ""
        table.append(
            f'<tr{row_class}>'
            f'<th scope="row">{rows} × 1024</th><td>{baseline}</td><td>{fused}</td>'
            f'<td>{saved:+d}</td><td>{saved / baseline * 100:.2f}%</td>'
            f'<td>{baseline / fused:.3f}×</td>'
            f'<td>{pair["three_loop"]["ub_bytes"] / 1024:.4f}</td>'
            f'<td>{pair["gap2"]["ub_bytes"] / 1024:.0f}</td>'
            f'<td>{"超预算；仅趋势" if trend_only else "名义占用低于预算"}</td></tr>'
        )

    gap2 = experiment / "softmax_giantloopfusion_fp16_gap2"
    copy(gap2 / "README.md", "evidence/gap2_original_experiment.md")
    copy(gap2 / "golden_summary.json", "evidence/gap2_a5_golden_summary.json")
    golden = read_json(gap2 / "golden_summary.json")
    if not golden["golden_pass"] or golden["shape"] != [8, 1024]:
        raise ValueError("历史 A5 数值验证证据不符合预期")
    manifest = {
        "description": "历史实验材料的字节校验；不表示重新仿真或 A6 硬件验证",
        "files": files,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    page = TEMPLATE.replace("{{RESULT_ROWS}}", "\n".join(table))
    page = page.replace("{{SOURCE_ROWS}}", "\n".join(source_rows))
    runnable_rows = []
    for rows in (8, 12, 16, 24):
        prefix = f'a6_softmax_fp16_row_scaling/runnable_a6/rows{rows}'
        runnable_rows.append(
            f'<tr><th scope="row">{rows} × 1024</th>'
            f'<td><a href="{prefix}/no_membar.cce">无 membar 融合版 CCE</a></td>'
            f'<td><a href="{prefix}/three_loop.cce">三循环 CCE</a></td></tr>'
        )
    page = page.replace("{{RUNNABLE_ROWS}}", "\n".join(runnable_rows))
    report = archive / "a6_softmax_fusion_results_summary.html"
    report.write_text(page, encoding="utf-8")
    print(json.dumps({"report": str(report), "shapes": len(SHAPES),
                      "source_versions": len(source_rows),
                      "verified_files": len(files)}, ensure_ascii=False))


TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A6 Softmax 融合与多 Shape 实验结果汇总</title>
<style>
:root { color-scheme: light; font-family: system-ui, "Microsoft YaHei", sans-serif; color: #202624; background: #fff; }
* { box-sizing: border-box; letter-spacing: 0; }
body { margin: 0; font-size: 15px; line-height: 1.75; }
a { color: #14624d; text-underline-offset: 3px; overflow-wrap: anywhere; }
a:hover { color: #903649; }
a:focus-visible { outline: 3px solid #14624d; outline-offset: 3px; }
aside { position: fixed; inset: 0 auto 0 0; width: 230px; background: #f3f5f4; padding: 28px 20px; border-right: 1px solid #d8dfdb; overflow-y: auto; }
aside strong { display: block; font-size: 20px; margin-bottom: 8px; }
aside p { color: #53615a; font-size: 13px; }
nav a { display: block; padding: 9px 0; }
main { margin-left: 230px; max-width: 1500px; padding: 32px 38px 60px; min-width: 0; }
header { border-bottom: 2px solid #14624d; padding-bottom: 18px; }
h1 { font-size: 29px; line-height: 1.4; margin: 7px 0 12px; overflow-wrap: anywhere; }
h2 { font-size: 22px; margin: 0 0 15px; line-height: 1.5; }
p { margin: 10px 0; }
.eyebrow, .muted { color: #53615a; }
.eyebrow { font-size: 13px; margin: 0; }
section { padding: 28px 0; border-bottom: 1px solid #d8dfdb; scroll-margin-top: 16px; }
.notice { padding: 10px 15px; border-left: 4px solid #a14650; background: #fff4f4; }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
caption { text-align: left; margin-bottom: 10px; color: #53615a; }
th, td { text-align: right; border-bottom: 1px solid #d8dfdb; padding: 11px 12px; white-space: nowrap; }
thead th { background: #eef3f0; color: #344b40; }
th:first-child, td:last-child { text-align: left; }
tbody th { font-weight: 500; }
.trend-only { background: #fff1f2; }
.sources th, .sources td { text-align: left; }
code, pre { font-family: ui-monospace, Consolas, monospace; font-size: 14px; }
pre { background: #f3f5f4; border-left: 3px solid #14624d; padding: 16px; overflow: auto; line-height: 1.8; }
figure { margin: 20px 0 0; }
img { display: block; width: 100%; height: auto; }
figcaption { color: #53615a; font-size: 13px; margin-top: 10px; }
ul { padding-left: 23px; }
li { margin: 8px 0; }
footer { padding-top: 20px; color: #53615a; font-size: 13px; }
@media (max-width: 760px) {
  aside { position: static; width: auto; padding: 18px; border-right: 0; border-bottom: 1px solid #d8dfdb; }
  aside strong { font-size: 18px; }
  aside p { margin: 5px 0; }
  nav { display: flex; flex-wrap: wrap; gap: 0 20px; }
  nav a { padding: 5px 0; }
  main { margin: 0; padding: 23px 18px 40px; }
  h1 { font-size: 25px; }
  h2 { font-size: 20px; }
}
</style>
</head>
<body>
<aside>
<strong>Membar 实验结果</strong>
<p>A6 · Softmax 融合专题</p>
<nav aria-label="报告章节">
<a href="#overview">实验概览</a>
<a href="#schedule">两行错位流水</a>
<a href="#results">多 Shape 周期与 UB</a>
<a href="#ipc">计算完成 IPC</a>
<a href="#runnable">完整 A6 运行入口</a>
<a href="#limits">验证范围与限制</a>
<a href="#sources">各版本 CCE 与证据</a>
<a href="spill_candidate_alignment_summary.html">A5 实验汇总</a>
</nav>
</aside>
<main>
<header id="overview">
<p class="eyebrow">A6 / DV121 / dav-9201 · 历史 VfSim 局部 UB 地址依赖实验</p>
<h1>Softmax 融合与多 Shape 结果汇总</h1>
<p>固定 1024 列，对比三循环实现与 SUM→NORM 间隔两行的融合实现。输入与中间值为 FP32，输出为 FP16。</p>
<p class="notice"><strong>本页为历史模型预测，不是 A6 CAModel 实测。</strong>所有 shape 均启用局部 UB 地址依赖；本表不表示全局 membar 与局部同步的直接 A/B。此次仅整理材料，没有重新运行模型。</p>
</header>
<section id="schedule">
<h2>两行错位流水</h2>
<p>三循环分别完成所有行的最大值、exp 与求和、归一化。融合版将不同行的三个阶段交错，以两个向量、128 个元素为一个处理单元；外层行循环不手动展开。</p>
<pre>启动：MAX(0)
      MAX(1) + SUM(0)
      MAX(2) + SUM(1)
稳态：MAX(i) + SUM(i-1) + NORM(i-3)，i = 3 ... R-1
排空：SUM(R-1) + NORM(R-3)
      NORM(R-2)
      NORM(R-1)</pre>
<p>同一行的 SUM 与 NORM 相隔两次外层迭代，为求和、除法和倒数准备留出时间。倒数由两个寄存器状态传递；中间 exp 仍写入 UB，再由 NORM 读取，相关访存由局部地址依赖保护。</p>
<p>输出链为 <code>VMUL → VCVT(FP32→FP16) → VSTS(PK_B32)</code>，没有独立 VPACK。这里是浮点精度转换，不是 INT8 量化。</p>
<figure><a href="a6_softmax_fp16_row_scaling/plots/giant_loop_interleaving.png"><img src="a6_softmax_fp16_row_scaling/plots/giant_loop_interleaving.png" width="4160" height="1680" alt="Softmax giant-loop 128元素分段交错排列"></a>
<figcaption>上方对比按阶段集中排列与按128元素交错；下方展开稳态 pair=1 的20条实际指令。箭头仅表示源码顺序，不是数据依赖边或硬件周期。</figcaption></figure>
<p id="first-segment-note"><strong>首段初始化例外：</strong>第0段的 I0 只含两条 VLDS，直接初始化 max_acc0/max_acc1，不执行 VMAX；因此 steady loop 开头第一个计算指令是 VEXPDIF。VMAX 更新从内层 pair=1 开始。</p>
<p><a href="a6_softmax_fp16_row_scaling/plots/giant_loop_interleaving.svg">交错排列矢量图</a> · <a href="a6_softmax_fp16_row_scaling/plots/giant_loop_interleaving.md">逐段说明</a> · <a href="a6_softmax_fp16_row_scaling/plots/giant_loop_interleaving.json">源码核对记录</a></p>
<p>8×1024 的历史一行间距版本为 776 cycles，两行间距为 570 cycles，减少 206 cycles（26.55%）；这是另外一组同 shape 调度对照，不是下面多 shape 表的基线。</p>
</section>
<section id="results">
<h2>多 Shape 周期与 UB 占用</h2>
<div class="table-scroll" tabindex="0" role="region" aria-label="A6 多形状结果表">
<table>
<caption>周期口径：vf_end_cycle。节省周期 = 三循环 − 融合；周期降幅 = 节省 / 三循环；加速比 = 三循环 / 融合。UB 单位为 KiB（1024 bytes）。</caption>
<thead><tr><th scope="col">输入 Shape</th><th scope="col">三循环</th><th scope="col">两行错位融合</th><th scope="col">节省周期</th><th scope="col">周期降幅</th><th scope="col">加速比</th><th scope="col">三循环 UB</th><th scope="col">融合 UB</th><th scope="col">248 KiB 预算对照</th></tr></thead>
<tbody>{{RESULT_ROWS}}</tbody>
</table></div>
<p>UB 为当前完整 buffer 布局的名义占用：输入 4RC bytes + exp 中间值 4RC bytes + FP16 输出 2RC bytes；三循环另有 8R bytes 统计量。它不含对齐预留、VF 栈等额外开销。</p>
<p class="notice"><strong>32 行版本仅用于吞吐趋势。</strong>完整 buffer 约 320 KiB，超过 248 KiB 预算；不能据此声称该版本可直接在硬件上执行。24 行版本约 240 KiB，也需要进一步验证实际布局和额外开销。</p>
<p>在 8→32 行区间，三循环平均每新增一行约增加 66.42 cycles，融合版约增加 53.50 cycles。更低的持续处理成本使融合收益随行数增长而显现；启动与排空成本的解释是趋势判断，未逐事件分离验证。</p>
<figure><a href="a6_softmax_fp16_row_scaling/plots/row_scaling.png"><img src="a6_softmax_fp16_row_scaling/plots/row_scaling.png" width="1920" height="1120" alt="A6 三循环与两行错位融合的总周期随行数增长对比图"></a>
<figcaption>保留的历史总周期图仅绘制 8、16、32 行；完整五种 shape 见上表及下方 IPC。32 行点为超 UB 预算的模型趋势数据。</figcaption></figure>
</section>
<section id="ipc">
<h2>计算完成 IPC 对比</h2>
<p>按 10-cycle trailing window 统计每周期计算指令完成数，不是所有指令的发射 IPC。每条曲线以自己的首次计算完成为零点；五个子图纵轴统一为 0～3，横轴各自取范围，单位均为 cycle。图例周期为完整 VF 时间。</p>
<figure><a href="a6_softmax_fp16_row_scaling/plots/ipc_five_rows_window10.png"><img src="a6_softmax_fp16_row_scaling/plots/ipc_five_rows_window10.png" width="3240" height="2700" alt="8、12、16、24、32 行的三循环与融合版计算完成 IPC 五联图" loading="lazy"></a>
<figcaption>从上到下为 8、12、16、24、32 行；最后一组仅供趋势观察。图表元数据与十组周期、计算事件数已核对。</figcaption></figure>
<p>IPC CSV：<a href="a6_softmax_fp16_row_scaling/plots/ipc_rows8_window10.csv">8 行</a> · <a href="a6_softmax_fp16_row_scaling/plots/ipc_rows12_window10.csv">12 行</a> · <a href="a6_softmax_fp16_row_scaling/plots/ipc_rows16_window10.csv">16 行</a> · <a href="a6_softmax_fp16_row_scaling/plots/ipc_rows24_window10.csv">24 行</a> · <a href="a6_softmax_fp16_row_scaling/plots/ipc_rows32_window10.csv">32 行</a></p>
</section>
<section id="runnable">
<h2>完整 A6 运行入口</h2>
<p>下方历史 CCE 链接是 VF-only 快照，只接收 UB 地址，不能直接由 host 启动。这里另提供 8/12/16/24×1024 的八份完整独立 CCE，保留原 VF，增加 GM→UB NDDMA、UB→GM scatter 和外层同步。32 行按要求不提供完整入口。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="完整可运行 CCE 索引"><table>
<thead><tr><th scope="col">Shape</th><th scope="col">两行错位融合</th><th scope="col">三循环同步基线</th></tr></thead>
<tbody>{{RUNNABLE_ROWS}}</tbody></table></div>
<p>无 membar 版的 VF 内没有同步指令；三循环版保留原有两条循环间 mem_bar(VST_VLD)，不是给融合版加 barrier。每份 CCE 适配 microbenchmark-cce-direct/run_test.py 的 MB_KERNEL / MB_TENSOR；输入 FP32、输出 FP16，只使用 block 0。三循环另为 memMax/memSum 分配空间，避免输出覆盖统计量。</p>
<p><a href="a6_softmax_fp16_row_scaling/runnable_a6/run_softmax_a6.py">启动及自动精度检查脚本</a> · <a href="a6_softmax_fp16_row_scaling/runnable_a6/check_softmax.py">独立 FP16 softmax golden</a> · <a href="a6_softmax_fp16_row_scaling/runnable_a6/README.md">完整运行命令与环境说明</a></p>
<p><a href="a6_softmax_fp16_row_scaling/runnable_a6/validation/shape_kernel_compile.json">八份 CCE 编译记录</a> · <a href="a6_softmax_fp16_row_scaling/runnable_a6/cases.json">源码校验与 UB 布局</a>。旧 8 行逐行加 barrier 对照保留在目录根部，不是本表的三循环基线。</p>
<p class="notice">八份完整 CCE 已通过 A6 kernel 编译和设备链接，但本机缺少 A6 runtime 模型库，尚未通过 A6 CAModel 数值验证。原 runner 的 MB_GOLDEN 不支持 FP16；请使用配套启动脚本强制检查真实输入/输出，不以正常退出代替精度通过。24 行还需在目标环境确认额外 VF 栈开销。</p>
</section>
<section id="limits">
<h2>验证范围与限制</h2>
<ul>
<li><strong>不是 A6 硬件精度验证。</strong>五种 shape 没有运行 A6 CAModel，也没有可以用于计算周期预测精度的 A6 golden。</li>
<li>历史 8×1024 gap2 原版另做过 A5 CAModel 数值验证：8192 个 FP16 输出通过，最大相对误差约 0.04672%，A5 VF 时间 634 cycles。这不等于本次多 shape VF-only 源码逐项通过，更不能作为 A6 实测时间。</li>
<li>多 shape 已检查每行 16 条 VEXPDIF、16 条 VMUL、16 条 VCVT 和 1 条 VDIV，以及 FP16 输出起始地址完整无重复；这些是结构与地址覆盖检查，不是数值执行验证。</li>
<li>正式 DV121 配置保持 68 个物理向量寄存器、SHQ 深度 44，未为候选修改资源配置。历史 forwarding/II fallback 警告保留，当前参数补齐或模型变化后需重新预测。</li>
<li>最终 PK_B32 输出 store 的 span 当时仍为 unknown_span；该输出对象独立且没有后续 load。因此不能称为全部访存都精确建模，融合版精确范围覆盖率为 80%。</li>
<li>三循环预测移除两条全局 membar 并启用局部地址依赖；融合源码本身没有显式 membar。两者的指令数量和资源生命周期不同，不能把差异全部归因于同步方式。</li>
<li>本页只归档源码、预测摘要、模型警告和 IPC 证据，不伪造 A6 CAModel 日志，也不复制体积较大的逐周期调试历史。</li>
</ul>
<p><a href="a6_softmax_fp16_row_scaling/evidence/gap2_a5_golden_summary.json">历史 A5 数值验证</a> · <a href="a6_softmax_fp16_row_scaling/evidence/original_experiment.md">多 Shape 原始记录</a> · <a href="a6_softmax_fp16_row_scaling/evidence/gap2_original_experiment.md">两行错位原始记录</a></p>
</section>
<section id="sources">
<h2>各版本 CCE 与证据</h2>
<div class="table-scroll" tabindex="0" role="region" aria-label="源码和预测材料索引"><table class="sources">
<thead><tr><th scope="col">Shape / 版本</th><th scope="col">源码</th><th scope="col">结果</th><th scope="col">覆盖边界</th></tr></thead>
<tbody>{{SOURCE_ROWS}}</tbody></table></div>
<p>这些是历史 VF-only 源码快照，不是包含 host、UB 分配和编译运行环境的硬件完整用例。文件名 <code>kernel.cce</code> 仅统一归档命名，源码字节未修改。</p>
<p><a href="a6_softmax_fp16_row_scaling/evidence/summary.json">十组原始预测汇总</a> · <a href="a6_softmax_fp16_row_scaling/plots/ipc_five_rows_window10.json">IPC 元数据</a> · <a href="a6_softmax_fp16_row_scaling/manifest.json">文件 SHA256 清单</a></p>
</section>
<footer>整理日期：2026-10-19。历史周期与图表保持原样；报告和所有引用材料均位于 membar_results 内。</footer>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vfsim-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build(args.vfsim_root.resolve(), args.archive_root.resolve())
