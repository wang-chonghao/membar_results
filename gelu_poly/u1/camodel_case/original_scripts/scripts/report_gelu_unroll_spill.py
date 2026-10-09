import collections
import json
import math
from pathlib import Path
import re
import shutil
import struct

ROOT = Path('/mnt/e/vfsimulator')
OUT = ROOT / 'results/ub_address_dependency_experiment/gelu_poly_i96_unroll_spill_20260911'


def ub_accesses(path):
    records = {}
    current = None
    head = re.compile(r'\[info\]\s+\[(\d+)\]\s+(RV_\w+), pc: (0x[0-9a-fA-F]+), id: (\d+)')
    with path.open(errors='replace') as f:
        for line in f:
            match = head.search(line)
            if match:
                cycle, op, pc, iid = match.groups()
                current = records.setdefault((op, pc, int(iid)), dict(
                    op=op, pc=pc, inst_id=int(iid), cycle=int(cycle), min_address=None, max_address=None))
            elif line.startswith('[info] ['):
                current = None
            if current is not None:
                addresses = [int(v, 16) for v in re.findall(r'Address ([0-9a-fA-F]+)', line)]
                if addresses:
                    current['min_address'] = min(addresses + ([current['min_address']] if current['min_address'] is not None else []))
                    current['max_address'] = max(addresses + ([current['max_address']] if current['max_address'] is not None else []))
    return [v for v in records.values() if v['min_address'] is not None]


summaries = []
failures = []
for case in OUT.glob('u*'):
    build_log = case / 'camodel/build.log'
    if not (case / 'summary.json').exists() and build_log.exists() and 'fatal error:' in build_log.read_text():
        failure = dict(unroll=int(case.name[1:]), iterations=96, elements=6144,
                       status='build_failed', golden_pass=None, build_error=build_log.read_text())
        (case / 'summary.json').write_text(json.dumps(failure, indent=2))
for path in sorted(OUT.glob('u*/summary.json'), key=lambda p: int(p.parent.name[1:])):
    s = json.loads(path.read_text())
    if s.get('status') == 'build_failed':
        failures.append(s)
        continue
    case = path.parent
    pc_text = (case / 'compiled_pc_order.txt').read_text()
    vregs = sorted(set(map(int, re.findall(r'V[dnm]\[(\d+)\]', pc_text))))
    accesses = []
    for direction in ('rd', 'wr'):
        accesses += ub_accesses(case / f'camodel/core0.veccore0.ub.{direction}_log.dump')
    timing = s['timing']
    accesses = [a for a in accesses if timing['first_vf_start'] <= a['cycle'] <= timing['last_vf_end']]
    extra = [a for a in accesses if not (
        (0 <= a['min_address'] <= a['max_address'] < 0x6000)
        or (0x10000 <= a['min_address'] <= a['max_address'] < 0x16000))]
    spills = [a for a in extra if a['op'].startswith(('RV_VLD', 'RV_VST'))]
    s['extra_scalar_load_count'] = sum(a['op'] == 'RV_SLDI' for a in extra)
    s['extra_scalar_store_count'] = sum(a['op'] == 'RV_SSTI' for a in extra)
    s['architectural_vregs_used'] = vregs
    s['spill_load_count'] = sum(a['op'].startswith('RV_VLD') for a in spills)
    s['spill_store_count'] = sum(a['op'].startswith('RV_VST') for a in spills)
    s['spill_pcs'] = sorted({a['pc'] for a in spills}, key=lambda pc: int(pc, 16))
    s['spill_min_byte_address'] = min((a['min_address'] for a in spills), default=None)
    s['spill_max_observed_byte_address'] = max((a['max_address'] for a in spills), default=None)
    mem = json.loads((case / 'memory_records.json').read_text())
    s['barrier_types'] = dict(collections.Counter(re.search(r'type:\s*(\w+)', r['text']).group(1)
        for r in mem if r['op'] == 'RV_SMEM_BAR'))
    s['output_bitwise_equal_u1'] = (case / 'camodel/output0.bin').read_bytes() == (OUT / 'u1/camodel/output0.bin').read_bytes()
    assert len((case / 'camodel/output0.bin').read_bytes()) == 6144 * 4
    actual = struct.unpack('<6144f', (case / 'camodel/output0.bin').read_bytes())
    golden = struct.unpack('<6144f', (case / 'camodel/golden.bin').read_bytes())
    assert all(math.isfinite(x) for x in actual + golden)
    assert all(abs(a - g) <= 1e-6 or abs(a - g) <= 1e-3 * abs(g) for a, g in zip(actual, golden))
    assert s['load_count'] - s['spill_load_count'] == 96
    assert s['store_count'] - s['spill_store_count'] == 96
    (case / 'spill_ub_accesses.json').write_text(json.dumps(spills, indent=2))
    path.write_text(json.dumps(s, indent=2))
    summaries.append(s)
(OUT / 'summary.json').write_text(json.dumps(sorted(summaries + failures, key=lambda s: s['unroll']), indent=2))
lines = ['# GeLU Poly I96：A5 Unroll 与架构寄存器 Spill', '',
    '## 实验口径', '',
    '- 基线：`cce_code/GeLU_poly.dsl`，入口 `foo_add`，固定 96 次循环，每次 64 个 FP32，共 6144 个元素。',
    '- 唯一改动：`#pragma unroll(1)` 中的因子，依次测试 1、2、4、6、8、12、16；不改算术、UB 布局或同步。',
    '- A5：`dav-c310-vec` / `Ascend950PR_9599`；固定 `-O2 -mllvm -cce-aicore-vec-misched=0 --cce-simd-vf-fusion=false`。',
    '- VF cycles = instr_log 最后 VF end - instr_popped_log 首次 VF start；不计 host 墙钟时间和 VF 之外的搬运时间。',
    '- golden：host 中原有 GeLU polynomial FP32 参考实现，6144 个元素逐一检查；阈值 abs=1e-6、rel=1e-3，未调整。',
    '- 原源码只有 96 次 load、96 次 store，无 VF 内 membar。新增访存须结合 UB 实际地址确认 spill。',
    '- 输入 UB `[0x0, 0x6000)`、输出 UB `[0x10000, 0x16000)`；本实验额外访问高地址栈区的 VST/VLD 是编译器生成的 spill/reload。',
    '- 下表访存及 membar 均为完整 VF 的动态条数，不是静态 PC 数。架构 V 寄存器列为编译产物实际使用的不同编号数，不是物理寄存器数。',
    '', '## 结果', '',
    '| Unroll | CAmodel cycles | Golden | 架构 V 寄存器数 | 总 VLD / VST | 额外 spill VST / VLD | Membar |',
    '| --- | ---: | --- | ---: | --- | --- | ---: |']
for s in summaries:
    lines.append(f"| {s['unroll']} | {s['timing']['vf_total_cycles']} | {'PASS' if s['golden_pass'] else 'FAIL'} | {len(s['architectural_vregs_used'])} | {s['load_count']} / {s['store_count']} | {s['spill_store_count']} / {s['spill_load_count']} | {s['membar_count']} |")
for s in failures:
    lines.append(f"| {s['unroll']} | 编译失败，无有效时间 | 未运行 | 无完整产物 | - | - | - |")
lines += ['', '## 结论', '',
    '- 本轮测试点中 U6 最快（1670 cycles）；U8 使用全部 V0–V31，但仍未发生 spill。U12 首次观察到 spill。这不等于已经验证 U9/U10/U11 的精确门槛。',
    '- U12 是可用于下一阶段实验的真实算子：结果正确，但耗时 6037 cycles，为 U6 的约 3.62 倍。',
    '- U12 展开组数为 8（96/12）；每组额外 20 次向量写回、20 次重载及 40 条 membar。全程为 160/160/320，其中 VLD_VST 和 VST_VLD 各 160 条。',
    '- U12 的向量 spill 指令形式：144 条 RV_VSTI + 16 条 RV_VSTS；重载为 144 条 RV_VLDI + 16 条 RV_VLDS，均访问原始输入/输出之外的高地址 UB 栈区。',
    '- U12 另外还有 32 条 RV_SSTI / 32 条 RV_SLDI，用于标量保存/恢复。因此不能把全部时间差直接视为可由 membar 优化消除的开销；后续回放必须保留编译产物，并说明标量建模边界。',
    '- U16 编译报错：33 个 Vector Slots，Total Spilled Byte Size=8480，超过 VF stack size=6144。没有改变编译参数扩大栈，也没有用 VfSim 结果冒充 CAmodel 时间。',
    '- 原始算术计算指令动态数量在六个成功版本中保持一致；所有输出与 U1 逐位相同。', '']
lines += ['', '## 每次测量记录', '']
previous = None
best = None
for s in summaries:
    u, cycles = s['unroll'], s['timing']['vf_total_cycles']
    lines += [f'### U{u}', '',
        f"- VF start={s['timing']['first_vf_start']}，end={s['timing']['last_vf_end']}；{s['checks'][0]}",
        f"- 与 U1 输出逐位一致：{s['output_bitwise_equal_u1']}。",
        f"- 相比前一个候选：{cycles-previous if previous is not None else 0:+d} cycles；相比此前最优：{cycles-best if best is not None else 0:+d} cycles。",
        f"- 源码：[U{u}](u{u}/source/gelu_poly_i96_u{u}.cce)；[逐 PC 指令](u{u}/compiled_pc_order.txt)；[spill 实际 UB 地址](u{u}/spill_ub_accesses.json)。",
        f"- 同步类型：{s['barrier_types']}；源码 SHA256：`{s['source_sha256']}`。", '']
    previous = cycles
    best = min(best, cycles) if best is not None else cycles
for s in failures:
    lines += [f"### U{s['unroll']} 编译失败", '', '```text', s['build_error'].strip(), '```', '']
lines += ['## 复现', '',
    '脚本、host/golden 实现与构建脚本保存在 `scripts/`；各候选的 build.log、run.log、输入/输出/golden 二进制，以及 core0 的 instr_log、instr_popped、IDU、EXU、UB 读写日志保存在 `u*/camodel/`。',
    '运行需要本机 CANN CAmodel 工具链；脚本使用 `/tmp/vfsim_gelu_unroll_spill_20260911/` 作为构建目录。', '',
    '```bash', 'python3 scripts/gelu_unroll_spill_scan.py 1 2 4 6 8 12 16',
    'python3 scripts/report_gelu_unroll_spill.py', '```', '',
    '本轮仅测 CAmodel 和编译器 spill。尚未回放编译后的指令到 VfSim，也尚未测量局部 UB 依赖解除能带来的收益。', '']
(OUT / 'README.md').write_text('\n'.join(lines))
if Path(__file__).resolve() != (OUT / 'scripts' / Path(__file__).name).resolve():
    shutil.copy2(__file__, OUT / 'scripts' / Path(__file__).name)
for s in summaries:
    print(s['unroll'], s['timing']['vf_total_cycles'], len(s['architectural_vregs_used']),
          s['spill_store_count'], s['spill_load_count'], s['membar_count'], s['spill_min_byte_address'])
