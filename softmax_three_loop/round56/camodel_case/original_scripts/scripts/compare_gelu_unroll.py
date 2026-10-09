import json
from pathlib import Path
import sys
import shutil

ROOT = Path('/mnt/e/vfsimulator')
sys.path.insert(0, str(ROOT))
from api.cce_adapter import parse_cce_canonical_vf_info
from api.simulator_costmodel import CoreVfCostModel

CASE = ROOT / 'results/ub_address_dependency_experiment/gelu_poly_i96_unroll_spill_20260911'
rows = []
for u in (1, 2, 4, 6, 8, 12):
    sub = CASE / f'u{u}'
    vf = parse_cce_canonical_vf_info(sub / f'source/gelu_poly_i96_u{u}.cce',
                                   kernel_name='gelu_simd_ub', loop_params={'repeat_times': 96})
    result = CoreVfCostModel(soc='dav-3510', out_dir=sub / 'vfsim_source_global',
                            ub_dependency_mode='disabled').run_vf_info(vf)
    measured = json.loads((sub / 'summary.json').read_text())['timing']['vf_total_cycles']
    predicted = result['vf_end_cycle']
    row = dict(unroll=u, camodel_cycles=measured, vfsim_source_cycles=predicted,
               source_accuracy_percent=100*(1-abs(predicted-measured)/measured), result=result)
    if u == 12:
        replay = json.loads((sub / 'compiled_replay/ub_comparison/summary.json').read_text())
        row['vfsim_compiled_global_cycles'] = replay['baseline']['vf_end_cycle']
        row['compiled_accuracy_percent'] = 100*(1-abs(row['vfsim_compiled_global_cycles']-measured)/measured)
    rows.append(row)
    (CASE / 'camodel_vfsim_comparison.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)
if Path(__file__).resolve() != (CASE / 'scripts/compare_gelu_unroll.py').resolve():
    shutil.copy2(__file__, CASE / 'scripts/compare_gelu_unroll.py')
lines = ['# GeLU I96：CAmodel 与 A5 VfSim 全局模式对比', '',
         '精度 = (1 - abs(VfSim - CAmodel) / CAmodel) * 100%。固定 A5/DV100，68 个物理寄存器，启用当前全局 membar 时序；不是局部 UB 依赖模式。', '',
         '| Unroll | CAmodel cycles | VfSim 原始 CCE cycles | 原始 CCE 精度 | VfSim 编译后 spill 回放 cycles | 回放精度 |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for row in rows:
    extra = f"{row['vfsim_compiled_global_cycles']} | {row['compiled_accuracy_percent']:.2f}%" if 'vfsim_compiled_global_cycles' in row else '- | -'
    lines.append(f"| {row['unroll']} | {row['camodel_cycles']} | {row['vfsim_source_cycles']} | {row['source_accuracy_percent']:.2f}% | {extra} |")
lines += ['', 'U1/U2/U4/U6/U8 未发现编译器向量 spill。U12 编译器新增 160 次 VST、160 次 VLD 和 320 条 membar，原始 CCE 中没有这些指令，因此必须区分原始输入与编译后回放。',
          'U12 局部 UB 依赖实验为 1656 cycles，属于假设硬件能力优化结果，不应对原硬件 CAmodel 6037 cycles 计算预测精度。',
          '全部六个 CAmodel 版本 golden PASS；此处精度指周期预测精度，不是数值正确率。', '']
(CASE / 'camodel_vfsim_comparison.md').write_text('\n'.join(lines))
