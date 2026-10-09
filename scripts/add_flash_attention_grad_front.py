"""Append the validated extracted FlashAttention Grad VF without rebuilding history."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys


CASE_ID = 'flash_attention_grad_front'


def collect_case(builder, repository):
    experiments = repository/'results/flash_attention_grad_front_unroll2'
    baseline = repository/'results/membar_candidate_baselines_20261008/flash_attention_grad_front'
    bank = experiments/'bank_aware_u2'
    bank_results = {row['case']:row for row in builder.read(bank/'summary.json')['reports']}
    variants = []
    definitions = [
        ('u1', '原始单循环 U1', baseline),
        ('u2_original', '原序直接展开 U2', bank/'original_order/flash_attention_grad_front'),
        ('u2_tree_local', '最短生命周期重排 U2', experiments/'lifetime_optimized_u2/flash_attention_grad_front'),
        ('u2_concentrated_spill', '集中加载 U2 · 编译器 spill', experiments/'manual_u2/flash_attention_grad_front'),
        ('u2_bank_window1', 'Bank-aware U2 · 预取 1 组', bank/'bank_window_1/flash_attention_grad_front'),
        ('u2_bank_window2', 'Bank-aware U2 · 预取 2 组', bank/'bank_window_2/flash_attention_grad_front'),
        ('u2_bank_window3', 'Bank-aware U2 · 预取 3 组', bank/'bank_window_3/flash_attention_grad_front'),
    ]
    for variant_id,label,folder in definitions:
        summary = builder.read(folder/'evidence/summary.json')
        audit = builder.read(folder/'evidence/alignment_audit.json')
        assert summary['validation']['passed'] and audit['register_dependency_graph_equal']
        spilled = variant_id=='u2_concentrated_spill'
        global_model = experiments/'manual_u2/ub_comparison/global' if spilled else folder/'vfsim_hardware'
        local_model = experiments/'manual_u2/ub_comparison/local' if spilled else None
        comparison = builder.read(experiments/'manual_u2/ub_comparison/comparison.json') if spilled else None
        note = ('保留实际编译器插入的 84 次向量保存、84 次重载和 168 条 mem_bar。'
                '全局 3208 / 局部 306 来自相同 PC 对齐指令流；306 是假设硬件具备局部 UB 地址识别能力的预测。'
                '源码预测 208 没有包含 spill，不能用于本二进制的精度评估。'
                '另有 48 对标量栈保存/重载，其流水时序未建模；UB bank 冲突也未建模。') if spilled else (
                '无 vector/predicate/scalar spill，输出与原始 U1 逐字节一致。'
                '预测使用按本版本实际日志还原的 PC 指令流；未增加固定启动补偿。')
        if variant_id=='u2_tree_local':
            note += '仅缩短生命周期，没有规避连续 load 的 bank 竞争，读 bank 冲突 129 条。'
        elif variant_id=='u2_original':
            note += '直接复制两次原循环体，不是把两行全部 load 集中到前面；读 bank 冲突 4 条。'
        elif variant_id.startswith('u2_bank_window'):
            note += '保持 src=0、grad=0x8000，交错访问 bank 半区；读 bank 冲突为 0。'
        evidence = [folder/'evidence/summary.json',folder/'evidence/validation.json',
                    folder/'evidence/alignment_audit.json',folder/'evidence/commands.json']
        if spilled:
            evidence.append(experiments/'manual_u2/ub_comparison/comparison.json')
        variant = builder.make_variant(CASE_ID,variant_id,label,
            folder/'source/kernel.cce',folder/'camodel',summary['camodel'],
            source_cycles=summary['vfsim_source'],global_cycles=summary['vfsim_hardware'],
            local_cycles=comparison['local']['vf_end_cycle'] if comparison else None,
            global_model=global_model,local_model=local_model,
            source_model=folder/'vfsim' if spilled else None,
            replay=folder/'source/hardware_equivalent.cce',spill=(84,84) if spilled else (0,0),
            membars=168 if spilled else 0,note=note,evidence=evidence)
        variant['vector_arch_registers'] = len(audit['vector_registers'])
        variant['load_bank_conflicts'] = None
        bank_key = {'u2_original':'original_order','u2_bank_window1':'bank_window_1',
                    'u2_bank_window2':'bank_window_2','u2_bank_window3':'bank_window_3'}.get(variant_id)
        if bank_key:
            variant['load_bank_conflicts'] = bank_results[bank_key]['ub_bank_conflict_records']
        elif variant_id=='u2_tree_local':
            variant['load_bank_conflicts'] = 129
        if spilled:
            target = builder.DEST/CASE_ID/variant_id/'source'
            hardware = target/'hardware_equivalent.cce'
            header = ('// 实际硬件执行的 VF 指令流等效 CCE，包含编译器插入的向量 spill 和 mem_bar。\n'
                      '// 依据同版本 CAModel 日志按 PC 还原；标量地址效果已折入，标量流水未建模。\n')
            hardware.write_text(header+(folder/'source/hardware_equivalent.cce').read_text(),encoding='utf-8')
            mapping = builder.read(folder/'source/pc_mapping.json')
            counts = Counter()
            for row in mapping:
                counts[row['opcode']] += row['dynamic_count']
            recorded = summary['hardware_op_counts']
            assert all(recorded[op]==count for op,count in counts.items())
            assert counts['RV_SMEM_BAR']==168
            digest = hashlib.sha256(hardware.read_bytes()).hexdigest()
            mapping_path = target/'hardware_pc_mapping.json'
            mapping_path.write_text(json.dumps(dict(modeled_pc_coverage_complete=True,
                dynamic_instruction_counts=dict(counts),sha256=digest,pc_mapping=mapping,
                unrepresented_opcodes={op:count for op,count in recorded.items() if op not in counts},
                register_dependency_graph_equal=True,dynamic_load_store_addresses_checked=368,
                scope='vector/predicate/LSU/membar; scalar address effects folded, scalar timing excluded'),
                ensure_ascii=False,indent=2),encoding='utf-8')
            for path in (hardware,mapping_path):
                builder.MANIFEST.append(dict(path=str(path.relative_to(builder.DEST)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            variant['hardware_equivalent'] = dict(path=str(hardware.relative_to(builder.DEST)),
                mapping=str(mapping_path.relative_to(builder.DEST)),pc_coverage=True,
                camodel_cycles=summary['camodel'],prediction_used=True)
            variant['sources'].insert(0,dict(label='实际硬件执行指令流 · 等效 CCE（含 spill）',
                                            path=str(hardware.relative_to(builder.DEST))))
        (builder.DEST/CASE_ID/variant_id/'result.json').write_text(
            json.dumps(variant,ensure_ascii=False,indent=2),encoding='utf-8')
        variants.append(variant)
    case = dict(id=CASE_ID,title='FlashAttention Grad 前段',subtitle='U2 spill、指令重排与 UB bank',
        shape='提取 VF：FP32，8 行 × 768，输出 8 个行点积；不是完整 FlashAttention Grad 算子。',
        default='u2_concentrated_spill',
        headline='集中加载 U2：全局预测 3208、局部预测 306；bank-aware 重排的本轮最佳实测为 228 cycle。',
        notes=['原仓库是 AscendC 源码，本 case 为手动翻译的提取 VF；不能将其时间称为完整算子时间。',
               'U2 并非必然 spill：原序直接复制两次循环体，实测 234，无 spill；集中加载两行的版本实测 3237，发生 spill。',
               '本轮三个 bank-aware 窗口实测 228 / 229 / 234；保持 UB 布局不变，读 bank 冲突均为 0，没有新增 mem_bar。',
               '228 是 CAModel 实测；306 是假设性局部同步预测。耗时数值比为 74.51%，不是已经验证的硬件性能比。',
               '保持 cap7，未修改正式预测核心、forwarding 或参数。参数版本为 MultiSoC-membar 1ff6ada。'],
        variants=variants,metrics_columns=[dict(key='vector_arch_registers',label='向量架构寄存器'),
                                          dict(key='load_bank_conflicts',label='UB 读 bank 冲突')],
        comparison=dict(title='重排实测与局部同步预测的数值比较',rows=[
            dict(label='Bank-aware 重排 U2，本轮最佳',cycles=228,basis='CAModel 实测，无 spill'),
            dict(label='集中加载、保留 spill，只替换同步语义',cycles=306,basis='VfSim 局部依赖预测，未有对应硬件实测')],
            note='228 / 306 = 74.51%，仅是跨口径的耗时数值比较；不能据此确认真实硬件加速 1.34 倍。'
                 '同指令流模型内的同步收益应比较 3208 / 306 = 10.48 倍，仍是实验预测。'),
        overlay=True,overlay_default=False,overlay_variant_ids=['u1','u2_original','u2_bank_window1'],
        overlay_label='U1 / 原序 U2 / bank-aware U2 实测',
        overlay_figure=f'{CASE_ID}/figures/camodel_u1_original_u2_bank_aware.png',
        overlay_figure_label='U1 / 原序 U2 / bank-aware U2 实测叠图')
    (builder.DEST/CASE_ID/'summary.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    return case


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--vfsim-root',type=Path,required=True)
    args=parser.parse_args()
    import build_archive as builder
    builder.ROOT=args.vfsim_root
    builder.DEST=args.vfsim_root/'results/membar_results'
    builder.MANIFEST=[]
    metadata=builder.DEST/'metadata'
    data=builder.read(metadata/'data.json')
    old_manifest=builder.read(metadata/'manifest.json')
    case=collect_case(builder,args.vfsim_root)
    data['cases']=[item for item in data['cases'] if item['id']!=CASE_ID]+[case]
    data['generated']='2026-10-09'
    (metadata/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    (metadata/'data.js').write_text('window.MEMBAR_REPORT='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    manifest=[item for item in old_manifest if not item['path'].startswith(CASE_ID+'/')]+builder.MANIFEST
    (metadata/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    index=builder.read(metadata/'hardware_equivalent_index.json')
    index=[item for item in index if not item['file'].startswith(CASE_ID+'/')]
    spilled=next(item for item in case['variants'] if item['vector_store'])
    mapping=builder.read(builder.DEST/spilled['hardware_equivalent']['mapping'])
    index.append(dict(case=case['title'],variant=spilled['label'],file=spilled['hardware_equivalent']['path'],
                      membars=168,covered_pcs=len(mapping['pc_mapping'])))
    (metadata/'hardware_equivalent_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    with (builder.DEST/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','variant','CAModel','源码预测','全局预测','局部预测','全局精度%',
                         '模型内加速倍数','vector写','vector读','predicate写','predicate读','动态Membar'])
        for item in data['cases']:
            for variant in item['variants']:
                writer.writerow([item['title'],variant['label']]+[variant[key] for key in (
                    'camodel','source_cycles','global_cycles','local_cycles','accuracy','speedup','vector_store',
                    'vector_load','predicate_store','predicate_load','membars')])
    print('Added:',case['title'],len(case['variants']),'variants;',len(manifest),'evidence hashes')


if __name__=='__main__':main()
