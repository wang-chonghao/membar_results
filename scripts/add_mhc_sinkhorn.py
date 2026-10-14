"""汇总 MHC 多形状结果，不重建其他 case。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


CASE_ID = 'mhc_sinkhorn'
VARIANTS = (('ub','UB 版'),('resident','驻留 U1'),('resident_u2','驻留 U2'),
            ('resident_u4','驻留 U4'),('resident_u6','驻留 U6 · 自动 spill'),
            ('element_major','Element-major · 仅实测'))


def experiment_paths(repository, matrices, ident):
    if matrices == 8:
        if ident in ('resident_u2', 'resident_u4'):
            unroll = ident.removeprefix('resident_')
            cam = repository/f'results/mhc_resident_manual_{unroll}_20261010'/unroll
        else:
            cam = repository/'results/mhc_ascendc_alignment_20261009'/ident/'cce'
    else:
        cam = repository/f'results/mhc_generic_{matrices}_20261010'/ident/'cce'
    model_name = 'mhc_vfsim_membar_compare_36_20261010' if matrices == 36 else 'mhc_vfsim_membar_compare_20261010'
    model = repository/'results'/model_name
    return cam, model


def hardware_cce(repository,cam,model):
    sys.path.insert(0,str(repository))
    sys.path.insert(0,str(repository/'cce_code/membar_candidate_baselines'))
    import predict_mhc_log_aligned as decoder
    summary = json.loads((cam/'evidence/summary.json').read_text())
    calibrated = summary['matrices'] == 128
    if not calibrated:
        sys.path.insert(0,str(model/'model_snapshot'))
    from api.frontend.adapter_ir import AdapterInstruction,AdapterLoop,AdapterMembar
    rows = decoder.read_rows(cam/'camodel/core0.veccore0.instr_popped_log.dump')
    tree,by_pc,ordinary = decoder.loop_tree(rows)
    decoder.annotate_index_vectors(ordinary,summary)
    plt_remaining = {}
    for row in ordinary:
        if calibrated and row['op']=='RV_PLT':
            counter=decoder.register(row['detail'],'Sn')
            row['plt_remaining']=plt_remaining.get(counter,summary['matrices'] if calibrated else 4)
            plt_remaining[counter]=decoder.bits(row['detail'],'Sn',include_high=False)
    program,origins = decoder.build_program(tree,by_pc,summary,core_only=calibrated)
    registers = sorted((name for name,value in program.values.items() if value.storage=='Register'),key=lambda n:int(n[1:]))
    predicates = sorted((name for name,value in program.values.items() if value.storage=='PredicateRegister'),key=lambda n:int(n[1:]))
    memory = sorted(name for name,value in program.values.items() if value.storage=='UB')
    reg = lambda name: name.lower()
    def literal(value):
        number = struct.unpack('<f',struct.pack('<I',value & 0xffffffff))[0]
        text = format(number,'.9g')
        return text+('f' if '.' in text or 'e' in text else '.0f')
    mapping = []
    def translate(node):
        pc = int(node.attributes['compiled_pc'],16)
        row = by_pc[pc][0];detail = row['detail'];op = node.name
        inputs = [name for name in node.src if program.values[name].storage=='Register']
        mask = next((reg(name) for name in node.src if program.values[name].storage=='PredicateRegister'),None)
        dst = next((reg(name) for name in node.dst if program.values[name].storage in ('Register','PredicateRegister')),None)
        args = [reg(name) for name in inputs]
        access = node.memory_accesses[0] if node.memory_accesses else None
        lines = []
        if access:
            pointer = access.value_id;offset = access.offset
            if op == 'VLDAS':
                statement = f'vldas(u0, (__ubuf__ int32_t *)({pointer}+({offset})));'
            elif op == 'VLDUS':
                temp = f'ptr_{pc:x}'
                lines.append(f'__ubuf__ float *{temp}={pointer}+({offset});')
                statement = f'vldus((vector_s32 &){dst},u0,(__ubuf__ int32_t *&){temp},64,POST_UPDATE);'
            elif op == 'VGATHER2':
                statement = f'vgather2((vector_s32 &){dst},(__ubuf__ int32_t *)({pointer}+({offset})),(vector_u32 &){args[0]},{mask});'
            elif op == 'VSCATTER':
                statement = f'vscatter({args[0]},{pointer}+({offset}),(vector_u32 &){args[1]},{mask});'
            elif op == 'VLDS':
                if access.update_mode=='post_update':
                    temp = f'ptr_{pc:x}'
                    lines.append(f'__ubuf__ float *{temp}={pointer}+({offset});')
                    statement = f'vlds({dst},{temp},{node.attributes["compiled_post_delta"]//4},NORM,POST_UPDATE);'
                else:
                    dist = 'BRC_B32' if 'dist:BRC_B32' in detail else 'NORM'
                    statement = f'vlds({dst},{pointer},{offset},{dist});'
            elif op == 'VSTS':
                if 'dist:NORM_B8' in detail:
                    statement = f'vsts((vector_u8 &){args[0]},(__ubuf__ uint8_t *){pointer},({offset})*4,NORM_B8,{mask});'
                elif access.update_mode=='post_update':
                    temp = f'ptr_{pc:x}'
                    lines.append(f'__ubuf__ float *{temp}={pointer}+({offset});')
                    statement = f'vsts({args[0]},{temp},{node.attributes["compiled_post_delta"]//4},NORM_B32,{mask},POST_UPDATE);'
                else:
                    statement = f'vsts({args[0]},{pointer},{offset},NORM_B32,{mask});'
            else:
                raise ValueError(op)
        elif op.startswith('PSET_'):
            statement = f'{dst}=pset_{op.removeprefix("PSET_").lower()}(PAT_ALL);'
        elif op=='PLT':
            active = row['plt_remaining'] if calibrated else 4
            assert not calibrated or all(r['plt_remaining']==active for r in by_pc[pc])
            lines.append(f'uint32_t remaining_{pc:x}={active};')
            statement = f'{dst}=plt_b32(remaining_{pc:x},POST_UPDATE);'
        elif op=='VCI':
            assert decoder.bits(detail,'Sn')==0
            statement = f'vci((vector_s32 &){dst},0,std::integral_constant<Order,static_cast<Order>(0)>());'
        elif op=='VMOV':statement = f'vmov({dst},{args[0]});'
        elif op=='VDUP':
            if inputs:statement = f'vdup({dst},{args[0]},{mask},POS_LOWEST,MODE_ZEROING);'
            else:statement = f'vdup({dst},{literal(decoder.bits(detail,"Sn"))},{mask},MODE_ZEROING);'
        else:
            if op in ('VMULS','VADDS'):
                values = {decoder.bits(r['detail'],'Sm') for r in by_pc[pc]}
                assert len(values)==1
                scalar = next(iter(values))
                args.append(f'(uint32_t){scalar}' if node.form=='uint32' else literal(scalar))
            if node.form=='uint32':
                dst=f'(vector_u32 &){dst}'
                args=[f'(vector_u32 &){arg}' if arg.startswith('v') else arg for arg in args]
            statement = f'{op.lower()}('+','.join([dst,*args,mask,'MODE_ZEROING'])+');'
        lines.append(statement)
        mapping.append(dict(pc=hex(pc),opcode=row['op'],cce='\n'.join(lines),
            dynamic_count=len(by_pc[pc]),byte_addresses=[decoder.addresses(r,summary['matrices'],summary['variant'])[0] for r in by_pc[pc]] if access else None))
        return [f'// PC {hex(pc)}: {row["op"]}',*lines]
    def emit(nodes,depth):
        lines = []
        pad = '    '*depth
        for node in nodes:
            if isinstance(node,AdapterLoop):
                lines.append(pad+f'for (uint16_t {node.induction_variable}=0;{node.induction_variable}<{node.count};++{node.induction_variable}) {{')
                lines.extend(emit(node.body,depth+1));lines.append(pad+'}')
            elif isinstance(node,AdapterMembar):
                # Barrier nodes do not carry attributes; match their static PC by traversal order.
                index = len(mapping)
                ordered = sorted((pc,r[0]) for pc,r in by_pc.items() if r[0]['op']!='RV_VLOOPv2')
                pc,row = ordered[index]
                assert row['op']=='RV_SMEM_BAR'
                statement = f'mem_bar({node.type});'
                mapping.append(dict(pc=hex(pc),opcode=row['op'],cce=statement,dynamic_count=len(by_pc[pc])))
                lines.extend([pad+f'// PC {hex(pc)}: RV_SMEM_BAR',pad+statement])
            else:
                assert isinstance(node,AdapterInstruction)
                lines.extend(pad+line for line in translate(node))
        return lines
    body = emit(program.context,2)
    counts = Counter()
    for item in mapping:counts[item['opcode']]+=item['dynamic_count']
    assert counts==Counter(row['op'] for row in ordinary)
    audit_path = model/'comparison.json' if calibrated else model/'m36/resident_u6/comparison.json'
    audit = json.loads(audit_path.read_text())
    assert audit['audits']['global']['register_producer_graph_equal'] and audit['audits']['local']['register_producer_graph_equal']
    description = '128 矩阵 element-major U2' if calibrated else '36 矩阵驻留 U6'
    spill_note = '234 条 spill store、234 条 spill load 和总计 468 条 mem_bar' if calibrated else '457 条 spill store、570 条 spill load 和总计 1064 条 mem_bar'
    lines = [f'// MHC {description}：实际底层硬件 VF 指令流的等效 CCE，包含自动 spill。',
        '// 按同目录 CAModel 日志的 PC、寄存器读写、循环和实际地址还原。',
        f'// 包含 {spill_note}。',
        '// 只表达 VF 计算、访存与同步；标量地址效果已折入，不表达标量时序或 GM wrapper。',
        '// 表中预测使用对应日志重建的 CanonicalVfInfo，不是重新解析或编译这份说明用 CCE。',
        '// 再次编译可能重新分配寄存器，不能宣称新二进制已与原机器码对齐。',
        '// UB 对象的原始起点：'+', '.join(f'{name}=0x{origins[name]:x}' for name in memory),
        '#include <type_traits>',
        'inline [aicore] void hardware_equivalent_vf('+', '.join('__ubuf__ float *'+name for name in memory)+') {',
        '    __VEC_SCOPE__ {',
        '        vector_f32 '+','.join(reg(name) for name in registers)+';',
        '        vector_bool '+','.join(reg(name) for name in predicates)+';',
        '        vector_align u0;',*body,'    }','}']
    code='\n'.join(lines)+'\n'
    return code,dict(modeled_pc_coverage_complete=True,dynamic_instruction_counts=dict(counts),
        pc_mapping=mapping,register_dependency_graph_equal=True,
        dynamic_load_store_addresses_checked=audit['audits']['local']['ordering_32b_checks'] if calibrated else audit['audits']['local']['memory_range_checks'],
        sha256=hashlib.sha256(code.encode()).hexdigest(),
        scope='VF vector/predicate/align/memory/membar; scalar address effects folded, scalar timing excluded')


def collect_calibrated_variants(builder,repository):
    archive=repository/'results/mhc_element_major_128_20261012'
    calibration=repository/'results/indexed_memory_calibration_20261012'
    local=archive/'local_dependency'
    variants=[]
    for ident,unroll in (('element_major',1),('element_major_u2',2)):
        cam=archive/ident/'cce';prediction=calibration/'prediction'/ident
        summary=builder.read(cam/'evidence/summary.json')
        result=builder.read(prediction/'comparison.json')
        ab=builder.read(local/'comparison.json') if unroll==2 else None
        assert summary['validation']['passed'] and summary['matrices']==128
        assert result['native_cycles']==result['python_cycles']
        assert result['dynamic_pc_order_equal'] and result['register_producer_graph_equal']
        if ab:assert ab['global_cycles']==result['python_cycles']
        evidence=[cam/'evidence'/name for name in ('summary.json','validation.json','experiment.json','layout.json','commands.json','spill_summary.json')]
        evidence.extend([prediction/'comparison.json',prediction/'canonical_input.json',calibration/'measurements.csv',calibration/'结果说明.txt'])
        if ab:evidence.extend(local/name for name in ('comparison.json','config_hashes.json','dependency_edges.json','memory_records.json','local_canonical.json'))
        note='128 个 4x4 FP32 矩阵，20 轮，保留 RMS 输入归一化；2048 个有效输出及全部保护区通过校验。全局预测 Python/C++ 完全一致。'
        if ab:note+='保留 234 次向量保存、234 次重载及 115 次 VMOV；原硬件含 468 条 mem_bar。局部预测采用 32B RAW/WAR/WAW，地址解析 550/550，依赖核查通过；另有 6 对标量栈访存，标量时序未建模。'
        else:note+='无向量 spill、无 mem_bar；局部依赖未运行，不用全局值代填。'
        ident_out=f'm128_{ident}'
        variant=builder.make_variant(CASE_ID,ident_out,f'128 矩阵 · Element-major U{unroll}'+(' · 自动 spill' if ab else ''),
            cam/'source/kernel.cce',cam/'camodel',summary['camodel'],
            global_cycles=result['python_cycles'],local_cycles=ab['local_cycles'] if ab else None,
            global_model=prediction/'python',local_model=local/'local' if ab else None,
            spill=(234,234) if ab else (0,0),membars=result['hardware_membars'],note=note,evidence=evidence)
        variant.update(matrix_count=128,vector_arch_registers=32 if ab else 23,ub_bytes=summary['layout']['ub_bytes'])
        destination=builder.DEST/CASE_ID/ident_out
        for name in ('host.cpp','reference/mhc_pre_sinkhorn_base_arch35.h'):
            builder.copy_file(cam/'source'/name,destination/'source'/name)
        for name in ('input.bin','golden.bin','output.bin'):
            builder.copy_file(cam/'data'/name,destination/'data'/name)
        for mode,folder in [('global',prediction/'python')]+([('local',local/'local')] if ab else []):
            for name in ('start_by_cycle.json','done_by_cycle.json','model_warnings.json','membar_history.json'):
                builder.copy_file(folder/name,destination/'model'/mode/name)
        for name in ('isa.json','forwarding.json','InitiationInterval.json','instruction_catalog.json','uarch.json'):
            builder.copy_file(calibration/'configs_used'/name,destination/'configs_used'/name)
        if ab:
            code,mapping=hardware_cce(repository,cam,local)
            file=destination/'source/hardware_equivalent.cce'
            file.write_text(code,encoding='utf-8')
            path=destination/'source/hardware_pc_mapping.json'
            path.write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
            builder.copy_file(file,destination/'source/compiled_pc_replay.cce')
            variant['hardware_equivalent']=dict(path=str(file.relative_to(builder.DEST)),mapping=str(path.relative_to(builder.DEST)),
                pc_coverage=True,camodel_cycles=summary['camodel'],prediction_used=False,
                prediction_note='预测使用对应日志重建的 CanonicalVfInfo，已核对动态 PC、寄存器依赖和 32B 访存次序；这份等效 CCE 未重新编译验证。局部结果仅为 Python 实验预测。')
            variant['sources'].insert(0,dict(label='实际硬件 VF · 等效 CCE（含自动 spill）',path=str(file.relative_to(builder.DEST))))
            variant['sources'][1]['label']='编译前实验源码'
            variant['evidence'].append(str(path.relative_to(builder.DEST)))
        (destination/'result.json').write_text(json.dumps(variant,ensure_ascii=False,indent=2),encoding='utf-8')
        variants.append(variant)
    optimized=repository/'results/mhc_element_major_128_optimization_20261012/staged_early_u1'
    if (optimized/'prediction/comparison.json').is_file():
        variants.append(collect_optimized_variant(builder,optimized))
    return variants


def collect_optimized_variant(builder,archive):
    cam=archive/'cce';prediction=archive/'prediction';local=archive/'local_dependency_prediction'
    summary=builder.read(cam/'evidence/summary.json')
    optimization=builder.read(cam/'evidence/optimization.json')
    result=builder.read(prediction/'comparison.json');ab=builder.read(local/'comparison.json')
    assert summary['validation']['passed'] and summary['matrices']==128
    assert optimization['outputs_byte_identical_to_u1']
    assert optimization['vector_spill_stores']==optimization['vector_spill_loads']==0
    assert result['python_cycles']==result['native_cycles']==ab['global_cycles']
    assert result['camodel']==ab['camodel']==summary['camodel']
    assert result['dynamic_pc_order_equal'] and result['register_producer_graph_equal']
    assert result['native_audit']['register_producer_graph_equal']
    assert all(audit['dynamic_opcodes_and_pcs_equal'] and audit['register_producer_graph_equal'] for audit in ab['audits'].values())
    hashes=builder.read(prediction/'config_hashes.json')
    assert builder.read(local/'config_hashes.json')==hashes
    note='128 个 4x4 FP32 矩阵，初始化 1 轮＋后续 4+15 轮，交错四组归一化链；2048 个有效输出及保护区通过，输出与原 U1 逐字节一致。实际汇编使用 27 个向量架构寄存器，无自动向量/谓词 spill，仅 1 条 mem_bar。显式保存 32 次、读回 32 次，8KiB scratch 复用原 result 空间，公共 UB 布局仍为 30976B（不含 VF 栈）。预测用编译后日志重建 CanonicalVfInfo，2946 条指令的 PC 与寄存器依赖核查通过，Python/C++ 全局结果一致。局部为 Python 32B RAW/WAR/WAW 实验，162 条访存全部解析，移除 1 条屏障，只少 37 cycle；不是优化硬件实测。标量时序未建模，GSU 与 cap7 参数未改。'
    evidence=[cam/'evidence'/name for name in ('summary.json','validation.json','experiment.json','optimization.json','spill_summary.json','layout.json','commands.json')]
    evidence += [prediction/name for name in ('comparison.json','canonical_input.json','config_hashes.json','native_run.log')]
    evidence += [local/name for name in ('comparison.json','config_hashes.json','dependency_edges.json','memory_records.json','local_canonical.json')]
    ident='m128_element_major_split4_15'
    variant=builder.make_variant(CASE_ID,ident,'128 矩阵 · Element-major 4+15 切分＋重排',
        cam/'source/kernel.cce',cam/'camodel',summary['camodel'],global_cycles=result['python_cycles'],
        local_cycles=ab['local_cycles'],global_model=prediction/'python',local_model=local/'local',
        membars=result['hardware_membars'],note=note,evidence=evidence)
    variant.update(matrix_count=128,vector_arch_registers=optimization['vector_arch_registers'],
        ub_bytes=summary['layout']['ub_bytes'],explicit_scratch_bytes=8192,explicit_scratch_loads=32,explicit_scratch_stores=32)
    variant['sources'][0]['label']='实际编译 CCE · 4+15 切分重排'
    destination=builder.DEST/CASE_ID/ident
    for name in ('host.cpp','reference/mhc_pre_sinkhorn_base_arch35.h'):
        builder.copy_file(cam/'source'/name,destination/'source'/name)
    for name in ('input.bin','golden.bin','output.bin'):
        builder.copy_file(cam/'data'/name,destination/'data'/name)
    for mode,folder in (('global',prediction/'python'),('native',prediction/'native'),('local',local/'local')):
        for name in ('start_by_cycle.json','done_by_cycle.json','model_warnings.json','membar_history.json'):
            builder.copy_file(folder/name,destination/'model'/mode/name)
    for name,digest in hashes.items():
        assert hashlib.sha256((prediction/'configs_used'/name).read_bytes()).hexdigest()==digest
        builder.copy_file(prediction/'configs_used'/name,destination/'configs_used'/name)
    (destination/'result.json').write_text(json.dumps(variant,ensure_ascii=False,indent=2),encoding='utf-8')
    return variant


def refresh_camodel_curves(builder,case):
    for variant in case['variants']:
        duration,curve=builder.camodel_curve(builder.DEST/CASE_ID/variant['id']/'camodel')
        assert duration==variant['camodel']
        variant['series']=[curve if series['key']=='camodel' else series for series in variant['series']]


def apply_calibrated_element_major(builder,repository,case):
    prediction_root=repository/'results/mhc_element_major_calibrated_shapes_verified_20261012'
    calibration=repository/'results/indexed_memory_calibration_20261012'
    if not prediction_root.exists():return
    for matrices in (8,36,64):
        prediction=prediction_root/f'm{matrices}'
        result=builder.read(prediction/'comparison.json')
        hashes=builder.read(prediction/'config_hashes.json')
        assert all(hashlib.sha256((calibration/'configs_used'/name).read_bytes()).hexdigest()==checksum for name,checksum in hashes.items())
        assert result['python_cycles']==result['native_cycles']
        assert result['dynamic_pc_order_equal'] and result['register_producer_graph_equal']
        assert result['native_audit']['dynamic_pc_order_equal'] and result['native_audit']['register_producer_graph_equal']
        variant=next(v for v in case['variants'] if v['matrix_count']==matrices and 'element_major' in v['id'])
        assert result['camodel']==variant['camodel'] and result['hardware_membars']==0
        variant.update(label=f'{matrices} 矩阵 · Element-major U1 · 校准预测',
            global_cycles=result['python_cycles'],local_cycles=None,accuracy=round(result['accuracy_percent'],4),speedup=None,
            series=[s for s in variant['series'] if s['key']=='camodel']+[builder.model_curve(prediction/'python','global')])
        variant.pop('prediction_status_note',None)
        variant['note']=f'{matrices} 个 4x4 FP32 矩阵，20 轮，保留 RMS 输入归一化；{matrices*16} 个有效输出及保护区校验通过。新校准参数下 Python/C++ 全局结果一致，动态指令/PC 与寄存器 producer 图均核查通过；无 mem_bar，未跑局部依赖预测。标量时序未建模，部分 forwarding/II 仍使用 fallback，详见参数 warning。'
        if matrices<64:variant['note']+='采用完整 64-lane 稀疏 GSU 固定参数，尚未按本例有效 lane 数缩短时序。'
        if matrices==64:variant['note']+='精度按当前 CCE 的 CAModel 2760 cycle 比较；原 AscendC 为 2797，不混用两个基线。'
        destination=builder.DEST/CASE_ID/variant['id']
        evidence=[]
        for name in ('comparison.json','canonical_input.json','config_hashes.json','native_run.log'):
            evidence.append(builder.copy_file(prediction/name,destination/'evidence/calibrated_20261012'/name))
        for name in ('measurements.csv','结果说明.txt'):
            evidence.append(builder.copy_file(calibration/name,destination/'evidence/calibrated_20261012'/name))
        variant['evidence']=[p for p in variant['evidence'] if '/calibrated_20261012/' not in p]+evidence
        for mode,folder in (('global',prediction/'python'),('native',prediction/'native')):
            for name in ('start_by_cycle.json','done_by_cycle.json','model_warnings.json','membar_history.json'):
                builder.copy_file(folder/name,destination/'model'/mode/name)
        local=prediction/'local_dependency'
        if (local/'comparison.json').is_file():
            ab=builder.read(local/'comparison.json')
            assert ab['global_cycles']==result['python_cycles'] and ab['camodel']==variant['camodel']
            assert ab['matrices']==matrices and ab['removed_dynamic_membars']==0 and ab['fallback_memory_instructions']==0
            assert builder.read(local/'config_hashes.json')==hashes
            assert all(audit['dynamic_opcodes_and_pcs_equal'] and audit['register_producer_graph_equal'] for audit in ab['audits'].values())
            variant.update(local_cycles=ab['local_cycles'],speedup=round(ab['modeled_speedup'],4))
            variant['series'].append(builder.model_curve(local/'local','local'))
            variant['note']=variant['note'].replace('无 mem_bar，未跑局部依赖预测。',
                f'无 mem_bar；已跑 Python 32B RAW/WAR/WAW 局部实验，预测 {ab["local_cycles"]} cycle，模型内加速 {ab["modeled_speedup"]:.2f}x，地址解析 {ab["resolved_memory_instructions"]}/{ab["resolved_memory_instructions"]}，核查通过。C++ 仅核对全局预测，局部结果不是优化硬件的实测。')
            prefix='/local_dependency_20261012/'
            variant['evidence']=[path for path in variant['evidence'] if prefix not in path]
            for name in ('comparison.json','config_hashes.json','dependency_edges.json','memory_records.json','global_canonical.json','local_canonical.json'):
                variant['evidence'].append(builder.copy_file(local/name,destination/'evidence/local_dependency_20261012'/name))
            for name in ('start_by_cycle.json','done_by_cycle.json','model_warnings.json','membar_history.json'):
                builder.copy_file(local/'local'/name,destination/'model/local'/name)
        for name in hashes:builder.copy_file(calibration/'configs_used'/name,destination/'configs_used'/name)
        (destination/'result.json').write_text(json.dumps(variant,ensure_ascii=False,indent=2),encoding='utf-8')


def describe_case(case):
    case.update(id=CASE_ID,subtitle='8 / 36 / 64 / 128 矩阵 · UB / 驻留 / element-major',
        shape='提取 VF：8、36、64、128 个 4x4 FP32 矩阵，hc_mult=4，20 轮 Sinkhorn；不是完整多核 MHC 算子。',
        headline='UB 版在 8 / 36 / 64 矩阵下模型内加速约 2.5x；128 矩阵 element-major U2 局部依赖预测为 4145 cycle，加速 2.68x。')
    calibrated=all(v['global_cycles'] is not None for v in case['variants'] if v['matrix_count'] in (8,36,64) and 'element_major' in v['id'])
    notes=[note for note in case['notes'] if not note.startswith(('128 矩阵','新校准 Element-major'))]
    if calibrated:
        notes=[note for note in notes if not note.startswith(('8/36/64 矩阵 Element-major','Element-major 额外保留'))]
    frozen_scope='8/36/64 矩阵的 UB / 驻留版' if calibrated else '8/36/64 矩阵'
    frozen_note=frozen_scope+'使用此前冻结的 Python 实验模型，保持 cap7、lookahead=8、min_count=1，不修改正式预测核心。'
    notes=[re.sub(r'^\d+ 个 CAModel',f'{len(case["variants"])} 个 CAModel',note)
           if '冻结的 Python 实验模型' not in note else frozen_note for note in notes]
    notes.extend([
        '128 矩阵 U1/U2 使用新校准参数：VGATHER2.b32 latency=35 / GSU interval=16，VSCATTER.b32 latency=43 / interval=32，来自全 64 lane 稀疏访存测试；并非通用 bank/coalescing 模型。历史 UB / 驻留版保留冻结结果，不能和新校准预测当作统一参数版本比较。',
        '128 矩阵 U1：CAModel 4950、全局预测 5082，精度 97.33%；U2：11446、11118，精度 97.13%。U2 局部预测 4145，同流节省 6973 cycle（62.72%），不是优化硬件的 CAModel 实测。',
        '128 矩阵 U2 按 32B RAW/WAR/WAW 保留编译器 spill 与共享 GSU 约束，550 条访存全部解析；核对 1872 / 3088 / 3520 条约束。局部路径仅 Python 实验，未声称 C++ 同步支持。',
        '128 矩阵两版本的输入、golden 和输出逐字节相同；公共 UB 布局均为 30976B（30.25KiB），不含另行使用的 VF 栈。U1 未跑局部预测；IPC 默认仍为 8 矩阵 UB，使用 100-cycle 窗口。'])
    if any(v['id']=='m128_element_major_split4_15' for v in case['variants']):
        notes.extend([
            '128 矩阵新增 4+15 切分＋重排：CAModel 4600、全局预测 4364（精度 94.87%）、局部预测 4327 cycle。无自动 spill，仅 1 条 mem_bar；局部只少 37 cycle，模型内加速 1.009x。',
            '128 矩阵切分版使用 8KiB 显式 scratch 和 32 次保存/32 次读回，不是没有中间 UB 访存。与原 U1 的实测比较为 4950/4600≈1.076x；这是实际代码优化，与 U2 的假设性局部模型内 2.68x 收益不是同一种比较。',
            '128 矩阵的 IPC 排除 Gather/Scatter 访存，与全页“向量计算完成”的统计口径一致；只修正事件统计，不改变任何总周期或模型参数。'])
    if calibrated:
        notes.extend([
            '新校准 Element-major：8 / 36 / 64 矩阵的 CAModel 为 2146 / 2425 / 2760 cycle，Python/C++ 预测均为 2586，精度 79.50% / 93.36% / 93.70%。三种输入均通过 1442 条动态指令的 PC、寄存器依赖核查，使用和 128 矩阵相同的配置校验值。',
            '新校准 Element-major 目前不按有效 lane 数调整 GSU 时序。8 / 36 / 64 矩阵各为一个 SIMD batch，建模指令序列与 form 相同，所以预测相同；参数覆盖不等于 partial-mask 时序已经准确。尾部实测 Gather latency 为 22 / 28 / 35、Scatter 为 16 / 29 / 43，而正式配置仍统一为 35 / 43。',
            '新校准 Element-major 无 mem_bar、无向量 spill；未运行的局部结果留空，不用全局值代填。数值校验沿用对应 CAModel 的有效输出及保护区结果；本次不重编译或重跑 CAModel。部分 forwarding / II 仍使用默认参数，保留全部 warning。'])
        for variant in case['variants']:
            if variant['matrix_count'] in (8,36,64) and 'element_major' in variant['id'] and variant['local_cycles'] is not None:
                notes.append(f'新校准 Element-major {variant["matrix_count"]} 矩阵已补跑 Python 32B 局部地址依赖实验：全局 {variant["global_cycles"]}、局部 {variant["local_cycles"]} cycle，模型内加速 {variant["speedup"]:.2f}x。原流没有 mem_bar，不能认为它会获得屏障优化收益；保留真实访存依赖、当前 GSU 参数和独立核查结果。')
    case['notes']=notes
    return case


def collect_case(builder,repository):
    variants=[]
    experiments = [(matrices, ident, label) for matrices in (8, 36, 64)
                   for ident, label in VARIANTS if matrices == 36 or ident != 'resident_u6']
    for matrices,ident,label in experiments:
        cam,model=experiment_paths(repository,matrices,ident)
        summary=builder.read(cam/'evidence/summary.json')
        # 保留既有 36 矩阵链接，新形状使用显式前缀。
        variant_id=ident if matrices==36 else f'm{matrices}_{ident}'
        label=f'{matrices} 矩阵 · {label}'
        popped=(cam/'camodel/core0.veccore0.instr_popped_log.dump').read_text()
        vector_registers=set(re.findall(r'V[dnm]\[(\d+)\]',popped))
        prediction=None if ident=='element_major' else model/f'm{matrices}'/ident
        result=builder.read(prediction/'comparison.json') if prediction else None
        assert summary['validation']['passed'] and summary['matrices']==matrices
        evidence=[cam/'evidence'/name for name in ('summary.json','validation.json','experiment.json','layout.json','commands.json')
                  if (cam/'evidence'/name).is_file()]
        if result:evidence.extend([prediction/'comparison.json',prediction/'canonical_input.json',prediction/'compiled_loop_tree.json'])
        note=f'{matrices} 个 4x4 FP32 矩阵，20 轮；全部 {matrices*16} 个有效输出及保护区校验通过。'
        if ident=='element_major':
            note+='保留 RMS 输入归一化；原实现无 membar。'
            if matrices==36:note+='按要求未运行 VfSim。'
            else:note+='此前默认参数试跑未通过地址顺序核查且 Gather/Scatter 未校准，不作为有效预测。'
            if matrices==64:note+='原 AscendC 为 2797 cycle，对应 CCE 为 2760 cycle，并非周期完全对齐。'
        else:note+='全局/局部使用相同编译后指令与 SSA 数据流；地址依赖核对通过，标量 ASU 时序未建模。'
        if ident=='resident_u6':note+='保留 457 条 spill store、570 条 spill load 和 1064 条 membar；另有 4 次标量参数栈读取，不据此认定标量寄存器溢出。'
        variant=builder.make_variant(CASE_ID,variant_id,label,cam/'source/kernel.cce',cam/'camodel',summary['camodel'],
            global_cycles=result['global_cycles'] if result else None,local_cycles=result['local_cycles'] if result else None,
            global_model=prediction/'global' if prediction else None,local_model=prediction/'local' if prediction else None,
            spill=(457,570) if ident=='resident_u6' else (0,0),membars=summary['hardware_op_counts'].get('RV_SMEM_BAR',0),note=note,evidence=evidence)
        variant['matrix_count']=matrices;variant['vector_arch_registers']=len(vector_registers)
        if ident=='element_major' and matrices!=36:
            variant['prediction_status_note']='未提供有效预测：默认参数试跑未通过核查'
        destination=builder.DEST/CASE_ID/variant_id
        source_files=['host.cpp']
        reference='reference/mhc_pre_sinkhorn_base_arch35.h'
        if (cam/'source'/reference).is_file():source_files.append(reference)
        for name in source_files:
            builder.copy_file(cam/'source'/name,destination/'source'/name)
        for name in ('input.bin','golden.bin','output.bin'):
            builder.copy_file(cam/'data'/name,destination/'data'/name)
        if prediction:
            for mode in ('global','local'):
                for name in ('start_by_cycle.json','done_by_cycle.json','model_warnings.json','membar_history.json'):
                    builder.copy_file(prediction/mode/name,destination/'model'/mode/name)
        if ident=='resident_u6':
            code,mapping=hardware_cce(repository,cam,model)
            file=destination/'source/hardware_equivalent.cce';file.write_text(code,encoding='utf-8')
            path=destination/'source/hardware_pc_mapping.json';path.write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
            builder.copy_file(file,destination/'source/compiled_pc_replay.cce')
            for item in (file,path):builder.MANIFEST.append(dict(path=str(item.relative_to(builder.DEST)),sha256=hashlib.sha256(item.read_bytes()).hexdigest()))
            variant['hardware_equivalent']=dict(path=str(file.relative_to(builder.DEST)),mapping=str(path.relative_to(builder.DEST)),
                pc_coverage=True,camodel_cycles=summary['camodel'],prediction_used=False,
                prediction_note='表中预测使用该实际日志重建的 CanonicalVfInfo，已核对 PC、寄存器依赖和全部访存地址；这份等效 CCE 未重新编译验证。')
            variant['sources'].insert(0,dict(label='实际硬件 VF · 等效 CCE（含自动 spill）',path=str(file.relative_to(builder.DEST))))
            variant['sources'][1]['label']='编译前实验源码'
            variant['evidence'].append(str(path.relative_to(builder.DEST)))
        (destination/'result.json').write_text(json.dumps(variant,ensure_ascii=False,indent=2),encoding='utf-8')
        variants.append(variant)
    case=dict(id=CASE_ID,title='MHC Sinkhorn',subtitle='8 / 36 / 64 矩阵 · UB / 驻留 / element-major',
        plot_window=100,plot_ipc_method='向量计算指令完成事件，100-cycle trailing window，各曲线以首条计算完成为 cycle 0；不含 LSU、Membar、PSET、scalar。',
        shape='提取 VF：8、36、64 个 4x4 FP32 矩阵，hc_mult=4，20 轮 Sinkhorn；不是完整多核 MHC 算子。',
        default='m8_ub',headline='UB 版在 8 / 36 / 64 矩阵下的局部依赖模型内收益分别为 2.54x / 2.51x / 2.50x。',
        notes=['16 个 CAModel 版本均通过数值及保护区校验；每个形状内的逐行实现输出逐字节一致。',
            '8 矩阵为早期固定形状实验，36/64 为通用矩阵数实现；不同形状的输入样本不完全相同，比较实现时应保持同形状。',
            '驻留版 U4 在 8 / 36 / 64 矩阵下的 CAModel 为 5763 / 25640 / 45217 cycle。只有 36 矩阵补测 U6，实测 26288，包含自动 spill。',
            '36 矩阵局部预测中 U6 比 U4 少约 7.8% 周期，但 U4/U6 全局预测偏高约 11.6%/10.3%，不能当作优化硬件的实测结论。',
            '使用此前冻结的 Python 实验模型，保持 cap7、lookahead=8、min_count=1，不修改正式预测核心。',
            '快照按字节重叠建立 RAW/WAR；这些逐行/栈访问额外通过 32B RAW/WAR/WAW 检查，不代表已通用支持 32B STORE/STORE 依赖。',
            'Element-major 额外保留 RMS 归一化，沿矩阵数使用 SIMD lane；仅列 CAModel 2146 / 2425 / 2760。36 矩阵未跑 VfSim，8/64 的未校准试跑不计入有效预测。',
            '64 矩阵 element-major 原 AscendC 为 2797 cycle，归档的对应 CCE 为 2760；数值通过，但周期并非完全对齐。未纳入编译失败的驻留 U8。'],
        variants=variants,metrics_columns=[dict(key='matrix_count',label='矩阵数'),dict(key='vector_arch_registers',label='向量架构寄存器')])
    case['variants'].extend(collect_calibrated_variants(builder,repository))
    apply_calibrated_element_major(builder,repository,case)
    refresh_camodel_curves(builder,case)
    describe_case(case)
    (builder.DEST/CASE_ID/'summary.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    return case


def refresh_report_manifest(builder,manifest):
    metadata=builder.DEST/'metadata'
    changed=[metadata/name for name in ('data.json','data.js','hardware_equivalent_index.json','mhc_browser_verification.json')]
    changed.extend(builder.DEST/name for name in ('comparison.csv','使用说明.txt','assets/report.js'))
    changed.extend(builder.DEST/'scripts'/name for name in ('add_mhc_sinkhorn.py','build_archive.py','test_report_metrics.py'))
    changed.extend([builder.DEST/CASE_ID/'summary.json',builder.DEST/CASE_ID/'结果说明.txt'])
    changed.extend((builder.DEST/CASE_ID).glob('*/result.json'))
    changed.extend((builder.DEST/CASE_ID).glob('*/figures/ipc.png'))
    changed.extend((builder.DEST/CASE_ID).glob('m128_*/source/hardware*'))
    manifest.pop('scripts/add_mhc_sinkhorn_36.py',None)
    for path in changed:
        if path.is_file():
            relative=str(path.relative_to(builder.DEST))
            manifest[relative]=dict(path=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (metadata/'manifest.json').write_text(json.dumps(list(manifest.values()),ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vfsim-root',type=Path,required=True);args=parser.parse_args()
    import build_archive as builder
    builder.ROOT=args.vfsim_root;builder.DEST=args.vfsim_root/'results/membar_results';builder.MANIFEST=[]
    metadata=builder.DEST/'metadata';data=builder.read(metadata/'data.json');manifest=builder.read(metadata/'manifest.json')
    old='mhc_sinkhorn_36'
    def migrate(value):
        if isinstance(value,dict):return {key:migrate(item) for key,item in value.items()}
        if isinstance(value,list):return [migrate(item) for item in value]
        if isinstance(value,str):
            if value==old:return CASE_ID
            if value.startswith(old+'/'):return CASE_ID+value[len(old):]
            prefix='results/membar_results/'+old+'/'
            if value.startswith(prefix):return value.replace(prefix,'results/membar_results/'+CASE_ID+'/',1)
        return value
    if (builder.DEST/old).exists():
        assert not (builder.DEST/CASE_ID).exists(), '旧目录和新目录同时存在，需人工确认'
        (builder.DEST/old).rename(builder.DEST/CASE_ID)
    data=migrate(data);manifest=migrate(manifest)
    case=next(item for item in data['cases'] if item['id']==CASE_ID)
    case['variants']=[v for v in case['variants'] if v.get('matrix_count')!=128]+collect_calibrated_variants(builder,args.vfsim_root)
    apply_calibrated_element_major(builder,args.vfsim_root,case)
    refresh_camodel_curves(builder,case)
    describe_case(case)
    for variant in case['variants']:
        file=builder.DEST/CASE_ID/variant['id']/'result.json'
        file.write_text(json.dumps(variant,ensure_ascii=False,indent=2),encoding='utf-8')
    (builder.DEST/CASE_ID/'summary.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    data['generated']='2026-10-12'
    (metadata/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    (metadata/'data.js').write_text('window.MEMBAR_REPORT='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    manifest={entry['path']:entry for entry in manifest}
    manifest.update({entry['path']:entry for entry in builder.MANIFEST})
    index=migrate(builder.read(metadata/'hardware_equivalent_index.json'))
    index=[entry for entry in index if not entry['file'].startswith(CASE_ID+'/')]
    for spill in (v for v in case['variants'] if v['vector_store']):
        mapping=builder.read(builder.DEST/spill['hardware_equivalent']['mapping'])
        index.append(dict(case=case['title'],variant=spill['label'],file=spill['hardware_equivalent']['path'],membars=spill['membars'],covered_pcs=len(mapping['pc_mapping'])))
    (metadata/'hardware_equivalent_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    with (builder.DEST/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['case','variant','CAModel','源码预测','全局预测','局部预测','全局精度%','模型内加速倍数','vector写','vector读','predicate写','predicate读','动态Membar'])
        for item in data['cases']:
            for variant in item['variants']:
                writer.writerow([item['title'],variant['label']]+[variant[key] for key in ('camodel','source_cycles','global_cycles','local_cycles','accuracy','speedup','vector_store','vector_load','predicate_store','predicate_load','membars')])
    # 仅刷新本次生成的元数据和新增材料；旧日志的原始校验值保留。
    refresh_report_manifest(builder,manifest)
    print('Updated',case['title'],len(case['variants']),'variants;',len(manifest),'evidence hashes')


if __name__=='__main__':main()
