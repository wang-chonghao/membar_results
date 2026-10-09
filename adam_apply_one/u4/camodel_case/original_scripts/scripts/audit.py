import json
from pathlib import Path
import re
import sys
from collections import Counter

ROOT = Path('/mnt/e/vfsimulator')
OUT = ROOT/'results/ub_address_dependency_experiment/adam_apply_one_fused_i48_unroll'
def memory_accesses(path):
    records = {}
    current = None
    header = re.compile(r'\[info\]\s+\[(\d+)\]\s+(RV_\w+), pc: (0x[0-9a-fA-F]+), id: (\d+)')
    for line in path.read_text(errors='replace').splitlines():
        match = header.search(line)
        if match:
            cy, op, pc, iid = match.groups()
            current = records.setdefault((op, pc, int(iid)), dict(op=op, pc=pc, inst_id=int(iid),
                                        cycle=int(cy), min_address=None, max_address=None))
        elif line.startswith('[info] ['):
            current = None
        if current is not None:
            addresses = [int(v, 16) for v in re.findall(r'Address ([0-9a-fA-F]+)', line)]
            if addresses:
                current['min_address'] = min(addresses + ([] if current['min_address'] is None else [current['min_address']]))
                current['max_address'] = max(addresses + ([] if current['max_address'] is None else [current['max_address']]))
    return [record for record in records.values() if record['min_address'] is not None]
results=[]
for file in sorted(OUT.glob('u*/summary.json'), key=lambda p:int(p.parent.name[1:])):
    info=json.loads(file.read_text())
    info['unroll_applied'] = info['compile_returncode']==0 and 'loop not unrolled' not in (file.parent/'camodel/compile.log').read_text()
    if info.get('camodel_returncode')==0:
        accesses=[]
        for name in ('core0.veccore0.ub.rd_log.dump','core0.veccore0.ub.wr_log.dump'):
            accesses += memory_accesses(file.parent/'camodel'/name)
        accesses=[a for a in accesses if a['op'].startswith(('RV_VLD','RV_VST'))]
        spill=[a for a in accesses if not (0 <= a['min_address'] <= a['max_address'] < 0x150c0)]
        info['spill_loads_by_address']=sum(a['op'].startswith('RV_VLD') for a in spill)
        info['spill_stores_by_address']=sum(a['op'].startswith('RV_VST') for a in spill)
        info['spill_min_address']=min((a['min_address'] for a in spill),default=None)
        info['spill_max_address']=max((a['max_address'] for a in spill),default=None)
        assert info['extra_loads']==info['spill_loads_by_address'],info
        assert info['extra_stores']==info['spill_stores_by_address'],info
        for name,count in {'VMUL':288,'VADD':144,'VDIV':48,'VSQRT':48,'VSUB':48}.items():
            assert info['opcode_counts'][name]==count,info
        (file.parent/'spill_ub_accesses.json').write_text(json.dumps(spill,indent=2)+'\n')
        log=(file.parent/'camodel/core0.veccore0.instr_log.dump').read_text()
        pcs={}
        for line in log.splitlines():
            m=re.search(r'\(PC: (0x[\da-fA-F]+)\).*\bRV_\w+',line)
            if m: pcs.setdefault(int(m[1],16),line)
        (file.parent/'compiled_pc_order.txt').write_text('\n'.join(pcs[k] for k in sorted(pcs))+'\n')
        file.write_text(json.dumps(info,indent=2)+'\n')
    model_dir=file.parent/'vfsim'
    if info.get('vfsim_cycles') is not None:
        done=[json.loads(line) for line in (model_dir/'done_by_cycle.json').read_text().splitlines() if line.strip()]
        model_counts=Counter(d['op'] for d in done)
        assert model_counts == {'VLDS':198,'VSTS':144,'VMUL':288,'VADD':144,'VDIV':48,'VSQRT':48,'VSUB':48}, model_counts
        assert len({d['inst_id'] for d in done}) == 918
        info['vfsim_dynamic_opcode_counts']=dict(model_counts)
        warnings=json.loads((model_dir/'model_warnings.json').read_text())
        info['vfsim_fallback_warnings']=warnings['instruction_fallback_warnings']
        events=[json.loads(line) for line in (model_dir/'idu_to_ooo.json').read_text().splitlines() if line.strip()]
        info['vfsim_min_idu_visible_free_pregs']=min(e['vreg'] for e in events)
    file.write_text(json.dumps(info,indent=2)+'\n')
    results.append(info)
(OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
print('U CAmodel VfSim vregs spill-LD spill-ST membar golden')
for i in results:
    print(i['unroll'],i.get('vf_cycles'),i.get('vfsim_cycles'),len(i.get('architectural_vregs',[])),i.get('spill_loads_by_address'),
          i.get('spill_stores_by_address'),i.get('membar_count'),i.get('golden_pass'))
