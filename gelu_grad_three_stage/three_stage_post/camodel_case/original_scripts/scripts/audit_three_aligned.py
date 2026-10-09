from pathlib import Path
from collections import Counter
import csv,json,re,subprocess,sys
repo=Path('/mnt/e/vfsimulator')
new=Path('/tmp/gelu-three-stage-log-aligned')
old=repo/'results/ub_address_dependency_experiment/GeLU_grad/experiments/three_stage_post/three_stage_u8_post_update/logs/camodel'
def extract(folder):
    static={};dynamic={}
    for line in (folder/'core0.veccore0.instr_popped_log.dump').read_text().splitlines():
        m=re.search(r'\(PC: (0x[0-9a-f]+)\).*?\(Binary: (0x[0-9a-f]+)\).*?\(ID:\s*(\d+)\) (RV_\w+)(.*)',line)
        if m:
            pc,binary,iid,op,detail=m.groups()
            entry=dict(pc=int(pc,16),binary=binary,op=op,detail=detail)
            static.setdefault(entry['pc'],entry);dynamic[int(iid)]=entry
    ordered=[static[p] for p in sorted(static)]
    loops=[]
    for j,i in enumerate(ordered):
        if i['op']=='RV_VLOOPv2':
            n=int(re.search(r'instr_num\s*:\s*(\d+)',i['detail'])[1])
            iterations=int(re.search(r'Sn\[\d+\]=0x([0-9a-f]+)',i['detail'])[1],16)
            loops.append(dict(iterations=iterations,body=ordered[j+1:j+1+n]))
    done=(folder/'core0.veccore0.instr_log.dump').read_text()
    cycles=int(re.search(r'vf_execute_time:\s*(\d+)',done)[1])
    return dict(static=ordered,loops=loops,cycles=cycles,
                counts=dict(Counter(i['op'] for i in dynamic.values())),
                spill=dict(Counter(i['op'] for i in dynamic.values() if 'Sn[95]' in i['detail'])))
a,b=extract(old),extract(new)
assert len(a['loops'])==len(b['loops'])==3
comparisons=[]
for i,(x,y) in enumerate(zip(a['loops'],b['loops'])):
    comparisons.append(dict(stage=i+1,iterations=y['iterations'],body_count=len(y['body']),
                            binary_equal=[r['binary'] for r in x['body']]==[r['binary'] for r in y['body']],
                            opcodes_equal=[r['op'] for r in x['body']]==[r['op'] for r in y['body']]))
for name,data in (('original',a),('rewritten',b)):
    with (new/f'{name}_pc.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['pc','binary','op','first_observed_operands'])
        w.writerows((hex(i['pc']),i['binary'],i['op'],i['detail']) for i in data['static'])
cmd=[sys.executable,str(repo/'main.py'),'--cce',str(new/'kernel.cce'),'--out_dir',str(new/'vfsim')]
p=subprocess.run(cmd,cwd=repo,capture_output=True,text=True,timeout=180)
(new/'vfsim.log').write_text(p.stdout+p.stderr)
assert p.returncode==0,p.stdout+p.stderr
prediction=int(re.search(r'VF end cycle \(with drain\) =\s*(\d+)',p.stdout)[1])
events=[json.loads(l) for l in (new/'vfsim/start_by_cycle.json').read_text().splitlines()]
counts=Counter(r['op'] for r in events)
assert counts['VMOV']==128 and counts['VDUP']==4 and counts['VLDS']==448 and counts['VSTS']==256
assert all(len(r['src'])==len(r['dst'])==1 and r['preg_src'][0] is not None for r in events if r['op']=='VMOV')
bars=json.loads((new/'vfsim/membar_history.json').read_text())
assert sum(r['event']=='issue' for r in bars)==2
delta={op:b['counts'].get(op,0)-a['counts'].get(op,0) for op in set(a['counts'])|set(b['counts']) if b['counts'].get(op,0)!=a['counts'].get(op,0)}
summary=dict(original_camodel=a['cycles'],new_camodel=b['cycles'],vfsim=prediction,
             accuracy_percent=100*(1-abs(prediction-b['cycles'])/b['cycles']),loops=comparisons,
             dynamic_count_delta=delta,camodel_counts=b['counts'],camodel_spill=b['spill'],
             vfsim_counts=dict(counts),validation=json.loads((new/'validation.json').read_text()))
(new/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
