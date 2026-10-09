import csv,json,re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

ROOT=Path('/mnt/e/vfsimulator')
OUT=Path('/tmp/gelu-explicit-vmov-validation')
PAT=re.compile(r'\(PC: (0x[0-9a-f]+)\).*?\(Binary: (0x[0-9a-f]+)\).*?\(ID:\s*(\d+)\) (RV_\w+)(.*)')
def extract(folder):
    static={}; dynamic={}
    for line in (folder/'core0.veccore0.instr_popped_log.dump').read_text().splitlines():
        m=PAT.search(line)
        if not m:continue
        pc,binary,iid,op,operands=m.groups()
        item=dict(pc=int(pc,16),binary=binary,op=op,operands=operands)
        static.setdefault(int(pc,16),item)
        assert static[int(pc,16)]['binary']==binary
        dynamic[int(iid)]=item
    ordered=[static[pc] for pc in sorted(static)]
    loops=[i for i in ordered if i['op']=='RV_VLOOPv2']
    assert len(loops)==1
    loop=loops[0]
    length=int(re.search(r'instr_num\s*:\s*(\d+)',loop['operands'])[1])
    iterations=int(re.search(r'Sn\[\d+\]=0x([0-9a-f]+)',loop['operands'])[1],16)
    start=ordered.index(loop)+1
    body=ordered[start:start+length]
    assert len(body)==length
    logs=(folder/'core0.veccore0.instr_log.dump').read_text()
    cycles=re.findall(r'vf_execute_time:\s*(\d+)',logs)
    assert len(cycles)==1
    ops=Counter(i['op'] for i in dynamic.values())
    spill=Counter(i['op'] for i in dynamic.values() if 'Sn[95]' in i['operands'])
    return dict(static=ordered,body=body,iterations=iterations,cycles=int(cycles[0]),
                ops=dict(ops),spill=dict(spill),dynamic=dynamic)

def reg_graph(items):
    # Normalize physical register names, retaining their reuse and operand roles.
    aliases={}
    result=[]
    for item in items:
        def rename(match):
            role,n=match.groups()
            bank=role[0]
            key=(bank,n)
            if key not in aliases:
                aliases[key]=sum(k[0]==bank for k in aliases)
            return f'{role}[{aliases[key]}]'
        operands=re.sub(r'\b(V[dnm]|P[dg])\[(\d+)\]',rename,item['operands'])
        # Scalar register numbers vary; retain their values (constants and addresses).
        operands=re.sub(r'\b(S(?:d|n2?|m2?))\[\d+\]',r'\1',operands)
        result.append((item['op'],operands.strip()))
    return result

rows=[]
for u in (1,2,4,8):
    folder=OUT/f'u{u}'
    if not (folder/'validation.json').exists():continue
    old=extract(ROOT/f'results/ub_address_dependency_experiment/GeLU_grad/experiments/baseline/u{u}/logs/camodel')
    new=extract(folder)
    ops_delta={op:new['ops'].get(op,0)-old['ops'].get(op,0) for op in sorted(set(old['ops'])|set(new['ops']))
               if new['ops'].get(op,0)!=old['ops'].get(op,0)}
    seq=lambda x:[(i['op'],i['binary']) for i in x]
    body_graph_old=reg_graph(old['body']); body_graph_new=reg_graph(new['body'])
    row=dict(unroll=u,original_cycles=old['cycles'],cycles=new['cycles'],iterations=new['iterations'],
             body_instructions=len(new['body']),original_body_instructions=len(old['body']),
             static_instructions=len(new['static']),original_static_instructions=len(old['static']),
             exact_static_pc_binary_equal=old['static']==new['static'],
             static_opcode_binary_sequence_equal=seq(old['static'])==seq(new['static']),
             body_opcode_sequence_equal=[i['op'] for i in old['body']]==[i['op'] for i in new['body']],
             body_binary_sequence_equal=seq(old['body'])==seq(new['body']),
             body_normalized_register_graph_equal=body_graph_old==body_graph_new,
             dynamic_op_count_delta=ops_delta,original_spill=old['spill'],spill=new['spill'],
             original_membar=old['ops'].get('RV_SMEM_BAR',0),membar=new['ops'].get('RV_SMEM_BAR',0),
             validation=json.loads((folder/'validation.json').read_text()))
    (folder/'alignment.json').write_text(json.dumps(row,indent=2))
    for name,data in (('original',old),('rewritten',new)):
        with (folder/f'{name}_pc.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['pc','binary','opcode','first_observed_operands'])
            w.writerows((hex(i['pc']),i['binary'],i['op'],i['operands']) for i in data['static'])
    matcher=SequenceMatcher(a=[i['op'] for i in old['static']],b=[i['op'] for i in new['static']],autojunk=False)
    changes=[]
    for tag,a,b,c,d in matcher.get_opcodes():
        if tag!='equal':changes.append(dict(kind=tag,original=old['static'][a:b],rewritten=new['static'][c:d]))
    (folder/'static_opcode_diff.json').write_text(json.dumps(changes,indent=2))
    rows.append(row)
    print(json.dumps(row),flush=True)
(OUT/'summary.json').write_text(json.dumps(rows,indent=2))
