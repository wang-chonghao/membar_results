import csv,json,re
from collections import Counter
from pathlib import Path
repo=Path('/mnt/e/vfsimulator')
work=Path('/tmp/gelu-u8-explicit-spill')
old=repo/'results/ub_address_dependency_experiment/GeLU_grad/experiments/explicit_vmov_validation/u8/rewritten_pc.csv'
reference=list(csv.DictReader(old.open()))
static={};dynamic={}
for line in (work/'core0.veccore0.instr_popped_log.dump').read_text().splitlines():
    m=re.search(r'\(PC: (0x[0-9a-f]+)\).*?\(Binary: (0x[0-9a-f]+)\).*?\(ID:\s*(\d+)\) (RV_\w+)(.*)',line)
    if m:
        pc,binary,iid,op,detail=m.groups()
        row=dict(pc=pc,binary=binary,opcode=op,first_observed_operands=detail)
        static.setdefault(int(pc,16),row);dynamic[int(iid)]=row
ordered=[static[k] for k in sorted(static)]
with (work/'rewritten_pc.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(reference[0]));w.writeheader();w.writerows(ordered)
def loop(rows):
    i=next(i for i,r in enumerate(rows) if r['opcode']=='RV_VLOOPv2')
    s=rows[i]['first_observed_operands']
    n=int(re.search(r'instr_num\s*:\s*(\d+)',s)[1])
    count=int(re.search(r'Sn\[\d+\]=0x([0-9a-f]+)',s)[1],16)
    return count,rows[i+1:i+1+n]
a,ab=loop(reference);b,bb=loop(ordered)
def semantic_trace(rows,body):
    values={};signatures=[]
    def field(s,name):
        m=re.search(r'\b'+name+r'\[(\d+)\](?:=0x([0-9a-f]+))?',s)
        return (int(m[1]),int(m[2],16) if m[2] else None) if m else None
    def key(s,name):
        f=field(s,name)
        return name[0]+str(f[0]) if f else None
    def value(s,name):
        k=key(s,name)
        return values.get(k,('undefined',k))
    for r in rows:
        if r['opcode']=='RV_VLOOPv2':break
        s=r['first_observed_operands'];op=r['opcode']
        if op=='RV_PSET':values[key(s,'Pd')]=('pset',re.search(r'Dtype:\s*(\w+)',s)[1],re.search(r'#pattern=(\d+)',s)[1])
        elif op=='RV_VDUPS':values[key(s,'Vd')]=('constant',field(s,'Sn')[1] | (field(s,'Sn2')[1]<<16))
        elif op=='RV_VMOV':values[key(s,'Vd')]=value(s,'Vn')
    for iteration in range(2):
        for j,r in enumerate(body):
            op=r['opcode'];s=r['first_observed_operands'];inputs=[];dst=None;extra={}
            if op=='RV_SMEM_BAR':extra['type']=s.split(':')[-1].strip()
            elif op in ('RV_VLDS','RV_VLDI','RV_VSTS','RV_VSTI','RV_PLDI','RV_PSTI'):
                load=op in ('RV_VLDS','RV_VLDI','RV_PLDI')
                pred=op in ('RV_PLDI','RV_PSTI')
                operand='Pd' if pred else 'Vd'
                if load:dst=key(s,operand)
                else:inputs.append(value(s,operand))
                if field(s,'Pg'):inputs.append(value(s,'Pg'))
                address=field(s,'Sn')[1] | ((field(s,'Sn2')[1]<<16) if field(s,'Sn2') else 0)
                offset=re.search(r'#offset=(\d+)',s)
                post='#p=1' in s
                if offset:
                    off=int(offset[1]);off=(off if off<128 else off-256)*32
                else:off=field(s,'Sm')[1]
                # POST_UPDATE logs show the updated base. Normal accesses show the original base.
                extra=dict(address=address-off if post else address+off,post_update=post,
                           increment=off if post else 0,dist=re.search(r'dist:(\w+)',s)[1])
            else:
                dst=key(s,'Pd') if op=='RV_VCMP_EQ' else key(s,'Vd')
                if op=='RV_VAXPY':inputs.append(value(s,'Vd'))
                for name in ('Vn','Vm','Pg'):
                    if field(s,name):inputs.append(value(s,name))
                if field(s,'Sm'):extra['scalar_bits']=field(s,'Sm')[1] | ((field(s,'Sm2')[1]<<16) if field(s,'Sm2') else 0)
            signatures.append(dict(iteration=iteration,index=j,opcode=op,inputs=inputs,extra=extra))
            if dst:values[dst]=('definition',iteration,j)
    return signatures
sa,sb=semantic_trace(reference,ab),semantic_trace(ordered,bb)
diffs=[dict(original=x,rewritten=y) for x,y in zip(sa,sb) if x!=y]
(work/'semantic_differences.json').write_text(json.dumps(diffs,indent=2))
def normalized(trace):
    aliases={'RV_VLDS':'RV_VLDI','RV_VSTS':'RV_VSTI'}
    return [dict(r,opcode=aliases.get(r['opcode'],r['opcode'])) for r in trace]
ops=lambda rs:[r['opcode'] for r in rs]
summary=dict(original_cycles=2553,new_cycles=int(re.search(r'vf_execute_time:\s*(\d+)',(work/'core0.veccore0.instr_log.dump').read_text())[1]),
    original_iterations=a,new_iterations=b,original_body_size=len(ab),new_body_size=len(bb),
    loop_opcode_sequence_equal=ops(ab)==ops(bb),
    loop_binary_equal=[r['binary'] for r in ab]==[r['binary'] for r in bb],
    original_body_counts=dict(Counter(ops(ab))),new_body_counts=dict(Counter(ops(bb))),
    dynamic_counts=dict(Counter(ops(dynamic.values()))),
    semantic_trace_equal=sa==sb,semantic_difference_count=len(diffs),
    normalized_dependency_and_first_address_trace_equal=normalized(sa)==normalized(sb),
    validation=json.loads((work/'validation.json').read_text()))
(work/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
