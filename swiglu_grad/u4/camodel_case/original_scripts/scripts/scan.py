"""Scan compiler unroll while keeping SwiGLUGrad arithmetic and golden fixed."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[3]
BASE = OUT.parent / 'u1'
sys.path.insert(0, str(ROOT))
from codex_optimization_skill.scripts.parse_camodel_vf import parse


def save(path, data):
    path.write_text(json.dumps(data, indent=2))


def command(args, log, *, cwd=ROOT, env=None, timeout=600):
    start = time.monotonic()
    with log.open('w') as f:
        p = subprocess.run(args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, time.monotonic() - start


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
    return [r for r in records.values() if r['min_address'] is not None]


def analyze(build, case, summary):
    logs = case / 'camodel'
    for p in build.iterdir():
        if p.is_file() and (p.name.startswith('core0.veccore0.') or p.suffix == '.bin' or p.name == 'run_simexec_env.sh'):
            shutil.copy2(p, logs / p.name)
    timing = parse(build)
    summary['timing'] = timing
    pattern = re.compile(r'\[info\]\s+\[(\d+)\]\s+\(PC:\s*(0x[0-9a-fA-F]+)\).*?\(ID:\s*(\d+)\)\s+(RV_\w+)\b')
    records = {}
    for line in (logs/'core0.veccore0.instr_log.dump').read_text().splitlines():
        m = pattern.search(line)
        if m and timing['first_vf_start'] <= int(m[1]) <= timing['last_vf_end']:
            records.setdefault((m[2],int(m[3]),m[4]),dict(pc=m[2],op=m[4],text=line))
    counts = collections.Counter(r['op'] for r in records.values())
    pcs = {}
    for r in records.values():
        pcs.setdefault(r['pc'], r['text'])
    (case/'compiled_pc_order.txt').write_text('\n'.join(pcs[p] for p in sorted(pcs,key=lambda p:int(p,16)))+'\n')
    summary['architectural_vregs'] = sorted(set(int(v) for line in pcs.values() for v in re.findall(r'V[dnm]\[(\d+)\]',line)))
    summary['opcode_counts'] = dict(sorted(counts.items()))
    summary['load_count'] = sum(n for op,n in counts.items() if op.startswith('RV_VLD'))
    summary['store_count'] = sum(n for op,n in counts.items() if op.startswith('RV_VST'))
    summary['membar_count'] = counts['RV_SMEM_BAR']
    accesses = memory_accesses(logs/'core0.veccore0.ub.rd_log.dump') + memory_accesses(logs/'core0.veccore0.ub.wr_log.dump')
    accesses = [a for a in accesses if timing['first_vf_start'] <= a['cycle'] <= timing['last_vf_end']]
    bases = (0, 0x8000, 0x10000, 0x18000, 0x20000)
    extra = [a for a in accesses if not any(base <= a['min_address'] <= a['max_address'] < base+0x6000 for base in bases)]
    spill = [a for a in extra if a['op'].startswith(('RV_VLD','RV_VST'))]
    save(case/'spill_ub_accesses.json',spill)
    summary['spill_load_count'] = sum(a['op'].startswith('RV_VLD') for a in spill)
    summary['spill_store_count'] = sum(a['op'].startswith('RV_VST') for a in spill)
    summary['spill_address_min'] = min((a['min_address'] for a in spill), default=None)
    summary['spill_address_max'] = max((a['max_address'] for a in spill), default=None)
    assert summary['load_count']-summary['spill_load_count']==288, summary
    assert summary['store_count']-summary['spill_store_count']==192, summary
    def floats(name):
        data = (logs/name).read_bytes()
        assert len(data)==6144*4, (name,len(data))
        return struct.unpack('<6144f',data)
    a,b,g=(floats('input%d.bin'%i) for i in range(3))
    checks=[]
    for out in range(2):
        errors=[]
        mismatch=0
        for x,y,dy,value in zip(a,b,g,floats('output%d.bin'%out)):
            s=1/(1+math.exp(-x))
            golden=dy*y*s*(1+x*(1-s)) if out==0 else dy*x*s
            err=abs(value-golden)
            mismatch+=not (math.isfinite(value) and (err<=1e-6 or err<=1e-3*abs(golden)))
            errors.append(err)
        checks.append(dict(output=out,elements=6144,mismatches=mismatch,max_abs_error=max(errors)))
    summary['golden_checks']=checks
    summary['golden_pass']=all(c['mismatches']==0 for c in checks)
    assert summary['golden_pass'], checks


def run(u):
    case=OUT/f'u{u}'
    if (case/'summary.json').exists():
        raise RuntimeError(f'Refusing to overwrite existing result: {case}')
    for d in ('source','camodel','vfsim/source'):
        (case/d).mkdir(parents=True,exist_ok=True)
    text=(BASE/'source/swiglu_grad_fp32_i96_u1_compiler_aligned.cce').read_text()
    assert text.count('#pragma unroll(1)')==1 and 96%u==0
    text=text.replace('#pragma unroll(1)',f'#pragma unroll({u})')
    source=case/f'source/swiglu_grad_fp32_i96_u{u}.cce'
    source.write_text(text)
    summary=dict(unroll=u,iterations=96,elements_per_tensor=6144,source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                 status='started',model_commit='4ff5c0e')
    print(f'U{u}: VfSim source',flush=True)
    rc,wall=command([sys.executable,str(ROOT/'main.py'),'--soc','dav-3510','--cce',str(source),
        '--cce-kernel','swiglu_grad_vf','--out_dir',str(case/'vfsim/source')],case/'vfsim/run.log')
    summary['vfsim_returncode']=rc
    if rc==0:
        log=(case/'vfsim/run.log').read_text()
        summary['vfsim_source_cycles']=int(re.search(r'VF end cycle \(with drain\) = (\d+)',log)[1])
        warn=case/'vfsim/source/model_warnings.json'
        summary['vfsim_warnings']=json.loads(warn.read_text()) if warn.exists() else {}
    buildroot=Path('/tmp/vfsim_swiglu_unroll_20260914')/f'u{u}'
    stem=source.stem
    build=buildroot/f'{stem}_native_simexec'
    env=dict(os.environ,BUILD_ROOT=str(buildroot),MAIN_CPP=str(OUT/'scripts/native_runtime_swiglu_grad_main.cpp'),
             CCEC_EXTRA_FLAGS='-mllvm -cce-aicore-vec-misched=0',CORE_ARCH='dav-c310-vec',NPU_TYPE='Ascend950PR_9599')
    print(f'U{u}: build',flush=True)
    rc,wall=command(['bash',str(OUT/'scripts/build_camodel.sh'),str(source),stem],case/'camodel/build.log',env=env)
    if rc:
        summary.update(status='build_failed',build_returncode=rc)
    else:
        print(f'U{u}: CAmodel + dual golden',flush=True)
        rc,wall=command(['bash','-c',f'source ./run_simexec_env.sh\nexec ./{stem}_simexec ./{stem}_mix.o swiglu_grad_kernel 3 2 6144'],
                        case/'camodel/run.log',cwd=build)
        summary.update(camodel_returncode=rc,camodel_wall_seconds=wall,status='run_failed')
        if rc==0:
            analyze(build,case,summary)
            summary['status']='ok'
            summary['source_prediction_accuracy_percent']=100*(1-abs(summary['vfsim_source_cycles']-summary['timing']['vf_total_cycles'])/summary['timing']['vf_total_cycles'])
    save(case/'summary.json',summary)
    save(OUT/'summary.json',[json.loads(p.read_text()) for p in sorted(OUT.glob('u*/summary.json'),key=lambda p:int(p.parent.name[1:]))])
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('unroll',type=int,nargs='+')
    args=ap.parse_args()
    for name in ('build_camodel.sh','native_runtime_swiglu_grad_main.cpp'):
        shutil.copy2(BASE/'scripts'/name,OUT/'scripts'/name)
    shutil.copytree(ROOT/'configs/socs/DV100',OUT/'config_snapshot/DV100',dirs_exist_ok=True)
    for u in args.unroll:
        run(u)
