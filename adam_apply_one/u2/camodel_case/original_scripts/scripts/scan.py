import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path('/mnt/e/vfsimulator')
OUT = ROOT/'results/ub_address_dependency_experiment/adam_apply_one_fused_i48_unroll'
SDK = Path('/home/lenovo/Ascend/ascend-toolkit/cann-9.0.0-beta.1')
BASE = ROOT/'cce_code/adam_apply_one_fp32_fused_2240.cce'
OLD = ROOT/'results/ub_address_dependency_experiment/adam_apply_one_fp32_scalar_i96/u1/scripts'

def command(args, log, cwd=None, env=None):
    t = time.monotonic()
    with log.open('w') as f:
        p = subprocess.run(args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    return p.returncode, time.monotonic()-t

def scan(u, run):
    case = OUT/f'u{u}'
    for d in ('source','camodel'):
        (case/d).mkdir(parents=True, exist_ok=True)
    src = case/f'source/adam_i48_u{u}.cce'
    text = BASE.read_text()
    assert text.count('#pragma unroll(1)') == 1
    assert 48 % u == 0
    text = text.replace('#pragma unroll(1)', f'#pragma unroll({u})')
    text = text.replace('i < 35', 'i < 48').replace('2240', '3072').replace('8960', '12288')
    old_addresses = [0x2300*j for j in range(7)] + [0xf500+32*j for j in range(6)]
    new_addresses = [0x3000*j for j in range(7)] + [0x15000+32*j for j in range(6)]
    addresses = dict(zip(old_addresses, new_addresses))
    text = re.sub(r'get_imm\((0x[0-9a-f]+)\)',
                  lambda m: f'get_imm(0x{addresses[int(m[1],16)]:04x})', text)
    src.write_text(text)
    obj = case/'camodel/kernel.o'
    flags = ['-g','-std=c++17','-O2','-I/usr/include/c++/11','-I/usr/include/aarch64-linux-gnu/c++/11',
             '--cce-aicore-arch=dav-c310-vec','--cce-aicore-only',
             '-mllvm','-cce-aicore-function-stack-size=16000',
             '-mllvm','-cce-aicore-record-overflow=false', '-mllvm','-cce-aicore-addr-transform',
             '-mllvm','-cce-aicore-jump-expand=true','--cce-simd-vf-fusion=false',
             '-mllvm','-cce-aicore-vec-misched=0']
    rc, wall = command([str(SDK/'x86_64-linux/bin/ccec'),'-c',*flags,str(src),'-o',str(obj)], case/'camodel/compile.log')
    info = dict(unroll=u, iterations=48, elements=3072, full_groups=48//u, remainder=0,
                compile_returncode=rc, compile_seconds=wall)
    info['unroll_applied'] = rc==0 and 'loop not unrolled' not in (case/'camodel/compile.log').read_text()
    if rc == 0 and not info['unroll_applied']:
        raise RuntimeError(f'U{u}: unroll not applied')
    if rc == 0 and run:
        buildroot = Path('/tmp/adam_fused_unroll_i48')/f'u{u}'
        stem = f'adam_i48_u{u}'
        build = buildroot/f'{stem}_native_simexec'
        env = dict(os.environ, BUILD_ROOT=str(buildroot), MAIN_CPP=str(OLD/'native_runtime_adam_main.cpp'),
                   CCEC_EXTRA_FLAGS='-mllvm -cce-aicore-vec-misched=0')
        rc,_ = command(['bash', str(OLD/'build_camodel.sh'),str(src),stem],case/'camodel/build.log',env=env)
        info['build_returncode'] = rc
        if rc == 0:
            print(f'U{u}: running CAmodel',flush=True)
            rc,wall = command(['bash','-c',f'source ./run_simexec_env.sh\nexec ./{stem}_simexec ./{stem}_mix.o adam_kernel 10 3 3072'],
                               case/'camodel/run.log',cwd=build)
            info.update(camodel_returncode=rc,camodel_wall_seconds=wall)
            for p in build.iterdir():
                if p.is_file() and (p.name.startswith('core0.veccore0.') or p.suffix=='.bin'):
                    dest = case/'camodel'/p.name
                    if not dest.exists(): shutil.copy2(p,dest)
            if rc == 0:
                log = (build/'core0.veccore0.instr_log.dump').read_text()
                pop = (build/'core0.veccore0.instr_popped_log.dump').read_text()
                starts=[int(re.search(r'\[(\d+)\]',s)[1]) for s in pop.splitlines() if re.search(r'\bVF\s+addr:',s)]
                ends=[int(re.search(r'\[(\d+)\]',s)[1]) for s in log.splitlines() if re.search(r'\bVF\s+addr:',s) and 'vf_execute_time' in s]
                assert len(starts)==len(ends)==1
                info['vf_cycles']=max(ends)-min(starts)
                counts=Counter(re.findall(r'\bRV_([A-Z0-9_]+)\b',log))
                info['opcode_counts']=dict(sorted(counts.items()))
                info['load_count']=sum(n for op,n in counts.items() if op.startswith('VLD'))
                info['store_count']=sum(n for op,n in counts.items() if op.startswith('VST'))
                info['extra_loads']=info['load_count']-198
                info['extra_stores']=info['store_count']-144
                info['membar_count']=counts['SMEM_BAR']
                info['architectural_vregs']=sorted({int(v) for v in re.findall(r'V[dnm]\[(\d+)\]',log)})
                info['golden_pass']='[CHECK] PASS all three outputs' in (case/'camodel/run.log').read_text()
                assert info['golden_pass']
    rc, wall = command(['python3', str(ROOT/'main.py'), '--cce',str(src),
                       '--cce-kernel','adam_fused_vf','--soc=dav-3510',
                       '--out_dir',str(case/'vfsim')],case/'vfsim_run.log',cwd=ROOT)
    model_log = (case/'vfsim_run.log').read_text()
    match = re.search(r'VF end cycle \(with drain\) = (\d+)',model_log)
    info.update(vfsim_returncode=rc, vfsim_cycles=int(match[1]) if match else None,
                vfsim_wall_seconds=wall)
    (case/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)
    results=[json.loads(p.read_text()) for p in OUT.glob('u*/summary.json')]
    (OUT/'summary.json').write_text(json.dumps(sorted(results,key=lambda x:x['unroll']),indent=2)+'\n')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('unroll',nargs='+',type=int)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    (OUT/'scripts').mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,OUT/'scripts/scan.py')
    for u in args.unroll: scan(u,args.run)
