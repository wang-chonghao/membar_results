import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/e/vfsimulator')
OUT = ROOT / 'results/ub_address_dependency_experiment/gelu_poly_i96_unroll_spill_20260911'
HELPERS = Path('/mnt/e/VfSimulator_agentOptimization/VfSimulator_RR/cce_code/softmax_manuel')
if (Path(__file__).parent / 'build_softmax_camodel.sh').exists():
    HELPERS = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from codex_optimization_skill.scripts.parse_camodel_vf import parse


def run(u):
    case = OUT / f'u{u}'
    source_dir = case / 'source'
    logs = case / 'camodel'
    source_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(exist_ok=True)
    original = (ROOT / 'cce_code/GeLU_poly.dsl').read_bytes()
    assert original.count(b'#pragma unroll(1)') == 1
    candidate = original.replace(b'#pragma unroll(1)', f'#pragma unroll({u})'.encode())
    source = source_dir / f'gelu_poly_i96_u{u}.cce'
    source.write_bytes(candidate)
    stem = source.stem
    build_root = Path('/tmp/vfsim_gelu_unroll_spill_20260911') / f'u{u}'
    build = build_root / f'{stem}_native_simexec'
    env = dict(os.environ, BUILD_ROOT=str(build_root),
               CCEC_EXTRA_FLAGS='-mllvm -cce-aicore-vec-misched=0',
               CORE_ARCH='dav-c310-vec', NPU_TYPE='Ascend950PR_9599')
    print(f'BUILD u{u}', flush=True)
    with (logs / 'build.log').open('w') as f:
        compiled = subprocess.run(['bash', str(HELPERS / 'build_softmax_camodel.sh'), str(source), stem],
                       env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
    if compiled.returncode:
        summary = dict(unroll=u, iterations=96, elements=6144, status='build_failed',
                       source_sha256=hashlib.sha256(candidate).hexdigest(),
                       golden_pass=None, returncode=compiled.returncode,
                       build_error=(logs / 'build.log').read_text())
        (case / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return
    print(f'RUN u{u}', flush=True)
    start = time.monotonic()
    with (build / 'run.log').open('w') as f:
        result = subprocess.run(['bash', '-c',
            f'source ./run_simexec_env.sh\nexec ./{stem}_simexec ./{stem}_mix.o foo_add 2 1 6144'],
            cwd=build, stdout=f, stderr=subprocess.STDOUT, timeout=240)
    wall = time.monotonic() - start
    for path in build.iterdir():
        if path.name.startswith('core0.veccore0.') or path.name in (
                'run.log', 'input0.bin', 'input1.bin', 'output0.bin', 'golden.bin', 'run_simexec_env.sh'):
            if path.is_file():
                shutil.copy2(path, logs / path.name)
    log = (build / 'run.log').read_text(errors='replace')
    check_lines = [line for line in log.splitlines() if '[CHECK]' in line or '[ERROR]' in line]
    timing = parse(build)
    records = []
    seen = set()
    pattern = re.compile(r'\[info\]\s+\[(\d+)\]\s+\(PC:\s*(0x[0-9a-fA-F]+)\).*?\(ID:\s*(\d+)\)\s+(RV_\w+)\b')
    for line in (build / 'core0.veccore0.instr_log.dump').read_text(errors='replace').splitlines():
        m = pattern.search(line)
        if not m:
            continue
        cycle, pc, iid, op = m.groups()
        if not timing['first_vf_start'] <= int(cycle) <= timing['last_vf_end']:
            continue
        key = (pc, iid, op)
        if key in seen:
            continue
        seen.add(key)
        records.append(dict(cycle=int(cycle), pc=pc, inst_id=int(iid), op=op, text=line))
    counts = collections.Counter(r['op'] for r in records)
    memory = [r for r in records if r['op'].startswith(('RV_VLD', 'RV_VST')) or r['op'] == 'RV_SMEM_BAR']
    (case / 'memory_records.json').write_text(json.dumps(memory, indent=2))
    (case / 'compiled_pc_order.txt').write_text('\n'.join(
        next(r['text'] for r in records if r['pc'] == pc)
        for pc in sorted({r['pc'] for r in records}, key=lambda x: int(x, 16))) + '\n')
    summary = dict(unroll=u, iterations=96, elements=6144,
                   source_sha256=hashlib.sha256(candidate).hexdigest(),
                   golden_pass=result.returncode == 0 and '[CHECK] PASS' in log,
                   checks=check_lines, returncode=result.returncode, wall_seconds=wall,
                   timing=timing, opcode_counts=dict(sorted(counts.items())),
                   load_count=sum(n for op, n in counts.items() if op.startswith('RV_VLD')),
                   store_count=sum(n for op, n in counts.items() if op.startswith('RV_VST')),
                   membar_count=counts['RV_SMEM_BAR'])
    (case / 'summary.json').write_text(json.dumps(summary, indent=2))
    summaries = [json.loads(p.read_text()) for p in OUT.glob('u*/summary.json')]
    summaries.sort(key=lambda x: x['unroll'])
    (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summary), flush=True)
    if not summary['golden_pass']:
        raise RuntimeError(f'u{u}: golden check failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('unroll', type=int, nargs='+')
    args = parser.parse_args()
    scripts = OUT / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ('build_softmax_camodel.sh', 'native_runtime_softmax_main.cpp'):
        if (HELPERS / name).resolve() != (scripts / name).resolve():
            shutil.copy2(HELPERS / name, scripts / name)
    if Path(__file__).resolve() != (scripts / Path(__file__).name).resolve():
        shutil.copy2(__file__, scripts / Path(__file__).name)
    for u in args.unroll:
        run(u)
