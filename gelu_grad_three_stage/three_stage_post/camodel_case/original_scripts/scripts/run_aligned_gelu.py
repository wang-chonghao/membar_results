import json,subprocess,shutil,os,struct,math,sys
from pathlib import Path
repo=Path('/mnt/e/vfsimulator')
three_stage=sys.argv[1]=='three'
unroll=1 if three_stage else int(sys.argv[1])
assert unroll in (1,2,4,8)
work=Path('/tmp/gelu-three-stage-log-aligned') if three_stage else Path(f'/tmp/gelu-explicit-vmov-validation/u{unroll}')
work.mkdir(parents=True,exist_ok=True)
src=repo/'cce_code/gelu_grad/log_aligned/three_stage_post_u1.cce' if three_stage else repo/f'cce_code/gelu_grad/log_aligned/u{unroll}.cce'
shutil.copy2(src,work/'kernel.cce')
base=repo/f'results/ub_address_dependency_experiment/GeLU_grad/experiments/baseline/u{unroll}'
for name in ('input.bin','golden.bin'):
    shutil.copy2(base/'data'/name,work/name)
cann=Path('/home/lenovo/Ascend/ascend-toolkit/cann-9.0.0-beta.1')
libs=[cann/'tools/simulator/Ascend950PR_9599/lib',cann/'lib64',cann/'x86_64-linux/devlib',cann/'x86_64-linux/devlib/device']
env=dict(os.environ,ASCEND_TOOLKIT_HOME=str(cann),NPU_TYPE='Ascend950PR_9599',LD_LIBRARY_PATH=':'.join(map(str,libs)))
commands=[([str(cann/'bin/ccec'),'-g','-std=c++17','-c','-O2','kernel.cce','-o','kernel_aiv.o',
            '-I/usr/include/c++/11','-I/usr/include/aarch64-linux-gnu/c++/11',
            '--cce-aicore-arch=dav-c310-vec','--cce-aicore-only','--cce-simd-vf-fusion=false','-mllvm','-cce-aicore-vec-misched=0'],'compile.log'),
          ([str(cann/'bin/ld.lld'),'-Ttext=0','kernel_aiv.o','-static','-o','kernel.o'],'link.log'),
          (['/tmp/vfsim-gelu-grad-baseline-post-20260929/host',str(work/'kernel.o'),'gelu_grad_fp32'],'camodel.log')]
(work/'commands.json').write_text(json.dumps(commands,indent=2))
for cmd,log in commands:
    with (work/log).open('w') as f:
        p=subprocess.run(cmd,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    if p.returncode:raise RuntimeError((work/log).read_text())
data=struct.unpack('<8192f',(work/'input.bin').read_bytes())
out=struct.unpack('<4096f',(work/'output.bin').read_bytes())
expected=[]
for dy,x in zip(data[:4096],data[4096:]):
    t=math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3))
    expected.append(dy*0.5*(1+t+x*(1-t*t)*math.sqrt(2/math.pi)*(1+3*0.044715*x*x)))
err=[abs(a-b) for a,b in zip(out,expected)]
valid=dict(passed=all(math.isfinite(a) and e<=2e-6+2e-4*abs(b) for a,b,e in zip(out,expected,err)),
           elements=4096,max_abs_error=max(err),atol=2e-6,rtol=2e-4,
           reference_bitwise_equal=(work/'output.bin').read_bytes()==(base/'data/output.bin').read_bytes())
(work/'validation.json').write_text(json.dumps(valid,indent=2))
assert valid['passed'] and valid['reference_bitwise_equal']
print(valid)
