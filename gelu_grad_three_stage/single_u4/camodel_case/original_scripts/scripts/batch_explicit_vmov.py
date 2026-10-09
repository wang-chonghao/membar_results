import subprocess,sys
for u in (1,2,4,8):
    print(f'Running explicit VMOV u{u}',flush=True)
    subprocess.run([sys.executable,'/tmp/run_aligned_gelu.py',str(u)],check=True)
