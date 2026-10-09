import hashlib,json,shutil,tarfile
from pathlib import Path
repo=Path('/mnt/e/vfsimulator')
work=Path('/tmp/gelu-u8-explicit-spill')
dest=repo/'results/ub_address_dependency_experiment/GeLU_grad/experiments/u8_explicit_spill'
dest.mkdir(parents=True,exist_ok=True)
manifest=[]
def copy(src,relative):
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,target)
    a=hashlib.sha256(src.read_bytes()).hexdigest();b=hashlib.sha256(target.read_bytes()).hexdigest()
    assert a==b
    manifest.append(dict(path=relative,sha256=a))
copy(repo/'cce_code/gelu_grad/log_aligned/U8_EXPLICIT_SPILL_RESULTS.md','README.md')
copy(work/'kernel.cce','code/kernel.cce')
for name in ('summary.json','validation.json','rewritten_pc.csv','semantic_differences.json','commands.json'):
    copy(work/name,name)
copy(repo/'results/ub_address_dependency_experiment/GeLU_grad/experiments/explicit_vmov_validation/u8/rewritten_pc.csv','original_pc.csv')
for name in ('input.bin','golden.bin','output.bin'):copy(work/name,'data/'+name)
for name in ('compile.log','link.log','camodel.log','core0.veccore0.instr_log.dump','core0.veccore0.instr_popped_log.dump','core0.veccore0.rvec.EXU.dump','core0.veccore0.rvec.LSU.dump'):
    copy(work/name,'logs/'+name)
for name in ('generate_u8_spill.py','run_aligned_gelu.py','audit_u8_spill.py','archive_u8_spill.py'):
    copy(Path('/tmp')/name,'scripts/'+name)
with tarfile.open(dest/'full_camodel.tar.gz','w:gz') as t:t.add(work,arcname='run')
with tarfile.open(dest/'full_camodel.tar.gz') as t:
    members=t.getnames();assert any(n.endswith('.vcd') for n in members)
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(destination=str(dest),verified_files=len(manifest),archive_members=len(members))))
