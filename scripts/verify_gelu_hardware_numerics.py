import argparse,json,re,sys
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description='逐条解释 GeLU Grad 硬件等效 CCE，比较原 CAModel 输出。')
parser.add_argument('--vfsim-root',type=Path,required=True,help='VfSimulator 仓库路径，需包含原实验输入与输出。')
repo=parser.parse_args().vfsim_root.resolve()
sys.path.insert(0,str(repo))
from api.cce_adapter import _split_args
root=Path(__file__).resolve().parents[1]
archives=repo/'results/ub_address_dependency_experiment/GeLU_grad/experiments'
checks=[]
for version,archive in (('u8',archives/'explicit_vmov_validation/u8'),('u8_explicit',archives/'u8_explicit_spill')):
    folder=root/'gelu_grad'/version
    audit=json.loads((folder/'source/hardware_pc_mapping.json').read_text(encoding='utf-8'))
    code=(folder/'source/hardware_equivalent.cce').read_text(encoding='utf-8')
    mem=bytearray(0x40000)
    mem[:32768]=(archive/'data/input.bin').read_bytes()
    regs={};pred={};ptr={}
    origins={'dy':0,'x':0x4000,'out':0x8000}
    for name,base,offset in re.findall(r'float \*(ptr_\w+) = (dy|x|out) \+ (\d+);',code):
        ptr[name]=origins[base]+4*int(offset)
    def reg(arg):return re.search(r'\br\d+\b',arg)[0]
    def f32(value):return np.float32(value.removesuffix('f'))
    def address(arg,offset):
        if arg in ptr:return ptr[arg]
        return 0x40000+int(offset)*(4 if 'float' in arg else 1)
    def execute(node):
        statement=node['cce']
        call=re.search(r'(\w+)\((.*)\);',statement)
        op=call[1];args=_split_args(call[2])
        if op.startswith('pset'):
            dest=statement.split('=')[0].strip();width=int(op.split('b')[1])
            pred[dest]=np.zeros(256,dtype=bool);pred[dest][::width//8]=True
        elif op=='vdup':regs[args[0]]=np.full(64,f32(args[1]),dtype=np.float32)
        elif op=='vmov':regs[args[0]]=regs[args[1]].copy()
        elif op in ('vlds','vsts'):
            addr=address(args[1],args[2])
            if op=='vlds':regs[reg(args[0])]=np.frombuffer(mem,dtype='<f4',count=64,offset=addr).copy()
            else:mem[addr:addr+256]=regs[reg(args[0])].astype('<f4').tobytes()
            if args[-1]=='POST_UPDATE':ptr[args[1]]+=4*int(args[2])
        elif op=='psts':
            addr=address(args[1],args[2]);mem[addr:addr+32]=np.packbits(pred[args[0]],bitorder='little').tobytes()
        elif op=='plds':
            addr=address(args[1],args[2]);pred[args[0]]=np.unpackbits(np.frombuffer(mem,dtype=np.uint8,count=32,offset=addr),bitorder='little').astype(bool)
        elif op=='vaxpy':
            regs[args[0]]=(regs[args[0]].astype(np.float64)+regs[args[1]].astype(np.float64)*float(f32(args[2]))).astype(np.float32)
        elif op=='vmul':regs[args[0]]=regs[args[1]]*regs[args[2]]
        elif op=='vadds':regs[args[0]]=regs[args[1]]+f32(args[2])
        elif op=='vadd':regs[args[0]]=regs[args[1]]+regs[args[2]]
        elif op=='vdiv':regs[args[0]]=regs[args[1]]/regs[args[2]]
        elif op=='vexp':regs[args[0]]=np.exp(regs[args[1]])
        elif op=='vcmp_eq':
            pred[args[0]]=np.zeros(256,dtype=bool)
            pred[args[0]][::4]=regs[args[1]]==regs[args[2]]
        elif op=='vsel':regs[args[0]]=np.where(pred[args[3]][::4],regs[args[1]],regs[args[2]])
        elif op!='mem_bar':raise ValueError(op)
    nodes=audit['pc_mapping']
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for node in nodes:
            if node['dynamic_count']==1:execute(node)
        for _ in range(8):
            for node in nodes:
                if node['dynamic_count']==8:execute(node)
    output=np.frombuffer(mem,dtype='<f4',count=4096,offset=0x8000).copy()
    expected=np.fromfile(archive/'data/output.bin',dtype='<f4')
    errors=np.abs(output.astype(float)-expected.astype(float))
    assert np.all(np.isfinite(output)) and np.all(errors<=2e-6+2e-4*np.abs(expected)),(version,errors.max())
    checks.append(dict(version=version,elements=4096,passed=True,max_abs_error_vs_camodel=float(errors.max()),
                       method='按生成的等效 CCE 逐条解释，保留 spill 和 predicate bitmaps；比较既有 CAModel 输出'))
(root/'gelu_grad/hardware_numerical_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
