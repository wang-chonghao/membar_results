import csv,re
from pathlib import Path

repo=Path('/mnt/e/vfsimulator')
rows=list(csv.DictReader((repo/'results/ub_address_dependency_experiment/GeLU_grad/experiments/explicit_vmov_validation/u8/rewritten_pc.csv').open()))
start=next(i for i,r in enumerate(rows) if r['opcode']=='RV_VLOOPv2')
body=rows[start+1:start+193]
lines=['#define __aicore__ [aicore]', '', '// Reconstructed from u8 static PC order, including explicit spill traffic.',
       '__attribute__((always_inline)) inline [aicore] void gelu_grad_vf(',
       '    __ubuf__ float *dy, __ubuf__ float *x, __ubuf__ float *out, int stride) {']
for group in ('dy','x','out'):
    for i in range(8):lines.append(f'    __ubuf__ float *{group}{i} = {group} + {64*i};')
lines+=['    __ubuf__ unsigned int *spill = (__ubuf__ unsigned int *)get_imm(0x40000);',
        '    __VEC_SCOPE__ {','        vector_bool p0 = pset_b8(PAT_ALL);','        vector_bool p1 = pset_b32(PAT_ALL);',
        '        vector_bool p2, p3, p4, p5, p6, p7;',
        '        vector_f32 '+', '.join(f'v{i}' for i in range(32))+';',
        '        vdup(v0, 1.0f, p1, MODE_ZEROING);','        vdup(v1, 0.0f, p1, MODE_ZEROING);',
        '        vdup(v2, -1.595769121605730711759f, p1, MODE_ZEROING);',
        '        vdup(v3, 1.595769121605730711759f, p1, MODE_ZEROING);',
        '        vmov(v7, v2);','        vmov(v8, v3);',
        '        #pragma unroll(1)','        for (int batch = 0; batch < 8; ++batch) {']
ptrs={87:'dy0',88:'x0',89:'out0',**{64+i:f'dy{i+1}' for i in range(7)},**{71+i:f'x{i+1}' for i in range(7)},**{78+i:f'out{i+1}' for i in range(7)}}
for r in body:
    op=r['opcode'];s=r['first_observed_operands']
    def reg(name):return int(re.search(name+r'\[(\d+)\]',s)[1])
    def v(name):return 'v'+str(reg(name))
    def p(name):return 'p'+str(reg(name))
    if op=='RV_SMEM_BAR': code=f'mem_bar({s.split(":")[-1].strip()});'
    elif op=='RV_VLDS':code=f'vlds({v("Vd")}, {ptrs[reg("Sn")]}, stride, NORM, POST_UPDATE);'
    elif op=='RV_VSTS':code=f'vsts({v("Vd")}, {ptrs[reg("Sn")]}, stride, NORM_B32, p1, POST_UPDATE);'
    elif op in ('RV_VSTI','RV_VLDI','RV_PSTI','RV_PLDI'):
        offset=int(re.search(r'#offset=(\d+)',s)[1]); offset=(offset-256)*8
        if op=='RV_VSTI':code=f'vsts((vector_u8 &){v("Vd")}, (__ubuf__ unsigned char *)spill, {offset*4}, NORM_B8, p0);'
        elif op=='RV_VLDI':code=f'vlds({v("Vd")}, (__ubuf__ float *)spill, {offset}, NORM);'
        elif op=='RV_PSTI':code=f'psts({p("Pd")}, spill, {offset*4}, NORM);'
        else:code=f'plds({p("Pd")}, spill, {offset*4}, NORM);'
    elif op=='RV_VMOV':code=f'vmov({v("Vd")}, {v("Vn")});'
    elif op=='RV_VAXPY':
        scalar='-0.0713548162726002527220f' if reg('Sm')==14 else '0.2140644488178007f'
        code=f'vaxpy({v("Vd")}, {v("Vn")}, {scalar}, p1, MODE_ZEROING);'
    elif op=='RV_VADDS':code=f'vadds({v("Vd")}, {v("Vn")}, 1.0f, p1, MODE_ZEROING);'
    elif op=='RV_VEXP':code=f'vexp({v("Vd")}, {v("Vn")}, p1, MODE_ZEROING);'
    elif op=='RV_VCMP_EQ':code=f'vcmp_eq({p("Pd")}, {v("Vn")}, {v("Vm")}, p1);'
    elif op=='RV_VSEL':code=f'vsel({v("Vd")}, {v("Vn")}, {v("Vm")}, {p("Pg")});'
    elif op in ('RV_VMUL','RV_VDIV','RV_VADD'):code=f'{op[3:].lower()}({v("Vd")}, {v("Vn")}, {v("Vm")}, p1, MODE_ZEROING);'
    else:raise ValueError(op)
    lines.append(f'            {code} // {r["pc"]}')
lines+=['        }','    }','}','']
original=(repo/'cce_code/gelu_grad/log_aligned/u8.cce').read_text()
lines.append(original[original.index('extern "C"'):].replace('gelu_grad_vf(dy, x, out);', 'gelu_grad_vf(dy, x, out, get_imm(512));'))
(repo/'cce_code/gelu_grad/log_aligned/u8_explicit_spill.cce').write_text('\n'.join(lines))
