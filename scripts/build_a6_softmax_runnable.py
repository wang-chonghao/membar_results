"""从已归档的 gap2 VF 生成适配 microbenchmark-cce-direct 的 A6 完整入口。"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def build(archive):
    source = archive / 'a6_softmax_fp16_row_scaling/rows8/gap2/source/kernel.cce'
    original = source.read_bytes()
    text = original.decode('utf-8')
    target = archive / 'a6_softmax_fp16_row_scaling/runnable_a6'
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for with_bar in (False, True):
        body = text
        if with_bar:
            pattern = r'(?m)^(\s*)vlds\(norm_exp0, memB, (?:norm_row|penultimate_norm_row|final_norm_row) \* cols, NORM\);'
            body, count = re.subn(pattern, r'\1mem_bar(VST_VLD);\n\g<0>', body)
            if count != 4:
                raise ValueError('应在四个静态 NORM 行入口插入同步')
        name = 'softmax_8x1024_a6' + ('_membar' if with_bar else '')
        header = f'''// A6 完整运行入口；原始 VF 快照与历史预测保持不变。
// 无 MB_GOLDEN：按行 softmax 和 FP16 由 check_softmax.py 独立校验。
// MB_KERNEL {name}
// MB_TENSOR input input fp32 8x1024
// MB_TENSOR output output fp16 8x1024
#if __NPU_ARCH__ != 9201
#error "需要 A6 dav-920r1-vec 编译目标"
#endif

'''
        wrapper = f'''
extern "C" __global__ __aicore__
void {name}(__gm__ float* input, __gm__ half* output)
{{
    constexpr uint32_t elements = 8 * 1024;
    constexpr uint32_t outputBytes = elements * sizeof(half);
    // [0, 0x8000) 输入；[0x8000, 0x10000) exp；
    // [0x10000, 0x14000) FP16 输出；[0x14000, 0x14020) offset 表预留。
    __ubuf__ float* memA = (__ubuf__ float*)get_imm(0x0000);
    __ubuf__ float* memB = (__ubuf__ float*)get_imm(0x8000);
    __ubuf__ half* memOutput = (__ubuf__ half*)get_imm(0x10000);
    __ubuf__ uint32_t* offsets = (__ubuf__ uint32_t*)get_imm(0x14000);

    nddma_desc inputDesc(nddma_desc::loop_desc(elements, 1, 1));
    nddma_out_to_ub(memA, input, 0, inputDesc, 0, NEAREST_PADDING, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    attention_softmax_8row_deferbar1024_simd_ub_fixed(memA, memB, memOutput);

    offsets[0] = 0;
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    // 按字节搬运 16384B，uint32_t 指针转换不改变 FP16 数据。
    scatter_ubuf_to_gm(reinterpret_cast<__gm__ uint32_t*>(output),
                       reinterpret_cast<__ubuf__ uint32_t*>(memOutput),
                       0, 0, offsets, 0, outputBytes, 0, 1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}}
'''
        destination = target / f'{name}.cce'
        destination.write_text(header+body+wrapper, encoding='utf-8')
        records.append({'kernel': destination.name, 'kernel_name': name,
                        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                        'internal_membar': with_bar})
    (target / 'source_provenance.json').write_text(json.dumps({
        'vf_source': '../rows8/gap2/source/kernel.cce',
        'vf_source_sha256': hashlib.sha256(original).hexdigest(),
        'shape': [8, 1024], 'ub_reserved_bytes': 81952,
        'copyin': 'nddma_out_to_ub, fp32 8192 elements',
        'copyout': 'scatter_ubuf_to_gm, 16384 bytes, offset_type=0, source_stride=0',
        'artifacts': records,
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    build(parser.parse_args().archive_root)
