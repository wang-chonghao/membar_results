"""将已归档的多 shape VF 包装成独立 A6 kernel，不修改历史 VF。"""

import argparse
import hashlib
import json
from pathlib import Path


ROWS = (8, 12, 16, 24)
COLS = 1024
UB_BUDGET = 248 * 1024
VF_NAME = 'attention_softmax_8row_deferbar1024_simd_ub_fixed'


def render_kernel(name, body, rows, scalar_bytes):
    input_bytes = rows * COLS * 4
    output_bytes = rows * COLS * 2
    output_base = input_bytes * 2 + scalar_bytes
    offsets_base = output_base + output_bytes
    reserved_bytes = offsets_base + 32
    if reserved_bytes > UB_BUDGET or offsets_base % 32:
        raise ValueError('UB 布局超出预算或未按 32B 对齐')
    header = f'''// A6 完整运行入口；GM 输入 FP32，GM 输出 FP16，block_dim=1。
// 原 VF 保留不变；按行 softmax 由 check_softmax.py 独立校验。
// MB_KERNEL {name}
// MB_TENSOR input input fp32 {rows}x{COLS}
// MB_TENSOR output output fp16 {rows}x{COLS}
#if __NPU_ARCH__ != 9201
#error "需要 A6 dav-920r1-vec 编译目标"
#endif

'''
    wrapper = f'''
extern "C" __global__ __aicore__
void {name}(__gm__ float* input, __gm__ half* output)
{{
    if (get_block_idx() != 0) {{
        return;
    }}
    constexpr uint32_t elements = {rows} * {COLS};
    constexpr uint32_t outputBytes = elements * sizeof(half);
    // memB 的 exp 后另留 {scalar_bytes}B 给三循环的 memMax/memSum。
    // UB 共预留 {reserved_bytes}B，不含编译器额外 VF 栈。
    __ubuf__ float* memA = (__ubuf__ float*)get_imm(0x0000);
    __ubuf__ float* memB = (__ubuf__ float*)get_imm(0x{input_bytes:x});
    __ubuf__ half* memOutput = (__ubuf__ half*)get_imm(0x{output_base:x});
    __ubuf__ uint32_t* offsets = (__ubuf__ uint32_t*)get_imm(0x{offsets_base:x});

    nddma_desc inputDesc(nddma_desc::loop_desc(elements, 1, 1));
    nddma_out_to_ub(memA, input, 0, inputDesc, 0, NEAREST_PADDING, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    {VF_NAME}(memA, memB, memOutput);

    offsets[0] = 0;
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    // outputBytes 是 FP16 输出的字节数，uint32_t 转换不改变位模式。
    scatter_ubuf_to_gm(reinterpret_cast<__gm__ uint32_t*>(output),
                       reinterpret_cast<__ubuf__ uint32_t*>(memOutput),
                       0, 0, offsets, 0, outputBytes, 0, 1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}}
'''
    return header + body + wrapper


def build(archive):
    experiment = archive / 'a6_softmax_fp16_row_scaling'
    target = experiment / 'runnable_a6'
    records = []
    for rows in ROWS:
        for variant, source_variant in (('no_membar', 'gap2'), ('three_loop', 'three_loop')):
            source_path = Path(f'rows{rows}/{source_variant}/source/kernel.cce')
            original = (experiment / source_path).read_bytes()
            body = original.decode('utf-8')
            expected_bars = 0 if variant == 'no_membar' else 2
            if f'constexpr int rows = {rows};' not in body or body.count('mem_bar(') != expected_bars:
                raise ValueError(f'历史 VF 的 shape 或同步数量不符：{source_path}')
            scalar_bytes = rows * 8 if variant == 'three_loop' else 0
            name = f'softmax_{rows}x{COLS}_a6_{variant}'
            destination = target / f'rows{rows}/{variant}.cce'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(render_kernel(name, body, rows, scalar_bytes), encoding='utf-8')
            input_bytes, output_bytes = rows * COLS * 4, rows * COLS * 2
            output_base = input_bytes * 2 + scalar_bytes
            offsets_base = output_base + output_bytes
            records.append({
                'shape': [rows, COLS], 'variant': variant,
                'kernel': destination.relative_to(target).as_posix(), 'kernel_name': name,
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'vf_source': '../' + source_path.as_posix(),
                'vf_source_sha256': hashlib.sha256(original).hexdigest(),
                'static_membar_count': expected_bars, 'block_dim': 1,
                'ub_regions': [
                    {'name': 'memA', 'start': 0, 'end': input_bytes},
                    {'name': 'memB_exp', 'start': input_bytes, 'end': 2 * input_bytes},
                    {'name': 'memB_scalars', 'start': 2 * input_bytes, 'end': output_base},
                    {'name': 'memOutput', 'start': output_base, 'end': offsets_base},
                    {'name': 'offsets', 'start': offsets_base, 'end': offsets_base + 32},
                ],
                'ub_reserved_bytes': offsets_base + 32,
            })
    (target / 'cases.json').write_text(json.dumps({
        'ub_budget_bytes': UB_BUDGET,
        'note': '仅完整 kernel 源码与编译验证；没有 A6 CAModel 数值通过结论。32 行不在本次范围内。',
        'cases': records,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'已生成 {len(records)} 份完整 CCE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    build(parser.parse_args().archive_root)
