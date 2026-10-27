"""绘制 giant-loop 的 128 元素分段交错示意，放大图对应实际 CCE 指令顺序。"""

import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'a6_softmax_fp16_row_scaling'
SOURCE = EXPERIMENT / 'runnable_a6/rows8/no_membar.cce'
OUT = EXPERIMENT / 'plots'
STAGES = {
    'M': ('MAX · 第3行', '#d5e7d5', '#35733e'),
    'S': ('EXP/SUM · 第2行', '#dce9fb', '#315b97'),
    'N': ('NORM · 第0行', '#fce4d5', '#9b5038'),
}
OP_COLORS = {'load': '#f6d844', 'compute': '#65a3e8', 'store': '#21bd64'}


def setup_font():
    for name in ('msyh.ttc', 'msyhbd.ttc'):
        path = Path('/mnt/c/Windows/Fonts') / name
        if path.is_file():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                         'svg.fonttype': 'path'})


def arrow(ax, start, end, color='#616971', width=1.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
                                linewidth=width, color=color))


def bracket(ax, left, right, y, label, color, above=True):
    ax.plot([left, left, right, right], [y-0.1, y, y, y-0.1], color=color, linewidth=1.2)
    ax.text((left+right)/2, y+(0.13 if above else -0.28), label,
            fontsize=12, color=color, ha='center', va='bottom' if above else 'top')


def cell(ax, x, y, stage, segment):
    _, fill, edge = STAGES[stage]
    ax.add_patch(Rectangle((x, y), 0.99, 0.61, facecolor=fill, edgecolor=edge, linewidth=0.9))
    label = 'I0' if stage == 'M' and segment == 0 else f'{stage}{segment}'
    ax.text(x+0.495, y+0.305, label, fontsize=12, fontweight='bold',
            ha='center', va='center', color='#18242e')


def source_ops():
    text = SOURCE.read_text(encoding='utf-8')
    steady = text.index('// Steady state:')
    first_start = text.index('vlds(max_acc0, memA, pipe_i * cols, NORM);', steady)
    first_stop = text.index('#pragma unroll(1)', first_start)
    first_ops = re.findall(r'^\s*(v\w+)\(', text[first_start:first_stop], flags=re.MULTILINE)
    assert first_ops == ['vlds', 'vlds',
                         'vlds', 'vlds', 'vexpdif', 'vexpdif', 'vadd', 'vsts', 'vsts',
                         'vlds', 'vlds', 'vmul', 'vmul', 'vcvt', 'vsts', 'vcvt', 'vsts']
    assert next(op for op in first_ops if op not in ('vlds', 'vsts')) == 'vexpdif'
    start = text.index('int max_offset = pipe_i * cols + 128 * pair;')
    stop = text.index('\n            }', start)
    body = text[start:stop]
    stages = [body[:body.index('int sum_offset')],
              body[body.index('int sum_offset'):body.index('int norm_offset')],
              body[body.index('int norm_offset'):]]
    ops = [re.findall(r'^\s*(v\w+)\(', stage, flags=re.MULTILINE) for stage in stages]
    expected = [
        ['vlds', 'vlds', 'vmax', 'vmax'],
        ['vlds', 'vlds', 'vexpdif', 'vexpdif', 'vadd', 'vadd', 'vsts', 'vsts'],
        ['vlds', 'vlds', 'vmul', 'vmul', 'vcvt', 'vsts', 'vcvt', 'vsts'],
    ]
    assert ops == expected, ops
    return ops, first_ops


def main():
    setup_font()
    verified_ops, first_ops = source_ops()
    fig, ax = plt.subplots(figsize=(26, 10.5))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98)
    ax.set(xlim=(-1.0, 28), ylim=(-0.1, 11.5))
    ax.axis('off')
    ax.text(13.5, 11.2, 'Softmax giant-loop：128 个元素一段，三个阶段交错排列',
            fontsize=23, fontweight='bold', ha='center')
    ax.text(13.5, 10.7, '稳态示例：MAX(第3行) + EXP/SUM(第2行) + NORM(第0行)；行号从0开始',
            fontsize=14, ha='center', color='#535d67')
    for stage, x in (('M', 2.3), ('S', 10.4), ('N', 19.0)):
        name, fill, edge = STAGES[stage]
        ax.add_patch(Rectangle((x, 9.95), 0.42, 0.35, facecolor=fill, edgecolor=edge))
        ax.text(x+0.62, 10.12, f'{stage} = {name}', fontsize=14, va='center')

    ax.text(-0.6, 9.45, '① 按阶段集中安排（对照示意）', fontsize=16, fontweight='bold')
    base, step = 0.25, 1.07
    for phase, stage in enumerate(('M', 'S', 'N')):
        for segment in range(8):
            cell(ax, base+(phase*8+segment)*step, 8.12, stage, segment)
        bracket(ax, base+phase*8*step, base+(phase*8+7)*step+0.99, 8.91,
                f'{STAGES[stage][0]}：先连续处理整行8段', STAGES[stage][2])
    ax.text(13.2, 7.75, 'I0 初始化 MAX → M1 → … → M7，然后 S0 → … → S7，最后 N0 → … → N7',
            fontsize=12, color='#535d67', ha='center')

    ax.text(-0.6, 7.38, '② 当前 giant-loop：第0段先初始化；第1～7段按 M、S、N 交错', fontsize=16, fontweight='bold')
    for segment in range(8):
        for phase, stage in enumerate(('M', 'S', 'N')):
            cell(ax, base+(segment*3+phase)*step, 5.99, stage, segment)
        left = base+segment*3*step
        right = left+2*step+0.99
        bracket(ax, left, right, 6.78, f'第{segment}段', '#5b6570')
        ax.text((left+right)/2, 5.77, f'元素 {segment*128}～{segment*128+127}',
                fontsize=10.5, color='#59636d', ha='center', va='top')
    arrow(ax, (0.25, 5.25), (26.0, 5.25), color='#68727d')
    ax.text(13.2, 5.06, '从左到右是源码指令顺序；每格是一组指令，不代表相同周期或相同指令数',
            fontsize=12, ha='center', va='top', color='#535d67')

    ax.text(-0.6, 4.25, '③ 放大第1段（元素128～255）：实际 load / compute / store 排列',
            fontsize=16, fontweight='bold')
    labels = [
        ('VLDS', '前64'), ('VLDS', '后64'), ('VMAX', 'acc0'), ('VMAX', 'acc1'),
        ('VLDS', '前64'), ('VLDS', '后64'), ('VEXPDIF', 'exp0'), ('VEXPDIF', 'exp1'),
        ('VADD', '段内合并'), ('VADD', '累加sum'), ('VSTS', 'exp0'), ('VSTS', 'exp1'),
        ('VLDS', 'exp0'), ('VLDS', 'exp1'), ('VMUL', '归一化0'), ('VMUL', '归一化1'),
        ('VCVT', 'FP16'), ('VSTS', '输出0'), ('VCVT', 'FP16'), ('VSTS', '输出1'),
    ]
    assert [op.lower() for op, _ in labels] == sum(verified_ops, [])
    xpos = [0.25+1.34*i for i in range(len(labels))]
    y, radius = 2.55, 0.47
    for i, (op, value) in enumerate(labels):
        kind = 'load' if op == 'VLDS' else 'store' if op == 'VSTS' else 'compute'
        ax.add_patch(Circle((xpos[i], y), radius, facecolor=OP_COLORS[kind], edgecolor='#536371', linewidth=1.15))
        ax.text(xpos[i], y, f'{op}\n{value}', fontsize=9 if op == 'VEXPDIF' else 10,
                fontweight='bold', ha='center', va='center')
        if i < len(labels)-1:
            arrow(ax, (xpos[i]+radius+0.03, y), (xpos[i+1]-radius-0.03, y))
    for left, right, stage in ((0, 3, 'M'), (4, 11, 'S'), (12, 19, 'N')):
        bracket(ax, xpos[left]-radius, xpos[right]+radius, 3.30,
                f'{stage}1：{STAGES[stage][0]}', STAGES[stage][2])
    ax.text((xpos[0]+xpos[3])/2, 1.73, '读取原始输入 → 更新 MAX 累加器', ha='center', fontsize=11)
    ax.text((xpos[4]+xpos[11])/2, 1.73, '读取输入 → exp 与 sum 累加 → exp 写回 UB', ha='center', fontsize=11)
    ax.text((xpos[12]+xpos[19])/2, 1.73, '读取旧行 exp → 乘倒数 → 转 FP16 → 写输出', ha='center', fontsize=11)
    ax.text(13.3, 1.13, '每段128个元素 = 两组64个 FP32；MAX/SUM 仍对整行1024个元素归约，不是每段单独 softmax。',
            ha='center', fontsize=13, fontweight='bold')
    ax.text(13.3, 0.57, '圆节点：黄色 load · 蓝色 compute · 绿色 store；箭头仅表示源码排列，不是数据依赖边，也不是硬件周期。',
            ha='center', fontsize=11, color='#626b75')
    ax.text(13.3, 0.08, 'I0 只含两条 VLDS，不含 VMAX，故首个计算为 VEXPDIF；放大图仅对应 pair=1。省略流水启动/排空与行末归约。',
            ha='center', fontsize=10.5, color='#7a828a')
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'giant_loop_interleaving.{suffix}', dpi=160, facecolor='white')
    plt.close(fig)
    (OUT / 'giant_loop_interleaving.json').write_text(json.dumps({
        'source': SOURCE.relative_to(ROOT).as_posix(),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'steady_state_pipe_i': 3, 'row_mapping': {'MAX': 3, 'EXP_SUM': 2, 'NORM': 0},
        'columns': 1024, 'segment_elements': 128, 'vector_elements': 64,
        'zoom_pair': 1, 'zoom_element_range': [128, 256],
        'zoom_stage_opcodes': verified_ops,
        'first_segment_opcodes': first_ops,
        'first_compute_opcode': 'vexpdif',
        'note': '源码顺序示意，非硬件时序图；I0 是 VLDS 初始化，不含 VMAX。省略流水启动排空和行末归约。',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Generated giant_loop_interleaving.png/.svg; verified first segment (17 ops) and pair=1 (20 ops).')


if __name__ == '__main__':
    main()
