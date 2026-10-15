# U2 含 Spill 指令顺序图

[总览 PNG](../../../figure/DAG/u2_instruction_order.png) · [总览 PDF](u2_instruction_order.pdf)

沿用 U1 的泳道形式，横轴严格保持首轮双行展开组的程序顺序，不按依赖重新排序，也不是执行 cycle。
首组共有 230 条真实指令，重复 4 次；全部 48 条标量辅助/保存恢复及 42 条 Membar 均保留，不折叠计数。
每组有 48 条输入 load、21 次向量 spill store、21 次 reload、24 VMUL、22 VADD、2 VCADD、2 VSTUS。
黄色为输入/reload，绿色为保存/输出，蓝色为计算，红色菱形为 Membar，灰色为标量辅助。
曲线仅绘制向量寄存器 RAW 和 spill UB RAW；没有把所有屏障扇出、标量、谓词依赖叠在图中。完整依赖见 [原始图数据](u2_concentrated_spill_graph.json)。
总览保留全部节点但不在小圆内挤入编号；细节图每段 46 条，节点写全局连续编号，没有重新从 1 编号。
跨段依赖在目标所在图下方列出“目标←来源”；跨段输出可在后续目标段找到，对照 [JSON](u2_instruction_order.json) 保留全部边。
循环外 PSET、地址初始化及最后 VSTAS 不包含在 230 条循环体中，完整记录保存在原始图数据中。
每张图均为 2560×1440（16:9），另附 SVG/PDF。原始 PC、动态 ID、读写寄存器和地址见 [指令对照表](u2_instruction_order.csv)。

| 分段 | 指令编号 | PNG | PDF |
| --- | --- | --- | --- |
| 1 | 1–46 | [查看](../../../figure/DAG/u2_instruction_order_part01.png) | [下载](u2_instruction_order_part01.pdf) |
| 2 | 47–92 | [查看](../../../figure/DAG/u2_instruction_order_part02.png) | [下载](u2_instruction_order_part02.pdf) |
| 3 | 93–138 | [查看](../../../figure/DAG/u2_instruction_order_part03.png) | [下载](u2_instruction_order_part03.pdf) |
| 4 | 139–184 | [查看](../../../figure/DAG/u2_instruction_order_part04.png) | [下载](u2_instruction_order_part04.pdf) |
| 5 | 185–230 | [查看](../../../figure/DAG/u2_instruction_order_part05.png) | [下载](u2_instruction_order_part05.pdf) |
