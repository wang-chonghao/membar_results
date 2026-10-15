# U1 指令顺序图

[PNG](../../../figure/DAG/u1_instruction_order.png) · [SVG](../../../figure/DAG/u1_instruction_order.svg) · [PDF](u1_instruction_order.pdf)

横轴为第一轮编译后循环体的程序顺序编号，不是发射顺序、硬件 ID 或 cycle。节点不按依赖重新排列。

左侧按指令类别分泳道，圆点内是编号；真实 PC、动态 ID、寄存器读写及 UB 地址见 [完整对照表](u1_instruction_order.csv)。

1–24：24 条输入 VLDS；25–36：12 条 VMUL；37–47：11 条 VADD；48：VCADD；49：VSTUS。

图中保留循环体内全部 49 条指令和 48 条向量 RAW 依赖，连线用曲线。循环实际重复 8 次。

循环外 PSET、地址初始化、最终 VSTAS 及其依赖未展示在本图；完整记录保存在 [原始图数据](u1_graph.json)。
