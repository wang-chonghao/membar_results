# Bank-aware U2 指令顺序 DAG

根据同版本 CAModel 日志按动态 ID、PC 重建，未按依赖重新排序。保留首组全部 98 条指令和 96 条向量 RAW；两行各 49 条，硬件重复四次。
行号由实际 UB 地址和向量数据流推导；PNG/SVG 在 figure/DAG，PDF 和编号对照在本目录。
循环外 PSET、标量准备和最终 VSTAS 不在顺序图中，完整动态依赖保存在 [原始图数据](u2_bank_aware_graph.json)。
实测 228 cycles，VfSim 全局预测 204 cycles；数据来自既有验证记录，本次未重新仿真。

| 范围 | PNG | SVG |
| --- | --- | --- |
| 总览 1–98 | [查看](../../../figure/DAG/u2_bank_aware_instruction_order.png) | [查看](../../../figure/DAG/u2_bank_aware_instruction_order.svg) |
| 前半 1–49 | [查看](../../../figure/DAG/u2_bank_aware_instruction_order_part01.png) | [查看](../../../figure/DAG/u2_bank_aware_instruction_order_part01.svg) |
| 后半 50–98 | [查看](../../../figure/DAG/u2_bank_aware_instruction_order_part02.png) | [查看](../../../figure/DAG/u2_bank_aware_instruction_order_part02.svg) |

[指令编号与 PC](u2_bank_aware_instruction_order.csv) · [分段依赖](u2_bank_aware_instruction_order.json)
