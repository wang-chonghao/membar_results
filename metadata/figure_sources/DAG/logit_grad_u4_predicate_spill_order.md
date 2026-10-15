# Logit Grad U4 Predicate Spill DAG

依据纯 CCE 复测的实际编译日志、Canonical 操作数及动态地址核验记录绘制；不是源码级无 spill 示意图。
首组 144 条指令：8 输入 VLD、24 谓词生成/计算、13 PSTS、14 PLDS、44 向量计算、4 输出 VST、27 Membar、10 标量辅助。重复四次，总计 52 PSTS、56 PLDS、108 Membar。
保留完整首组的指令顺序；四路展开不是四张独立图。细节图按连续 48 条指令分页，跨段输入在图下方给出。
紫线为谓词 RAW，灰线为向量 RAW，绿虚线为 spill UB RAW。红菱形保留全部屏障节点，屏障扇出不叠加于图片，避免掩盖数据依赖。
图中节点是架构寄存器动态读写及 UB 保存/恢复，不是物理寄存器分配图，也不是执行时间轴。
循环外 PSET、VDUP 常量和标量初始化不包含在 144 条循环体内，相关输入依赖及完整 590 条 RV 记录保存在 JSON 中。
本次未重新预测周期。既有测量：CAModel 2025、VfSim 全局 2037、局部 393 cycles。

| 范围 | PNG | SVG |
| --- | --- | --- |
| 总览 1–144 | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order.png) | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order.svg) |
| 细节 1–48 | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order_part01.png) | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order_part01.svg) |
| 细节 49–96 | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order_part02.png) | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order_part02.svg) |
| 细节 97–144 | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order_part03.png) | [查看](../../../figure/DAG/logit_grad_u4_predicate_spill_order_part03.svg) |

[编号、PC、操作数与字节地址](logit_grad_u4_predicate_spill_order.csv) · [依赖及来源校验值](logit_grad_u4_predicate_spill_order.json)
