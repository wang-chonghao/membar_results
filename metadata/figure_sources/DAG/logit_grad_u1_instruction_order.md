# Logit Grad U1 指令顺序 DAG

基于纯 CCE 复测的实际编译日志及 Canonical 操作数重建；横轴是程序顺序，不是执行 cycle。
首轮全部 22 条指令：2 条输入 VLD、6 条谓词生成/计算、11 条向量计算、1 条输出 VST、2 条标量辅助。循环执行 16 次；无 spill、无 Membar。
紫线为谓词 RAW，灰线为向量 RAW。循环外的 PSET、4 条 VDUP 常量及其输入依赖保存在 JSON 中，并在图片底部列出。
既有测量：CAModel 226、VfSim 全局 240 cycles。本次没有重跑预测。

| 范围 | PNG | SVG |
| --- | --- | --- |
| 首轮 1–22 | [查看](../../../figure/DAG/logit_grad_u1_instruction_order.png) | [查看](../../../figure/DAG/logit_grad_u1_instruction_order.svg) |

[编号、PC、操作数与字节地址](logit_grad_u1_instruction_order.csv) · [依赖及来源校验值](logit_grad_u1_instruction_order.json)
